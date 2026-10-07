"""Aggregator logic: given all sites' payloads for this round, produce the reply.

The aggregator keeps the boosting state between rounds: the agreed bin edges, the init score, the
finished trees and the tree being grown. Sites only ever get split decisions and leaf values back, and,
in the reply that ends training, the finished model and the training report.
"""

import json
from dataclasses import asdict

import numpy as np

from fl_lightgbm.binning import BinMapper, Categories, agree_bin_mapper, agree_categorical_bin_mapper, grid
from fl_lightgbm.encoding import SPARSE, Encoding
from fl_lightgbm.grower import TreeGrower
from fl_lightgbm.histogram import Histogram, feature_offsets
from fl_lightgbm.central import central_baseline_params
from fl_lightgbm.model import Tree, write_model
from fl_lightgbm.objective import METRIC, init_score, label_weights
from fl_lightgbm.params import Params
from fl_lightgbm.rounds import RoundCounter


class SchemaError(ValueError):
    """Sites hold different columns, or different columns are categorical."""


class Aggregator:
    def __init__(self, params: Params, bounds: list[tuple[float, float]] | None = None,
                 encoding: Encoding = SPARSE):
        """`bounds`: per-feature (lo, hi) from the config, in place of the sites' min/max.
        `encoding`: how the sites send grid counts and histograms."""
        self.params = params
        self.bounds = bounds
        self.encoding = encoding
        self.rounds = RoundCounter(params.round_budget, "aggregator")
        self._round = 0
        self.trees: list[Tree] = []
        self.delivery_round: int | None = None  # the round whose reply delivered the model
        self.training_loss: list[float] = []  # after each tree
        self._grower: TreeGrower | None = None

    @property
    def rounds_left(self) -> int:
        return self.rounds.left

    @property
    def padding_rounds(self) -> int:
        """The rounds after the one whose reply delivered the model; at least one (ADR 0005)."""
        assert self.delivery_round is not None
        return self.params.round_budget - self.delivery_round

    def reply(self, payloads: list[dict]) -> dict:
        """`payloads`: one per site; from setup round 2 on, under secure aggregation, their sum as one
        payload. The aggregator only ever adds the payloads up, so it decodes either alike."""
        self.rounds.take()
        self._round += 1
        if self._round == 1:
            return self._setup(payloads)
        if self._round == 2:
            self._agree_bin_edges(payloads)
            self._grower = TreeGrower(self.bin_mappers, self.params) if self.params.num_iterations else None
            return {
                "bin_upper_bounds": [m.upper_bounds.tolist() for m in self.bin_mappers],
                "missing_types": [m.missing_type for m in self.bin_mappers],
                "most_freq_bins": [m.most_freq_bin for m in self.bin_mappers],  # the bins sites may leave out
                "bin_categories": [m.categories.tolist() if m.categories is not None else None
                                   for m in self.bin_mappers],
                **self._split_reply([], None),
            }
        self._take_tree_totals(payloads)
        if self._grower is None:
            return self._split_reply([], None)
        return self._grow(self._grower, payloads)

    def _setup(self, payloads: list[dict]) -> dict:
        self.num_sites = len(payloads)
        self.feature_names = payloads[0]["columns"]
        for i, p in enumerate(payloads):
            if p["columns"] != self.feature_names:
                raise SchemaError(f"site {i} has columns {p['columns']}, site 0 has {self.feature_names}")
            if [c is None for c in p["categories"]] != [c is None for c in payloads[0]["categories"]]:
                raise SchemaError(f"site {i} has other categorical columns than site 0")
        # The category sets merged, so that every site can bin every category, even one only another site has.
        categories = [None if c is None else sorted(set().union(*(p["categories"][f] for p in payloads)))
                      for f, c in enumerate(payloads[0]["categories"])]
        if self.bounds is None:
            lo = np.fmin.reduce([p["min"] for p in payloads], axis=0)  # a site's NaN: no value in that column
            hi = np.fmax.reduce([p["max"] for p in payloads], axis=0)
            self.feature_ranges = list(zip(lo.tolist(), hi.tolist()))
        elif len(self.bounds) != len(self.feature_names):
            raise ValueError(f"bounds are configured for {len(self.bounds)} features, the sites have "
                             f"{len(self.feature_names)}")
        else:
            self.feature_ranges = [(float(lo), float(hi)) for lo, hi in self.bounds]
        self.grids = [grid(lo, hi) if c is None else Categories(np.array(c, dtype=np.int64))
                      for (lo, hi), c in zip(self.feature_ranges, categories)]
        self.num_data = sum(p["n"] for p in payloads)
        sum_label = sum(p["sum_label"] for p in payloads)
        self.init_score = init_score(sum_label, self.num_data, self.params)
        reply = {
            "bounds": [list(r) for r in self.feature_ranges],
            "categories": categories,
            "init_score": self.init_score,
        }
        if self.params.objective == "binary":
            reply["label_weights"] = list(label_weights(int(sum_label), self.num_data, self.params))
        return reply

    def _agree_bin_edges(self, payloads: list[dict]) -> None:
        """Merge the sites' grid, category and missing-value counts and fix the agreed bin edges."""
        offsets = feature_offsets(np.array([len(g.values) for g in self.grids]))
        counts = np.sum([self.encoding.decode_counts(p["grid_counts"], offsets[-1]) for p in payloads], axis=0)
        na_counts = np.sum([p["na_counts"] for p in payloads], axis=0)
        self.bin_mappers: list[BinMapper] = []
        for f, g in enumerate(self.grids):
            cells, na_count = counts[offsets[f]:offsets[f + 1]], int(na_counts[f])
            if isinstance(g, Categories):
                self.bin_mappers.append(agree_categorical_bin_mapper(g, cells, na_count, self.num_data, self.params))
            else:
                self.bin_mappers.append(agree_bin_mapper(g, cells, na_count, self.num_data, self.params))

    def _take_tree_totals(self, payloads: list[dict]) -> None:
        """Put the true row counts into the tree finished last round, and record the training loss after it.

        The children of a tree's last split only have LightGBM's hessian-based estimates until now,
        and for logloss these differ from the true counts serial LightGBM writes.
        """
        if "leaf_counts" not in payloads[0]:
            return
        self.trees[-1].leaf_count = np.sum([p["leaf_counts"] for p in payloads], axis=0).tolist()
        self.training_loss.append(sum(self.encoding.decode_sum(p["loss_sum"]) for p in payloads) / self.num_data)

    def _grow(self, grower: TreeGrower, payloads: list[dict]) -> dict:
        (leaf,) = grower.request  # v1 asks for one leaf per round
        site_leaves = [p["leaves"][0] for p in payloads]
        site_hists = [self.encoding.decode_histogram(s["histogram"], grower.num_bins.sum()) for s in site_leaves]
        hist = Histogram(
            grad=np.sum([h.grad for h in site_hists], axis=0),
            hess=np.sum([h.hess for h in site_hists], axis=0),
            count=sum(h.count for h in site_hists),
        )
        sum_grad = sum(self.encoding.decode_sum(s["sum_grad"]) for s in site_leaves)
        sum_hess = sum(self.encoding.decode_sum(s["sum_hess"]) for s in site_leaves)
        split = grower.grow(leaf, hist, sum_grad, sum_hess)
        splits = [split] if split else []
        if not grower.done:
            return self._split_reply(splits, None)
        return self._split_reply(splits, self._finish_tree(grower))

    def _finish_tree(self, grower: TreeGrower) -> list[float] | None:
        """Close the tree as GBDT::TrainOneIter does; return the leaf values the sites add to their scores."""
        if grower.num_leaves == 1:
            # LightGBM stops training. Only a first tree is kept, as a constant tree of the init score. The
            # sites' scores hold the init score already; they add 0 and send the loss in the closing round.
            self._grower = None
            if not self.trees:
                self.trees.append(Tree(leaf_value=[self.init_score], leaf_count=[self.num_data]))
                return [0.0]
            return None
        tree = grower.finished_tree()
        tree.shrink(self.params.learning_rate)
        leaf_values = list(tree.leaf_value)
        if not self.trees and self.init_score != 0.0:
            tree.add_bias(self.init_score)
        self.trees.append(tree)
        trees_left = len(self.trees) < self.params.num_iterations
        self._grower = TreeGrower(self.bin_mappers, self.params) if trees_left else None
        return leaf_values

    def _split_reply(self, splits: list[dict], leaf_values: list[float] | None) -> dict:
        """The split just applied, the leaf values of a finished tree and the leaves to histogram next;
        once training has ended, also the model and the training report, delivered this once."""
        reply: dict = {
            "splits": splits,
            "leaf_values": leaf_values,
            "histogram_leaves": list(self._grower.request) if self._grower else [],
        }
        # Training has ended once no tree is growing and the sites owe no tree totals; the model is final.
        if self._grower is None and leaf_values is None and self.delivery_round is None:
            self.delivery_round = self._round
            reply |= {"model": self.model(), "report": self.report()}
        return reply

    def model(self) -> str:
        ranges = [None if m.is_trivial else (m.min_val, m.max_val) for m in self.bin_mappers]
        return write_model(self.trees, self.feature_names, ranges, self.params.objective,
                           [m.categories for m in self.bin_mappers])

    def report(self) -> str:
        """The training report as JSON: the parameters, the number of sites, the rounds used and the padding
        rounds, the training loss after every tree, and the further LightGBM parameters with which the
        forced-bins file trains the central baseline."""
        return json.dumps({
            "params": asdict(self.params),
            "num_sites": self.num_sites,
            "round_budget": self.params.round_budget,
            "rounds_used": self.delivery_round,
            "padding_rounds": self.padding_rounds,
            "metric": METRIC[self.params.objective],
            "training_loss": self.training_loss,
            "central_baseline": central_baseline_params(self.bin_mappers, self.params.max_bin, self.num_data),
        }, indent=2)
