"""Site logic: given the aggregator's previous reply, produce this round's payload.

The first round is setup round 1 (schema, min/max unless bounds are configured, category sets, row count,
Σlabel). The second is setup round 2: the site counts its rows on each feature's grid between the global
bounds, or per category of the merged category sets, and its missing values, from which the aggregator
agrees the bin edges. The third is setup round 3: the site bins its data with the agreed bin edges and
counts its rows per bin of each numerical feature, from which the aggregator fixes the most frequent bins.
Every later round is a split round, the closing round or a padding round. The round after a tree is
complete carries that tree's per-leaf row counts and the site's training loss after it. The reply that
ends training delivers the finished model and the training report, which the site keeps.
"""

from collections.abc import Sequence

import numpy as np

from fl_lightgbm.binning import BinMapper, Categories, bin_values, category_codes, grid, missing_bin
from fl_lightgbm.encoding import SPARSE, Encoding
from fl_lightgbm.histogram import leaf_histogram, most_freq_bin_positions
from fl_lightgbm.objective import gradients, label_sum, loss_sum
from fl_lightgbm.params import Params
from fl_lightgbm.rounds import RoundCounter


class Site:
    def __init__(self, X: np.ndarray, y: np.ndarray, params: Params, feature_names: list[str],
                 bounds: list[tuple[float, float]] | None = None, encoding: Encoding = SPARSE,
                 categorical: Sequence[int] = ()):
        """`bounds`: per-feature (lo, hi) from the config, in place of sharing the site's min/max.
        `encoding`: how grid counts and histograms are sent; the aggregator must hold the same.
        `categorical`: the indices of the categorical features."""
        self.X = np.asarray(X, dtype=np.float64)
        self.labels = np.asarray(y, dtype=np.float32)  # LightGBM keeps labels as float32
        self.params = params
        self.feature_names = list(feature_names)
        self.bounds = bounds
        self.encoding = encoding
        self.categorical = set(categorical)
        for name, x, (lo, hi) in zip(self.feature_names, self.X.T, bounds or []):
            if np.any((x < lo) | (x > hi)):
                raise ValueError(f"{name} has values outside its configured bounds [{lo}, {hi}]")
        self.rounds = RoundCounter(params.round_budget, "site")
        self._round = 0
        self.model: str | None = None  # LightGBM text model, once delivered
        self.report: str | None = None  # JSON training report, once delivered

    @property
    def rounds_left(self) -> int:
        return self.rounds.left

    def payload(self, reply: dict | None) -> dict:
        self.rounds.take()
        self._round += 1
        if self._round == 1:
            return self._setup_payload()
        assert reply is not None
        if self._round == 2:
            self.score = np.full(len(self.labels), reply["init_score"])
            self.label_weights = tuple(reply.get("label_weights", (1.0, 1.0)))
            return self._grid_payload(reply["bounds"], reply["categories"])
        if self._round == 3:
            return self._bin_payload(reply)
        if self._round == 4:
            self._agree(reply)
        return self._split_round(reply)

    def _setup_payload(self) -> dict:
        payload = {
            "columns": self.feature_names,
            "n": len(self.labels),
            "sum_label": label_sum(self.labels, self.params),
            # Per feature: its categories, or None for a numerical feature.
            "categories": [np.unique(codes[codes >= 0]).tolist() if codes is not None else None
                           for codes in self._category_codes()],
        }
        if self.bounds is None:
            payload["min"] = np.fmin.reduce(self.X, axis=0).tolist()  # NaN only if the whole column is
            payload["max"] = np.fmax.reduce(self.X, axis=0).tolist()
        return payload

    def _category_codes(self) -> list[np.ndarray | None]:
        return [category_codes(x, self.params) if f in self.categorical else None for f, x in enumerate(self.X.T)]

    def _grid_payload(self, bounds: list[list[float]], categories: list[list[int] | None]) -> dict:
        """Rows per grid cell, or per category of a categorical feature, all features in one flat array,
        and missing values per feature (for a categorical feature, the rows of its NaN bin)."""
        counts, na_counts = [], []
        for (lo, hi), merged, codes, x in zip(bounds, categories, self._category_codes(), self.X.T):
            if codes is None:
                counts.append(grid(lo, hi).counts(x))
                na_counts.append(int(np.isnan(x).sum()))
            else:
                counts.append(Categories(np.array(merged, dtype=np.int64)).counts(codes))
                na_counts.append(int(np.sum(codes < 0)))
        return {
            "grid_counts": self.encoding.encode_counts(np.concatenate(counts)),
            "na_counts": na_counts,
        }

    def _bin_payload(self, reply: dict) -> dict:
        """Bin the data with the agreed bin edges; the rows per bin of each numerical feature, all in one flat
        array."""
        # The most frequent bin, which binning does not use, comes with the next reply.
        mappers = [BinMapper(np.asarray(bounds), missing_type, 0,
                             None if categories is None else np.array(categories, dtype=np.int64))
                   for bounds, missing_type, categories in zip(
                       reply["bin_upper_bounds"], reply["missing_types"], reply["bin_categories"])]
        self.num_bins = np.array([m.num_bins for m in mappers])
        self.binned = bin_values(self.X, mappers)
        self.missing_bins = np.array([-1 if m.is_categorical else missing_bin(m.upper_bounds, m.missing_type)
                                      for m in mappers])
        counts = [m.counts(x) for m, x in zip(mappers, self.X.T) if not m.is_categorical]
        return {"bin_counts": self.encoding.encode_counts(np.concatenate([np.zeros(0, dtype=np.int64), *counts]))}

    def _agree(self, reply: dict) -> None:
        self.most_freq_bins = most_freq_bin_positions(self.num_bins, np.array(reply["most_freq_bins"]))
        self.leaf = np.zeros(len(self.labels), dtype=np.int32)
        self._compute_gradients()

    def _compute_gradients(self) -> None:
        self.gradients, self.hessians = gradients(self.labels, self.score, self.params, self.label_weights)

    def _split_round(self, reply: dict) -> dict:
        if "model" in reply:
            self.model, self.report = reply["model"], reply["report"]
        for s in reply["splits"]:
            bins = self.binned[:, s["feature"]]
            if "left_bins" in s:  # categorical
                goes_right = ~np.isin(bins, s["left_bins"])
            else:
                missing = bins == self.missing_bins[s["feature"]]
                goes_right = np.where(missing, not s["default_left"], bins > s["threshold_bin"])
            self.leaf[(self.leaf == s["leaf"]) & goes_right] = s["right_leaf"]
        payload: dict = {}
        if reply["leaf_values"] is not None:
            leaf_values = np.asarray(reply["leaf_values"])
            payload["leaf_counts"] = np.bincount(self.leaf, minlength=len(leaf_values)).tolist()
            self.score += leaf_values[self.leaf]
            payload["loss_sum"] = self.encoding.encode_sum(loss_sum(self.labels, self.score, self.params))
            self.leaf[:] = 0
            self._compute_gradients()
        if reply["histogram_leaves"]:
            payload["leaves"] = [self._leaf_payload(leaf) for leaf in reply["histogram_leaves"]]
        return payload  # empty in a padding round

    def _leaf_payload(self, leaf: int) -> dict:
        rows = np.flatnonzero(self.leaf == leaf)
        hist = leaf_histogram(self.binned, self.num_bins, self.gradients, self.hessians, rows)
        return {  # no leaf id: the aggregator requested the leaf, and secure aggregation would sum it
            "histogram": self.encoding.encode_histogram(hist, self.most_freq_bins),
            "sum_grad": self.encoding.encode_sum(float(np.sum(self.gradients[rows].astype(np.float64)))),
            "sum_hess": self.encoding.encode_sum(float(np.sum(self.hessians[rows].astype(np.float64)))),
        }
