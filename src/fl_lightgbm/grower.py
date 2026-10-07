"""Leaf-wise growth of one tree on the aggregator, as SerialTreeLearner::Train does it.

Each call to `grow` takes the summed histogram of the leaf asked for last time, derives its sibling
by subtraction, finds the best split of both, and applies the best split over all leaves. Afterwards
`request` names the leaf whose histogram is needed next; it is empty once the tree is done.
Leaf numbering follows Tree::Split: the left child keeps the parent's leaf index, the right child
gets the next free one, and the new internal node gets index `num_leaves - 1`.
"""

import numpy as np

from fl_lightgbm.binning import BinMapper
from fl_lightgbm.histogram import Histogram, fix_histogram
from fl_lightgbm.model import Tree, categorical_decision_type, decision_type
from fl_lightgbm.params import Params
from fl_lightgbm.split import SplitInfo, find_best_split, leaf_output


class TreeGrower:
    def __init__(self, bin_mappers: list[BinMapper], params: Params):
        self.bin_mappers = bin_mappers
        self.num_bins = np.array([mapper.num_bins for mapper in bin_mappers])
        self.most_freq_bins = np.array([mapper.most_freq_bin for mapper in bin_mappers])
        self.params = params
        self.request = [0]
        self.done = False
        # per leaf
        self.sum_grad: list[float] = [0.0]
        self.sum_hess: list[float] = [0.0]
        self.count: list[int] = [0]
        self.output: list[float] = [0.0]
        self.parent: list[int] = [-1]
        self.depth: list[int] = [0]
        self.best: list[SplitInfo | None] = [None]
        self.splittable: list[np.ndarray | None] = [None]
        self.hist: dict[int, Histogram] = {}
        # per internal node
        self.tree = Tree(leaf_value=[], leaf_count=[])
        self._last_split: tuple[int, int, Histogram, np.ndarray | None] | None = None

    @property
    def num_leaves(self) -> int:
        return len(self.output)

    def grow(self, leaf: int, hist: Histogram, sum_grad: float, sum_hess: float) -> dict | None:
        """Take the requested leaf's summed histogram; return the split applied, if any."""
        if self._last_split is None:
            self._grow_root(hist, sum_grad, sum_hess)
        else:
            self._grow_children(leaf, hist, *self._last_split)
        return self._split_best_leaf()

    def _grow_root(self, hist: Histogram, sum_grad: float, sum_hess: float) -> None:
        self.sum_grad[0], self.sum_hess[0], self.count[0] = sum_grad, sum_hess, hist.count
        self.output[0] = float(leaf_output(sum_grad, sum_hess, self.params))
        self._search(0, self._fixed(0, hist), candidates=None)

    def _grow_children(self, leaf: int, hist: Histogram, left: int, right: int,
                       parent_hist: Histogram, parent_splittable: np.ndarray | None) -> None:
        sibling = right if leaf == left else left
        self.count[leaf] = hist.count
        self.count[sibling] = parent_hist.count - hist.count
        # BeforeFindBestSplit: neither child is searched at `max_depth`, or if both are too small to split.
        if 0 < self.params.max_depth <= self.depth[left]:
            return
        if max(self.count[left], self.count[right]) < 2 * self.params.min_data_in_leaf:
            return
        # SerialTreeLearner builds the histogram of the child with fewer rows (the right one on equal
        # counts), fixes it, and subtracts it from the parent's for the other child. When that child is
        # the sibling, parent minus requested child stands in for its own histogram; the fixed bin is
        # rebuilt from the leaf totals anyway. The two are equal while the float64 sums of float32
        # gradients are exact, which they usually are. When they are not, the bins differ by rounding,
        # and a tie that rounding decides (such as the two scan directions finding the same partition)
        # can still go the other way than in LightGBM.
        smaller = left if self.count[left] < self.count[right] else right
        larger = right if smaller == left else left
        built = hist if smaller == leaf else Histogram(
            parent_hist.grad - hist.grad, parent_hist.hess - hist.hess, self.count[sibling])
        smaller_hist = self._fixed(smaller, built)
        larger_hist = Histogram(parent_hist.grad - smaller_hist.grad, parent_hist.hess - smaller_hist.hess,
                                self.count[larger])
        self._search(smaller, smaller_hist, parent_splittable)
        self._search(larger, larger_hist, parent_splittable)

    def _fixed(self, leaf: int, hist: Histogram) -> Histogram:
        return fix_histogram(hist, self.sum_grad[leaf], self.sum_hess[leaf], self.num_bins, self.most_freq_bins)

    def _search(self, leaf: int, hist: Histogram, candidates: np.ndarray | None) -> None:
        self.hist[leaf] = hist
        self.best[leaf], self.splittable[leaf] = find_best_split(
            hist, self.sum_grad[leaf], self.sum_hess[leaf], self.bin_mappers, self.params, candidates)

    def _split_best_leaf(self) -> dict | None:
        best_leaf, best = 0, None
        for leaf, split in enumerate(self.best):  # ArgMax: the first leaf wins ties
            if split is not None and split.beats(best):
                best_leaf, best = leaf, split
        if best is None or best.gain <= 0.0:
            self._finish()
            return None
        return self._split(best_leaf, best)

    def _split(self, leaf: int, s: SplitInfo) -> dict:
        node, right = self.num_leaves - 1, self.num_leaves
        t = self.tree
        parent = self.parent[leaf]
        if parent >= 0:
            if t.left_child[parent] == ~leaf:
                t.left_child[parent] = node
            else:
                t.right_child[parent] = node
        t.split_feature.append(s.feature)
        t.split_gain.append(float(np.float32(s.gain + self.params.min_gain_to_split)))  # as Tree::Split stores it: a float
        feature = self.bin_mappers[s.feature]
        if feature.categories is not None:  # Tree::SplitCategorical: the threshold is the index of the category set
            t.threshold.append(float(t.add_category_set(feature.categories[np.array(s.left_bins) - 1].tolist())))
            t.decision_type.append(categorical_decision_type(feature.missing_type))
        else:
            t.threshold.append(float(feature.upper_bounds[s.threshold]))
            t.decision_type.append(decision_type(s.default_left, feature.missing_type))
        t.left_child.append(~leaf)
        t.right_child.append(~right)
        t.internal_value.append(self.output[leaf])
        t.internal_weight.append(s.left_sum_hess + s.right_sum_hess)
        t.internal_count.append(self.count[leaf])

        self._last_split = (leaf, right, self.hist.pop(leaf), self.splittable[leaf])
        self.parent[leaf] = node
        self.parent.append(node)
        self.depth[leaf] += 1
        self.depth.append(self.depth[leaf])
        self.sum_grad[leaf], self.sum_hess[leaf] = s.left_sum_grad, s.left_sum_hess
        self.sum_grad.append(s.right_sum_grad)
        self.sum_hess.append(s.right_sum_hess)
        self.output[leaf] = s.left_output
        self.output.append(s.right_output)
        # Hessian-based estimates. Serial LightGBM uses the true counts here; the next histogram brings
        # them. The children of a tree's last split get theirs in the round after the tree is complete
        # (the aggregator's `_take_leaf_counts`). The child the sites histogram is still chosen by the
        # estimates; `_grow_children` then follows LightGBM's choice by the true counts.
        self.count[leaf] = s.left_count
        self.count.append(s.right_count)
        self.best[leaf] = None
        self.best.append(None)
        self.splittable.append(None)

        if self.num_leaves == self.params.num_leaves:
            self._finish()
        else:
            self.request = [leaf if s.left_count < s.right_count else right]
        if s.left_bins is not None:
            return {"leaf": leaf, "feature": s.feature, "left_bins": s.left_bins, "right_leaf": right}
        return {"leaf": leaf, "feature": s.feature, "threshold_bin": s.threshold, "default_left": s.default_left,
                "right_leaf": right}

    def _finish(self) -> None:
        self.done = True
        self.request = []

    def finished_tree(self) -> Tree:
        """The grown tree, before shrinkage."""
        t = self.tree
        t.leaf_value = list(self.output)
        t.leaf_weight = list(self.sum_hess)
        t.leaf_count = list(self.count)
        return t
