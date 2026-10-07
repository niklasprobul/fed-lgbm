"""Best split of one leaf from its aggregated histogram, with LightGBM's rules.

Mirrors FeatureHistogram::FindBestThreshold (feature_histogram.hpp) for numerical features: a
right-to-left scan that sends missing values left and, for a feature with missing values, a
left-to-right scan that sends them right; the L1/L2-regularised gain with `max_delta_step` and
`min_gain_to_split`; and the hessian-based count estimate for `min_data_in_leaf`. Categorical features
follow FindBestThresholdCategoricalInner (feature_histogram.cpp).
"""

from dataclasses import dataclass, replace

import numpy as np

from fl_lightgbm.binning import MISSING_NAN, MISSING_NONE, MISSING_ZERO, BinMapper
from fl_lightgbm.histogram import Histogram, feature_offsets, subtract_in_order
from fl_lightgbm.params import Params

# LightGBM's kEpsilon (meta.h): the float literal 1e-15f, widened to double.
EPSILON = float(np.float32(1e-15))


@dataclass
class SplitInfo:
    feature: int
    threshold: int  # bin index; rows with bin <= threshold go left
    gain: float  # less `min_gain_to_split`, as LightGBM compares gains; Tree::Split adds it back
    left_output: float
    right_output: float
    left_count: int  # estimated from hessians, as LightGBM does
    right_count: int
    left_sum_grad: float
    left_sum_hess: float
    right_sum_grad: float
    right_sum_hess: float
    default_left: bool  # where the missing values go
    left_bins: list[int] | None = None  # categorical: the bins that go left; `threshold` is then unused

    def beats(self, other: "SplitInfo | None") -> bool:
        """SplitInfo::operator>: higher gain wins, then the smaller feature index."""
        if other is None:
            return True
        if self.gain != other.gain:
            return self.gain > other.gain
        return self.feature < other.feature


def _threshold_l1(sum_grad, l1: float):
    """ThresholdL1: Σg shrunk towards 0 by `lambda_l1`; Σg itself when it is 0."""
    return np.sign(sum_grad) * np.maximum(0.0, np.abs(sum_grad) - l1)


def leaf_output(sum_grad, sum_hess, params: Params):
    """CalculateSplittedLeafOutput without smoothing: L1/L2-regularised, clipped to `max_delta_step`."""
    output = -_threshold_l1(sum_grad, params.lambda_l1) / (sum_hess + params.lambda_l2)
    if params.max_delta_step > 0:
        output = np.where(np.abs(output) > params.max_delta_step, np.sign(output) * params.max_delta_step, output)
    return output


def _leaf_gain(sum_grad, sum_hess, params: Params):
    """GetLeafGain: with `max_delta_step`, from the clipped output (GetLeafGainGivenOutput)."""
    sg = _threshold_l1(sum_grad, params.lambda_l1)
    if params.max_delta_step <= 0:
        return (sg * sg) / (sum_hess + params.lambda_l2)
    output = leaf_output(sum_grad, sum_hess, params)
    return -(2.0 * sg * output + (sum_hess + params.lambda_l2) * output * output)


def _round_int(x: np.ndarray) -> np.ndarray:
    """Common::RoundInt: static_cast<int>(x + 0.5f)."""
    return np.trunc(x + 0.5).astype(np.int64)


def find_best_split(
    hist: Histogram,
    sum_grad: float,
    sum_hess: float,
    bin_mappers: list[BinMapper],
    params: Params,
    candidates: np.ndarray | None = None,
) -> tuple[SplitInfo | None, np.ndarray]:
    """The leaf's best split over the `candidates` features (default: all), and which features can split.

    `sum_grad`, `sum_hess` and `hist.count` are the leaf totals. Features outside `candidates` are not
    scanned and are reported as not splittable, as LightGBM does for features its parent could not split.
    """
    num_features = len(bin_mappers)
    if candidates is None:
        candidates = np.ones(num_features, dtype=bool)
    offsets = feature_offsets(np.array([mapper.num_bins for mapper in bin_mappers]))
    sum_hess = sum_hess + 2 * EPSILON
    # BeforeNumerical: a split must beat the leaf's own gain by `min_gain_to_split`.
    leaf = _Leaf(sum_grad, sum_hess, hist.count, _leaf_gain(sum_grad, sum_hess, params) + params.min_gain_to_split)

    best = None
    splittable = np.zeros(num_features, dtype=bool)
    for f in np.flatnonzero(candidates):
        mapper = bin_mappers[f]
        if mapper.num_bins < 2:
            continue
        g = hist.grad[offsets[f]:offsets[f + 1]]
        h = hist.hess[offsets[f]:offsets[f + 1]]
        best_threshold = _best_categories if mapper.is_categorical else _best_threshold
        split = best_threshold(int(f), mapper, g, h, leaf, params)
        if split is None:
            continue
        splittable[f] = True
        if split.beats(best):
            best = split
    return best, splittable


@dataclass
class _Leaf:
    sum_grad: float
    sum_hess: float  # plus 2 * kEpsilon, as FindBestThreshold passes it on
    num_data: int
    min_gain_shift: float

    @property
    def cnt_factor(self) -> float:
        return self.num_data / self.sum_hess


def _best_threshold(feature: int, mapper: BinMapper, g: np.ndarray, h: np.ndarray, leaf: _Leaf,
                    params: Params) -> SplitInfo | None:
    """FuncForNumricalL3: which scans run for this feature, and the better of their results.

    A later scan replaces an earlier one only if strictly better, so the right-to-left scan (missing
    values left) wins ties.
    """
    two_scans = mapper.num_bins > 2 and mapper.missing_type != MISSING_NONE
    scans = (True, False) if two_scans else (True,)
    # With a NaN bin and one other, the single scan can only put the NaN bin on the right.
    nan_goes_right = not two_scans and mapper.missing_type == MISSING_NAN
    best, best_gain = None, -np.inf  # output->gain, kMinScore before the first scan
    for reverse in scans:
        found = _scan(g, h, mapper, leaf, params, reverse,
                      skip_default_bin=two_scans and mapper.missing_type == MISSING_ZERO,
                      na_as_missing=two_scans and mapper.missing_type == MISSING_NAN)
        if found is None or not found[0] > best_gain + leaf.min_gain_shift:
            continue
        gain, threshold, left_grad, left_hess, left_count = found
        best_gain = gain - leaf.min_gain_shift
        best = SplitInfo(
            feature=feature,
            threshold=threshold,
            gain=float(best_gain),
            left_output=float(leaf_output(left_grad, left_hess, params)),
            right_output=float(leaf_output(leaf.sum_grad - left_grad, leaf.sum_hess - left_hess, params)),
            left_count=left_count,
            right_count=leaf.num_data - left_count,
            left_sum_grad=float(left_grad),
            left_sum_hess=float(left_hess - EPSILON),
            right_sum_grad=float(leaf.sum_grad - left_grad),
            right_sum_hess=float(leaf.sum_hess - left_hess - EPSILON),
            default_left=reverse and not nan_goes_right,
        )
    return best


def _scan(g: np.ndarray, h: np.ndarray, mapper: BinMapper, leaf: _Leaf, params: Params, reverse: bool,
          skip_default_bin: bool, na_as_missing: bool) -> tuple[float, int, float, float, int] | None:
    """FindBestThresholdSequentially: the best (gain, threshold bin, left Σg, left Σh, left count) of one scan.

    The scan adds one bin at a time to the near side (right when `reverse`, else left); the far side is
    the leaf total minus the near side, so skipped bins (the NaN bin, the default bin for Zero) go far.
    A histogram does not hold bin 0 when it is the most frequent bin (LightGBM's `offset`): a
    left-to-right scan then starts after it, or, with NaN as missing, derives it from the leaf totals.
    cumsum adds sequentially, matching the C++ loop.
    """
    nb = mapper.num_bins
    offset = 1 if mapper.most_freq_bin == 0 else 0
    cnt = _round_int(h * leaf.cnt_factor)
    start_grad, start_hess, start_count, evaluate_start = 0.0, EPSILON, 0, False
    if reverse:
        bins = np.arange(nb - 1 - int(na_as_missing), 0, -1)
    else:
        bins = np.arange(offset, nb - 1)
        if na_as_missing and offset == 1:
            start_grad = subtract_in_order(leaf.sum_grad, g[1:])
            start_hess = subtract_in_order(leaf.sum_hess - EPSILON, h[1:])
            start_count = leaf.num_data - int(cnt[1:].sum())
            evaluate_start = True
    if skip_default_bin:
        bins = bins[bins != mapper.default_bin]

    first = 0 if evaluate_start else 1
    near_grad = np.cumsum(np.concatenate([[start_grad], g[bins]]))[first:]
    near_hess = np.cumsum(np.concatenate([[start_hess], h[bins]]))[first:]
    near_count = (start_count + np.concatenate([[0], np.cumsum(cnt[bins])]))[first:]
    thresholds = np.concatenate([[0], bins - 1 if reverse else bins])[first:]
    far_grad = leaf.sum_grad - near_grad
    far_hess = leaf.sum_hess - near_hess
    far_count = leaf.num_data - near_count

    # `continue` while the near side is too small, `break` once the far side is.
    near_ok = (near_count >= params.min_data_in_leaf) & (near_hess >= params.min_sum_hessian_in_leaf)
    far_short = (far_count < params.min_data_in_leaf) | (far_hess < params.min_sum_hessian_in_leaf)
    stops = np.flatnonzero(near_ok & far_short)
    end = stops[0] if len(stops) else len(near_ok)

    if reverse:
        left_grad, left_hess, left_count, right_grad, right_hess = far_grad, far_hess, far_count, near_grad, near_hess
    else:
        left_grad, left_hess, left_count, right_grad, right_hess = near_grad, near_hess, near_count, far_grad, far_hess
    with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 only past the break, never used
        gain = _leaf_gain(left_grad, left_hess, params) + _leaf_gain(right_grad, right_hess, params)
    valid = np.zeros(len(gain), dtype=bool)
    valid[:end] = near_ok[:end] & (gain[:end] > leaf.min_gain_shift)
    if not valid.any():
        return None
    i = np.argmax(np.where(valid, gain, -np.inf))  # first strictly better in scan order
    return float(gain[i]), int(thresholds[i]), float(left_grad[i]), float(left_hess[i]), int(left_count[i])


def _best_categories(feature: int, mapper: BinMapper, g: np.ndarray, h: np.ndarray, leaf: _Leaf,
                     params: Params) -> SplitInfo | None:
    """FindBestThresholdCategoricalInner: which categories go left; bin 0 (NaN and the categories not
    kept) and every category not chosen go right.

    Up to `max_cat_to_onehot` bins, one category goes left. Otherwise the categories with at least
    `cat_smooth` rows are sorted by Σg / (Σh + cat_smooth), and the left side is a prefix of that order,
    taken from either end, of at most `max_cat_threshold` categories, with `cat_l2` added to `lambda_l2`.
    A prefix is only considered once it has `min_data_per_group` rows more than the last one considered.
    The first strictly better candidate wins, so the ascending order wins ties.
    """
    cnt = _round_int(h * leaf.cnt_factor)
    bins = range(1, mapper.num_bins)
    candidates: list[tuple[float, list[int], float, float, int]] = []  # gain, left bins, left Σg, left Σh, left count
    if mapper.num_bins <= params.max_cat_to_onehot:
        for t in bins:
            if cnt[t] < params.min_data_in_leaf or h[t] < params.min_sum_hessian_in_leaf:
                continue
            if leaf.num_data - cnt[t] < params.min_data_in_leaf:
                continue
            other_hess = leaf.sum_hess - h[t] - EPSILON
            if other_hess < params.min_sum_hessian_in_leaf:
                continue
            gain = _leaf_gain(leaf.sum_grad - g[t], other_hess, params) + _leaf_gain(g[t], h[t] + EPSILON, params)
            candidates.append((gain, [t], g[t], h[t] + EPSILON, int(cnt[t])))
    else:
        params = replace(params, lambda_l2=params.lambda_l2 + params.cat_l2)
        used = [t for t in bins if cnt[t] >= params.cat_smooth]
        ctr = [g[t] / (h[t] + params.cat_smooth) for t in used]
        used = [used[i] for i in np.argsort(ctr, kind="stable")]
        max_num_cat = min(params.max_cat_threshold, (len(used) + 1) // 2)
        for order in (used, used[::-1]):
            left_grad, left_hess, left_count, group_count = 0.0, EPSILON, 0, 0
            for i, t in enumerate(order[:max_num_cat]):
                left_grad += g[t]
                left_hess += h[t]
                left_count += int(cnt[t])
                group_count += int(cnt[t])
                if left_count < params.min_data_in_leaf or left_hess < params.min_sum_hessian_in_leaf:
                    continue
                right_count = leaf.num_data - left_count
                if right_count < params.min_data_in_leaf or right_count < params.min_data_per_group:
                    break
                if leaf.sum_hess - left_hess < params.min_sum_hessian_in_leaf:
                    break
                if group_count < params.min_data_per_group:
                    continue
                group_count = 0
                gain = (_leaf_gain(left_grad, left_hess, params)
                        + _leaf_gain(leaf.sum_grad - left_grad, leaf.sum_hess - left_hess, params))
                candidates.append((gain, order[:i + 1], left_grad, left_hess, left_count))

    best = None
    for candidate in candidates:
        if candidate[0] > leaf.min_gain_shift and (best is None or candidate[0] > best[0]):
            best = candidate
    if best is None:
        return None
    gain, left_bins, left_grad, left_hess, left_count = best
    return SplitInfo(
        feature=feature,
        threshold=-1,
        gain=float(gain - leaf.min_gain_shift),
        left_output=float(leaf_output(left_grad, left_hess, params)),
        right_output=float(leaf_output(leaf.sum_grad - left_grad, leaf.sum_hess - left_hess, params)),
        left_count=left_count,
        right_count=leaf.num_data - left_count,
        left_sum_grad=float(left_grad),
        left_sum_hess=float(left_hess - EPSILON),
        right_sum_grad=float(leaf.sum_grad - left_grad),
        right_sum_hess=float(leaf.sum_hess - left_hess - EPSILON),
        default_left=False,
        left_bins=[int(t) for t in left_bins],
    )
