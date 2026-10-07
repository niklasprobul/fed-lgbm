"""The fine grid (ADR 0004, superseded by ADR 0008): about 4,096 equal-width cells per feature, each cell taken
as one distinct value, and LightGBM's GreedyFindBin run on the merged cells. The federation no longer uses it;
it stays as an arm of the rebinning benchmark (`benchmark_rebinning.py`) and of `leak_shifted_bins.py`. A copy of
the code as it was before #26, kept as it was rather than shared with `fl_lightgbm.binning`.
"""

from dataclasses import dataclass

import numpy as np

from fl_lightgbm.binning import (MISSING_NAN, MISSING_NONE, MISSING_ZERO, ZERO_THRESHOLD, BinMapper, default_bin,
                                 with_most_freq_bin)
from fl_lightgbm.params import Params

GRID_CELLS = 4096


@dataclass(frozen=True)
class FineGrid:
    """A feature's fine equal-width grid between its global bounds.

    The cells have width (hi - lo) / GRID_CELLS (fewer, wider cells for bounds too narrow for that many
    distinct doubles) and are laid out from 0, so that no cell holds values of both signs. Each cell
    stands for one distinct value, its centre. The boundary between two neighbouring cells is where
    GreedyFindBin puts a bin bound between their values, so every bound it chooses either is a cell
    boundary or lies in a run of empty cells, and the rows per agreed bin are sums of cells.
    Zeros (|x| <= kZeroThreshold, as LightGBM samples them) and NaN are not in the grid.
    """

    values: np.ndarray  # one per cell, ascending
    lo: float  # the global bounds, NaN if no site has a value
    hi: float

    def counts(self, x: np.ndarray) -> np.ndarray:
        """Rows per cell. A value beyond the outermost cell of its sign counts in that cell."""
        x = x[np.abs(x) > ZERO_THRESHOLD]  # also drops NaN
        boundaries = _midpoint_bound(self.values[:-1], self.values[1:])
        return np.bincount(np.searchsorted(boundaries, x, side="left"), minlength=len(self.values))


def fine_grid(lo: float, hi: float) -> FineGrid:
    """The grid between a feature's global bounds; a single cell if they are equal, none if they are NaN.

    Bounds so narrow, and so far from 0, that the cells' values would not be distinct doubles get fewer
    cells: the cell count is halved until the bound GreedyFindBin puts between neighbouring cells lies
    below the upper one's value, which also puts them more than one ulp apart, so FindBin would not merge
    them. Bounds too close for even two such cells get a single cell.
    """
    num_cells = GRID_CELLS
    while lo < hi and num_cells >= 1:
        width = (hi - lo) / num_cells
        if width > 0:  # a subnormal range divided into many cells underflows to 0
            cells = np.arange(np.floor(lo / width), np.ceil(hi / width))  # cell k holds (k * width, (k + 1) * width]
            values = (cells + 0.5) * width
            if np.all(_midpoint_bound(values[:-1], values[1:]) < values[1:]):
                return FineGrid(values, lo, hi)
        num_cells //= 2
    return FineGrid(np.array([lo]) if abs(lo) > ZERO_THRESHOLD else np.empty(0), lo, hi)


def fine_grid_bin_mapper(grid: FineGrid, cell_counts: np.ndarray, na_count: int, num_data: int,
                         params: Params) -> BinMapper:
    """BinMapper::FindBin on the merged grid, with the cells as distinct values.

    `cell_counts` and `na_count` are summed over the sites, `num_data` is the global row count; the
    rows in neither are zeros. Without `use_missing`, and with `zero_as_missing`, NaN is read as zero.
    """
    if not params.use_missing:
        missing_type = MISSING_NONE
    elif params.zero_as_missing:
        missing_type = MISSING_ZERO
    else:
        missing_type = MISSING_NAN if na_count > 0 else MISSING_NONE
    na_cnt = na_count if missing_type == MISSING_NAN else 0
    non_empty = cell_counts > 0
    values, counts = grid.values[non_empty].tolist(), cell_counts[non_empty].tolist()
    zero_cnt = num_data - sum(counts) - na_cnt

    # Distinct values with zero put in between the signs: FindBin adds it when there are zeros, when
    # there are no other values, and between negative and positive values even without zeros. The
    # cells' values do not merge as close doubles do in FindBin, because `fine_grid` keeps them more than
    # one ulp apart.
    num_negative = sum(v < 0.0 for v in values)
    with_zero = zero_cnt > 0 or not values or 0 < num_negative < len(values)
    zero = ([0.0], [zero_cnt]) if with_zero else ([], [])
    distinct = values[:num_negative] + zero[0] + values[num_negative:]
    distinct_counts = counts[:num_negative] + zero[1] + counts[num_negative:]

    if missing_type == MISSING_NAN:
        bounds = _find_bin_with_zero_as_one_bin(distinct, distinct_counts, params.max_bin - 1, num_data - na_cnt,
                                                params.min_data_in_bin) + [np.nan]
    else:
        bounds = _find_bin_with_zero_as_one_bin(distinct, distinct_counts, params.max_bin, num_data,
                                                params.min_data_in_bin)
        if missing_type == MISSING_ZERO and len(bounds) == 2:
            missing_type = MISSING_NONE

    cnt_in_bin = np.zeros(len(bounds), dtype=np.int64)
    i_bin = 0
    for v, c in zip(distinct, distinct_counts):
        while v > bounds[i_bin] and i_bin < len(bounds) - 1:
            i_bin += 1
        cnt_in_bin[i_bin] += c
    if missing_type == MISSING_NAN:
        cnt_in_bin[-1] = na_cnt

    edges = np.array(bounds)
    lo, hi = (0.0 if abs(b) <= ZERO_THRESHOLD else float(b) for b in (grid.lo, grid.hi))  # NaN stays NaN
    min_val, max_val = (float(np.fmin(lo, 0.0)), float(np.fmax(hi, 0.0))) if with_zero else (lo, hi)
    mapper = BinMapper(edges, missing_type, default_bin(edges, missing_type), min_val=min_val, max_val=max_val)
    return with_most_freq_bin(mapper, cnt_in_bin, num_data)


def _find_bin_with_zero_as_one_bin(distinct: list[float], counts: list[int], max_bin: int, total_cnt: int,
                                   min_data_in_bin: int) -> list[float]:
    """FindBinWithZeroAsOneBin: negative values, zero and positive values get bins of their own."""
    left_cnt_data = sum(c for v, c in zip(distinct, counts) if v <= -ZERO_THRESHOLD)
    right_cnt_data = sum(c for v, c in zip(distinct, counts) if v > ZERO_THRESHOLD)
    cnt_zero = sum(counts) - left_cnt_data - right_cnt_data
    left_cnt = next((i for i, v in enumerate(distinct) if v > -ZERO_THRESHOLD), len(distinct))

    bounds: list[float] = []
    if left_cnt > 0 and max_bin > 1:
        left_max_bin = max(1, int(left_cnt_data / (total_cnt - cnt_zero) * (max_bin - 1)))
        bounds = _greedy_find_bin(distinct[:left_cnt], counts[:left_cnt], left_max_bin, left_cnt_data, min_data_in_bin)
        if bounds:
            bounds[-1] = -ZERO_THRESHOLD

    right_start = next((i for i in range(left_cnt, len(distinct)) if distinct[i] > ZERO_THRESHOLD), -1)
    right_max_bin = max_bin - 1 - len(bounds)
    if right_start >= 0 and right_max_bin > 0:
        right = _greedy_find_bin(distinct[right_start:], counts[right_start:], right_max_bin, right_cnt_data,
                                 min_data_in_bin)
        return bounds + [ZERO_THRESHOLD] + right
    return bounds + [np.inf]


def _greedy_find_bin(distinct: list[float], counts: list[int], max_bin: int, total_cnt: int,
                     min_data_in_bin: int) -> list[float]:
    """GreedyFindBin: bin upper bounds over sorted distinct values, the last one +inf."""
    bounds: list[float] = []
    n = len(distinct)
    if n <= max_bin:
        cur_cnt_inbin = 0
        for i in range(n - 1):
            cur_cnt_inbin += counts[i]
            if cur_cnt_inbin >= min_data_in_bin:
                val = float(_midpoint_bound(distinct[i], distinct[i + 1]))
                if not bounds or not _check_double_equal_ordered(bounds[-1], val):
                    bounds.append(val)
                    cur_cnt_inbin = 0
        return bounds + [np.inf]

    if min_data_in_bin > 0:
        max_bin = max(min(max_bin, total_cnt // min_data_in_bin), 1)
    mean_bin_size = total_cnt / max_bin
    rest_bin_cnt, rest_sample_cnt = max_bin, total_cnt
    is_big = [c >= mean_bin_size for c in counts]
    for c, big in zip(counts, is_big):
        if big:
            rest_bin_cnt -= 1
            rest_sample_cnt -= c
    mean_bin_size = _divide(rest_sample_cnt, rest_bin_cnt)
    upper_bounds, lower_bounds = [np.inf] * max_bin, [np.inf] * max_bin

    bin_cnt = 0
    lower_bounds[0] = distinct[0]
    cur_cnt_inbin = 0
    for i in range(n - 1):
        if not is_big[i]:
            rest_sample_cnt -= counts[i]
        cur_cnt_inbin += counts[i]
        if is_big[i] or cur_cnt_inbin >= mean_bin_size or (
                is_big[i + 1] and cur_cnt_inbin >= _cpp_max(1.0, mean_bin_size * 0.5)):
            upper_bounds[bin_cnt] = distinct[i]
            bin_cnt += 1
            lower_bounds[bin_cnt] = distinct[i + 1]
            if bin_cnt >= max_bin - 1:
                break
            cur_cnt_inbin = 0
            if not is_big[i]:
                rest_bin_cnt -= 1
                mean_bin_size = _divide(rest_sample_cnt, rest_bin_cnt)
    bin_cnt += 1

    for i in range(bin_cnt - 1):
        val = float(_midpoint_bound(upper_bounds[i], lower_bounds[i + 1]))
        if not bounds or not _check_double_equal_ordered(bounds[-1], val):
            bounds.append(val)
    return bounds + [np.inf]


def _midpoint_bound(a, b):
    """The bin bound GreedyFindBin puts between neighbouring distinct values: just above their midpoint."""
    return np.nextafter((a + b) / 2.0, np.inf)


def _check_double_equal_ordered(a: float, b: float) -> bool:
    """Common::CheckDoubleEqualOrdered: b, known to be >= a, is at most one ulp above it."""
    return b <= np.nextafter(a, np.inf)


def _divide(a: int, b: int) -> float:
    """C++ double division, which gives inf or NaN rather than raising when b is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / b)


def _cpp_max(a: float, b: float) -> float:
    """std::max, which returns `a` when either is NaN."""
    return b if a < b else a
