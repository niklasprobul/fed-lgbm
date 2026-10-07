"""Agreed bin edges and the mapping of raw values to bins.

Edges are LightGBM bin upper bounds: a value falls in the first bin whose upper bound is >= the value,
and the last bound is +inf. A feature whose missing type is NaN has one more bin after that for NaN,
with the upper bound NaN, as BinMapper::FindBin appends it.

A categorical feature has one bin per kept category instead, after bin 0, which holds NaN, negative
values and every category not kept.
"""

from dataclasses import dataclass, replace

import numpy as np

from fl_lightgbm.params import Params

# LightGBM's kZeroThreshold (meta.h): the float literal 1e-35f, widened to double.
ZERO_THRESHOLD = float(np.float32(1e-35))

# LightGBM's MissingType, by the names dump_model uses.
MISSING_NONE, MISSING_ZERO, MISSING_NAN = "None", "Zero", "NaN"

# BinMapper::FindBin keeps the bin of 0 as most frequent bin unless another bin holds this share of rows.
SPARSE_THRESHOLD = 0.7

# Cells of the fine grid per feature (ADR 0004).
GRID_CELLS = 4096


@dataclass(frozen=True)
class BinMapper:
    """One feature's agreed bin edges, as LightGBM's BinMapper holds them."""

    upper_bounds: np.ndarray  # empty for a categorical feature
    missing_type: str
    most_freq_bin: int
    categories: np.ndarray | None = None  # categorical: the category of bins 1, 2, … (bin_2_categorical_ without bin 0)
    # Numerical: the range LightGBM's model writes in `feature_infos` (min_val_, max_val_), which includes
    # the 0 its bin finding adds. With configured bounds, the configured bounds instead.
    min_val: float = np.nan
    max_val: float = np.nan

    @property
    def is_categorical(self) -> bool:
        return self.categories is not None

    @property
    def is_trivial(self) -> bool:
        """BinMapper::is_trivial_ without `feature_pre_filter`, which the central baseline turns off: one
        bin, so the feature is never split, and LightGBM's dataset leaves it out."""
        return self.num_bins <= 1

    @property
    def num_bins(self) -> int:
        return 1 + len(self.categories) if self.categories is not None else len(self.upper_bounds)

    @property
    def default_bin(self) -> int:
        """The bin of 0 (BinMapper::default_bin_)."""
        return int(self.value_to_bin(np.zeros(1))[0])

    def value_to_bin(self, values: np.ndarray) -> np.ndarray:
        """BinMapper::ValueToBin."""
        if self.categories is None:
            return _value_to_bin(values, self.upper_bounds, self.missing_type)
        return _category_to_bin(values, self.categories)


@dataclass(frozen=True)
class Grid:
    """A feature's fine equal-width grid between its global bounds, on which the sites count their rows.

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


def grid(lo: float, hi: float) -> Grid:
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
                return Grid(values, lo, hi)
        num_cells //= 2
    return Grid(np.array([lo]) if abs(lo) > ZERO_THRESHOLD else np.empty(0), lo, hi)


def agree_bin_mapper(grid: Grid, cell_counts: np.ndarray, na_count: int, num_data: int, params: Params) -> BinMapper:
    """Setup round 2: BinMapper::FindBin on the merged grid, with the cells as distinct values.

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
    # cells' values do not merge as close doubles do in FindBin, because `grid` keeps them more than one
    # ulp apart.
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
    min_val, max_val = _value_range(grid.lo, grid.hi, with_zero)
    return BinMapper(edges, missing_type, _most_freq_bin(cnt_in_bin, default_bin(edges, missing_type), num_data),
                     min_val=min_val, max_val=max_val)


def _value_range(lo: float, hi: float, with_zero: bool) -> tuple[float, float]:
    """BinMapper::FindBin's min_val_ and max_val_ from the global bounds: a bound within kZeroThreshold is
    0, as LightGBM samples it, and the 0 it puts among the distinct values widens the range."""
    lo, hi = (0.0 if abs(b) <= ZERO_THRESHOLD else float(b) for b in (lo, hi))  # NaN stays NaN
    return (float(np.fmin(lo, 0.0)), float(np.fmax(hi, 0.0))) if with_zero else (lo, hi)


def _most_freq_bin(cnt_in_bin: np.ndarray, zero_bin: int, num_data: int) -> int:
    """The bin with the most rows, unless it holds less than SPARSE_THRESHOLD of them: then the bin of 0."""
    most_freq = int(np.argmax(cnt_in_bin))  # ArrayArgs::ArgMax: the first maximum
    if most_freq != zero_bin and cnt_in_bin[most_freq] / num_data < SPARSE_THRESHOLD:
        return zero_bin
    return most_freq


@dataclass(frozen=True)
class Categories:
    """A categorical feature's categories merged over the sites (setup round 1), on which the sites count
    their rows in setup round 2, as the grid of a numerical feature."""

    values: np.ndarray  # ascending

    def counts(self, codes: np.ndarray) -> np.ndarray:
        """Rows per category, from `category_codes`; the rows of the NaN bin (-1) are not counted."""
        codes = codes[codes >= 0]
        return np.bincount(np.searchsorted(self.values, codes), minlength=len(self.values))


def category_codes(x: np.ndarray, params: Params) -> np.ndarray:
    """The category BinMapper::FindBin counts each value of a categorical feature in: the value cast to
    int, or -1 for the NaN bin. Negative values go to the NaN bin, and so does NaN, unless it is read as
    0 (without `use_missing`, or with `zero_as_missing`).

    One case is not reproduced: FindBin puts a zero between negative and positive values even when no
    row is 0, so with negative values and no row of category 0, LightGBM may keep an empty bin for it.
    """
    nan_code = -1.0 if params.use_missing and not params.zero_as_missing else 0.0
    codes = np.where(np.isnan(x), nan_code, np.trunc(x))  # static_cast<int> truncates towards 0
    return np.where(codes < 0, -1, codes).astype(np.int64)


def agree_categorical_bin_mapper(categories: Categories, counts: np.ndarray, na_count: int, num_data: int,
                                 params: Params) -> BinMapper:
    """Setup round 2: BinMapper::FindBin for a categorical feature, on the summed category counts.

    Categories are kept in order of descending count (ties: the smaller category first) until they
    cover 99 % of the rows that are not in the NaN bin and there are `max_bin` bins; a category with
    fewer than `min_data_in_bin` rows stops it, unless at most one category is kept so far.
    """
    order = np.argsort(-counts, kind="stable")
    cut_cnt = int(float(np.float32(num_data - na_count) * np.float32(0.99)) + 0.5)  # RoundInt of a float product
    max_bin = min(len(categories.values) + (na_count > 0), params.max_bin)
    kept: list[int] = []
    cnt_in_bin = [0]
    used_cnt = 0
    for i in order:
        if not (used_cnt < cut_cnt or 1 + len(kept) < max_bin):
            break
        if counts[i] < params.min_data_in_bin and len(kept) > 1:
            break
        kept.append(int(categories.values[i]))
        cnt_in_bin.append(int(counts[i]))
        used_cnt += int(counts[i])
    # MissingType::None marks a feature whose bins hold every category and no NaN.
    missing_type = MISSING_NONE if len(kept) == len(categories.values) and na_count == 0 else MISSING_NAN
    cnt_in_bin[0] = num_data - used_cnt
    mapper = BinMapper(np.empty(0), missing_type, 0, np.array(kept, dtype=np.int64))
    return replace(mapper, most_freq_bin=_most_freq_bin(np.array(cnt_in_bin), mapper.default_bin, num_data))


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


def default_bin(edges: np.ndarray, missing_type: str) -> int:
    return int(_value_to_bin(np.zeros(1), edges, missing_type)[0])


def missing_bin(edges: np.ndarray, missing_type: str) -> int:
    """The bin whose rows follow a split's default direction: the bin of 0 for Zero, the last bin for
    NaN, and -1 (no bin) for None."""
    if missing_type == MISSING_ZERO:
        return default_bin(edges, missing_type)
    if missing_type == MISSING_NAN:
        return len(edges) - 1
    return -1


def _value_to_bin(values: np.ndarray, edges: np.ndarray, missing_type: str) -> np.ndarray:
    """BinMapper::ValueToBin for numerical features: NaN goes to the NaN bin, or is read as 0 without one."""
    nan = np.isnan(values)
    if missing_type == MISSING_NAN:
        return np.where(nan, len(edges) - 1, np.searchsorted(edges[:-1], values, side="left"))
    return np.searchsorted(edges, np.where(nan, 0.0, values), side="left")


def _category_to_bin(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """BinMapper::ValueToBin for categorical features: bin 0 for NaN, negative values and categories not kept."""
    codes = np.trunc(values)  # static_cast<int>; NaN and negative values match no category
    found = np.isin(codes, categories)
    order = np.argsort(categories)
    bins = np.zeros(len(values), dtype=np.int64)
    bins[found] = order[np.searchsorted(categories[order], codes[found])] + 1
    return bins


def bin_values(X: np.ndarray, mappers: list[BinMapper]) -> np.ndarray:
    """Map every value to its bin."""
    return np.column_stack([m.value_to_bin(X[:, f]) for f, m in enumerate(mappers)]).astype(np.int32)
