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

# Cells of the grid per feature (ADR 0008).
GRID_CELLS = 64


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

    def counts(self, values: np.ndarray) -> np.ndarray:
        """Rows per bin, which the sites count in setup round 3."""
        return np.bincount(self.value_to_bin(values), minlength=self.num_bins)


@dataclass(frozen=True)
class Grid:
    """A feature's equal-width grid between its global bounds, on which the sites count their rows in
    setup round 2 and from which the aggregator interpolates the bin edges.

    Cell k holds the values in (bounds[k], bounds[k + 1]], the first cell also `lo` itself. The cells have
    width (hi - lo) / GRID_CELLS (fewer, wider cells for bounds too narrow for that many distinct doubles)
    and are laid out from 0, so that no cell holds values of both signs; the outermost cells end at the
    bounds. Zeros (|x| <= kZeroThreshold, as LightGBM samples them) and NaN are not in the grid.
    """

    bounds: np.ndarray  # ascending: lo, the boundaries between cells, hi; empty without cells
    lo: float  # the global bounds, NaN if no site has a value
    hi: float

    @property
    def num_cells(self) -> int:
        return max(len(self.bounds) - 1, 0)

    def counts(self, x: np.ndarray) -> np.ndarray:
        """Rows per cell."""
        x = x[np.abs(x) > ZERO_THRESHOLD]  # also drops NaN
        return np.bincount(np.searchsorted(self.bounds[1:-1], x, side="left"), minlength=self.num_cells)


def grid(lo: float, hi: float) -> Grid:
    """The grid between a feature's global bounds; a single cell if they are equal, none if they are NaN or 0.

    Bounds so narrow, and so far from 0, that the boundaries between cells would not be distinct doubles
    get fewer cells: the cell count is halved until they are.
    """
    num_cells = GRID_CELLS
    while lo < hi and num_cells >= 1:
        # (hi - lo) / num_cells, without overflowing for bounds near the largest doubles; num_cells is a power of 2
        width = hi / num_cells - lo / num_cells
        if width > 0:  # a subnormal range divided into many cells underflows to 0
            inner = np.arange(np.floor(lo / width) + 1, np.ceil(hi / width)) * width  # cell k ends at (k + 1) * width
            bounds = np.concatenate([[lo], inner[(lo < inner) & (inner < hi)], [hi]])
            if np.all(np.diff(bounds) > 0):
                return Grid(bounds, lo, hi)
        num_cells //= 2
    return Grid(np.array([lo, hi]) if lo < hi or abs(lo) > ZERO_THRESHOLD else np.empty(0), lo, hi)


def agree_bin_mapper(grid: Grid, cell_counts: np.ndarray, na_count: int, num_data: int, params: Params) -> BinMapper:
    """Setup round 2: equal-count bin edges interpolated from the merged grid, with the values taken as
    spread evenly within each cell.

    LightGBM, given these edges as forced bins (FindBinWithPredefinedBin), keeps them as they are: zero
    has a bin of its own, there are at most `max_bin` bins (one of them for NaN), and the nonzero rows get
    at most one bin per `min_data_in_bin` rows, one more where 0 splits one. `cell_counts` and `na_count`
    are summed over the sites, `num_data` is the global row count; the rows in neither are zeros. Without
    `use_missing`, and with `zero_as_missing`, NaN is read as zero. The most frequent bin is the bin of 0
    until setup round 3 has counted the rows per bin (`with_most_freq_bin`).
    """
    if not params.use_missing:
        missing_type = MISSING_NONE
    elif params.zero_as_missing:
        missing_type = MISSING_ZERO
    else:
        missing_type = MISSING_NAN if na_count > 0 else MISSING_NONE
    max_bin = params.max_bin - 1 if missing_type == MISSING_NAN else params.max_bin
    num_negative = int(cell_counts[grid.bounds[1:] <= 0.0].sum())
    num_nonzero = int(cell_counts.sum())
    zero_bounds = _zero_bounds(num_negative > 0, num_nonzero > num_negative, max_bin)
    num_edges = min(max_bin - len(zero_bounds), num_nonzero // params.min_data_in_bin) - 1
    edges = np.unique(np.concatenate([_equal_count_edges(grid, cell_counts, num_edges), zero_bounds]))

    bounds = np.append(edges, np.inf)
    if missing_type == MISSING_NAN:
        bounds = np.append(bounds, np.nan)
    elif missing_type == MISSING_ZERO and len(bounds) == 2:
        missing_type = MISSING_NONE
    na_cnt = na_count if missing_type == MISSING_NAN else 0
    # FindBin puts a 0 among the distinct values when there are zeros, when there are no other values, and
    # between negative and positive values even without zeros.
    with_zero = num_data - num_nonzero - na_cnt > 0 or num_nonzero == 0 or 0 < num_negative < num_nonzero
    min_val, max_val = _value_range(grid.lo, grid.hi, with_zero)
    return BinMapper(bounds, missing_type, default_bin(bounds, missing_type), min_val=min_val, max_val=max_val)


def _zero_bounds(negative: bool, positive: bool, max_bin: int) -> list[float]:
    """The bounds FindBinWithPredefinedBin puts around zero before any forced bound, where there are
    negative or positive values: both with at least 3 bins, with 2 only the one below 0 if there are
    negative values. Without nonzero values there are none, and LightGBM, given no forced bound, finds none."""
    if not (negative or positive) or max_bin < 2:
        return []
    if max_bin == 2:
        return [-ZERO_THRESHOLD if negative else ZERO_THRESHOLD]
    return [-ZERO_THRESHOLD] * negative + [ZERO_THRESHOLD] * positive


def _equal_count_edges(grid: Grid, cell_counts: np.ndarray, num_edges: int) -> np.ndarray:
    """`num_edges` edges that split the grid's rows into bins of equal count, with the values taken as spread
    evenly within each cell. Edges within kZeroThreshold of 0 are left out: zero has a bin of its own."""
    if num_edges < 1:
        return np.empty(0)
    cum = np.concatenate([[0], np.cumsum(cell_counts)])
    targets = np.arange(1, num_edges + 1) * cum[-1] / (num_edges + 1)
    cell = np.searchsorted(cum, targets, side="left") - 1  # the cell each target lies in, which is not empty
    inside = (targets - cum[cell]) / (cum[cell + 1] - cum[cell])
    edges = grid.bounds[cell] + inside * (grid.bounds[cell + 1] - grid.bounds[cell])
    return edges[np.abs(edges) > ZERO_THRESHOLD]


def _value_range(lo: float, hi: float, with_zero: bool) -> tuple[float, float]:
    """BinMapper::FindBin's min_val_ and max_val_ from the global bounds: a bound within kZeroThreshold is
    0, as LightGBM samples it, and the 0 it puts among the distinct values widens the range."""
    lo, hi = (0.0 if abs(b) <= ZERO_THRESHOLD else float(b) for b in (lo, hi))  # NaN stays NaN
    return (float(np.fmin(lo, 0.0)), float(np.fmax(hi, 0.0))) if with_zero else (lo, hi)


def with_most_freq_bin(mapper: BinMapper, cnt_in_bin: np.ndarray, num_data: int) -> BinMapper:
    """Setup round 3: the feature's most frequent bin from its rows per agreed bin, summed over the sites."""
    return replace(mapper, most_freq_bin=_most_freq_bin(cnt_in_bin, mapper.default_bin, num_data))


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
    return with_most_freq_bin(BinMapper(np.empty(0), missing_type, 0, np.array(kept, dtype=np.int64)),
                              np.array(cnt_in_bin), num_data)


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
