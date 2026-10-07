import re
import struct
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from baseline import central_dataset
from fl_lightgbm.binning import (
    GRID_CELLS,
    MISSING_NAN,
    MISSING_NONE,
    MISSING_ZERO,
    BinMapper,
    Categories,
    agree_bin_mapper,
    agree_categorical_bin_mapper,
    category_codes,
    grid,
)
from fl_lightgbm.params import Params

# LightGBM's MissingType enum values.
LIGHTGBM_MISSING_TYPES = {0: MISSING_NONE, 1: MISSING_ZERO, 2: MISSING_NAN}


def saved_bin_mapper(raw: bytes, mapper: BinMapper) -> tuple[str, int]:
    """Missing type and most frequent bin of the bin mapper in a saved LightGBM dataset whose upper bounds,
    or categories, are exactly `mapper`'s. BinMapper::SaveBinaryToFile writes nine fields of 8 bytes each
    (num_bin, missing_type, is_trivial, sparse_rate, bin_type, min_val, max_val, default_bin,
    most_freq_bin), then the upper bounds as doubles or the category of each bin as int32, -1 for the
    first bin of a categorical feature. The NaN bin's bound is stored as 2.0: FindBin appends `NaN`, which
    in bin.cpp names the enum value MissingType::NaN, not a double; ValueToBin never reads it."""
    if mapper.is_categorical:
        needle = np.array([-1, *mapper.categories], dtype="<i4").tobytes()
    else:
        needle = np.nan_to_num(np.asarray(mapper.upper_bounds, dtype=np.float64), nan=2.0, posinf=np.inf).tobytes()
    starts = [m.start() - 72 for m in re.finditer(re.escape(needle), raw)]
    found = [s for s in starts if s >= 0 and struct.unpack_from("<i", raw, s)[0] == mapper.num_bins]
    assert len(found) == 1, f"LightGBM has no bin mapper like {mapper}"
    missing_type = struct.unpack_from("<i", raw, found[0] + 8)[0]
    most_freq_bin = struct.unpack_from("<I", raw, found[0] + 64)[0]
    return LIGHTGBM_MISSING_TYPES[missing_type], most_freq_bin


def saved(dataset: lgb.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.bin"
        dataset.construct().save_binary(str(path))
        return path.read_bytes()


def assert_same_as_lightgbm(dataset: lgb.Dataset, mappers: list[BinMapper]):
    raw = saved(dataset)
    for f, m in enumerate(mappers):
        assert dataset.feature_num_bin(f) == m.num_bins, f"feature {f}"
        assert saved_bin_mapper(raw, m) == (m.missing_type, m.most_freq_bin), f"feature {f}"


def mixed_features(seed, n=1200):
    """One feature per case the bin finding treats differently."""
    rng = np.random.default_rng(seed)
    nan_share = lambda x, share: np.where(rng.random(n) < share, np.nan, x)  # noqa: E731
    return np.column_stack([
        rng.normal(size=n),  # both signs, more distinct values than bins
        rng.uniform(0.5, 4.0, size=n),  # positive only
        -rng.exponential(size=n),  # negative only
        np.where(rng.random(n) < 0.6, 0.0, rng.normal(size=n)),  # mostly zero
        np.where(rng.random(n) < 0.75, 3.0, rng.uniform(0.0, 10.0, size=n)),  # a non-zero bin holds over 70 %
        rng.integers(-2, 4, size=n).astype(float),  # few distinct values, fewer than bins
        rng.standard_cauchy(size=n),  # long tails: most grid cells empty
        nan_share(rng.normal(size=n), 0.2),
        nan_share(rng.uniform(1.0, 2.0, size=n), 0.8),  # the NaN bin holds most rows
        nan_share(np.where(rng.random(n) < 0.5, 0.0, rng.normal(size=n)), 0.3),
    ])


def agree(X, sites, params):
    """Both setup rounds for the bin edges, without the transport: global bounds, then summed grid counts."""
    mappers = []
    for f in range(X.shape[1]):
        lo = np.fmin.reduce([np.fmin.reduce(X[rows, f]) for rows in sites])
        hi = np.fmax.reduce([np.fmax.reduce(X[rows, f]) for rows in sites])
        g = grid(lo, hi)
        counts = sum(g.counts(X[rows, f]) for rows in sites)
        na_count = sum(int(np.isnan(X[rows, f]).sum()) for rows in sites)
        mappers.append(agree_bin_mapper(g, counts, na_count, len(X), params))
    return mappers


def grid_data(X, sites):
    """The pooled rows as the grid sees them: each row replaced by its grid cell's value, zeros and NaN kept."""
    columns = []
    for f in range(X.shape[1]):
        x = X[:, f]
        g = grid(np.fmin.reduce(x), np.fmax.reduce(x))
        counts = sum(g.counts(x[rows]) for rows in sites)
        na_count = int(np.isnan(x).sum())
        zeros = len(x) - counts.sum() - na_count
        columns.append(np.concatenate([np.repeat(g.values, counts), np.zeros(zeros), np.full(na_count, np.nan)]))
    return np.column_stack(columns)


SETTINGS = [
    {},
    {"max_bin": 16},
    {"max_bin": 63, "min_data_in_bin": 1},
    {"min_data_in_bin": 25},
    {"zero_as_missing": True},
    {"use_missing": False},
]


@pytest.mark.parametrize("settings", SETTINGS)
def test_agreed_edges_equal_lightgbm_binning_of_the_grid_cells(settings):
    X = mixed_features(seed=1)
    sites = np.array_split(np.random.default_rng(2).permutation(len(X)), 3)

    mappers = agree(X, sites, Params.from_dict(settings))

    lgb_params = {**settings, "bin_construct_sample_cnt": len(X), "feature_pre_filter": False,
                  "enable_bundle": False, "verbose": -1}
    assert_same_as_lightgbm(lgb.Dataset(grid_data(X, sites), np.zeros(len(X)), params=lgb_params), mappers)


@pytest.mark.parametrize("settings", SETTINGS)
def test_most_frequent_bin_and_missing_type_equal_lightgbm_on_the_pooled_rows(settings):
    """The central baseline bins the real rows with the agreed edges; its bin mapper must agree with ours."""
    X = mixed_features(seed=3)
    sites = np.array_split(np.arange(len(X)), 4)

    mappers = agree(X, sites, Params.from_dict(settings))

    with central_dataset(X, np.zeros(len(X)), mappers, settings) as dataset:
        assert_same_as_lightgbm(dataset, mappers)


def categorical_features(seed, n=1500):
    """One categorical feature per case the categorical bin rules treat differently."""
    rng = np.random.default_rng(seed)
    nan_share = lambda x, share: np.where(rng.random(n) < share, np.nan, x)  # noqa: E731
    return np.column_stack([
        rng.choice(4, size=n, p=[0.1, 0.2, 0.3, 0.4]),  # few categories: all kept
        rng.geometric(0.05, size=n),  # a long tail of rare categories: the 99 % cut and min_data_in_bin stop
        np.where(rng.random(n) < 0.8, 0, rng.integers(1, 6, size=n)),  # category 0 holds most rows
        np.where(rng.random(n) < 0.75, 7, rng.integers(0, 6, size=n)),  # another category holds over 70 %
        rng.choice([-2.0, -0.5, 0.0, 1.0, 2.7, 3.0], size=n),  # -2 goes to the NaN bin, -0.5 and 2.7 are cast to 0 and 2
        nan_share(rng.integers(0, 30, size=n), 0.3),
        nan_share(rng.geometric(0.1, size=n), 0.02),  # NaN and other categories share the first bin
    ]).astype(np.float64)


def agree_categorical(X, sites, params):
    """Both setup rounds for categorical features, without the transport: merged category sets, then
    summed category counts."""
    mappers = []
    for x in X.T:
        codes = [category_codes(x[rows], params) for rows in sites]
        categories = Categories(np.unique(np.concatenate([c[c >= 0] for c in codes])))
        counts = sum(categories.counts(c) for c in codes)
        na_count = sum(int(np.sum(c < 0)) for c in codes)
        mappers.append(agree_categorical_bin_mapper(categories, counts, na_count, len(x), params))
    return mappers


@pytest.mark.parametrize("settings", [*SETTINGS, {"min_data_in_bin": 1}])  # max_bin then decides
def test_categorical_bin_mapping_equals_lightgbm_on_the_pooled_rows(settings):
    X = categorical_features(seed=4)
    sites = np.array_split(np.random.default_rng(5).permutation(len(X)), 3)

    mappers = agree_categorical(X, sites, Params.from_dict(settings))

    lgb_params = {**settings, "bin_construct_sample_cnt": len(X), "feature_pre_filter": False,
                  "enable_bundle": False, "verbose": -1}
    dataset = lgb.Dataset(X, np.zeros(len(X)), categorical_feature=list(range(X.shape[1])), params=lgb_params)
    assert_same_as_lightgbm(dataset, mappers)


@pytest.mark.parametrize("lo, hi", [
    (5.0, 5.0 + 1e-12),
    (1e8, 1e8 + 1e-6),
    (-1e8 - 1e-6, -1e8),
    (1e8, np.nextafter(1e8, 2e8)),  # too close for two cells
    (1e-320, 2e-320),  # subnormal: the width of GRID_CELLS cells underflows to 0
])
def test_grid_between_bounds_below_the_resolution_of_doubles_has_fewer_distinct_cells(lo, hi):
    values = grid(lo, hi).values

    assert 1 <= len(values) < GRID_CELLS
    assert np.all(values[1:] > np.nextafter(values[:-1], np.inf))  # not equal in Common::CheckDoubleEqualOrdered


@pytest.mark.parametrize("lo, hi", [(0.0, 1.0), (-3.0, 7.0), (-1e-300, 1e300), (1e8, 1e8 + 1.0), (2.5, 4.0)])
def test_grid_between_ordinary_bounds_has_the_full_number_of_cells_of_equal_width(lo, hi):
    values = grid(lo, hi).values

    assert len(values) in (GRID_CELLS, GRID_CELLS + 1)  # laid out from 0, so the bounds may cut a cell at each end
    np.testing.assert_allclose(np.diff(values), (hi - lo) / GRID_CELLS, rtol=1e-6)


def test_most_frequent_bin_equals_lightgbm_on_rows_just_above_neighbouring_narrow_grid_cells():
    """Cells two ulps apart (4,096 between 1e6 and 1e6 + 1e-6) are distinct doubles, but the bound between
    them is the upper cell's value, so that cell's rows would be counted in the bin below it."""
    lo, hi = 1e6, 1e6 + 1e-6
    values = grid(lo, hi).values
    k = len(values) // 2
    x = np.where(np.arange(1000) < 200, np.nextafter(values[k], np.inf), np.nextafter(values[k + 1], np.inf))
    x[:2] = lo, hi
    X = x[:, None]

    mappers = agree(X, np.array_split(np.arange(len(X)), 3), Params.from_dict({}))

    with central_dataset(X, np.zeros(len(X)), mappers, {}) as dataset:
        assert_same_as_lightgbm(dataset, mappers)
