"""The fine grid that the rebinning benchmark keeps as an arm: its edges are LightGBM's own binning of the cells."""

import lightgbm as lgb
import numpy as np
import pytest

from baseline import central_dataset
from fine_grid import GRID_CELLS, fine_grid, fine_grid_bin_mapper
from fl_lightgbm.params import Params
from test_binning import SETTINGS, assert_same_as_lightgbm, mixed_features


def agree(X, sites, params):
    """The fine grid's bin edges, without the transport: global bounds, then summed grid counts."""
    mappers = []
    for f in range(X.shape[1]):
        lo = np.fmin.reduce([np.fmin.reduce(X[rows, f]) for rows in sites])
        hi = np.fmax.reduce([np.fmax.reduce(X[rows, f]) for rows in sites])
        g = fine_grid(lo, hi)
        counts = sum(g.counts(X[rows, f]) for rows in sites)
        na_count = sum(int(np.isnan(X[rows, f]).sum()) for rows in sites)
        mappers.append(fine_grid_bin_mapper(g, counts, na_count, len(X), params))
    return mappers


def grid_data(X, sites):
    """The pooled rows as the grid sees them: each row replaced by its grid cell's value, zeros and NaN kept."""
    columns = []
    for f in range(X.shape[1]):
        x = X[:, f]
        g = fine_grid(np.fmin.reduce(x), np.fmax.reduce(x))
        counts = sum(g.counts(x[rows]) for rows in sites)
        na_count = int(np.isnan(x).sum())
        zeros = len(x) - counts.sum() - na_count
        columns.append(np.concatenate([np.repeat(g.values, counts), np.zeros(zeros), np.full(na_count, np.nan)]))
    return np.column_stack(columns)


@pytest.mark.parametrize("settings", SETTINGS)
def test_fine_grid_edges_equal_lightgbm_binning_of_the_grid_cells(settings):
    X = mixed_features(seed=1)
    sites = np.array_split(np.random.default_rng(2).permutation(len(X)), 3)

    mappers = agree(X, sites, Params.from_dict(settings))

    lgb_params = {**settings, "bin_construct_sample_cnt": len(X), "feature_pre_filter": False,
                  "enable_bundle": False, "verbose": -1}
    assert_same_as_lightgbm(lgb.Dataset(grid_data(X, sites), np.zeros(len(X)), params=lgb_params), mappers)


@pytest.mark.parametrize("settings", SETTINGS)
def test_fine_grid_most_frequent_bin_and_missing_type_equal_lightgbm_on_the_pooled_rows(settings):
    X = mixed_features(seed=3)
    sites = np.array_split(np.arange(len(X)), 4)

    mappers = agree(X, sites, Params.from_dict(settings))

    with central_dataset(X, np.zeros(len(X)), mappers, settings) as dataset:
        assert_same_as_lightgbm(dataset, mappers)


@pytest.mark.parametrize("lo, hi", [
    (5.0, 5.0 + 1e-12),
    (1e8, 1e8 + 1e-6),
    (-1e8 - 1e-6, -1e8),
    (1e8, np.nextafter(1e8, 2e8)),  # too close for two cells
    (1e-320, 2e-320),  # subnormal: the width of GRID_CELLS cells underflows to 0
])
def test_fine_grid_between_bounds_below_the_resolution_of_doubles_has_fewer_distinct_cells(lo, hi):
    values = fine_grid(lo, hi).values

    assert 1 <= len(values) < GRID_CELLS
    assert np.all(values[1:] > np.nextafter(values[:-1], np.inf))  # not equal in Common::CheckDoubleEqualOrdered
