import numpy as np
import pytest

from baseline import train_central, trees
from fl_lightgbm.binning import (
    Categories,
    agree_bin_mapper,
    agree_categorical_bin_mapper,
    bin_values,
    category_codes,
    grid,
    with_most_freq_bin,
)
from fl_lightgbm.histogram import Histogram, leaf_histogram
from fl_lightgbm.params import Params
from fl_lightgbm.split import find_best_split


def regression_data(seed, n=400, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = np.abs(X[:, 1]) + 0.3  # strictly positive feature: no negative zero-bound
    y = 3 * X[:, 2] - 2 * (X[:, 0] > 0.5) + X[:, 1] + rng.normal(scale=0.5, size=n)
    return X, y


def lightgbm_threshold(value):
    """The threshold as dump_model shows it: Common::AvoidInf turns +inf into 1e300."""
    return min(value, 1e300)


def agreed_bin_mappers(X, params, categorical=()):
    """The agreed bins of one site holding all rows: the setup rounds without the transport."""
    bin_mappers = []
    for f, x in enumerate(X.T):
        if f in categorical:
            codes = category_codes(x, params)
            categories = Categories(np.unique(codes[codes >= 0]))
            bin_mappers.append(agree_categorical_bin_mapper(categories, categories.counts(codes),
                                                            int(np.sum(codes < 0)), len(x), params))
            continue
        g = grid(np.fmin.reduce(x), np.fmax.reduce(x))
        mapper = agree_bin_mapper(g, g.counts(x), int(np.isnan(x).sum()), len(x), params)
        bin_mappers.append(with_most_freq_bin(mapper, mapper.counts(x), len(x)))
    binned = bin_values(X, bin_mappers)
    return bin_mappers, binned


@pytest.mark.parametrize("settings", [
    {},
    {"lambda_l2": 5.0},
    {"min_data_in_leaf": 150},
    {"max_bin": 16},
    {"lambda_l1": 100.0},
    {"max_delta_step": 0.5},  # clips both children's outputs
    {"lambda_l1": 100.0, "max_delta_step": 0.5},
    {"min_gain_to_split": 50.0},  # lowers the written gain
])
def test_chosen_split_equals_lightgbm_root_split(settings):
    X, y = regression_data(seed=3)
    lgb_params = {"num_leaves": 2, "num_iterations": 1, "learning_rate": 1.0, **settings}
    params = Params.from_dict(lgb_params)
    bin_mappers, binned = agreed_bin_mappers(X, params)
    edges = [f.upper_bounds for f in bin_mappers]
    num_bins = np.array([f.num_bins for f in bin_mappers])

    y32 = y.astype(np.float32)
    init_score = np.sum(y32.astype(np.float64)) / len(y)
    g = (init_score - y32.astype(np.float64)).astype(np.float32)
    h = np.ones(len(y), dtype=np.float32)
    # The aggregated histogram: the sum of three sites' histograms over their own rows.
    site_hists = [leaf_histogram(binned[rows], num_bins, g[rows], h[rows], np.arange(len(rows)))
                  for rows in np.array_split(np.arange(len(y)), 3)]
    hist = Histogram(sum(s.grad for s in site_hists), sum(s.hess for s in site_hists), len(y))

    split, _ = find_best_split(hist, g.astype(np.float64).sum(), float(len(y)), bin_mappers, params)

    root = trees(train_central(X, y, bin_mappers, lgb_params))[0]["tree_structure"]
    assert split.feature == root["split_feature"]
    assert edges[split.feature][split.threshold] == root["threshold"]
    # LightGBM stores gains as float32, and writes them with min_gain_to_split added back.
    assert np.float32(split.gain + params.min_gain_to_split) == root["split_gain"]
    assert split.left_output + init_score == pytest.approx(root["left_child"]["leaf_value"], rel=1e-9)
    assert split.right_output + init_score == pytest.approx(root["right_child"]["leaf_value"], rel=1e-9)
    assert (split.left_count, split.right_count) == (
        root["left_child"]["leaf_count"], root["right_child"]["leaf_count"])


def missing_value_data(seed, missing_y, n=400):
    """Feature 0 is NaN in a third of the rows, whose target is `missing_y`; the others follow x0 > 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * (X[:, 0] > 0) + 0.3 * X[:, 1] + rng.normal(scale=0.2, size=n)
    nan_rows = rng.random(n) < 1 / 3
    X[nan_rows, 0] = np.nan
    y[nan_rows] = missing_y + rng.normal(scale=0.2, size=nan_rows.sum())
    X[rng.random(n) < 0.1, 2] = np.nan  # a second feature with missing values that does not matter
    return X, y


def zero_data(seed, n=400):
    """Feature 0 is exactly 0 in a third of the rows, whose target is high."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * (X[:, 0] > 0.5) + rng.normal(scale=0.2, size=n)
    zero_rows = rng.random(n) < 1 / 3
    X[zero_rows, 0] = 0.0
    y[zero_rows] = 3.0
    return X, y


@pytest.mark.parametrize("data, settings, missing_type, default_left", [
    (missing_value_data(5, missing_y=2.0), {}, "NaN", False),  # NaN rows join the high side, on the right
    (missing_value_data(6, missing_y=0.0), {}, "NaN", True),  # NaN rows join the low side, on the left
    (missing_value_data(7, missing_y=2.0), {"min_data_in_leaf": 120}, "NaN", False),
    (missing_value_data(8, missing_y=2.0), {"use_missing": False}, "None", True),  # NaN is read as 0
    (zero_data(9), {"zero_as_missing": True}, "Zero", False),
])
def test_split_with_missing_values_equals_lightgbm_root_split(data, settings, missing_type, default_left):
    X, y = data
    lgb_params = {"num_leaves": 2, "num_iterations": 1, "learning_rate": 1.0, **settings}
    params = Params.from_dict(lgb_params)
    bin_mappers, binned = agreed_bin_mappers(X, params)

    y32 = y.astype(np.float32)
    init_score = np.sum(y32.astype(np.float64)) / len(y)
    g = (init_score - y32.astype(np.float64)).astype(np.float32)
    h = np.ones(len(y), dtype=np.float32)
    hist = leaf_histogram(binned, np.array([f.num_bins for f in bin_mappers]), g, h, np.arange(len(y)))

    split, _ = find_best_split(hist, g.astype(np.float64).sum(), float(len(y)), bin_mappers, params)

    root = trees(train_central(X, y, bin_mappers, lgb_params))[0]["tree_structure"]
    assert (root["missing_type"], root["default_left"]) == (missing_type, default_left)  # the case is what it claims
    assert split.feature == root["split_feature"]
    assert lightgbm_threshold(bin_mappers[split.feature].upper_bounds[split.threshold]) == root["threshold"]
    assert split.default_left == root["default_left"]
    assert np.float32(split.gain) == root["split_gain"]
    assert split.left_output + init_score == pytest.approx(root["left_child"]["leaf_value"], rel=1e-9)
    assert split.right_output + init_score == pytest.approx(root["right_child"]["leaf_value"], rel=1e-9)
    assert (split.left_count, split.right_count) == (
        root["left_child"]["leaf_count"], root["right_child"]["leaf_count"])


def categorical_data(seed, few_effect, many_effect, n=1200):
    """Feature 0 has three categories, feature 1 has 40 with NaN in some rows and a few rare categories,
    feature 2 is numerical; the target follows category 1 of feature 0 and every category of feature 1."""
    rng = np.random.default_rng(seed)
    few = rng.integers(0, 3, size=n).astype(np.float64)
    many = rng.integers(0, 40, size=n).astype(np.float64)
    many[rng.random(n) < 0.05] = np.nan
    many[rng.random(n) < 0.01] = 100 + rng.integers(0, 5)  # fewer rows than min_data_in_bin: bin 0
    y = few_effect * (few == 1) + many_effect * rng.normal(size=105)[np.nan_to_num(many, nan=104).astype(int) % 105]
    X = np.column_stack([few, many, rng.normal(size=n)])
    return X, y + 0.1 * X[:, 2] + rng.normal(scale=0.2, size=n)


def categories_left(mapper, split):
    """The categories a split sends left, as dump_model writes a categorical threshold."""
    return "||".join(str(c) for c in sorted(mapper.categories[np.array(split.left_bins) - 1]))


@pytest.mark.parametrize("effects, settings, feature, num_categories", [
    ((3.0, 0.3), {}, 0, 1),  # three categories and bin 0: one category against the rest
    ((0.0, 1.0), {}, 1, None),  # sorted by Σg / (Σh + cat_smooth)
    ((0.0, 1.0), {"max_cat_threshold": 4}, 1, 4),  # four categories of about 30 rows: one group of 100
    ((0.0, 1.0), {"max_cat_to_onehot": 64}, 1, 1),
    ((0.0, 1.0), {"cat_smooth": 30.0}, 1, None),  # categories with fewer rows are left out of the scan
    ((0.0, 1.0), {"cat_l2": 0.0}, 1, None),
    ((0.0, 1.0), {"min_data_per_group": 250}, 1, None),
    ((0.0, 1.0), {"min_data_in_leaf": 450}, 1, None),
    ((0.0, 1.0), {"lambda_l1": 20.0, "max_delta_step": 0.5, "min_gain_to_split": 5.0}, 1, None),
    ((3.0, 0.3), {"min_sum_hessian_in_leaf": 350.0}, 0, 1),
])
def test_categorical_split_equals_lightgbm_root_split(effects, settings, feature, num_categories):
    X, y = categorical_data(11, *effects)
    lgb_params = {"num_leaves": 2, "num_iterations": 1, "learning_rate": 1.0, **settings}
    params = Params.from_dict(lgb_params)
    bin_mappers, binned = agreed_bin_mappers(X, params, categorical=(0, 1))

    y32 = y.astype(np.float32)
    init_score = np.sum(y32.astype(np.float64)) / len(y)
    g = (init_score - y32.astype(np.float64)).astype(np.float32)
    h = np.ones(len(y), dtype=np.float32)
    hist = leaf_histogram(binned, np.array([f.num_bins for f in bin_mappers]), g, h, np.arange(len(y)))

    split, _ = find_best_split(hist, g.astype(np.float64).sum(), float(len(y)), bin_mappers, params)

    root = trees(train_central(X, y, bin_mappers, lgb_params))[0]["tree_structure"]
    assert (root["split_feature"], root["decision_type"]) == (feature, "==")  # the case is what it claims
    if num_categories is not None:
        assert len(root["threshold"].split("||")) == num_categories
    assert split.feature == root["split_feature"]
    assert categories_left(bin_mappers[split.feature], split) == root["threshold"]
    assert split.default_left == root["default_left"]
    assert np.float32(split.gain + params.min_gain_to_split) == root["split_gain"]
    assert split.left_output + init_score == pytest.approx(root["left_child"]["leaf_value"], rel=1e-9)
    assert split.right_output + init_score == pytest.approx(root["right_child"]["leaf_value"], rel=1e-9)
    assert (split.left_count, split.right_count) == (
        root["left_child"]["leaf_count"], root["right_child"]["leaf_count"])


def test_no_split_when_min_data_in_leaf_cannot_be_met():
    X, y = regression_data(seed=4, n=30)
    params = Params.from_dict({"min_data_in_leaf": 20})
    bin_mappers, binned = agreed_bin_mappers(X, params)
    g = (y.mean() - y).astype(np.float32)
    hist = leaf_histogram(binned, np.array([f.num_bins for f in bin_mappers]), g, np.ones(30, np.float32), np.arange(30))

    split, splittable = find_best_split(hist, g.astype(np.float64).sum(), 30.0, bin_mappers, params)

    assert split is None
    assert not splittable.any()
