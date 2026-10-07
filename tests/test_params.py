import numpy as np
import pytest

from fl_lightgbm.params import Params
from fl_lightgbm.simulator import simulate


def tables():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(np.float64)
    return [(X[:50], y[:50]), (X[50:], y[50:])]


@pytest.mark.parametrize("name, value", [
    ("bagging_fraction", 0.8),
    ("bagging_freq", 1),
    ("pos_bagging_fraction", 0.5),
    ("neg_bagging_fraction", 0.5),
    ("bagging_by_query", True),
    ("data_sample_strategy", "goss"),
    ("boosting", "goss"),
    ("boosting", "dart"),
    ("feature_fraction", 0.5),
    ("feature_fraction_bynode", 0.5),
    ("extra_trees", True),
    ("monotone_constraints", [1, 0, 0]),
    ("interaction_constraints", [[0, 1], [2]]),
    ("linear_tree", True),
    ("path_smooth", 1.0),
    ("use_quantized_grad", True),
    ("weight_column", "name:w"),
])
def test_unsupported_parameters_stop_the_run_before_the_first_round(name, value):
    with pytest.raises(ValueError, match=name):
        simulate({"num_iterations": 1, name: value}, tables())


@pytest.mark.parametrize("name, default", [
    ("bagging_fraction", 1.0),
    ("bagging_freq", 0),
    ("data_sample_strategy", "bagging"),
    ("boosting", "gbdt"),
    ("feature_fraction", 1.0),
    ("extra_trees", False),
    ("monotone_constraints", []),
    ("path_smooth", 0.0),
])
def test_unsupported_parameters_at_their_lightgbm_default_are_accepted(name, default):
    assert Params.from_dict({name: default}) == Params()


@pytest.mark.parametrize("objective", ["multiclass", "regression_l1", "huber", "quantile", "lambdarank", "l2"])
def test_other_objectives_stop_the_run_before_the_first_round(objective):
    with pytest.raises(ValueError, match="objective"):
        simulate({"objective": objective, "num_iterations": 1}, tables())


@pytest.mark.parametrize("name", ["n_estimators", "min_child_samples", "not_a_parameter"])  # aliases are unknown too
def test_unknown_parameter_names_stop_the_run(name):
    with pytest.raises(ValueError, match=name):
        simulate({"num_iterations": 1, name: 10}, tables())


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_scale_pos_weight_must_be_positive(value):
    with pytest.raises(ValueError, match="scale_pos_weight"):
        Params.from_dict({"objective": "binary", "scale_pos_weight": value})


def test_missing_parameters_get_lightgbm_defaults():
    p = Params.from_dict({})

    assert (p.objective, p.num_iterations, p.num_leaves, p.learning_rate) == ("regression", 100, 31, 0.1)
    assert (p.min_data_in_leaf, p.min_sum_hessian_in_leaf, p.max_depth) == (20, 1e-3, -1)
    assert (p.lambda_l1, p.lambda_l2, p.max_delta_step, p.min_gain_to_split) == (0.0, 0.0, 0.0, 0.0)
    assert (p.max_cat_to_onehot, p.max_cat_threshold, p.cat_l2, p.cat_smooth, p.min_data_per_group) == (
        4, 32, 10.0, 10.0, 100)
    assert p.round_budget == 3 + 100 * 30 + 1 + 1  # setup, split, closing and one padding round


@pytest.mark.parametrize("name, value", [
    ("num_iterations", -1),
    ("learning_rate", 0.0),
    ("num_leaves", 1),
    ("num_leaves", 131073),
    ("max_bin", 1),
    ("min_data_in_bin", 0),
    ("min_data_in_leaf", -1),
    ("min_sum_hessian_in_leaf", -1e-3),
    ("lambda_l1", -1.0),
    ("lambda_l2", -1.0),
    ("min_gain_to_split", -1.0),
    ("max_cat_to_onehot", 0),
    ("max_cat_threshold", 0),
    ("cat_l2", -1.0),
    ("cat_smooth", -1.0),
    ("min_data_per_group", 0),
])
def test_values_lightgbm_refuses_stop_the_run(name, value):
    with pytest.raises(ValueError, match=name):
        Params.from_dict({name: value})


@pytest.mark.parametrize("params, num_leaves", [
    ({"max_depth": 3}, 8),  # LightGBM lowers num_leaves to 2^max_depth when num_leaves is not given
    ({"max_depth": 6}, 31),
    ({"max_depth": 3, "num_leaves": 12}, 12),
])
def test_max_depth_without_num_leaves_lowers_the_leaf_budget_and_round_budget(params, num_leaves):
    p = Params.from_dict({"num_iterations": 10, **params})

    assert p.num_leaves == num_leaves
    assert p.round_budget == 3 + 10 * (num_leaves - 1) + 1 + 1
