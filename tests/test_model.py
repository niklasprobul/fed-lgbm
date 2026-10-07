import lightgbm as lgb
import numpy as np
import pytest

from fl_lightgbm.model import Tree, categorical_decision_type, decision_type, write_model


def two_trees():
    # Tree 0: root splits feature 1 at 0.5; its right leaf (1) splits feature 0 at -0.25 into leaves 1 and 2.
    first = Tree(
        split_feature=[1, 0],
        split_gain=[12.5, 3.25],
        threshold=[0.5, -0.25],
        decision_type=[decision_type(True, "None")] * 2,
        left_child=[-1, -2],
        right_child=[1, -3],
        leaf_value=[1.5, -0.75, 2.0],
        leaf_weight=[10.0, 6.0, 4.0],
        leaf_count=[10, 6, 4],
        internal_value=[0.9, 0.35],
        internal_weight=[20.0, 10.0],
        internal_count=[20, 10],
    )
    constant = Tree(leaf_value=[0.125], leaf_count=[20])
    return [first, constant]


def walk(tree, x):
    if not tree.split_feature:
        return tree.leaf_value[0]
    node = 0
    while node >= 0:
        go_left = x[tree.split_feature[node]] <= tree.threshold[node]
        node = tree.left_child[node] if go_left else tree.right_child[node]
    return tree.leaf_value[~node]


def test_written_model_loads_and_predicts_like_walking_the_trees():
    trees = two_trees()
    text = write_model(trees, feature_names=["a", "b"], feature_ranges=[(-1.0, 1.0), (0.0, 2.0)])

    booster = lgb.Booster(model_str=text)

    X = np.random.default_rng(5).uniform(-1, 2, size=(50, 2))
    expected = [sum(walk(t, x) for t in trees) for x in X]
    np.testing.assert_allclose(booster.predict(X), expected, rtol=1e-15)
    assert booster.feature_name() == ["a", "b"]
    assert booster.num_trees() == 2


def one_split(missing_type, default_left, threshold=0.5):
    return Tree(
        split_feature=[0],
        split_gain=[1.0],
        threshold=[threshold],
        decision_type=[decision_type(default_left, missing_type)],
        left_child=[-1],
        right_child=[-2],
        leaf_value=[-1.0, 1.0],
        leaf_weight=[5.0, 5.0],
        leaf_count=[5, 5],
        internal_value=[0.0],
        internal_weight=[10.0],
        internal_count=[10],
    )


@pytest.mark.parametrize("missing_type, default_left, x, expected", [
    ("NaN", False, np.nan, 1.0),  # NaN follows the default direction, right
    ("NaN", True, np.nan, -1.0),
    ("Zero", False, 0.0, 1.0),  # 0 is missing: right, although 0 <= 0.5
    ("Zero", True, 0.0, -1.0),
    ("None", True, np.nan, -1.0),  # without a missing type NaN is read as 0, which is <= 0.5
    ("None", False, 0.75, 1.0),
])
def test_written_model_routes_missing_values_by_the_default_direction(missing_type, default_left, x, expected):
    text = write_model([one_split(missing_type, default_left)], feature_names=["a"], feature_ranges=[(-1.0, 1.0)])

    booster = lgb.Booster(model_str=text)

    assert booster.predict(np.array([[x]]))[0] == expected
    node = booster.dump_model()["tree_info"][0]["tree_structure"]
    assert (node["missing_type"], node["default_left"]) == (missing_type, default_left)


def test_written_model_loads_an_infinite_threshold():
    """A split of NaN against every other value has the last regular bin's upper bound, +inf, as threshold."""
    text = write_model([one_split("NaN", False, threshold=np.inf)], feature_names=["a"], feature_ranges=[(-1.0, 1.0)])

    booster = lgb.Booster(model_str=text)

    np.testing.assert_array_equal(booster.predict(np.array([[1e308], [np.nan]])), [-1.0, 1.0])


def test_written_model_keeps_gains_and_counts():
    text = write_model(two_trees(), feature_names=["a", "b"], feature_ranges=[(-1.0, 1.0), (0.0, 2.0)])

    root = lgb.Booster(model_str=text).dump_model()["tree_info"][0]["tree_structure"]

    assert root["split_gain"] == 12.5
    assert root["internal_count"] == 20
    assert root["left_child"]["leaf_count"] == 10
    assert root["right_child"]["internal_value"] == 0.35


def categorical_split(categories_left, missing_type="NaN"):
    tree = one_split("None", False)
    tree.threshold = [float(tree.add_category_set(categories_left))]
    tree.decision_type = [categorical_decision_type(missing_type)]
    return tree


@pytest.mark.parametrize("x, expected", [
    (1.0, -1.0),
    (3.0, -1.0),
    (35.0, -1.0),  # in the bitset's second word
    (3.7, -1.0),  # cast to int
    (0.0, 1.0),
    (2.0, 1.0),
    (34.0, 1.0),
    (100.0, 1.0),  # beyond the bitset
    (np.nan, 1.0),
    (-1.0, 1.0),
])
def test_written_categorical_split_sends_its_categories_left_and_every_other_value_right(x, expected):
    text = write_model([categorical_split([3, 35, 1])], feature_names=["a"], feature_ranges=[(0.0, 40.0)],
                       categories=[np.array([1, 3, 35, 2, 0])])

    booster = lgb.Booster(model_str=text)

    assert booster.predict(np.array([[x]]))[0] == expected
    node = booster.dump_model()["tree_info"][0]["tree_structure"]
    assert (node["decision_type"], node["threshold"], node["default_left"]) == ("==", "1||3||35", False)


def test_written_model_keeps_the_category_sets_of_several_categorical_splits_in_one_tree():
    # The root sends category 2 left to leaf 0; its right child sends 0 and 4 left to leaf 1, the rest to leaf 2.
    tree = Tree(
        split_feature=[0, 0],
        split_gain=[2.0, 1.0],
        threshold=[],
        decision_type=[categorical_decision_type("None")] * 2,
        left_child=[-1, -2],
        right_child=[1, -3],
        leaf_value=[-1.0, 0.5, 2.0],
        leaf_weight=[4.0, 3.0, 3.0],
        leaf_count=[4, 3, 3],
        internal_value=[0.0, 1.0],
        internal_weight=[10.0, 6.0],
        internal_count=[10, 6],
    )
    tree.threshold = [float(tree.add_category_set([2])), float(tree.add_category_set([0, 4]))]
    text = write_model([tree], feature_names=["a"], feature_ranges=[(0.0, 4.0)])

    booster = lgb.Booster(model_str=text)

    np.testing.assert_array_equal(booster.predict(np.array([[2.0], [0.0], [4.0], [1.0], [np.nan]])),
                                  [-1.0, 0.5, 0.5, 2.0, 2.0])
