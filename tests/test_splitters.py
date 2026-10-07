import numpy as np
import pytest

from fl_lightgbm.simulator import split_by_column, split_evenly, split_missing_category, split_skewed


def numbered(n, seed=0):
    """Rows whose first column is the row number, so that a split's rows can be traced back; a binary label."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.arange(n, dtype=np.float64), rng.integers(0, 4, size=n), rng.normal(size=n)])
    return X, (rng.random(n) < 0.4).astype(np.float64)


def assert_partition(X, y, tables):
    """Every row lands at exactly one site, with its own label."""
    rows = np.concatenate([Xs[:, 0] for Xs, _ in tables]).astype(int)
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(X)))
    for Xs, ys in tables:
        np.testing.assert_array_equal(Xs, X[Xs[:, 0].astype(int)])
        np.testing.assert_array_equal(ys, y[Xs[:, 0].astype(int)])


def test_even_split_gives_every_site_the_same_number_of_rows_give_or_take_one():
    X, y = numbered(103)

    tables = split_evenly(X, y, num_sites=5)

    assert_partition(X, y, tables)
    assert sorted(len(ys) for _, ys in tables) == [20, 20, 21, 21, 21]


def test_even_split_mixes_the_rows_and_depends_only_on_the_seed():
    X, y = numbered(100)

    tables = split_evenly(X, y, num_sites=4, seed=1)

    assert not np.array_equal(tables[0][0][:, 0], np.arange(25))
    np.testing.assert_array_equal(tables[0][0], split_evenly(X, y, num_sites=4, seed=1)[0][0])
    assert not np.array_equal(tables[0][0], split_evenly(X, y, num_sites=4, seed=2)[0][0])


def test_skewed_split_has_one_large_site_and_several_tiny_ones():
    X, y = numbered(1000)

    tables = split_skewed(X, y, num_sites=6, large_share=0.8)

    assert_partition(X, y, tables)
    assert [len(ys) for _, ys in tables] == [800, 40, 40, 40, 40, 40]


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_skewed_split_gives_the_tiny_sites_labels_from_low_to_high(objective):
    X, y = numbered(1000)
    if objective == "regression":
        y = np.random.default_rng(1).normal(size=1000)

    tables = split_skewed(X, y, num_sites=5)

    means = [ys.mean() for _, ys in tables[1:]]
    assert means == sorted(means)
    assert means[0] < y.mean() - y.std() / 2 and means[-1] > y.mean() + y.std() / 2
    assert abs(tables[0][1].mean() - y.mean()) < y.std() / 10  # the large site keeps the overall balance


def test_split_by_column_gives_one_site_per_value():
    X, y = numbered(200)
    cohorts = np.array(["b", "a", "c", "a"])[np.arange(200) % 4]

    tables = split_by_column(X, y, cohorts)

    assert_partition(X, y, tables)
    assert [len(ys) for _, ys in tables] == [100, 50, 50]  # in the order of the sorted values
    for (Xs, _), cohort in zip(tables, ["a", "b", "c"]):
        assert set(cohorts[Xs[:, 0].astype(int)]) == {cohort}


def test_split_by_a_feature_column_keeps_the_column():
    X, y = numbered(200)

    tables = split_by_column(X, y, X[:, 1])

    assert [set(Xs[:, 1]) for Xs, _ in tables] == [{0.0}, {1.0}, {2.0}, {3.0}]


def test_missing_category_split_leaves_one_category_out_of_the_first_site_only():
    X, y = numbered(400)

    tables = split_missing_category(X, y, num_sites=4, feature=1, category=2)

    assert_partition(X, y, tables)
    assert 2.0 not in tables[0][0][:, 1]
    assert all(2.0 in Xs[:, 1] for Xs, _ in tables[1:])
    assert {0.0, 1.0, 3.0} <= set(tables[0][0][:, 1])
