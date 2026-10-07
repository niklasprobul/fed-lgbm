import json
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pytest

from baseline import central_dataset, train_central, trees
from comparison import NEAR_TIE, SCAN_TIE, ZERO_BIN, assert_same_model
from fl_lightgbm.aggregator import Aggregator, SchemaError
from fl_lightgbm.encoding import DENSE, SPARSE, FixedPoint
from fl_lightgbm.objective import METRIC
from fl_lightgbm.params import Params
from fl_lightgbm.rounds import RoundCountError
from fl_lightgbm.simulator import run, simulate
from fl_lightgbm.site import Site


def synthetic(seed, n, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 3] = rng.uniform(0.5, 4.0, size=n)  # positive only
    X[:, 4] = -rng.exponential(size=n)  # negative only
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 3] - 0.5 * X[:, 4] + rng.normal(scale=0.3, size=n) + 5.0
    return X, y


def synthetic_binary(seed, n, p=6):
    X, y = synthetic(seed, n, p)
    return X, (y > np.median(y)).astype(np.float64)


def split_into_sites(X, y, sizes):
    bounds = np.cumsum([0, *sizes])
    return [(X[a:b], y[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


def skewed_sites(X, y, sizes, positive_shares, seed=0):
    """Sites of the given sizes, each with its own share of positive rows."""
    rng = np.random.default_rng(seed)
    pos, neg = rng.permutation(np.flatnonzero(y > 0)), rng.permutation(np.flatnonzero(y == 0))
    tables = []
    for size, share in zip(sizes, positive_shares):
        k = round(size * share)
        rows = np.concatenate([pos[:k], neg[:size - k]])
        pos, neg = pos[k:], neg[size - k:]
        tables.append((X[rows], y[rows]))
    return tables


def train_and_compare(params, tables, X_test, accepted=None, categorical=()):
    """Train federated and central; held-out rows are compared only where the comparison accepted no
    difference, because a held-out value in an empty range, or a missing value, may then go either way."""
    result = simulate(params, tables, categorical=categorical)
    X_train = np.vstack([X for X, _ in tables])
    y_train = np.concatenate([y for _, y in tables])

    ours = lgb.Booster(model_str=result.model)
    central = train_central(X_train, y_train, result.agreed_edges, params)
    differences = assert_same_model(ours, central, X_train, accepted)
    np.testing.assert_allclose(ours.predict(X_train), central.predict(X_train), rtol=1e-9)
    if not differences:
        np.testing.assert_allclose(ours.predict(X_test), central.predict(X_test), rtol=1e-9)
    return result


def test_simulated_run_equals_central_baseline():
    X, y = synthetic(seed=7, n=1500)
    params = {"num_iterations": 12, "num_leaves": 15, "learning_rate": 0.2, "lambda_l2": 1.0}

    train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])


def test_run_has_exactly_the_round_budget_and_pads_when_trees_stop_early():
    X, y = synthetic(seed=8, n=150)
    params = {"num_iterations": 5, "num_leaves": 31, "min_data_in_leaf": 20}

    result = simulate(params, split_into_sites(X, y, [70, 50, 30]))

    assert result.rounds == 3 + 5 * 30 + 1 + 1  # setup rounds, split rounds, closing round, one padding round
    assert result.padding_rounds > 0
    central = train_central(X, y, result.agreed_edges, params)
    assert_same_model(lgb.Booster(model_str=result.model), central, X)


def test_training_stops_like_lightgbm_when_no_tree_can_split():
    X, y = synthetic(seed=9, n=60)
    params = {"num_iterations": 4, "num_leaves": 4, "min_data_in_leaf": 40}  # 60 rows cannot give two leaves of 40

    result = simulate(params, split_into_sites(X, y, [30, 30]))

    assert result.padding_rounds == result.rounds - 5  # three setup rounds, one root histogram, the closing round
    central = train_central(X, y, result.agreed_edges, params)
    assert_same_model(lgb.Booster(model_str=result.model), central, X)


def test_simulated_binary_run_equals_central_baseline():
    X, y = synthetic_binary(seed=21, n=1500)
    params = {"objective": "binary", "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])

    # Every tree used its whole leaf budget, so the last tree's last leaf counts came from the closing round,
    # and only the padding round that always follows it was left.
    assert result.padding_rounds == 1


def test_binary_init_score_is_global_when_sites_differ_in_size_and_label_balance():
    X, y = synthetic_binary(seed=22, n=1500)
    tables = skewed_sites(X[:1200], y[:1200], sizes=[800, 150, 60], positive_shares=[0.3, 0.9, 0.05])
    params = {"objective": "binary", "num_iterations": 8, "num_leaves": 8}

    train_and_compare(params, tables, X[1200:])


@pytest.mark.parametrize("weighting", [{"is_unbalance": True}, {"scale_pos_weight": 3.5}])
def test_binary_label_weighting_equals_central_baseline(weighting):
    X, y = synthetic_binary(seed=23, n=1300)
    tables = skewed_sites(X[:1000], y[:1000], sizes=[500, 300, 200], positive_shares=[0.1, 0.3, 0.2])
    params = {"objective": "binary", "num_iterations": 8, "num_leaves": 10, **weighting}

    train_and_compare(params, tables, X[1000:])


# Each value binds for both objectives: the model differs from the one with LightGBM's default. The
# other supported parameters are set away from their defaults in the tests above and below.
@pytest.mark.parametrize("setting", [
    {"max_bin": 31},
    {"min_data_in_bin": 30},
    {"min_data_in_leaf": 60},
    {"min_sum_hessian_in_leaf": 30.0},
    {"lambda_l1": 10.0},
    {"lambda_l2": 10.0},
    {"max_delta_step": 0.3},
    {"min_gain_to_split": 5.0},
    {"max_depth": 3},
], ids=lambda setting: next(iter(setting)))
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_each_parameter_set_away_from_its_default_equals_central_baseline(objective, setting):
    X, y = with_missing_values(*synthetic(seed=71, n=1500), seed=72, shift=2.0)
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])

    result = train_and_compare({**params, **setting}, tables, X[1200:])

    assert result.model != simulate(params, tables).model


def test_max_depth_without_num_leaves_equals_central_baseline_with_fewer_rounds():
    X, y = synthetic(seed=73, n=1500)
    params = {"num_iterations": 10, "learning_rate": 0.3, "max_depth": 3}  # LightGBM lowers num_leaves to 8

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])

    assert result.rounds == 3 + 10 * 7 + 1 + 1


def leaf_counts(node):
    if "leaf_index" in node:
        return [node["leaf_count"]]
    return leaf_counts(node["left_child"]) + leaf_counts(node["right_child"])


def test_min_data_in_leaf_is_checked_on_lightgbms_count_estimate():
    # For logloss, LightGBM checks `min_data_in_leaf` on round(h · n / H), which differs from the true
    # count: it keeps leaves with fewer rows than the limit, and the federated model keeps the same ones.
    X, y = synthetic_binary(seed=81, n=1500)
    params = {"objective": "binary", "num_iterations": 10, "num_leaves": 31, "learning_rate": 0.3,
              "min_data_in_leaf": 50}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])

    counts = [c for tree in trees(lgb.Booster(model_str=result.model)) for c in leaf_counts(tree["tree_structure"])]
    assert min(counts) < 50


def with_missing_values(X, y, seed, shift, value=np.nan, features=(0, 2, 5), share=0.2):
    """`value` in a share of the rows of several features. The rows missing feature 0 get `shift` added
    to their target, so being missing pays off on the high side (shift > 0) or the low side (shift < 0)."""
    rng = np.random.default_rng(seed)
    X, y = X.copy(), y.copy()
    for f in features:
        X[rng.random(len(X)) < share, f] = value
    y[np.isnan(X[:, 0]) if np.isnan(value) else X[:, 0] == value] += shift
    return X, y


def split_kinds(booster):
    """(missing_type, default_left) of every split in the model."""
    kinds = set()

    def walk(node):
        if "split_index" in node:
            kinds.add((node["missing_type"], node["default_left"]))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in trees(booster):
        walk(tree["tree_structure"])
    return kinds


@pytest.mark.parametrize("objective", ["regression", "binary"])
@pytest.mark.parametrize("shift", [3.0, -3.0])
def test_missing_values_in_several_features_equal_central_baseline(objective, shift):
    X, y = with_missing_values(*synthetic(seed=31, n=1500), seed=32, shift=shift)
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    held_out = X[1200:]
    rows_with_missing_values = held_out[np.isnan(held_out).any(axis=1)]

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), rows_with_missing_values)

    # Both scan directions chose splits: missing values went left in some and right in others.
    assert {("NaN", True), ("NaN", False)} <= split_kinds(lgb.Booster(model_str=result.model))


def test_missing_values_in_a_positive_feature_equal_central_baseline_where_min_data_in_leaf_binds():
    # Feature 3 is positive, so its most frequent bin is bin 0, which LightGBM's histograms leave out:
    # the left-to-right scan derives it from the leaf totals, and its hessian-based count estimate
    # differs from the bin's own, which shows where `min_data_in_leaf` binds.
    X, y = with_missing_values(*synthetic(seed=501, n=1500), seed=601, shift=2.0, features=(3, 0), share=0.25)
    y = (y > np.median(y)).astype(np.float64)
    params = {"objective": "binary", "num_iterations": 8, "num_leaves": 31, "learning_rate": 0.3, "min_data_in_leaf": 40}

    train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])


def test_site_without_missing_values_in_a_feature_others_miss_equals_central_baseline():
    X, y = synthetic(seed=33, n=1500)
    X, y = with_missing_values(X, y, seed=34, shift=2.0, features=(0,))
    X[:500, 0] = np.where(np.isnan(X[:500, 0]), 0.5, X[:500, 0])  # site 0 has no missing values in feature 0
    X[1000:1200, 3] = np.where(np.arange(200) % 4 == 0, np.nan, X[1000:1200, 3])  # only site 2 misses feature 3
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 500, 200]), X[1200:])

    assert ("NaN", False) in split_kinds(lgb.Booster(model_str=result.model))


@pytest.mark.slow
def test_scan_ties_and_a_zero_bin_difference_of_a_long_binary_run_are_accepted():
    # Saturated to tiny hessians, where central LightGBM retrained on reordered rows flips the same
    # kind of ties against itself. Two more nodes have a zero bin that holds no rows of the leaf, so
    # the zero bin's residual moves the threshold across it.
    X, y = with_missing_values(*synthetic_binary(seed=49, n=3000), seed=50, shift=0.0)
    params = {"objective": "binary", "num_iterations": 200, "num_leaves": 31, "learning_rate": 0.3,
              "min_data_in_leaf": 5}

    train_and_compare(params, split_into_sites(X[:2400], y[:2400], [1000, 800, 600]), X[2400:],
                      accepted={SCAN_TIE: 7, ZERO_BIN: 2})


@pytest.mark.parametrize("settings, value, kind", [
    ({"use_missing": False}, np.nan, ("None", True)),  # NaN is read as 0
    ({"zero_as_missing": True}, 0.0, ("Zero", False)),
    ({"zero_as_missing": True}, np.nan, ("Zero", False)),  # NaN is read as 0, so it is missing too
])
def test_missing_value_settings_equal_central_baseline(settings, value, kind):
    X, y = with_missing_values(*synthetic(seed=35, n=1500), seed=36, shift=3.0, value=value)
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3, **settings}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])

    assert kind in split_kinds(lgb.Booster(model_str=result.model))


def sites_and_aggregator(params, tables):
    p = Params.from_dict(params)
    names = [f"Column_{i}" for i in range(tables[0][0].shape[1])]
    return [Site(X, y, p, names) for X, y in tables], Aggregator(p)


@pytest.mark.parametrize("extra", [1, -1])
def test_one_round_more_or_fewer_than_the_budget_is_an_error(extra):
    X, y = synthetic(seed=10, n=200)
    params = {"num_iterations": 3, "num_leaves": 4}
    sites, aggregator = sites_and_aggregator(params, split_into_sites(X, y, [100, 100]))

    with pytest.raises(RoundCountError):
        run(sites, aggregator, rounds=Params.from_dict(params).round_budget + extra)


@pytest.mark.parametrize("n, params", [
    (200, {"num_iterations": 3, "num_leaves": 4}),  # every tree uses its leaf budget: the closing round is the last but one
    (60, {"num_iterations": 4, "num_leaves": 4, "min_data_in_leaf": 40}),  # no tree can split: many padding rounds
    (60, {"num_iterations": 0}),  # no tree at all
])
def test_every_site_receives_the_model_and_report_before_the_last_round(n, params):
    X, y = synthetic(seed=10, n=n)
    sites, _ = sites_and_aggregator(params, split_into_sites(X, y, [n // 2, n // 2]))
    p = Params.from_dict(params)
    aggregator = RecordingAggregator(p)

    run(sites, aggregator, rounds=p.round_budget)

    for site in sites:
        assert site.model == aggregator.model()
        assert site.report == aggregator.report()
    # The last reply is a padding round's: FL-Net ships it in its finish message, so it stays small.
    assert len(json.dumps(aggregator.last_reply)) < 100


def test_sites_with_different_columns_stop_the_run():
    X, y = synthetic(seed=12, n=100)
    params = Params.from_dict({"num_iterations": 1, "num_leaves": 2})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    sites = [Site(X[:50], y[:50], params, names), Site(X[50:], y[50:], params, names[::-1])]

    with pytest.raises(SchemaError, match="site 1"):
        run(sites, Aggregator(params), rounds=params.round_budget)


class RecordingSite(Site):
    def __init__(self, *args):
        super().__init__(*args)
        self.replies = []
        self.payloads = []

    def payload(self, reply):
        if reply is not None:
            self.replies.append(reply)
        self.payloads.append(super().payload(reply))
        return self.payloads[-1]


class RecordingAggregator(Aggregator):
    def reply(self, payloads):
        self.last_reply = super().reply(payloads)
        return self.last_reply


SPLIT_REPLY_KEYS = {"splits", "leaf_values", "histogram_leaves"}


NUMERICAL_SPLIT_KEYS = {"leaf", "feature", "threshold_bin", "default_left", "right_leaf"}
CATEGORICAL_SPLIT_KEYS = {"leaf", "feature", "left_bins", "right_leaf"}


@pytest.mark.parametrize("objective, setup_keys", [
    ("regression", {"bounds", "categories", "init_score"}),
    ("binary", {"bounds", "categories", "init_score", "label_weights"}),
])
def test_sites_receive_only_split_decisions_and_leaf_values(objective, setup_keys):
    X, y = synthetic_categorical(seed=11, n=600, objective=objective)
    params = Params.from_dict({"objective": objective, "num_iterations": 3, "num_leaves": 6})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    sites = [RecordingSite(Xs, ys, params, names, None, SPARSE, CATEGORICAL)
             for Xs, ys in split_into_sites(X, y, [300, 300])]

    run(sites, Aggregator(params), rounds=params.round_budget)

    first_setup_reply, second_setup_reply, third_setup_reply, *split_replies = sites[0].replies
    assert first_setup_reply.keys() == setup_keys
    assert second_setup_reply.keys() == {"bin_upper_bounds", "missing_types", "bin_categories"}
    assert third_setup_reply.keys() == {"most_freq_bins"} | SPLIT_REPLY_KEYS
    # Besides them, one reply delivers the finished model and training report, which sites save.
    delivery = [i for i, reply in enumerate(split_replies) if "model" in reply]
    assert len(delivery) == 1
    assert split_replies[delivery[0]].keys() == SPLIT_REPLY_KEYS | {"model", "report"}
    split_keys = []
    for reply in split_replies:
        reply = {k: v for k, v in reply.items() if k not in ("model", "report")}
        assert reply.keys() == SPLIT_REPLY_KEYS
        split_keys += [split.keys() for split in reply["splits"]]
    assert NUMERICAL_SPLIT_KEYS in split_keys and CATEGORICAL_SPLIT_KEYS in split_keys
    assert all(keys in (NUMERICAL_SPLIT_KEYS, CATEGORICAL_SPLIT_KEYS) for keys in split_keys)


def test_agreed_edges_do_not_depend_on_how_rows_are_split_across_sites():
    X, y = with_missing_values(*synthetic(seed=39, n=1200), seed=40, shift=1.0)
    shuffled = np.random.default_rng(41).permutation(len(X))
    partitions = [
        split_into_sites(X, y, [1200]),
        split_into_sites(X, y, [400, 400, 400]),
        split_into_sites(X[shuffled], y[shuffled], [900, 250, 40, 10]),
    ]

    agreed = [simulate({"num_iterations": 1, "num_leaves": 2}, tables).agreed_edges for tables in partitions]

    for other in agreed[1:]:
        for ours, theirs in zip(agreed[0], other):
            np.testing.assert_array_equal(ours.upper_bounds, theirs.upper_bounds)
            assert (ours.missing_type, ours.most_freq_bin) == (theirs.missing_type, theirs.most_freq_bin)


# Per-feature bounds around `synthetic`: features 3 and 4 are positive and negative only.
BOUNDS = [(-6.0, 6.0)] * 3 + [(0.0, 5.0), (-12.0, 0.0), (-6.0, 6.0)]


def test_configured_bounds_replace_the_sites_min_max_and_equal_central_baseline():
    X, y = with_missing_values(*synthetic(seed=37, n=1500), seed=38, shift=2.0)
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    p = Params.from_dict(params)
    names = [f"Column_{i}" for i in range(X.shape[1])]
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])
    sites = [RecordingSite(Xs, ys, p, names, BOUNDS) for Xs, ys in tables]
    aggregator = Aggregator(p, BOUNDS)

    run(sites, aggregator, rounds=p.round_budget)

    assert not any({"min", "max"} & payload.keys() for site in sites for payload in site.payloads)
    assert sites[0].replies[0]["bounds"] == [list(b) for b in BOUNDS]
    ours = lgb.Booster(model_str=aggregator.model())
    central = train_central(X[:1200], y[:1200], aggregator.bin_mappers, params)
    assert_same_model(ours, central, X[:1200])
    np.testing.assert_allclose(ours.predict(X[1200:]), central.predict(X[1200:]), rtol=1e-9)


def test_a_site_with_values_outside_the_configured_bounds_refuses_to_start():
    X, y = synthetic(seed=13, n=100)
    names = [f"Column_{i}" for i in range(X.shape[1])]
    bounds = [*BOUNDS[:4], (0.0, 1.0), BOUNDS[5]]  # feature 4 is negative

    with pytest.raises(ValueError, match="Column_4"):
        Site(X, y, Params.from_dict({}), names, bounds)


def test_bounds_for_another_number_of_features_stop_the_run():
    X, y = synthetic(seed=14, n=100)

    with pytest.raises(ValueError, match="bounds are configured for 5 features"):
        simulate({"num_iterations": 1, "num_leaves": 2}, split_into_sites(X, y, [50, 50]), BOUNDS[:5])


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_sparse_encoding_gives_the_same_model_as_dense_float64(objective):
    # Missing values, a positive feature (its most frequent bin is bin 0) and a feature that is mostly
    # zero (its most frequent bin holds rows, and many of its bins stay empty in small leaves).
    X, y = with_missing_values(*synthetic(seed=51, n=1200), seed=52, shift=2.0)
    X[np.random.default_rng(53).random(len(X)) < 0.8, 1] = 0.0
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    tables = split_into_sites(X, y, [500, 380, 320])

    sparse, dense = simulate(params, tables), simulate(params, tables, encoding=DENSE)

    assert sparse.model == dense.model


class PayloadSizeSite(Site):
    """Records the JSON size of every payload that carries histograms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_round_sizes = []

    def payload(self, reply):
        payload = super().payload(reply)
        if payload.get("leaves"):
            self.split_round_sizes.append(len(json.dumps(payload)))
        return payload


def test_sparse_split_round_payloads_are_an_order_of_magnitude_smaller_than_dense_json(capsys):
    # 2,000 features that are zero in 95 % of the rows, like the microbiome taxa.
    rng = np.random.default_rng(61)
    X = rng.lognormal(size=(1200, 2000)) * (rng.random((1200, 2000)) > 0.95)
    y = X[:, :5].sum(axis=1) + rng.normal(size=1200)
    params = Params.from_dict({"num_iterations": 1, "num_leaves": 8})
    names = [f"Column_{i}" for i in range(X.shape[1])]

    sizes = {}
    for name, encoding in [("sparse", SPARSE), ("dense", DENSE)]:
        sites = [PayloadSizeSite(Xs, ys, params, names, encoding=encoding)
                 for Xs, ys in split_into_sites(X, y, [400, 400, 400])]
        run(sites, Aggregator(params, encoding=encoding), rounds=params.round_budget)
        sizes[name] = np.array(sites[0].split_round_sizes)

    ratios = sizes["dense"] / sizes["sparse"]  # the same rounds: both encodings grow the same tree
    with capsys.disabled():
        print(f"\nsplit-round payloads of site 0, 2,000 features, 95 % zeros, one tree of 8 leaves:\n"
              f"  root:       sparse {sizes['sparse'][0]:,} B, dense JSON {sizes['dense'][0]:,} B, {ratios[0]:.1f}x\n"
              f"  children:   sparse {sizes['sparse'][1:].tolist()} B, dense/sparse {np.round(ratios[1:], 1).tolist()}\n"
              f"  whole tree: sparse {sizes['sparse'].sum():,} B, dense JSON {sizes['dense'].sum():,} B, "
              f"{sizes['dense'].sum() / sizes['sparse'].sum():.1f}x")
    # The root round misses: a site's root leaf holds all its rows, so almost every bin is non-empty.
    assert ratios[0] > 1
    assert np.all(ratios[1:] >= 10)


def synthetic_categorical(seed, n, objective="regression"):
    """`synthetic` plus categorical features the target follows: one of three categories (6, split one
    against the rest), one of 60 (7, split by sorted categories) and one of two (8, whose two one-hot
    candidates are the same partition with exactly the same gain, so the first must win)."""
    X, y = synthetic(seed, n)
    rng = np.random.default_rng(seed + 1)
    few = rng.integers(0, 3, size=n)
    many = rng.integers(0, 60, size=n)
    two = rng.integers(0, 2, size=n)
    y = y + 1.5 * (few == 2) + rng.normal(size=60)[many] + 1.0 * two
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    return np.column_stack([X, few, many, two]), y


CATEGORICAL = (6, 7, 8)


def categorical_splits(booster):
    """(feature, the categories sent left) of every categorical split in the model."""
    splits = []

    def walk(node):
        if "split_index" in node:
            if node["decision_type"] == "==":
                splits.append((node["split_feature"], [int(c) for c in node["threshold"].split("||")]))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in trees(booster):
        walk(tree["tree_structure"])
    return splits


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_categorical_features_equal_central_baseline_in_one_hot_and_sorted_mode(objective):
    X, y = synthetic_categorical(seed=91, n=1500, objective=objective)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:],
                               categorical=CATEGORICAL)

    splits = categorical_splits(lgb.Booster(model_str=result.model))
    assert any(feature == 6 and len(left) == 1 for feature, left in splits)  # one category against the rest
    assert any(feature == 7 and len(left) > 1 for feature, left in splits)
    assert any(feature == 8 for feature, _ in splits)


def test_a_category_present_at_only_one_site_is_routed_like_central_baseline():
    X, y = synthetic_categorical(seed=92, n=1500)
    only_at_site_2 = np.flatnonzero(np.arange(1200) >= 1000)[::5]
    X[only_at_site_2, 7] = 60.0
    y[only_at_site_2] += 4.0
    held_out = X[1200:].copy()
    held_out[::3, 7] = 60.0
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 500, 200]), held_out,
                               categorical=CATEGORICAL)

    assert 60 in result.agreed_edges[7].categories
    # Some split sends category 60 left, so rows of that category take another path than unseen ones.
    assert any(feature == 7 and 60 in left for feature, left in categorical_splits(lgb.Booster(model_str=result.model)))


# Each value binds: the model differs from the one with LightGBM's default.
@pytest.mark.parametrize("setting", [
    {"max_cat_to_onehot": 64},  # feature 7 is split one category against the rest too
    {"max_cat_threshold": 5},
    {"cat_l2": 1.0},
    {"cat_smooth": 25.0},
    {"min_data_per_group": 30},
], ids=lambda setting: next(iter(setting)))
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_each_categorical_parameter_set_away_from_its_default_equals_central_baseline(objective, setting):
    X, y = synthetic_categorical(seed=93, n=1500, objective=objective)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])

    result = train_and_compare({**params, **setting}, tables, X[1200:], categorical=CATEGORICAL)

    assert result.model != simulate(params, tables, categorical=CATEGORICAL).model


def test_rows_with_categories_unseen_in_training_are_predicted_like_central_baseline():
    X, y = synthetic_categorical(seed=94, n=1500)
    X[np.random.default_rng(95).random(len(X)) < 0.05, 7] = np.nan
    held_out = X[1200:].copy()
    held_out[0::4, 7] = np.arange(len(held_out[0::4])) % 20 + 60  # categories no site has
    held_out[1::4, 6] = 3.0
    held_out[2::4, 7] = -2.0  # negative values: the NaN bin, as in training
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])

    train_and_compare(params, tables, held_out, categorical=CATEGORICAL)


def test_sites_marking_different_columns_categorical_stop_the_run():
    X, y = synthetic_categorical(seed=96, n=100)
    params = Params.from_dict({"num_iterations": 1, "num_leaves": 2})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    sites = [Site(X[:50], y[:50], params, names, categorical=CATEGORICAL),
             Site(X[50:], y[50:], params, names, categorical=(6,))]

    with pytest.raises(SchemaError, match="site 1"):
        run(sites, Aggregator(params), rounds=params.round_budget)


# The fixed-point encoding rounds each site's sums to 2^-26 (about 1.5e-8). With that exponent the
# model stays within 1e-6 (relative or absolute) of the central baseline: gains, leaf and internal
# values, and predictions. Counts stay exact.
FIXED_POINT_EXPONENT, FIXED_POINT_TOLERANCE = 26, 1e-6


def fixed_point_data(kind, objective):
    """Missing values, a positive feature and a mostly-zero one; or categorical features."""
    if kind == "categorical":
        return (*synthetic_categorical(seed=91, n=1500, objective=objective), CATEGORICAL)
    X, y = with_missing_values(*synthetic(seed=51, n=1500), seed=52, shift=2.0)
    X[np.random.default_rng(53).random(len(X)) < 0.8, 1] = 0.0
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    return X, y, ()


FIXED_POINT_PARAMS = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}
# The accepted differences each case of `fixed_point_data` reports at FIXED_POINT_EXPONENT.
FIXED_POINT_ACCEPTED = {("numerical", "regression"): {ZERO_BIN: 1}, ("numerical", "binary"): {},
                        ("categorical", "regression"): {ZERO_BIN: 1}, ("categorical", "binary"): {ZERO_BIN: 1}}


def train_fixed_point_and_compare(kind, objective, exponent, accepted=None, secure_aggregation=False):
    X, y, categorical = fixed_point_data(kind, objective)
    params = {"objective": objective, **FIXED_POINT_PARAMS}
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])

    result = simulate(params, tables, encoding=FixedPoint(exponent), categorical=categorical,
                      secure_aggregation=secure_aggregation)

    ours = lgb.Booster(model_str=result.model)
    central = train_central(X[:1200], y[:1200], result.agreed_edges, params)
    differences = assert_same_model(ours, central, X[:1200], accepted, tolerance=FIXED_POINT_TOLERANCE)
    np.testing.assert_allclose(json.loads(result.report)["training_loss"],
                               central_training_loss(X[:1200], y[:1200], result, params)[:10], rtol=FIXED_POINT_TOLERANCE)
    if not differences:
        for rows in (X[:1200], X[1200:]):
            np.testing.assert_allclose(ours.predict(rows), central.predict(rows), rtol=FIXED_POINT_TOLERANCE)


@pytest.mark.parametrize("kind", ["numerical", "categorical"])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_fixed_point_encoding_equals_central_baseline_within_its_tolerance(kind, objective):
    train_fixed_point_and_compare(kind, objective, FIXED_POINT_EXPONENT, FIXED_POINT_ACCEPTED[kind, objective])


def test_coarser_fixed_point_reports_a_split_that_differs_as_a_near_tie():
    # At 2^-22 the rounded sums pick another of two nearly equal candidates once.
    train_fixed_point_and_compare("numerical", "regression", exponent=22, accepted={NEAR_TIE: 1})


def integers_only(value) -> bool:
    if isinstance(value, dict):
        return all(integers_only(v) for v in value.values())
    if isinstance(value, list):
        return all(integers_only(v) for v in value)
    return isinstance(value, int) and not isinstance(value, bool)


def test_every_fixed_point_payload_holds_integers_only():
    X, y = synthetic_categorical(seed=97, n=600)
    params = Params.from_dict({"num_iterations": 3, "num_leaves": 6})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    encoding = FixedPoint(FIXED_POINT_EXPONENT)
    sites = [RecordingSite(Xs, ys, params, names, None, encoding, CATEGORICAL)
             for Xs, ys in split_into_sites(X, y, [300, 300])]

    run(sites, Aggregator(params, encoding=encoding), rounds=params.round_budget)

    # Setup round 1 shares column names, min/max and category sets, which are not summed (ADR 0003).
    _, grid_payload, bin_payload, *split_payloads = sites[0].payloads
    encoded = [grid_payload["grid_counts"], bin_payload["bin_counts"]] + \
        [leaf["histogram"] for p in split_payloads for leaf in p.get("leaves", [])]
    assert len(encoded) > 1
    for values in encoded:
        assert isinstance(values, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    # Nor does any other field carry text or a float, which FL-Net's SMPC would round to its own
    # decimal exponent: leaf totals and training losses are fixed-point too.
    assert any("loss_sum" in p for p in split_payloads)
    assert all(integers_only(p) for p in [grid_payload, bin_payload, *split_payloads])


def test_setup_rounds_send_64_cells_per_numerical_feature_then_the_rows_per_agreed_bin():
    X, y = synthetic_categorical(seed=98, n=600)
    params = Params.from_dict({"num_iterations": 2, "num_leaves": 4, "max_bin": 40})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    sites = [RecordingSite(Xs, ys, params, names, None, DENSE, CATEGORICAL)
             for Xs, ys in split_into_sites(X, y, [300] * 2)]

    result = run_and_keep(sites, params)

    numerical = [f for f in range(X.shape[1]) if f not in CATEGORICAL]
    categories = sum(len(np.unique(X[:, f])) for f in CATEGORICAL)
    for site in sites:
        _, grid_payload, bin_payload, *_ = site.payloads
        # Laid out from 0, the bounds may cut a cell at each end.
        assert 64 * len(numerical) <= len(grid_payload["grid_counts"]) - categories <= 65 * len(numerical)
        # Setup round 3: each numerical feature's rows per agreed bin, at most max_bin of them.
        bins = [result.bin_mappers[f].num_bins for f in numerical]
        assert len(bin_payload["bin_counts"]) == sum(bins) and max(bins) <= 40
        assert sum(bin_payload["bin_counts"]) == len(site.labels) * len(numerical)


def run_and_keep(sites, params):
    aggregator = Aggregator(params, encoding=sites[0].encoding)
    run(sites, aggregator, rounds=params.round_budget)
    return aggregator


def test_agreed_edges_do_not_depend_on_how_rows_are_split_across_sites():
    X, y = with_missing_values(*synthetic_categorical(seed=99, n=900), seed=100, shift=1.0)
    params = {"num_iterations": 0}

    edges = [simulate(params, tables, categorical=CATEGORICAL).forced_bins for tables in [
        [(X, y)], split_into_sites(X, y, [300] * 3), split_into_sites(X, y, [800, 60, 40]),
        split_into_sites(X[::-1], y[::-1], [450, 450])]]
    mappers = [simulate(params, split_into_sites(X, y, sizes), categorical=CATEGORICAL).agreed_edges
               for sizes in ([900], [100, 200, 600])]

    assert edges[1:] == edges[:1] * 3
    assert [(m.missing_type, m.most_freq_bin) for m in mappers[0]] == [(m.missing_type, m.most_freq_bin)
                                                                       for m in mappers[1]]


class PayloadRecordingAggregator(Aggregator):
    """Records the payloads it receives in every round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def reply(self, payloads):
        self.received.append(payloads)
        return super().reply(payloads)


def test_secure_aggregation_hands_the_aggregator_only_the_sum_from_setup_round_2_on():
    X, y = synthetic_categorical(seed=97, n=600)
    params = Params.from_dict({"num_iterations": 3, "num_leaves": 6})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    encoding = FixedPoint(FIXED_POINT_EXPONENT)
    sites = [Site(Xs, ys, params, names, None, encoding, CATEGORICAL) for Xs, ys in split_into_sites(X, y, [200] * 3)]
    aggregator = PayloadRecordingAggregator(params, encoding=encoding)

    run(sites, aggregator, rounds=params.round_budget, secure_aggregation=True)

    assert [len(p) for p in aggregator.received] == [3] + [1] * (params.round_budget - 1)
    # A leaf's histogram and totals are all the aggregator gets: it requested that leaf itself, and a
    # leaf id would have been summed.
    leaves = [leaf for (p,) in aggregator.received[2:] for leaf in p.get("leaves", [])]
    assert leaves and all(leaf.keys() == {"histogram", "sum_grad", "sum_hess"} for leaf in leaves)


@pytest.mark.parametrize("kind", ["numerical", "categorical"])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_secure_aggregation_gives_the_same_model_as_fixed_point_without_it(kind, objective):
    # Fixed-point integers sum exactly, so the aggregator decodes the same sums either way.
    X, y, categorical = fixed_point_data(kind, objective)
    params = {"objective": objective, **FIXED_POINT_PARAMS}
    tables = split_into_sites(X, y, [700, 500, 300])
    encoding = FixedPoint(FIXED_POINT_EXPONENT)

    plain = simulate(params, tables, encoding=encoding, categorical=categorical)
    secure = simulate(params, tables, encoding=encoding, categorical=categorical, secure_aggregation=True)

    assert (secure.model, secure.report) == (plain.model, plain.report)


@pytest.mark.parametrize("kind", ["numerical", "categorical"])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_secure_aggregation_equals_central_baseline_within_the_fixed_point_tolerance(kind, objective):
    train_fixed_point_and_compare(kind, objective, FIXED_POINT_EXPONENT, FIXED_POINT_ACCEPTED[kind, objective],
                                  secure_aggregation=True)


@pytest.mark.parametrize("encoding", [SPARSE, DENSE])
def test_secure_aggregation_without_fixed_point_stops_before_the_first_round(encoding):
    X, y = synthetic(seed=12, n=100)
    params = Params.from_dict({"num_iterations": 1, "num_leaves": 2})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    fixed_point = FixedPoint(FIXED_POINT_EXPONENT)

    for site_encoding, aggregator_encoding in [(encoding, encoding), (encoding, fixed_point), (fixed_point, encoding)]:
        sites = [RecordingSite(X[:50], y[:50], params, names, None, site_encoding),
                 RecordingSite(X[50:], y[50:], params, names, None, fixed_point)]
        with pytest.raises(ValueError, match="secure aggregation needs the FixedPoint payload encoding"):
            run(sites, Aggregator(params, encoding=aggregator_encoding), rounds=params.round_budget,
                secure_aggregation=True)
        assert sites[0].payloads == []


class TamperingSite(Site):
    """Changes its payload of one round."""

    def __init__(self, *args, tamper_round, tamper):
        super().__init__(*args)
        self.tamper_round, self.tamper = tamper_round, tamper

    def payload(self, reply):
        payload = super().payload(reply)
        if self._round == self.tamper_round:
            self.tamper(payload)
        return payload


@pytest.mark.parametrize("first_site, other_sites, problem", [
    (lambda p: p.update(note="x"), lambda p: p.update(note="x"), "holds 'x'"),
    (lambda p: p["leaves"][0]["histogram"].pop(), lambda p: None, "differs in shape"),
    (lambda p: p.update(extra=1), lambda p: None, "differs in shape"),
])
def test_a_payload_secure_aggregation_cannot_sum_stops_the_run_naming_the_round(first_site, other_sites, problem):
    X, y = synthetic(seed=12, n=300)
    params = Params.from_dict({"num_iterations": 2, "num_leaves": 3})
    names = [f"Column_{i}" for i in range(X.shape[1])]
    encoding = FixedPoint(FIXED_POINT_EXPONENT)
    sites = [TamperingSite(Xs, ys, params, names, None, encoding, tamper_round=4, tamper=tamper)
             for (Xs, ys), tamper in zip(split_into_sites(X, y, [100] * 3), [first_site, other_sites, other_sites])]

    with pytest.raises(ValueError, match=f"round 4: .* {problem}"):
        run(sites, Aggregator(params, encoding=encoding), rounds=params.round_budget, secure_aggregation=True)


def central_training_loss(X, y, result, params):
    """The central baseline's training loss after every iteration, as LightGBM records it."""
    losses = {}
    with central_dataset(X, y, result.agreed_edges, params) as dataset:
        lgb.train(dataset.params, dataset, valid_sets=[dataset], valid_names=["train"],
                  callbacks=[lgb.record_evaluation(losses)])
    return losses["train"][METRIC[params.get("objective", "regression")]]


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_report_has_the_central_baselines_training_loss_after_every_tree(objective):
    X, y = with_missing_values(*synthetic(seed=111, n=1200), seed=112, shift=2.0)
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = simulate(params, split_into_sites(X, y, [500, 380, 320]))

    report = json.loads(result.report)
    assert report["metric"] == METRIC[objective]
    assert len(report["training_loss"]) == 10
    np.testing.assert_allclose(report["training_loss"], central_training_loss(X, y, result, params), rtol=1e-9)


def test_report_counts_rounds_used_and_padding_rounds_to_the_round_budget():
    X, y = synthetic(seed=8, n=150)
    params = {"num_iterations": 5, "num_leaves": 31, "min_data_in_leaf": 20}

    result = simulate(params, split_into_sites(X, y, [70, 50, 30]))

    report = json.loads(result.report)
    assert report["params"] == asdict(Params.from_dict(params))
    assert report["num_sites"] == 3
    assert report["padding_rounds"] > 0
    assert report["rounds_used"] + report["padding_rounds"] == 3 + 5 * 30 + 1 + 1


def test_report_of_a_run_that_cannot_split_has_the_loss_of_the_init_score():
    X, y = synthetic(seed=9, n=60)
    params = {"num_iterations": 4, "num_leaves": 4, "min_data_in_leaf": 40}  # 60 rows cannot give two leaves of 40

    result = simulate(params, split_into_sites(X, y, [30, 30]))

    report = json.loads(result.report)
    assert report["rounds_used"] == 5  # three setup rounds, the root histogram and the closing round
    assert report["rounds_used"] + report["padding_rounds"] == result.rounds
    # LightGBM keeps one constant tree; its training loss stays the same after every iteration.
    np.testing.assert_allclose(report["training_loss"], central_training_loss(X, y, result, params)[:1], rtol=1e-9)


def test_report_of_a_run_that_stops_after_a_later_tree_has_the_loss_of_every_tree_kept():
    X, y = synthetic(seed=8, n=600)
    params = {"num_iterations": 10, "num_leaves": 4, "learning_rate": 1.0, "min_gain_to_split": 20.0}

    result = simulate(params, split_into_sites(X, y, [300, 300]))

    report = json.loads(result.report)
    assert len(report["training_loss"]) == 5
    # LightGBM records the loss of the last tree kept for every iteration after training stopped.
    central = central_training_loss(X, y, result, params)
    np.testing.assert_allclose(report["training_loss"], central[:5], rtol=1e-9)
    assert central[5:] == [central[4]] * 5


def train_from_export(result, X, y, directory):
    """LightGBM on the pooled rows with nothing but what a run writes next to the model: the forced-bins
    file and the report's parameters. Returns the booster and its training loss after every iteration."""
    report = json.loads(result.report)
    forced_bins = directory / "forcedbins.json"
    forced_bins.write_text(result.forced_bins)
    params = {**report["params"], **report["central_baseline"], "forcedbins_filename": str(forced_bins),
              "verbose": -1}
    # LightGBM's Python package takes the categorical features as a Dataset argument.
    dataset = lgb.Dataset(X, y, params=params, categorical_feature=params.pop("categorical_feature"))
    losses = {}
    booster = lgb.train(params, dataset, valid_sets=[dataset], valid_names=["train"],
                        callbacks=[lgb.record_evaluation(losses)], keep_training_booster=True)
    return booster, losses["train"][report["metric"]]


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_lightgbm_trained_with_only_the_exported_edges_reproduces_the_model(objective, tmp_path):
    X, y = synthetic_categorical(seed=113, n=1200, objective=objective)
    X, y = with_missing_values(X, y, seed=114, shift=0.0)  # NaN bins, which the forced bins leave out
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = simulate(params, split_into_sites(X, y, [500, 380, 320]), categorical=CATEGORICAL)

    assert all(np.isfinite(b) for f in json.loads(result.forced_bins) for b in f["bin_upper_bound"])
    central, training_loss = train_from_export(result, X, y, tmp_path)
    assert_same_model(lgb.Booster(model_str=result.model), central, X)
    np.testing.assert_allclose(json.loads(result.report)["training_loss"], training_loss, rtol=1e-9)


def test_a_feature_that_is_zero_at_every_site_equals_central_baseline(tmp_path):
    # Like a taxon no cohort has: one agreed bin, which LightGBM's `max_bin_by_feature` cannot take.
    X, y = synthetic(seed=115, n=1200)
    X[:, 2] = 0.0
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = simulate(params, split_into_sites(X, y, [500, 380, 320]))

    assert result.agreed_edges[2].num_bins == 1
    central, _ = train_from_export(result, X, y, tmp_path)
    assert_same_model(lgb.Booster(model_str=result.model), central, X)


def feature_infos(model: str) -> list[str | tuple[float, float]]:
    """The model's `feature_infos`, a numerical feature's range as numbers: LightGBM writes 17 significant
    digits, the model writer the shortest text that reads back as the same double."""
    line = next(line for line in model.splitlines() if line.startswith("feature_infos="))
    return [tuple(map(float, info[1:-1].split(":"))) if info.startswith("[") else info
            for info in line.removeprefix("feature_infos=").split(" ")]


@pytest.mark.parametrize("objective", ["regression", "binary"])
@pytest.mark.parametrize("settings, info", [
    ({}, (0.0, 0.0)),  # the bin of 0 and a NaN bin: LightGBM's bin finding puts 0 in a feature without values
    ({"zero_as_missing": True}, "none"),  # NaN is read as 0: one bin, which LightGBM's dataset leaves out
    ({"use_missing": False}, "none"),
])
def test_a_feature_that_is_nan_at_every_site_equals_central_baseline(objective, settings, info):
    X, y = synthetic(seed=116, n=1500)
    X[:, 2] = np.nan
    if objective == "binary":
        y = (y > np.median(y)).astype(np.float64)
    params = {"objective": objective, "num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3, **settings}
    tables = split_into_sites(X[:1200], y[:1200], [500, 380, 320])

    result = train_and_compare(params, tables, X[1200:])

    ours = lgb.Booster(model_str=result.model)
    central = train_central(X[:1200], y[:1200], result.agreed_edges, params)
    assert feature_infos(result.model)[2] == info
    assert feature_infos(result.model) == feature_infos(central.model_to_string())
    np.testing.assert_allclose(ours.predict(X[1200:], pred_contrib=True),
                               central.predict(X[1200:], pred_contrib=True), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("settings", [{}, {"zero_as_missing": True}, {"use_missing": False}])
def test_feature_infos_of_trivial_features_and_nan_read_as_zero_equal_central_baseline(settings):
    X, y = synthetic(seed=117, n=1200, p=12)
    some_nan = np.arange(len(X)) % 3 == 0
    X[:, 5] = 3.0  # one value
    X[:, 6] = 0.0  # one value, the bin of 0
    X[:, 7] = np.where(some_nan, np.nan, 2.0)  # read as 0, NaN widens the range to 0
    X[:, 8] = np.where(some_nan, np.nan, 0.0)
    X[:, 9] = 4.0  # one category
    X[:, 10] = np.nan  # no category
    X[:, 11] = np.where(np.arange(len(X)) % 5 == 0, -1e-40, X[:, 3])  # positive, but for values LightGBM reads as 0
    params = {"num_iterations": 3, "num_leaves": 7, "learning_rate": 0.3, **settings}

    result = simulate(params, split_into_sites(X, y, [500, 380, 320]), categorical=(9, 10))

    central = train_central(X, y, result.agreed_edges, params)
    assert feature_infos(result.model) == feature_infos(central.model_to_string())


def test_a_feature_narrower_than_the_grid_resolution_is_binned_and_equals_central_baseline():
    X, y = synthetic(seed=118, n=1500)
    X[:, 5] = 5.0 + np.random.default_rng(119).uniform(0.0, 1e-14, size=len(X))  # 11 doubles: too narrow for 64 cells
    y += 2.0 * (X[:, 5] > 5.0 + 5e-15)
    params = {"num_iterations": 10, "num_leaves": 12, "learning_rate": 0.3}

    result = train_and_compare(params, split_into_sites(X[:1200], y[:1200], [500, 380, 320]), X[1200:])

    lightgbm_bins = lgb.Dataset(X[:1200], y[:1200], params={"verbose": -1}).construct().feature_num_bin(5)
    assert result.agreed_edges[5].num_bins >= lightgbm_bins // 2  # not exactly LightGBM's: the grid has fewer cells
