"""The comparison with the central baseline accepts scan ties and zero-bin differences, and nothing else
(spec #16's "Equality" rule)."""

import numpy as np
import pytest

from comparison import NEAR_TIE, SCAN_TIE, ZERO_BIN, compare_trees
from fl_lightgbm.binning import ZERO_THRESHOLD


def leaf(index, count, value):
    return {"leaf_index": index, "leaf_value": value, "leaf_weight": float(count), "leaf_count": count}


def split(index, threshold, default_left, left, right, missing_type="NaN", gain=2.5):
    count = count_of(left) + count_of(right)
    return {"split_index": index, "split_feature": 0, "split_gain": gain, "threshold": threshold,
            "decision_type": "<=", "default_left": default_left, "missing_type": missing_type,
            "internal_value": 0.1, "internal_weight": float(count), "internal_count": count,
            "left_child": left, "right_child": right}


def count_of(node):
    return node.get("leaf_count", node.get("internal_count"))


# One split of 6 rows (leaf 0) against 4 rows (leaf 1); the leaf holds no missing values.
ROWS = np.array([0] * 6 + [1] * 4)


def stump(threshold=0.5, default_left=False, missing_type="NaN", gain=2.5, left_count=6, right_value=0.75):
    return split(0, threshold, default_left, leaf(0, left_count, -0.5), leaf(1, 4, right_value), missing_type, gain)


def test_equal_trees_have_no_accepted_differences():
    assert compare_trees(stump(), stump(), ROWS, ROWS) == []


@pytest.mark.parametrize("central", [
    stump(default_left=True),  # missing values would go the other way, but the leaf has none
    stump(threshold=0.6, default_left=True),  # also a threshold between bins that hold no rows of the leaf
])
def test_scan_tie_that_sends_every_training_row_the_same_way_is_accepted(central):
    assert compare_trees(stump(), central, ROWS, ROWS) == [("root", SCAN_TIE)]


@pytest.mark.parametrize("central", [
    stump(default_left=True, left_count=5),
    stump(default_left=True, gain=2.75),
    stump(default_left=True, right_value=0.8),
])
def test_a_scan_tie_with_another_count_gain_or_leaf_value_fails(central):
    with pytest.raises(AssertionError):
        compare_trees(stump(), central, ROWS, ROWS)


def test_a_threshold_that_moves_across_rows_of_the_leaf_fails():
    central_rows = ROWS.copy()
    central_rows[5] = 1  # the central threshold sends one more row right

    with pytest.raises(AssertionError, match="training row"):
        compare_trees(stump(), stump(threshold=0.4, default_left=True), ROWS, central_rows)


def test_another_threshold_with_the_same_default_direction_away_from_zero_fails():
    """Within one scan, bins that hold no rows of the leaf decide nothing: their sums are exactly 0."""
    with pytest.raises(AssertionError, match="neither a scan tie nor a zero-bin difference"):
        compare_trees(stump(), stump(threshold=0.6), ROWS, ROWS)


def test_another_default_direction_on_a_feature_without_missing_type_fails():
    with pytest.raises(AssertionError, match="neither a scan tie"):
        compare_trees(stump(missing_type="None"), stump(default_left=True, missing_type="None"), ROWS, ROWS)


@pytest.mark.parametrize("missing_type", ["None", "NaN"])
@pytest.mark.parametrize("ours, central", [
    (-ZERO_THRESHOLD, ZERO_THRESHOLD),
    (-ZERO_THRESHOLD, 0.12),  # the bin above the zero bin holds no rows of the leaf either
    (-0.3, ZERO_THRESHOLD),
])
def test_threshold_moved_across_an_empty_zero_bin_is_accepted(missing_type, ours, central):
    assert compare_trees(stump(threshold=ours, missing_type=missing_type),
                         stump(threshold=central, missing_type=missing_type), ROWS, ROWS) == [("root", ZERO_BIN)]


def test_threshold_moved_across_a_zero_bin_that_holds_rows_of_the_leaf_fails():
    central_rows = ROWS.copy()
    central_rows[5] = 1  # a row at 0 goes right with the central threshold

    with pytest.raises(AssertionError, match="training row"):
        compare_trees(stump(threshold=ZERO_THRESHOLD), stump(threshold=-ZERO_THRESHOLD), ROWS, central_rows)


# A split of missing values (B: rows 5 and 6) against all others (A), then a split of A into A1
# (rows 0 - 2) and A2 (rows 3 and 4). Ours keeps A on the left: A1 is leaf 0, A2 leaf 2, B leaf 1.
OURS = split(0, 1e300, False, split(1, -0.3, False, leaf(0, 3, -1.0), leaf(2, 2, -0.25)), leaf(1, 2, 1.0))
OUR_ROWS = np.array([0, 0, 0, 2, 2, 1, 1])


def mirrored(a1, a2):
    """The same tree written from the other side: B on the left keeps leaf 0, A on the right is leaf 1."""
    return split(0, -0.9, True, leaf(0, 2, 1.0), split(1, -0.3, False, leaf(a1, 3, -1.0), leaf(a2, 2, -0.25)))


def test_mirrored_split_with_consistently_renumbered_leaves_is_accepted():
    central_rows = np.array([1, 1, 1, 2, 2, 0, 0])

    assert compare_trees(OURS, mirrored(a1=1, a2=2), OUR_ROWS, central_rows) == [("root", SCAN_TIE)]


def test_mirrored_split_with_inconsistently_renumbered_leaves_fails():
    central_rows = np.array([2, 2, 2, 1, 1, 0, 0])

    with pytest.raises(AssertionError, match="leaf_index"):
        compare_trees(OURS, mirrored(a1=2, a2=1), OUR_ROWS, central_rows)


def test_mirrored_split_below_a_mirrored_split_renumbers_both_levels():
    # A's split is mirrored too: A2 on the left keeps leaf 1, A1 on the right is leaf 2.
    central = split(0, -0.9, True, leaf(0, 2, 1.0),
                    split(1, 0.4, True, leaf(1, 2, -0.25), leaf(2, 3, -1.0)))
    central_rows = np.array([2, 2, 2, 1, 1, 0, 0])

    assert compare_trees(OURS, central, OUR_ROWS, central_rows) == [
        ("root", SCAN_TIE), ("root.left_child", SCAN_TIE)]


# With a tolerance (a model trained on fixed-point sums), values may differ within it, and a split
# with another partition is accepted as a near-tie if its gain is within it too.
def test_values_within_the_tolerance_are_accepted():
    assert compare_trees(stump(), stump(gain=2.5 * (1 + 1e-7), right_value=0.75 * (1 + 1e-7)), ROWS, ROWS,
                         tolerance=1e-6) == []


def test_values_beyond_the_tolerance_fail():
    with pytest.raises(AssertionError, match="leaf_value"):
        compare_trees(stump(), stump(right_value=0.75 * (1 + 1e-5)), ROWS, ROWS, tolerance=1e-6)


def test_a_split_of_other_rows_with_a_gain_within_the_tolerance_is_a_near_tie():
    central_rows = ROWS.copy()
    central_rows[5] = 1
    central = split(0, 0.4, False, leaf(0, 5, -0.7), leaf(1, 5, 0.6), gain=2.5 * (1 + 1e-7))

    assert compare_trees(stump(), central, ROWS, central_rows, tolerance=1e-6) == [("root", NEAR_TIE)]


def test_a_threshold_moved_between_bins_without_rows_of_the_leaf_is_a_near_tie_with_a_tolerance():
    # The two thresholds send every row the same way: the bin between them holds only a rounding residual.
    assert compare_trees(stump(), stump(threshold=0.6, gain=2.5 * (1 + 1e-7)), ROWS, ROWS,
                         tolerance=1e-6) == [("root", NEAR_TIE)]


def test_a_split_of_other_rows_with_another_gain_fails_even_with_a_tolerance():
    central_rows = ROWS.copy()
    central_rows[5] = 1

    with pytest.raises(AssertionError, match="split_gain"):
        compare_trees(stump(), stump(threshold=0.4, gain=2.6), ROWS, central_rows, tolerance=1e-6)


def test_a_split_of_other_rows_without_a_tolerance_fails():
    central_rows = ROWS.copy()
    central_rows[5] = 1

    with pytest.raises(AssertionError, match="training row"):
        compare_trees(stump(), stump(threshold=0.4), ROWS, central_rows)
