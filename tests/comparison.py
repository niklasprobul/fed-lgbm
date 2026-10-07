"""What "equals the central baseline" means: spec #16's "Equality" rule, with its zero-bin differences
and scan ties; and, for a model trained on fixed-point sums, what "within a tolerance" means, with its
near-ties."""

import warnings
from collections import Counter

import lightgbm as lgb
import numpy as np
import pytest

from baseline import trees
from fl_lightgbm.binning import ZERO_THRESHOLD

EXACT = {"split_index", "split_feature", "decision_type", "missing_type", "internal_count", "leaf_count",
         "split_gain"}  # LightGBM stores gains as float32; ours are rounded the same way
CLOSE = {"internal_value", "internal_weight", "leaf_value", "leaf_weight"}
# The keys an accepted difference may change: how a split is written, not what it does to the rows.
WRITTEN_SPLIT = {"threshold", "default_left", "left_child", "right_child"}
# The keys a near-tie may change besides: which candidate split was chosen.
CHOSEN_SPLIT = {"split_feature", "decision_type", "missing_type"}

SCAN_TIE, ZERO_BIN, NEAR_TIE = "scan tie", "zero bin", "near-tie"


class _NearTie(Exception):
    """The trees grow apart from here: nothing below or after a near-tie can be compared."""


def assert_same_model(ours: lgb.Booster, central: lgb.Booster, X_train: np.ndarray,
                      accepted: dict[str, int] | None = None,
                      tolerance: float | None = None) -> list[tuple[str, str]]:
    """Assert the models equal and return the accepted differences as (node, kind). `accepted` gives
    how many of each kind there must be; by default none. With a `tolerance`, see `compare_trees`; the
    comparison ends at the first near-tie that sends training rows another way."""
    our_trees, central_trees = trees(ours), trees(central)
    our_rows = np.asarray(ours.predict(X_train, pred_leaf=True))
    central_rows = np.asarray(central.predict(X_train, pred_leaf=True))
    differences = []
    for i, (a, b) in enumerate(zip(our_trees, central_trees)):
        tree_differences, grew_apart = _compare_trees(a["tree_structure"], b["tree_structure"], our_rows[:, i],
                                                      central_rows[:, i], f"tree {i}", tolerance)
        differences += tree_differences
        if grew_apart:
            break
        assert a["num_leaves"] == b["num_leaves"], f"tree {i}"
    else:
        assert len(our_trees) == len(central_trees)
    if differences:
        warnings.warn(f"{len(differences)} accepted difference(s): "
                      + ", ".join(f"{kind} at {node}" for node, kind in differences))
    assert Counter(kind for _, kind in differences) == Counter(accepted or {}), f"accepted: {differences}"
    return differences


def compare_trees(ours: dict, central: dict, our_rows: np.ndarray, central_rows: np.ndarray,
                  path: str = "root", tolerance: float | None = None) -> list[tuple[str, str]]:
    """Assert two dumped trees equal and return the accepted differences as (node, kind).

    `our_rows` and `central_rows` hold the leaf each training row reaches in each tree. A split may be
    written differently only if it sends every training row the same way, and only as:
    - a scan tie, on a feature with missing type NaN or Zero: the other default direction (the other
      scan), possibly with the threshold moved between bins that hold no rows of the leaf; or the split
      written from the other side, its children swapped and the leaves below renumbered as Tree::Split
      numbers them;
    - a zero-bin difference: the same default direction, with the threshold moved across the empty
      zero bin, whose rounding residual decides where the scan stops.

    With a `tolerance` (for a model trained on fixed-point sums), gains and values may differ by that
    much, relative or absolute, and any other split whose gain is within the tolerance is a near-tie:
    fixed-point rounding picked another of two nearly equal candidates. If it sends the training rows
    another way, the trees grow apart and the comparison ends there. If it sends them the same way, the
    threshold moved between bins that hold no rows of the leaf: such a bin can still hold a rounding
    residual, because the aggregator derives a leaf's histogram by subtraction and a rounded sum differs
    from the sum of its rounded parts.
    """
    return _compare_trees(ours, central, our_rows, central_rows, path, tolerance)[0]


def _compare_trees(ours: dict, central: dict, our_rows: np.ndarray, central_rows: np.ndarray, path: str,
                   tolerance: float | None) -> tuple[list[tuple[str, str]], bool]:
    """`compare_trees`, and whether a near-tie made the trees grow apart."""
    differences: list[tuple[str, str]] = []

    def rows(node: dict, leaf_of_row: np.ndarray) -> np.ndarray:
        return np.isin(leaf_of_row, leaf_indices(node))

    def close(a: float, b: float) -> bool:
        if tolerance is None:
            return a == pytest.approx(b, rel=1e-9, abs=1e-12)
        return a == pytest.approx(b, rel=tolerance, abs=tolerance)

    def compare(o: dict, c: dict, renumber: dict[int, int], path: str) -> None:
        assert o.keys() == c.keys(), path
        if "leaf_index" in o:  # a tree of one leaf has none
            assert c["leaf_index"] == renumber.get(o["leaf_index"], o["leaf_index"]), f"{path}.leaf_index"
        may_differ = WRITTEN_SPLIT | CHOSEN_SPLIT if tolerance is not None else WRITTEN_SPLIT
        for key in o.keys() - may_differ - {"leaf_index"}:
            if key == "split_gain" and tolerance is not None:
                assert close(o[key], c[key]), f"{path}.{key}"
            elif key in EXACT:
                assert o[key] == c[key], f"{path}.{key}"
            elif key in CLOSE:
                assert close(o[key], c[key]), f"{path}.{key}"
            else:
                raise AssertionError(f"unexpected key {key} at {path}")
        if "split_index" not in o:
            return

        children = [(o["left_child"], c["left_child"]), (o["right_child"], c["right_child"])]
        if any(o[key] != c[key] for key in CHOSEN_SPLIT | {"threshold", "default_left"}):
            our_left = rows(o["left_child"], our_rows)
            same_sides = np.array_equal(our_left, rows(c["left_child"], central_rows))
            swapped = np.array_equal(our_left, rows(c["right_child"], central_rows))
            same_rule = all(o[key] == c[key] for key in CHOSEN_SPLIT)
            # Both kinds are numerical: a categorical split has one way to write it.
            with_missing_type = same_rule and o["missing_type"] in ("NaN", "Zero") and o["decision_type"] == "<="
            same_default = o["default_left"] == c["default_left"]
            if with_missing_type and same_sides and not same_default:
                differences.append((path, SCAN_TIE))
            elif with_missing_type and swapped:
                # The left child keeps the parent's leaf number and the right child gets a new one, so
                # the two numbers trade places below.
                kept, new = leftmost_leaf(o["left_child"]), leftmost_leaf(o["right_child"])
                renumber = {**renumber, kept: new, new: renumber.get(kept, kept)}
                children = [(o["left_child"], c["right_child"]), (o["right_child"], c["left_child"])]
                differences.append((path, SCAN_TIE))
            elif (same_rule and same_sides and same_default and o["decision_type"] == "<="
                  and across_zero_bin(o["threshold"], c["threshold"])):
                differences.append((path, ZERO_BIN))
            elif tolerance is not None:  # the gain is within it, checked above
                differences.append((path, NEAR_TIE))
                if not same_sides:
                    raise _NearTie
            else:
                raise AssertionError(
                    f"{path}: the split is written differently (threshold {o['threshold']} vs "
                    f"{c['threshold']}, default_left {o['default_left']} vs {c['default_left']}), and it is "
                    f"neither a scan tie nor a zero-bin difference that sends every training row the same way")
        compare(*children[0], renumber, f"{path}.left_child")
        compare(*children[1], renumber, f"{path}.right_child")

    try:
        compare(ours, central, {}, path)
    except _NearTie:
        return differences, True
    return differences, False


def across_zero_bin(a: float, b: float) -> bool:
    """One threshold at or below the zero bin's lower bound (-kZeroThreshold), the other at or above its top."""
    return min(a, b) <= -ZERO_THRESHOLD and max(a, b) >= ZERO_THRESHOLD


def leaf_indices(node: dict) -> list[int]:
    if "leaf_index" in node:
        return [node["leaf_index"]]
    return leaf_indices(node["left_child"]) + leaf_indices(node["right_child"])


def leftmost_leaf(node: dict) -> int:
    while "leaf_index" not in node:
        node = node["left_child"]
    return node["leaf_index"]
