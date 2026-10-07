"""Trees in LightGBM's text model format (GBDT::SaveModelToString, Tree::ToString)."""

from dataclasses import dataclass, field

import numpy as np

from fl_lightgbm.binning import MISSING_NAN, MISSING_NONE, MISSING_ZERO, ZERO_THRESHOLD

# decision_type bits (tree.h): bit 0 = categorical, bit 1 = default_left, bits 2-3 = missing type.
CATEGORICAL_MASK = 1
DEFAULT_LEFT_MASK = 2
MISSING_TYPE_CODE = {MISSING_NONE: 0, MISSING_ZERO: 1, MISSING_NAN: 2}

# ObjectiveFunction::ToString; `predict` applies the sigmoid only if the model names it.
OBJECTIVE_TEXT = {"regression": "regression", "binary": "binary sigmoid:1"}


@dataclass
class Tree:
    """One tree as LightGBM's Tree stores it. Children >= 0 are nodes, children < 0 are ~leaf."""

    leaf_value: list[float]
    leaf_count: list[int]
    leaf_weight: list[float] = field(default_factory=list)
    split_feature: list[int] = field(default_factory=list)
    split_gain: list[float] = field(default_factory=list)
    threshold: list[float] = field(default_factory=list)
    decision_type: list[int] = field(default_factory=list)
    left_child: list[int] = field(default_factory=list)
    right_child: list[int] = field(default_factory=list)
    internal_value: list[float] = field(default_factory=list)
    internal_weight: list[float] = field(default_factory=list)
    internal_count: list[int] = field(default_factory=list)
    cat_boundaries: list[int] = field(default_factory=lambda: [0])  # where each category set starts in cat_threshold
    cat_threshold: list[int] = field(default_factory=list)  # the category sets, as bitsets of uint32 words
    shrinkage: float = 1.0

    def add_category_set(self, categories) -> int:
        """Store the categories a categorical split sends left, as Tree::SplitCategorical does; return the
        set's index, which is the split's threshold."""
        words = [0] * (max(categories) // 32 + 1)  # Common::ConstructBitset
        for c in categories:
            words[c // 32] |= 1 << (c % 32)
        self.cat_threshold.extend(words)
        self.cat_boundaries.append(len(self.cat_threshold))
        return len(self.cat_boundaries) - 2

    def shrink(self, rate: float) -> None:
        """Tree::Shrinkage."""
        self.leaf_value = [_maybe_round_to_zero(v * rate) for v in self.leaf_value]
        self.internal_value = [_maybe_round_to_zero(v * rate) for v in self.internal_value]
        self.shrinkage *= rate

    def add_bias(self, bias: float) -> None:
        """Tree::AddBias, which puts the init score into the first tree."""
        self.leaf_value = [_maybe_round_to_zero(v + bias) for v in self.leaf_value]
        self.internal_value = [_maybe_round_to_zero(v + bias) for v in self.internal_value]
        self.shrinkage = 1.0


def decision_type(default_left: bool, missing_type: str) -> int:
    """A numerical split's decision_type, as Tree::Split sets it."""
    return (DEFAULT_LEFT_MASK if default_left else 0) | MISSING_TYPE_CODE[missing_type] << 2


def categorical_decision_type(missing_type: str) -> int:
    """A categorical split's decision_type, as Tree::SplitCategorical sets it: missing values go right."""
    return CATEGORICAL_MASK | MISSING_TYPE_CODE[missing_type] << 2


def _maybe_round_to_zero(x: float) -> float:
    """Tree::MaybeRoundToZero."""
    return 0.0 if -ZERO_THRESHOLD <= x <= ZERO_THRESHOLD else x


def _join(values) -> str:
    return " ".join(repr(float(v)) if isinstance(v, float) else str(v) for v in values)


def _tree_text(tree: Tree) -> str:
    num_cat = len(tree.cat_boundaries) - 1
    category_sets = [f"cat_boundaries={_join(tree.cat_boundaries)}",
                     f"cat_threshold={_join(tree.cat_threshold)}"] if num_cat else []
    return "\n".join([
        f"num_leaves={len(tree.leaf_value)}",
        f"num_cat={num_cat}",
        f"split_feature={_join(tree.split_feature)}",
        f"split_gain={_join(tree.split_gain)}",
        f"threshold={_join(tree.threshold)}",
        f"decision_type={_join(tree.decision_type)}",
        f"left_child={_join(tree.left_child)}",
        f"right_child={_join(tree.right_child)}",
        f"leaf_value={_join(tree.leaf_value)}",
        f"leaf_weight={_join(tree.leaf_weight)}",
        f"leaf_count={_join(tree.leaf_count)}",
        f"internal_value={_join(tree.internal_value)}",
        f"internal_weight={_join(tree.internal_weight)}",
        f"internal_count={_join(tree.internal_count)}",
        *category_sets,
        "is_linear=0",
        f"shrinkage={tree.shrinkage!r}",
    ]) + "\n"


def _feature_info(feature_range: tuple[float, float] | None, categories: np.ndarray | None) -> str:
    if feature_range is None:
        return "none"
    if categories is not None:
        return ":".join(str(c) for c in [-1, *categories])
    lo, hi = feature_range
    return f"[{float(lo)!r}:{float(hi)!r}]"


def write_model(
    trees: list[Tree],
    feature_names: list[str],
    feature_ranges: list[tuple[float, float] | None],
    objective: str = "regression",
    categories: list[np.ndarray | None] | None = None,
) -> str:
    """The model text `lightgbm.Booster(model_str=...)` loads. `feature_ranges` become `feature_infos`,
    except for a categorical feature, whose kept categories (`categories`, None for numerical features)
    are written as LightGBM's bin_info_string: -1 for bin 0, then the category of each further bin.
    A range of None marks a trivial feature, which LightGBM's dataset leaves out and writes as `none`."""
    categories = categories or [None] * len(feature_names)
    feature_infos = [_feature_info(r, cats) for r, cats in zip(feature_ranges, categories)]
    header = [
        "tree",
        "version=v4",
        "num_class=1",
        "num_tree_per_iteration=1",
        "label_index=0",
        f"max_feature_idx={len(feature_names) - 1}",
        f"objective={OBJECTIVE_TEXT[objective]}",
        f"feature_names={' '.join(feature_names)}",
        f"feature_infos={' '.join(feature_infos)}",
        "",
    ]
    body = [f"Tree={i}\n{_tree_text(t)}" for i, t in enumerate(trees)]
    return "\n".join(header) + "\n" + "\n".join(body) + "\nend of trees\n"
