"""LightGBM parameters, by their canonical names, with LightGBM's defaults."""

from dataclasses import dataclass, fields, replace

# LightGBM parameters that would give a different model than the federated one can, with the defaults
# that switch them off. Set to those defaults they are accepted, so central settings can be reused.
UNSUPPORTED = {
    "boosting": "gbdt",  # dart, rf and goss
    "data_sample_strategy": "bagging",  # goss
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "pos_bagging_fraction": 1.0,
    "neg_bagging_fraction": 1.0,
    "bagging_by_query": False,
    "feature_fraction": 1.0,
    "feature_fraction_bynode": 1.0,
    "extra_trees": False,
    "monotone_constraints": [],
    "interaction_constraints": [],
    "linear_tree": False,
    "path_smooth": 0.0,
    "use_quantized_grad": False,
    "weight_column": "",  # sample weights
}


@dataclass(frozen=True)
class Params:
    objective: str = "regression"
    num_iterations: int = 100
    num_leaves: int = 31
    learning_rate: float = 0.1
    max_bin: int = 255
    min_data_in_bin: int = 3
    min_data_in_leaf: int = 20
    min_sum_hessian_in_leaf: float = 1e-3
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    max_delta_step: float = 0.0
    min_gain_to_split: float = 0.0
    max_depth: int = -1
    use_missing: bool = True
    zero_as_missing: bool = False
    is_unbalance: bool = False
    scale_pos_weight: float = 1.0
    max_cat_to_onehot: int = 4
    max_cat_threshold: int = 32
    cat_l2: float = 10.0
    cat_smooth: float = 10.0
    min_data_per_group: int = 100

    @classmethod
    def from_dict(cls, params: dict) -> "Params":
        """Validate LightGBM parameters by their canonical names; missing ones get LightGBM's defaults."""
        params = dict(params)
        for name, default in UNSUPPORTED.items():
            if name in params and params.pop(name) != default:
                raise ValueError(f"unsupported parameter: {name} (only its default {default!r} is accepted)")
        unknown = sorted(set(params) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)} (LightGBM's canonical names are expected)")
        if params.get("objective", "regression") not in ("regression", "binary"):
            raise ValueError(f"unsupported objective: {params['objective']} (use regression or binary)")
        p = cls(**params)
        for name, ok, rule in (  # the CHECKs in LightGBM's config_auto.cpp
            ("num_iterations", p.num_iterations >= 0, ">= 0"),
            ("learning_rate", p.learning_rate > 0, "> 0"),
            ("num_leaves", 1 < p.num_leaves <= 131072, "in 2 – 131072"),
            ("max_bin", p.max_bin > 1, "> 1"),
            ("min_data_in_bin", p.min_data_in_bin > 0, "> 0"),
            ("min_data_in_leaf", p.min_data_in_leaf >= 0, ">= 0"),
            ("min_sum_hessian_in_leaf", p.min_sum_hessian_in_leaf >= 0, ">= 0"),
            ("lambda_l1", p.lambda_l1 >= 0, ">= 0"),
            ("lambda_l2", p.lambda_l2 >= 0, ">= 0"),
            ("min_gain_to_split", p.min_gain_to_split >= 0, ">= 0"),
            ("scale_pos_weight", p.scale_pos_weight > 0, "> 0"),
            ("max_cat_to_onehot", p.max_cat_to_onehot > 0, "> 0"),
            ("max_cat_threshold", p.max_cat_threshold > 0, "> 0"),
            ("cat_l2", p.cat_l2 >= 0, ">= 0"),
            ("cat_smooth", p.cat_smooth >= 0, ">= 0"),
            ("min_data_per_group", p.min_data_per_group > 0, "> 0"),
        ):
            if not ok:
                raise ValueError(f"{name} must be {rule}, got {getattr(p, name)}")
        if p.is_unbalance and abs(p.scale_pos_weight - 1.0) > 1e-6:  # as BinaryLogloss's constructor
            raise ValueError("is_unbalance and scale_pos_weight cannot be set at the same time")
        if p.max_depth > 0 and "num_leaves" not in params:  # as Config::CheckParamConflict
            p = replace(p, num_leaves=min(p.num_leaves, 2 ** p.max_depth))
        return p

    @property
    def round_budget(self) -> int:
        """Three setup rounds, one split round per possible split, the closing round and one padding round,
        so that the reply that delivers the model is never the last one (ADR 0005, 0008)."""
        return 3 + self.num_iterations * (self.num_leaves - 1) + 1 + 1
