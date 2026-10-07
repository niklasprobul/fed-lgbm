"""What the central baseline needs, besides the training parameters, to bin the pooled rows with exactly
the agreed bin edges and train them as the federation does: the forced-bins file and the parameters that
leave LightGBM no room for bins of its own."""

import json

import numpy as np

from fl_lightgbm.binning import BinMapper


def write_forced_bins(mappers: list[BinMapper]) -> str:
    """The agreed bin edges as LightGBM's forced-bins file (`forcedbins_filename`): each numerical feature's
    finite upper bounds. LightGBM's BinMapper::FindBin appends the last bound, +inf, and a NaN bin's bound
    after it itself. Categorical features take no forced bins: LightGBM bins them with its own rules,
    which the agreed categories follow."""
    return json.dumps([
        {"feature": f, "bin_upper_bound": [float(b) for b in m.upper_bounds if np.isfinite(b)]}
        for f, m in enumerate(mappers) if not m.is_categorical
    ])


def central_baseline_params(mappers: list[BinMapper], max_bin: int, num_data: int) -> dict:
    """The LightGBM parameters that, with the training parameters and the forced-bins file, make LightGBM
    bin the pooled rows with exactly the agreed edges and train them as the federation does."""
    return {
        # No room left for bins of LightGBM's own. A NaN bin counts: LightGBM reserves one of `max_bin` for it.
        # LightGBM takes no `max_bin` below 2; a feature of one bin holds one value, which gives one bin anyway.
        "max_bin_by_feature": [max_bin if m.is_categorical else max(m.num_bins, 2) for m in mappers],
        "categorical_feature": [f for f, m in enumerate(mappers) if m.is_categorical],
        "bin_construct_sample_cnt": num_data,  # bin all rows, not a sample
        "deterministic": True,
        "force_col_wise": True,
        "enable_bundle": False,
        "feature_pre_filter": False,
        "num_threads": 1,
    }
