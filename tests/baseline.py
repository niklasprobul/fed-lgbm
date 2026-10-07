"""The central baseline: LightGBM on the pooled rows, binned with exactly the agreed bin edges."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fl_lightgbm.binning import BinMapper
from fl_lightgbm.central import central_baseline_params, write_forced_bins


@contextmanager
def central_dataset(X: np.ndarray, y: np.ndarray, mappers: list[BinMapper], params: dict) -> Iterator[lgb.Dataset]:
    """The pooled rows, to be binned with the agreed edges as the run exports them: the forced-bins file
    and the parameters that leave LightGBM no room for bins of its own. LightGBM reads the forced-bins
    file on construction, so it exists only within the context."""
    with tempfile.TemporaryDirectory() as tmp:
        forced = Path(tmp) / "forcedbins.json"
        forced.write_text(write_forced_bins(mappers))
        full = {
            "objective": "regression",
            **params,
            **central_baseline_params(mappers, params.get("max_bin", 255), len(X)),
            "forcedbins_filename": str(forced),
            "verbose": -1,
        }
        categorical = full.pop("categorical_feature")  # LightGBM's Python package takes them as a Dataset argument
        yield lgb.Dataset(X, y, params=full, categorical_feature=categorical or "auto")


def train_central(X: np.ndarray, y: np.ndarray, mappers: list[BinMapper], params: dict) -> lgb.Booster:
    """Train LightGBM on the pooled rows with exactly the agreed edges."""
    with central_dataset(X, y, mappers, params) as dataset:
        # keep_training_booster: otherwise lgb.train reloads the model from its own text, which
        # prints split gains and internal values with only six significant digits.
        return lgb.train(dataset.params, dataset, keep_training_booster=True)


def trees(booster: lgb.Booster) -> list[dict]:
    return booster.dump_model()["tree_info"]
