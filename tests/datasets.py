"""Test data in the gitignored `data/` folder: the fetch script that fills it, and the loaders tests read it with.

Fill it with `uv run python tests/datasets.py`. Running it again downloads nothing that is there. The CRC
microbiome data has its own module, `tests/microbiome.py`, which the public mirror leaves out.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer
    return load_breast_cancer(as_frame=True).frame


def _california_housing() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing
    with tempfile.TemporaryDirectory() as home:
        return fetch_california_housing(data_home=home, as_frame=True).frame.rename(columns={"MedHouseVal": "target"})


def _adult() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    with tempfile.TemporaryDirectory() as home:
        # OpenML 1590 is "adult", version 2.
        return fetch_openml(data_id=1590, data_home=home, as_frame=True).frame.rename(columns={"class": "target"})


# One CSV per public dataset, the target in column `target`; missing values are empty fields.
PUBLIC = {
    "breast_cancer.csv": _breast_cancer,  # binary
    "california_housing.csv": _california_housing,  # regression
    "adult.csv": _adult,  # binary, categorical features and missing values
}


def fetch(data_dir: Path = DATA_DIR) -> None:
    """Fill `data_dir` with every public test dataset, leaving alone the files already there. Each file
    appears only once complete, so an interrupted run is picked up again by the next."""
    for name, load in PUBLIC.items():
        path = data_dir / name
        if not path.exists():
            print(f"downloading {name}")
            _write(path, lambda part: load().to_csv(part, index=False))


def _write(path: Path, write_to) -> None:
    part = path.with_name(path.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    write_to(part)
    part.rename(path)


def _data_file(name: str, script: str = "tests/datasets.py") -> Path:
    """The path of a file in `data/`; the calling test skips if it is not there."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{path} is missing: run `uv run python {script}` to fill data/")
    return path


@dataclass
class Dataset:
    X: np.ndarray  # float64, NaN where missing; categorical columns hold category codes 0, 1, …
    y: np.ndarray
    feature_names: list[str]
    categorical: list[str]  # the feature names to treat as categorical


def load_public(name: Literal["breast_cancer", "california_housing", "adult"]) -> Dataset:
    """A public dataset: every column but `target` is a feature, and the text columns are categorical.
    adult's target is 1 for an income above 50K."""
    table = pd.read_csv(_data_file(f"{name}.csv"))
    y = table.pop("target")
    if name == "adult":
        y = y == ">50K"
    categorical = [c for c in table.columns if not pd.api.types.is_numeric_dtype(table[c])]
    for c in categorical:
        table[c] = table[c].astype("category").cat.codes.replace(-1, np.nan)
    return Dataset(table.to_numpy(np.float64), y.to_numpy(np.float64), list(table.columns), categorical)


if __name__ == "__main__":
    fetch()
