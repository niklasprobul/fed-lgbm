"""The real-data equivalence suite: every public dataset under every site split that applies equals the central
baseline; the microbiome's cases are in `test_real_data_microbiome.py`. Slow; each case prints its runtime (run
with `uv run pytest -m slow tests/test_real_data.py`)."""

import time

import numpy as np
import pytest

from datasets import load_public
from fl_lightgbm.simulator import split_by_column, split_evenly, split_missing_category, split_skewed
from test_simulation import train_and_compare

pytestmark = pytest.mark.slow

PARAMS = {"num_iterations": 20, "num_leaves": 15, "learning_rate": 0.1}
HELD_OUT = 0.2  # of each dataset's rows, held out before the split

# "by column" applies where a dataset has a column with a few values to split by (breast cancer and California
# housing have none), "missing category" where it has a categorical feature.
SPLITS = ["even", "skewed", "by column", "missing category"]
CASES = [
    ("breast_cancer", "even"), ("breast_cancer", "skewed"),
    ("california_housing", "even"), ("california_housing", "skewed"),
    *(("adult", split) for split in SPLITS),
]
OBJECTIVE = {"breast_cancer": "binary", "california_housing": "regression", "adult": "binary"}
# The column "by column" splits by, and the categorical feature whose most frequent category the first site
# lacks in "missing category".
BY_COLUMN = {"adult": "race"}
MISSING_CATEGORY = {"adult": "occupation"}


def load(dataset):
    """The dataset and each row's site under "by column"."""
    data = load_public(dataset)
    by_column = data.X[:, data.feature_names.index(BY_COLUMN[dataset])] if dataset in BY_COLUMN else None
    return data, by_column


def tables_for(split, data, X, y, by_column, missing_category):
    """The sites' (X, y) tables under `split`."""
    if split == "even":
        return split_evenly(X, y, num_sites=5)
    if split == "skewed":
        return split_skewed(X, y, num_sites=6)
    if split == "by column":
        return split_by_column(X, y, by_column)
    feature = data.feature_names.index(missing_category)
    categories, counts = np.unique(X[~np.isnan(X[:, feature]), feature], return_counts=True)
    return split_missing_category(X, y, num_sites=5, feature=feature, category=categories[np.argmax(counts)])


def run_case(dataset, split, data, by_column, objective, missing_category, capsys):
    """Train `data` federated under `split` and centrally, and check they are equal. Returns the sites' tables."""
    held_out = np.random.default_rng(0).random(len(data.y)) < HELD_OUT
    X, y = data.X[~held_out], data.y[~held_out]
    tables = tables_for(split, data, X, y, None if by_column is None else by_column[~held_out], missing_category)
    categorical = [data.feature_names.index(name) for name in data.categorical]
    params = {"objective": objective, **PARAMS}

    start = time.perf_counter()
    # The default sparse encoding; held-out rows are predicted like the central baseline too.
    result = train_and_compare(params, tables, data.X[held_out], categorical=categorical)

    with capsys.disabled():
        print(f"\n{dataset}, {split}: {len(tables)} sites, {len(y):,} rows, {X.shape[1]:,} features, "
              f"{result.rounds} rounds: {time.perf_counter() - start:.1f} s federated and central")
    return tables


@pytest.mark.parametrize("dataset, split", CASES, ids=[f"{d}-{s}" for d, s in CASES])
def test_federated_run_equals_central_baseline(dataset, split, capsys):
    data, by_column = load(dataset)
    run_case(dataset, split, data, by_column, OBJECTIVE[dataset], MISSING_CATEGORY.get(dataset), capsys)
