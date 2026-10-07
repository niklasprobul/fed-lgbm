import numpy as np
import pandas as pd
import pytest

import datasets
from datasets import fetch, load_public


@pytest.mark.parametrize("name, shape, binary", [
    ("breast_cancer", (569, 30), True),
    ("california_housing", (20640, 8), False),
    ("adult", (48842, 14), True),
])
def test_public_datasets_have_numeric_features_and_a_numeric_target(name, shape, binary):
    data = load_public(name)
    assert data.X.shape == shape and data.X.dtype == np.float64
    assert len(data.feature_names) == shape[1]
    assert not np.isnan(data.y).any()
    if binary:
        assert set(np.unique(data.y)) == {0.0, 1.0}


def test_adult_has_categorical_features_as_codes_with_missing_values():
    data = load_public("adult")
    assert data.categorical == ["workclass", "education", "marital-status", "occupation", "relationship", "race",
                                "sex", "native-country"]
    assert data.y.mean() == pytest.approx(11687 / 48842)  # >50K
    for name in data.categorical:
        codes = data.X[:, data.feature_names.index(name)]
        present = codes[~np.isnan(codes)]
        assert np.array_equal(present, np.round(present)) and present.min() == 0
    assert np.isnan(data.X[:, data.feature_names.index("occupation")]).sum() == 2809


def test_without_data_tests_skip_and_name_the_fetch_script(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="tests/datasets.py"):
        load_public("adult")


def no_download():
    raise AssertionError("downloaded although the file is there")


def test_fetch_writes_public_datasets(monkeypatch, tmp_path):
    frame = pd.DataFrame({"x": [1.0, np.nan], "target": [0, 1]})
    monkeypatch.setattr(datasets, "PUBLIC", {name: lambda: frame for name in datasets.PUBLIC})
    data_dir = tmp_path / "data"

    fetch(data_dir)

    for name in datasets.PUBLIC:
        pd.testing.assert_frame_equal(pd.read_csv(data_dir / name), frame)


def test_fetch_again_downloads_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "PUBLIC", {name: no_download for name in datasets.PUBLIC})
    data_dir = tmp_path / "data"
    for name in datasets.PUBLIC:
        (data_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_text("already here")

    fetch(data_dir)

    assert all(p.read_text() == "already here" for p in data_dir.rglob("*.csv"))
