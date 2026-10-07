"""The FL-Net adapter (flnet_app.py): the client app and the aggregator on the SDK, delegating to the core."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from fl_lightgbm.params import Params
from fl_lightgbm.simulator import simulate
from flnet_app import AGGREGATOR, Config, FederatedLightGBM, FederatedLightGBMAggregator, Input, communication_ids
from pyfedappwrap.engine.config.config import ModeType
from pyfedappwrap.engine.config.system_config import local_runtime_paths, system_settings
from pyfedappwrap.engine.federated.models import FLNetLocalParticipantConfigDTO, FLNetLocalTestConfigDTO
from pyfedappwrap.engine.tests.federated.runner import LocalFederatedRunner
from pyfedappwrap.learning.base_app import DataValidatorService

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def tool_dir(monkeypatch, tmp_path):
    """The SDK reads app.yml and README.md from the working directory and writes the model to MODEL_DIR."""
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(system_settings, "model_dir", str(tmp_path / "model"))


def synthetic(seed, n, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y


def write_site_csv(path, X, y):
    table = pd.DataFrame(X, columns=[f"Column_{i}" for i in range(X.shape[1])])
    table["target"] = y
    table.to_csv(path, index=False)
    return path


class InProcessCommunicator:
    """One site talking straight to the adapter's aggregator, through JSON as on FL-Net; records every round."""

    def __init__(self):
        self.aggregator = FederatedLightGBMAggregator()
        self.communication_ids = []
        self.replies = []

    def aggregate(self, data, aggregator_name, communication_id=None, data_type=None, smpc=None, dp=None):
        self.communication_ids.append(communication_id)
        reply = self.aggregator.aggregate([json.loads(json.dumps(data))], n_clients=1)
        self.replies.append(json.loads(json.dumps(reply)))
        return SimpleNamespace(data=self.replies[-1])


def test_app_yml_offers_every_supported_lightgbm_parameter_with_the_config_and_lightgbm_defaults():
    """In TEST_MODE the coordinator gets only the config's defaults and the other sites app.yml's, so the
    two must agree, or the sites would budget different round counts."""
    hyperparams = {h["name"]: h["default"] for h in yaml.safe_load((REPO / "app.yml").read_text())["config"]["hyperparams"]}
    config = Config()

    assert hyperparams == {name: getattr(config, name) for name in hyperparams}
    assert set(hyperparams) == set(asdict(Params())) | {"target", "categorical_columns", "bounds", "communication_id"}
    assert config.lightgbm_params == {}  # all of LightGBM's defaults


def test_client_app_calls_aggregate_once_per_budgeted_round_with_unique_communication_ids(tmp_path):
    X, y = synthetic(seed=1, n=300)
    config = Config(num_iterations=4, num_leaves=6, learning_rate=0.3)
    app = FederatedLightGBM()
    app.config = config
    app.federated_client_id = "site-0"
    app.communicator = communicator = InProcessCommunicator()

    with local_runtime_paths(output_dir=str(tmp_path / "output")):
        output = app.run_train(Input(data=write_site_csv(tmp_path / "data.csv", X, y)))

    ids = communicator.communication_ids
    assert len(ids) == config.federated_rounds == 2 + 4 * 5 + 1 + 1
    assert len(set(ids)) == len(ids)
    assert all(len(i.encode()) <= 255 for i in ids)
    assert len(json.dumps(communicator.replies[-1])) < 100  # FL-Net ships the last reply in its finish message
    expected = simulate({"num_iterations": 4, "num_leaves": 6, "learning_rate": 0.3}, [(X, y)])
    assert output.model.read_text() == expected.model
    assert output.report.read_text() == expected.report


def test_prediction_on_a_site_csv_equals_lightgbm_predict_with_the_saved_model(tmp_path):
    X, y = synthetic(seed=3, n=400)
    X[::5, 1] = np.nan
    X[:, 3] = np.floor(np.abs(X[:, 3]) * 3)
    config = Config(num_iterations=3, num_leaves=5, learning_rate=0.3, categorical_columns="Column_3")
    trainer = FederatedLightGBM()
    trainer.config = config
    trainer.federated_client_id = "site-0"
    trainer.communicator = InProcessCommunicator()
    with local_runtime_paths(output_dir=str(tmp_path / "train-output")):
        trainer.run_train(Input(data=write_site_csv(tmp_path / "train.csv", X[:300], y[:300])))
    trainer._save()  # as at the end of an FL-Net training run
    csv = write_site_csv(tmp_path / "predict.csv", X[300:], y[300:])
    table = pd.read_csv(csv)
    table.insert(0, "sample_id", range(len(table)))  # columns that are not features are ignored
    table[table.columns[::-1]].to_csv(csv, index=False)
    app = FederatedLightGBM()  # a prediction run starts a fresh app, with only the model file in MODEL_DIR
    app.config = config

    with local_runtime_paths(output_dir=str(tmp_path / "output")):
        output = app.run_prediction(Input(data=csv))

    booster = lgb.Booster(model_file=str(Path(system_settings.model_dir) / "model.txt"))
    predictions = pd.read_csv(output.predictions, float_precision="round_trip")
    assert list(predictions.columns) == ["prediction"]
    np.testing.assert_array_equal(predictions["prediction"].to_numpy(), booster.predict(X[300:]))
    assert DataValidatorService().validate_output(output, ModeType.PREDICTION)[1]  # app.yml declares it


@pytest.mark.parametrize("model_text", [None, "not a model\n"])
def test_prediction_without_a_readable_model_file_names_model_dir(tmp_path, model_text):
    model_dir = Path(system_settings.model_dir)
    if model_text is not None:
        model_dir.mkdir()
        (model_dir / "model.txt").write_text(model_text)
    X, y = synthetic(seed=4, n=10)
    app = FederatedLightGBM()
    app.config = Config()

    with pytest.raises(ValueError, match="MODEL_DIR"):
        app.run_prediction(Input(data=write_site_csv(tmp_path / "predict.csv", X, y)))


def test_communication_ids_longer_than_255_bytes_are_refused():
    assert len(communication_ids("x" * 250, 3004)[-1].encode()) == 255
    with pytest.raises(ValueError, match="255 bytes"):
        communication_ids("x" * 251, 3004)


@pytest.mark.slow
def test_run_through_the_sdk_runner_gives_the_simulator_model_and_report_at_every_site(tmp_path):
    X, y = synthetic(seed=2, n=900)
    X[::7, 1] = np.nan  # missing values travel as empty CSV cells
    X[:, 3] = np.floor(np.abs(X[:, 3]) * 3)  # a categorical feature, coded as integers
    tables = [(X[:450], y[:450]), (X[450:750], y[450:750]), (X[750:], y[750:])]
    lightgbm_params = {"num_iterations": 5, "num_leaves": 7, "learning_rate": 0.3, "min_data_in_leaf": 10}
    hyper_params = {**lightgbm_params, "target": "target", "categorical_columns": "Column_3"}
    participants = []
    for i, (Xs, ys) in enumerate(tables):
        (tmp_path / f"site-{i}" / "data").mkdir(parents=True)
        write_site_csv(tmp_path / f"site-{i}" / "data" / "data.csv", Xs, ys)
        participants.append(FLNetLocalParticipantConfigDTO(
            participant_id=f"site-{i}", role="AGGREGATOR" if i == 0 else "CLIENT",  # site-0 is the coordinator
            base_dir=tmp_path / f"site-{i}", hyper_params=hyper_params, input_file_paths={"data": "data.csv"}))
    runner = LocalFederatedRunner(FLNetLocalTestConfigDTO(participants=participants, poll_interval=0.001,
                                                          timeout=30, max_polls=100_000),
                                  aggregators={AGGREGATOR: FederatedLightGBMAggregator()})

    results = runner.run({p.participant_id: FederatedLightGBM() for p in participants})

    expected = simulate(lightgbm_params, tables, categorical=[3])
    assert all(r.success for r in results.values()), results
    # The runner returns the coordinator's aggregator result, not its client app's outputs; those are in its folder.
    outputs = [(r.model, r.report) for r in (results["site-1"].result, results["site-2"].result)]
    outputs.append((tmp_path / "site-0" / "output" / "model.txt", tmp_path / "site-0" / "output" / "report.json"))
    for model, report in outputs:
        assert model.read_text() == expected.model
        assert report.read_text() == expected.report
    assert json.loads(expected.report)["num_sites"] == 3  # the coordinator trained too
    assert (Path(system_settings.model_dir) / "model.txt").read_text() == expected.model  # for FL-Net's upload


@pytest.mark.slow
def test_test_mode_smoke_run_trains_and_saves_the_model_at_every_participant(tmp_path):
    """FL-Net's build pipeline runs `python -m main` with TEST_MODE, as the Docker image does. It exits 0 even
    when the federated run fails, so the check is that every participant saved a model that loads."""
    for name in ["app.yml", "README.md", "main.py", "flnet_app.py", "test_data"]:
        (shutil.copytree if (REPO / name).is_dir() else shutil.copy)(REPO / name, tmp_path / name)

    run = subprocess.run([sys.executable, "-m", "main"], cwd=tmp_path, env={**os.environ, "TEST_MODE": "true"},
                         capture_output=True, text=True, timeout=300)

    assert run.returncode == 0, run.stderr
    assert run.stdout.count("Saving model to model.txt") == 3, run.stdout  # two sites and the coordinator
    assert lgb.Booster(model_file=str(tmp_path / "model" / "model.txt")).num_trees() == 100  # the defaults
