"""The FL-Net adapter: the SDK's client app and aggregator, delegating to the core's site and aggregator logic.

Every site, the coordinator's included, runs `FederatedLightGBM`; the coordinator also runs
`FederatedLightGBMAggregator`, which the SDK calls once per round with all sites' messages. A message is
the core's payload with the sender's site ID, so that the aggregator sums in the same order whatever order
the messages arrive in; the first round's message also carries the site's parameters and bounds, from
which the aggregator builds its core logic (the SDK gives it no configuration of its own).
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyfedappwrap.engine.config.system_config import system_settings
from pyfedappwrap.engine.federated.aggregator import AppAggregator
from pyfedappwrap.learning.federated import BaseFederatedApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from pydantic.dataclasses import dataclass

from fl_lightgbm.aggregator import Aggregator
from fl_lightgbm.params import Params
from fl_lightgbm.site import Site

AGGREGATOR = "fl-lightgbm"  # the name the aggregator is registered under in main.py
MODEL_FILE = "model.txt"  # under MODEL_DIR, where FL-Net uploads it from and where prediction finds it
MAX_COMMUNICATION_ID_BYTES = 255  # FL-Net's controller stores the length in one byte


@dataclass
class Config(AppConfig):
    """The hyperparameters in app.yml, with the same defaults: in TEST_MODE the coordinator gets only these."""
    target: str = "target"
    categorical_columns: str = ""  # comma-separated
    bounds: str = ""  # JSON object: feature name -> [lo, hi], for every feature
    communication_id: str = "fl-lightgbm"
    # LightGBM's parameters, with LightGBM's defaults, as in fl_lightgbm.params.Params
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

    @property
    def lightgbm_params(self) -> dict:
        """The LightGBM parameters set to other than their defaults. FL-Net always sends every hyperparameter,
        so this is how a run tells whether `num_leaves` was set, which decides whether `max_depth` lowers it."""
        defaults = Params()
        return {f.name: getattr(self, f.name) for f in fields(Params)
                if getattr(self, f.name) != getattr(defaults, f.name)}

    @property
    def federated_rounds(self) -> int:
        """The round budget; FL-Net's runtime reads it from the coordinator's config."""
        return Params.from_dict(self.lightgbm_params).round_budget

    def categorical(self) -> list[str]:
        return [name.strip() for name in self.categorical_columns.split(",") if name.strip()]

    def feature_bounds(self, features: list[str]) -> list[list[float]] | None:
        """The configured bounds in the order of `features`, or None if the sites share their min/max."""
        if not self.bounds.strip():
            return None
        given = json.loads(self.bounds)
        missing, unknown = sorted(set(features) - set(given)), sorted(set(given) - set(features))
        if missing or unknown:
            raise ValueError(f"bounds must cover exactly the features; missing: {missing}, unknown: {unknown}")
        return [[float(given[name][0]), float(given[name][1])] for name in features]


@dataclass
class Input(AppInputConfig):
    # The site's CSV, as a path, read here so that every float arrives exactly. Optional[...], not `| None`:
    # the SDK only recognises typing.Optional when it decides what to hand over.
    data: Optional[Path] = None


@dataclass
class Output(AppOutputConfig):
    model: Path | None = None
    report: Path | None = None
    predictions: Path | None = None


def communication_ids(base: str, rounds: int) -> list[str]:
    """One communication ID per round, each used once in the run, as FL-Net requires."""
    ids = [f"{base}-{r}" for r in range(1, rounds + 1)]
    if len(ids[-1].encode()) > MAX_COMMUNICATION_ID_BYTES:
        raise ValueError(f"communication ID {ids[-1]!r} is longer than {MAX_COMMUNICATION_ID_BYTES} bytes; "
                         f"shorten communication_id")
    return ids


def read_site_table(path: Path, target: str, categorical: list[str]) -> tuple[np.ndarray, np.ndarray, list[str],
                                                                              list[int]]:
    """The site's CSV as features X, target y, the feature names and the indices of the categorical features."""
    table = pd.read_csv(path, float_precision="round_trip")
    if target not in table.columns:
        raise ValueError(f"target column {target!r} is not in the data; its columns are {list(table.columns)}")
    features = table.drop(columns=[target])
    names = [str(c) for c in features.columns]
    unknown = sorted(set(categorical) - set(names))
    if unknown:
        raise ValueError(f"categorical columns {unknown} are not features in the data")
    text = [name for name in names if not pd.api.types.is_numeric_dtype(features[name])]
    if text:
        raise ValueError(f"columns {text} are not numeric; categorical columns must be coded as integers")
    return (features.to_numpy(np.float64), table[target].to_numpy(np.float64), names,
            [names.index(name) for name in categorical])


def read_prediction_table(path: Path, features: list[str]) -> np.ndarray:
    """The model's features from the site's CSV, in the model's order; other columns, such as the target, are
    ignored."""
    table = pd.read_csv(path, float_precision="round_trip")
    missing = [name for name in features if name not in table.columns]
    if missing:
        raise ValueError(f"the model's features {missing} are not in the data")
    return table[features].to_numpy(np.float64)


class FederatedLightGBM(BaseFederatedApp[Config, Input, Output]):
    def __init__(self) -> None:
        super().__init__()
        self.model: str | None = None  # LightGBM text model, once trained

    def run_train(self, data: Input) -> Output:
        assert self.config is not None and data.data is not None
        X, y, names, categorical = read_site_table(data.data, self.config.target, self.config.categorical())
        bounds = self.config.feature_bounds(names)
        site = Site(X, y, Params.from_dict(self.config.lightgbm_params), names, bounds, categorical=categorical)
        reply = None
        for round_nr, communication_id in enumerate(communication_ids(self.config.communication_id,
                                                                      self.config.federated_rounds)):
            message: dict[str, Any] = {"site": self.federated_client_id, "payload": site.payload(reply)}
            if round_nr == 0:
                message["config"] = {"params": self.config.lightgbm_params, "bounds": bounds}
            reply = self.communicator.aggregate(message, aggregator_name=AGGREGATOR,
                                                communication_id=communication_id).data
        assert site.model is not None and site.report is not None
        self.model = site.model
        return Output(model=self._write_output(MODEL_FILE, site.model),
                      report=self._write_output("report.json", site.report))

    def get_test_data(self) -> dict[str, Path]:
        """The input of FL-Net's TEST_MODE smoke run, in place of the SDK's random table, which has no target."""
        return {"data": Path(__file__).parent / "test_data" / "smoke.csv"}

    def _write_output(self, name: str, text: str) -> Path:
        path = self.resolve_local_output_path(name)
        path.write_text(text)
        return path

    def _save(self) -> str:
        assert self.model is not None
        model_dir = Path(system_settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / MODEL_FILE).write_text(self.model)
        return MODEL_FILE

    def _load(self, path: str) -> None:
        raise NotImplementedError("the SDK never calls _load")

    def run_prediction(self, data: Input) -> Output:
        """Local prediction with the model FL-Net placed in MODEL_DIR; the SDK never calls `_load`."""
        assert data.data is not None
        model_dir = Path(system_settings.model_dir)
        try:
            booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
        except lgb.basic.LightGBMError as error:
            raise ValueError(f"no readable model {MODEL_FILE} in MODEL_DIR ({model_dir}): {error}") from error
        X = read_prediction_table(data.data, booster.feature_name())
        return Output(predictions=self._write_output("predictions.csv",
                                                     pd.DataFrame({"prediction": booster.predict(X)}).to_csv(index=False)))


class FederatedLightGBMAggregator(AppAggregator):
    """Keeps the core aggregator, and with it the tree being built, between rounds."""

    def __init__(self) -> None:
        self.core: Aggregator | None = None

    def aggregate(self, data: list[Any], n_clients: int, meta: Any = None) -> Any:
        messages = sorted(data, key=lambda m: m["site"])
        if "config" in messages[0]:  # the first round of a run
            configs = [m["config"] for m in messages]
            if any(c != configs[0] for c in configs):
                raise ValueError(f"the sites run with different parameters or bounds: {configs}")
            bounds = configs[0]["bounds"]
            self.core = Aggregator(Params.from_dict(configs[0]["params"]),
                                   None if bounds is None else [(lo, hi) for lo, hi in bounds])
        assert self.core is not None
        return self.core.reply([m["payload"] for m in messages])
