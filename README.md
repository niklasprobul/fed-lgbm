# fl-lightgbm

Federated LightGBM for horizontally partitioned data: every site holds the same features for different rows. Sites share only summed gradient and hessian histograms. The aggregator chooses every split with LightGBM's own rules, and the result is an ordinary LightGBM text model that loads with `lightgbm.Booster`. It equals LightGBM trained on the pooled rows of all sites with the same bin edges and parameters.

It runs as an FL-Net federated tool and as a local simulation.

## On FL-Net

Each site provides one CSV with a header row. Name the target column in `target`; every other column is a feature. Categorical columns must be coded as non-negative integers and listed in `categorical_columns`. Missing values may be empty cells.

The other hyperparameters are LightGBM's, with LightGBM's names and defaults (`objective` is `regression` or `binary`). LightGBM parameters that would give a different model, such as bagging or `feature_fraction`, are not offered. The number of rounds is computed from the parameters: 3 setup rounds, `num_iterations × (num_leaves − 1)` split rounds, 1 closing round and 1 padding round. FL-Net takes about 1 – 2 s per round, so 100 trees of 31 leaves take roughly 45 – 90 minutes.

`bounds` is optional: a JSON object such as `{"age": [0, 120], "bmi": [10, 70]}` covering every feature. When it is given, the sites do not share their minimum and maximum.

Every site, the coordinator included, trains on its own rows, and every site ends with the same two outputs:

- `model`: the LightGBM text model;
- `report`: the training report (JSON) with the parameters, the number of sites, the rounds used and the padding rounds, the training loss after every tree, and the LightGBM parameters for training the central baseline.

A prediction run with a trained model is local to one site, with no federation. It reads the model's features from the site's CSV, ignoring other columns such as the target, and writes `predictions`: a CSV with one column, `prediction`, one row per input row in the same order (the predicted value for `regression`, the probability of class 1 for `binary`). The model is read from `MODEL_DIR/model.txt`, where FL-Net places it.

The tool is built from this repository: `Dockerfile`, `app.yml`, `main.py` and the adapter `flnet_app.py`, on the FL-Net SDK 0.7.12 (`FL-Net-Python-Tool-API`).

## Locally

```python
from fl_lightgbm.simulator import simulate, split_evenly

result = simulate({"objective": "binary", "num_iterations": 50}, split_evenly(X, y, num_sites=5))
booster = lightgbm.Booster(model_str=result.model)
```

`simulate` runs the same site and aggregator logic as FL-Net, with the same payloads and the same fixed round count.

## Development

```sh
uv sync
uv run pytest            # fast suite
uv run python tests/datasets.py  # fills the gitignored data/ folder for the real-data tests
uv run python tests/microbiome.py  # copies the CRC microbiome data there (not in the public mirror)
uv run pytest -m slow    # real-data and FL-Net runner tests
uv run python tests/benchmark.py  # binning, payload and round cost: writes docs/benchmarks.md (not in the public mirror)
uv run mypy
```

Build the FL-Net image and run its smoke test as FL-Net's pipeline does:

```sh
docker build -t fl-lightgbm .
docker run --rm -e TEST_MODE=true fl-lightgbm python -m main
```

Glossary: `CONTEXT.md`. Decisions: `docs/adr/`.
