FROM ghcr.io/fedlearnnet/python-tool-api/base-python-dep:0.7.12

# The base image brings the SDK and its dependencies; the pin keeps the SDK the adapter was written for.
RUN pip install --no-cache-dir FL-Net-Python-Tool-API==0.7.12

# The core package, with LightGBM pinned in pyproject.toml.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# The FL-Net adapter, its config and README (the SDK loads both), and the TEST_MODE smoke-run input.
COPY app.yml README.md main.py flnet_app.py ./
COPY test_data ./test_data

CMD ["python", "-m", "main"]
