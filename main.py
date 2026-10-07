"""FL-Net entry point: `python -m main`, also in the build pipeline's TEST_MODE smoke run."""

from pyfedappwrap.engine.runtime import FedDBEngine

from flnet_app import AGGREGATOR, FederatedLightGBM, FederatedLightGBMAggregator

if __name__ == "__main__":
    engine = FedDBEngine()
    engine.register_aggregator(FederatedLightGBMAggregator(), AGGREGATOR)
    engine.register_federated(FederatedLightGBM())
    engine.start()
    engine.wait_until_stop()
