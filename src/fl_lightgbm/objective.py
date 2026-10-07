"""The objectives: the init score and label weights from global sums, and each site's gradients.

L2 regression follows RegressionL2loss, binary logloss follows BinaryLogloss (sigmoid 1, no sample
weights). Binary counts a row as positive when its label is > 0, as LightGBM does.
"""

import math

import numpy as np

from fl_lightgbm.params import Params
from fl_lightgbm.split import EPSILON


def label_sum(labels: np.ndarray, params: Params) -> float:
    """A site's share of the sum BoostFromScore averages: Σlabel, or for binary the positive rows."""
    if params.objective == "binary":
        return float(np.count_nonzero(labels > 0))
    return float(np.sum(labels.astype(np.float64)))


def init_score(sum_label: float, num_data: int, params: Params) -> float:
    """GBDT::BoostFromAverage from global sums; it ignores a score within kEpsilon of 0."""
    score = sum_label / num_data
    if params.objective == "binary":
        p = max(min(score, 1.0 - EPSILON), EPSILON)
        score = math.log(p / (1.0 - p))
    return score if abs(score) > EPSILON else 0.0


def label_weights(num_positive: int, num_data: int, params: Params) -> tuple[float, float]:
    """BinaryLogloss::Init: the weights of negative and positive rows."""
    num_negative = num_data - num_positive
    negative, positive = 1.0, 1.0
    if params.is_unbalance and num_positive > 0 and num_negative > 0:
        if num_positive > num_negative:
            negative = num_positive / num_negative
        else:
            positive = num_negative / num_positive
    return negative, positive * params.scale_pos_weight


def gradients(
    labels: np.ndarray, score: np.ndarray, params: Params, weights: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """BinaryLogloss::GetGradients or RegressionL2loss::GetGradients: computed in double, stored as float32."""
    if params.objective == "binary":
        is_pos = labels > 0
        label = np.where(is_pos, 1.0, -1.0)
        weight = np.where(is_pos, weights[1], weights[0])
        response = -label / (1.0 + np.exp(label * score))
        abs_response = np.abs(response)
        return (response * weight).astype(np.float32), (abs_response * (1.0 - abs_response) * weight).astype(np.float32)
    return (score - labels.astype(np.float64)).astype(np.float32), np.ones(len(labels), dtype=np.float32)


# LightGBM's default metric for each objective, which the training report gives after every tree.
METRIC = {"regression": "l2", "binary": "binary_logloss"}


def loss_sum(labels: np.ndarray, score: np.ndarray, params: Params) -> float:
    """A site's share of the training loss: Σ over its rows of L2Metric's or BinaryLoglossMetric's loss.
    Like those metrics, it ignores the label weights; the aggregator divides by the global row count."""
    if params.objective == "binary":
        prob = 1.0 / (1.0 + np.exp(-score))
        p = np.where(labels > 0, prob, 1.0 - prob)  # the probability of the true label
        return float(np.sum(-np.log(np.where(p > EPSILON, p, EPSILON))))
    return float(np.sum((score - labels.astype(np.float64)) ** 2))
