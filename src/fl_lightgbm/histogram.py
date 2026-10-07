"""Gradient/hessian histograms of one leaf, built on a site.

All features share one flat array: feature f owns the slice
`offsets[f] : offsets[f] + num_bins[f]`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Histogram:
    grad: np.ndarray  # float64, Σg per bin, all features concatenated
    hess: np.ndarray  # float64, Σh per bin
    count: int  # rows in the leaf


def feature_offsets(num_bins: np.ndarray) -> np.ndarray:
    """Start of each feature's slice in the flat histogram, plus the total length at the end."""
    return np.concatenate([[0], np.cumsum(num_bins)])


def leaf_histogram(
    binned: np.ndarray,
    num_bins: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    rows: np.ndarray,
) -> Histogram:
    """Sum the float32 gradients and hessians of `rows` into float64 bins, like LightGBM."""
    offsets = feature_offsets(num_bins)
    num_features = binned.shape[1]
    flat_bins = (binned[rows] + offsets[:-1]).ravel()
    grad = np.bincount(flat_bins, np.repeat(gradients[rows].astype(np.float64), num_features), offsets[-1])
    hess = np.bincount(flat_bins, np.repeat(hessians[rows].astype(np.float64), num_features), offsets[-1])
    return Histogram(grad, hess, len(rows))


def most_freq_bin_positions(num_bins: np.ndarray, most_freq_bins: np.ndarray) -> np.ndarray:
    """Each feature's most frequent bin, as an index into the flat histogram."""
    return feature_offsets(num_bins)[:-1] + most_freq_bins


def fix_histogram(hist: Histogram, sum_grad: float, sum_hess: float, num_bins: np.ndarray,
                  most_freq_bins: np.ndarray) -> Histogram:
    """Dataset::FixHistogram: rebuild each feature's most frequent bin as the leaf total minus its other bins.

    LightGBM never accumulates that bin. A most frequent bin 0 is left as it is: LightGBM does not store
    it, and split finding never reads it.
    """
    grad, hess = hist.grad.copy(), hist.hess.copy()
    offsets = feature_offsets(num_bins)
    for f in np.flatnonzero(most_freq_bins > 0):
        mfb = offsets[f] + most_freq_bins[f]
        others = np.r_[offsets[f]:mfb, mfb + 1:offsets[f + 1]]
        grad[mfb] = subtract_in_order(sum_grad, grad[others])
        hess[mfb] = subtract_in_order(sum_hess, hess[others])
    return Histogram(grad, hess, hist.count)


def subtract_in_order(total: float, values: np.ndarray) -> float:
    """`total` minus each of `values` in turn, as a C++ `-=` loop does it (cumsum adds sequentially)."""
    return float(np.cumsum(np.concatenate([[total], -values]))[-1])
