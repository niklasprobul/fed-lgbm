"""Payload encodings: how a site writes its grid counts, histograms and other sums for the aggregator
(ADR 0003, 0006).

Both ends hold the same encoding; the site encodes, the aggregator decodes and sums. Every encoding
must give the same sums wherever the aggregator reads them.

- `SPARSE`, the plaintext default: each feature's most frequent bin is skipped (the aggregator rebuilds
  it from the leaf totals, as LightGBM's FixHistogram does), only non-empty bins and grid cells are
  listed, and the numbers travel as base64 little-endian binary. Grid counts have no most frequent bin
  to skip: zeros are not in the grid, the aggregator derives their count.
- `DENSE`: every bin and cell as a JSON list of float64 or int values; the plaintext reference.
- `FixedPoint(exponent)`, for secure aggregation (FL-Net SMPC, not switched on in v1): every bin and
  cell, in the same layout from every site, as one flat list of integers, because SMPC sums numbers
  only and refuses text. A leaf's Σg and Σh and a site's training loss travel as integers too, because
  SMPC would round a float to its own decimal exponent. A value v travels as round(v · 2^exponent), so a summed value is off by at
  most n_sites · 2^-exponent / 2. The scale is binary, unlike FL-Net's decimal SMPC exponent, so that
  decoded values and their sums are exact in float64 (ADR 0006, update for #9).
"""

import base64
from typing import Protocol

import numpy as np

from fl_lightgbm.histogram import Histogram

Encoded = dict | list  # JSON-ready


class Encoding(Protocol):
    def encode_counts(self, counts: np.ndarray) -> Encoded:
        """A site's rows per grid cell, all features in one flat array."""

    def decode_counts(self, encoded: Encoded, size: int) -> np.ndarray:
        """The `size` grid counts; a cell that was not sent is 0."""

    def encode_histogram(self, hist: Histogram, most_freq_bins: np.ndarray) -> Encoded:
        """A site's histogram of one leaf; `most_freq_bins` are the flat indices of the bins the aggregator
        rebuilds from the leaf totals."""

    def decode_histogram(self, encoded: Encoded, size: int) -> Histogram:
        """The histogram with `size` bins; a bin that was not sent is 0."""

    def encode_sum(self, value: float) -> float | int:
        """One number a site sums over its rows, such as a leaf's Σg or its training loss."""

    def decode_sum(self, encoded: float | int) -> float:
        """The number, as the aggregator adds it to the other sites'."""


class _Dense:
    def encode_counts(self, counts: np.ndarray) -> Encoded:
        return counts.tolist()

    def decode_counts(self, encoded: Encoded, size: int) -> np.ndarray:
        return np.asarray(encoded, dtype=np.int64)

    def encode_histogram(self, hist: Histogram, most_freq_bins: np.ndarray) -> Encoded:
        return {"count": hist.count, "grad": hist.grad.tolist(), "hess": hist.hess.tolist()}

    def decode_histogram(self, encoded: Encoded, size: int) -> Histogram:
        assert isinstance(encoded, dict)
        grad, hess = (np.asarray(encoded[k], dtype=np.float64) for k in ("grad", "hess"))
        return Histogram(grad, hess, encoded["count"])

    def encode_sum(self, value: float) -> float | int:
        return value

    def decode_sum(self, encoded: float | int) -> float:
        return float(encoded)


class _Sparse:
    def encode_counts(self, counts: np.ndarray) -> Encoded:
        cells = np.flatnonzero(counts)
        return {"cells": _b64(cells, "<u4"), "counts": _b64(counts[cells], "<u4")}

    def decode_counts(self, encoded: Encoded, size: int) -> np.ndarray:
        assert isinstance(encoded, dict)
        counts = np.zeros(size, dtype=np.int64)
        counts[_unb64(encoded["cells"], "<u4")] = _unb64(encoded["counts"], "<u4")
        return counts

    def encode_histogram(self, hist: Histogram, most_freq_bins: np.ndarray) -> Encoded:
        # A bin is empty when both sums are 0; leaving it out adds the same 0 to the aggregator's sum.
        listed = (hist.grad != 0) | (hist.hess != 0)
        listed[most_freq_bins] = False
        bins = np.flatnonzero(listed)
        return {"count": hist.count, "bins": _b64(bins, "<u4"),
                "grad": _b64(hist.grad[bins], "<f8"), "hess": _b64(hist.hess[bins], "<f8")}

    def decode_histogram(self, encoded: Encoded, size: int) -> Histogram:
        assert isinstance(encoded, dict)
        bins = _unb64(encoded["bins"], "<u4")
        grad, hess = np.zeros(size), np.zeros(size)
        grad[bins] = _unb64(encoded["grad"], "<f8")
        hess[bins] = _unb64(encoded["hess"], "<f8")
        return Histogram(grad, hess, encoded["count"])

    def encode_sum(self, value: float) -> float | int:
        return value

    def decode_sum(self, encoded: float | int) -> float:
        return float(encoded)


class FixedPoint:
    def __init__(self, exponent: int):
        self.exponent = exponent
        self.scale = 2.0 ** exponent

    def encode_counts(self, counts: np.ndarray) -> Encoded:
        return counts.tolist()  # integers already

    def decode_counts(self, encoded: Encoded, size: int) -> np.ndarray:
        return np.asarray(encoded, dtype=np.int64)

    def encode_histogram(self, hist: Histogram, most_freq_bins: np.ndarray) -> Encoded:
        """[count, Σg of every bin, Σh of every bin], the sums as fixed-point integers."""
        return [hist.count, *self._to_fixed(hist.grad), *self._to_fixed(hist.hess)]

    def decode_histogram(self, encoded: Encoded, size: int) -> Histogram:
        values = np.asarray(encoded[1:], dtype=np.float64) / self.scale
        return Histogram(values[:size], values[size:], int(encoded[0]))

    def encode_sum(self, value: float) -> float | int:
        return self._to_fixed(np.array([value]))[0]

    def decode_sum(self, encoded: float | int) -> float:
        return encoded / self.scale

    def _to_fixed(self, values: np.ndarray) -> list[int]:
        scaled = np.rint(values * self.scale)
        # Beyond 2^53 float64 no longer holds every integer, and neither the rounding bound nor exact sums hold.
        if np.any(np.abs(scaled) >= 2.0 ** 53):
            raise ValueError(f"a sum of {np.max(np.abs(values)):g} is too large for fixed-point "
                             f"exponent {self.exponent}")
        return scaled.astype(np.int64).tolist()


def _b64(values: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def _unb64(text: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


DENSE: Encoding = _Dense()
SPARSE: Encoding = _Sparse()
