import json
import math

import numpy as np
import pytest

from fl_lightgbm.encoding import DENSE, SPARSE, FixedPoint
from fl_lightgbm.histogram import Histogram, feature_offsets, fix_histogram, leaf_histogram, most_freq_bin_positions

NUM_BINS = np.array([4, 7, 1, 5, 6])
# Feature 0 skips bin 0 (LightGBM's `offset` case), the others a bin inside the feature.
MOST_FREQ_BINS = np.array([0, 3, 0, 4, 1])


def mostly_zero_site(rng, n):
    """Binned rows that fall mostly in each feature's most frequent bin, so many other bins stay empty."""
    binned = np.column_stack([np.where(rng.random(n) < 0.8, mfb, rng.integers(0, nb, size=n))
                              for nb, mfb in zip(NUM_BINS, MOST_FREQ_BINS)])
    gradients = rng.normal(size=n).astype(np.float32)
    hessians = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    return binned, gradients, hessians


def site_histograms(seed, sizes):
    rng = np.random.default_rng(seed)
    hists = []
    for n in sizes:
        binned, g, h = mostly_zero_site(rng, n)
        hists.append((leaf_histogram(binned, NUM_BINS, g, h, np.arange(n)), float(g.astype(np.float64).sum()),
                      float(h.astype(np.float64).sum())))
    return hists


SIZE = int(NUM_BINS.sum())
MOST_FREQ = most_freq_bin_positions(NUM_BINS, MOST_FREQ_BINS)


def through_json(encoded):
    return json.loads(json.dumps(encoded))


def test_dense_histogram_round_trips_exactly():
    (hist, _, _), = site_histograms(1, [40])

    decoded = DENSE.decode_histogram(through_json(DENSE.encode_histogram(hist, MOST_FREQ)), SIZE)

    np.testing.assert_array_equal(decoded.grad, hist.grad)
    np.testing.assert_array_equal(decoded.hess, hist.hess)
    assert decoded.count == hist.count


def test_sparse_histogram_round_trips_exactly_except_the_most_frequent_bins():
    (hist, _, _), = site_histograms(2, [12])
    assert np.sum(hist.hess == 0) > len(MOST_FREQ)  # some bins are empty besides the most frequent ones

    decoded = SPARSE.decode_histogram(through_json(SPARSE.encode_histogram(hist, MOST_FREQ)), SIZE)

    kept = np.ones(SIZE, dtype=bool)
    kept[MOST_FREQ] = False
    np.testing.assert_array_equal(decoded.grad[kept], hist.grad[kept])
    np.testing.assert_array_equal(decoded.hess[kept], hist.hess[kept])
    assert not decoded.grad[MOST_FREQ].any() and not decoded.hess[MOST_FREQ].any()
    assert decoded.count == hist.count


@pytest.mark.parametrize("encoding", [DENSE, SPARSE])
def test_grid_counts_round_trip_exactly(encoding):
    counts = np.random.default_rng(3).integers(0, 4, size=500) * (np.arange(500) % 7 == 0)

    decoded = encoding.decode_counts(through_json(encoding.encode_counts(counts)), len(counts))

    np.testing.assert_array_equal(decoded, counts)


def summed(encoding, hists):
    decoded = [encoding.decode_histogram(through_json(encoding.encode_histogram(h, MOST_FREQ)), SIZE)
               for h, _, _ in hists]
    return np.sum([d.grad for d in decoded], axis=0), np.sum([d.hess for d in decoded], axis=0), \
        sum(d.count for d in decoded)


def test_summed_sparse_payloads_equal_summed_dense_ones_once_the_aggregator_rebuilds_the_most_frequent_bins():
    hists = site_histograms(4, [40, 90, 7])
    sum_grad = sum(g for _, g, _ in hists)
    sum_hess = sum(h for _, _, h in hists)

    fixed = {}
    for name, encoding in [("dense", DENSE), ("sparse", SPARSE)]:
        grad, hess, count = summed(encoding, hists)
        assert count == 137
        fixed[name] = fix_histogram(Histogram(grad, hess, count), sum_grad, sum_hess, NUM_BINS, MOST_FREQ_BINS)

    # FixHistogram rebuilds every most frequent bin but bin 0, which split finding never reads.
    read = np.ones(SIZE, dtype=bool)
    read[feature_offsets(NUM_BINS)[:-1][MOST_FREQ_BINS == 0]] = False
    np.testing.assert_array_equal(fixed["sparse"].grad[read], fixed["dense"].grad[read])
    np.testing.assert_array_equal(fixed["sparse"].hess[read], fixed["dense"].hess[read])


@pytest.mark.parametrize("exponent", [4, 16, 26, 36])
@pytest.mark.parametrize("num_sites", [2, 7, 20])
def test_fixed_point_error_per_summed_value_stays_within_half_a_unit_per_site(exponent, num_sites):
    hists = site_histograms(exponent * num_sites, [30] * num_sites)
    encoding = FixedPoint(exponent)

    grad, hess, count = summed(encoding, hists)

    bound = num_sites * 0.5 * 2.0 ** -exponent
    exact_grad = [math.fsum(column) for column in zip(*(h.grad for h, _, _ in hists))]
    exact_hess = [math.fsum(column) for column in zip(*(h.hess for h, _, _ in hists))]
    assert np.max(np.abs(grad - exact_grad)) <= bound
    assert np.max(np.abs(hess - exact_hess)) <= bound
    assert count == 30 * num_sites
    if exponent == 4:  # the values really are rounded
        assert np.max(np.abs(grad - exact_grad)) > 2.0 ** -(exponent + 4)


def test_a_sum_too_large_for_the_fixed_point_exponent_is_refused_rather_than_rounded_wrongly():
    hist = Histogram(np.array([1.0, 2.0 ** 27]), np.array([1.0, 1.0]), 3)  # 2^27 · 2^26 = 2^53

    with pytest.raises(ValueError, match="exponent 26"):
        FixedPoint(26).encode_histogram(hist, np.array([], dtype=np.int64))
    FixedPoint(25).encode_histogram(hist, np.array([], dtype=np.int64))


@pytest.mark.parametrize("encoding", [DENSE, SPARSE])
def test_a_summed_number_round_trips_exactly(encoding):
    value = float(np.float32(-0.1)) * 1234.5

    assert encoding.decode_sum(through_json(encoding.encode_sum(value))) == value


@pytest.mark.parametrize("exponent", [4, 26])
def test_a_fixed_point_summed_number_is_an_integer_within_half_a_unit(exponent):
    encoding = FixedPoint(exponent)
    value = float(np.float32(-0.1)) * 1234.5

    encoded = through_json(encoding.encode_sum(value))

    assert isinstance(encoded, int)
    assert abs(encoding.decode_sum(encoded) - value) <= 0.5 * 2.0 ** -exponent


def test_a_summed_number_too_large_for_the_fixed_point_exponent_is_refused():
    with pytest.raises(ValueError, match="exponent 26"):
        FixedPoint(26).encode_sum(2.0 ** 27)
    FixedPoint(25).encode_sum(2.0 ** 27)
