import numpy as np

from fl_lightgbm.histogram import leaf_histogram

NUM_BINS = np.array([4, 7, 1, 5])


def random_site(rng, n):
    binned = np.column_stack([rng.integers(0, nb, size=n) for nb in NUM_BINS])
    gradients = rng.normal(size=n).astype(np.float32)
    hessians = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    return binned, gradients, hessians


def direct_sum(binned, values, rows):
    """Per feature and bin, the sum of `values` over the rows of the leaf, one bin at a time."""
    out = []
    for f, nb in enumerate(NUM_BINS):
        for b in range(nb):
            in_bin = rows[binned[rows, f] == b]
            out.append(np.sum(values[in_bin].astype(np.float64)))
    return np.array(out)


def test_site_histogram_equals_direct_sum_over_the_leaf_rows():
    rng = np.random.default_rng(1)
    binned, g, h = random_site(rng, 200)
    rows = np.flatnonzero(rng.random(200) < 0.4)

    hist = leaf_histogram(binned, NUM_BINS, g, h, rows)

    np.testing.assert_allclose(hist.grad, direct_sum(binned, g, rows), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hist.hess, direct_sum(binned, h, rows), rtol=1e-12, atol=1e-12)
    assert hist.count == len(rows)


def test_sum_over_sites_equals_pooled_histogram():
    rng = np.random.default_rng(2)
    sites = [random_site(rng, n) for n in (50, 120, 7)]
    site_hists = [leaf_histogram(b, NUM_BINS, g, h, np.arange(len(g))) for b, g, h in sites]

    pooled = leaf_histogram(
        np.vstack([s[0] for s in sites]),
        NUM_BINS,
        np.concatenate([s[1] for s in sites]),
        np.concatenate([s[2] for s in sites]),
        np.arange(177),
    )

    np.testing.assert_allclose(sum(s.grad for s in site_hists), pooled.grad, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sum(s.hess for s in site_hists), pooled.hess, rtol=1e-12, atol=1e-12)
    assert sum(s.count for s in site_hists) == pooled.count == 177
