"""In-memory simulation: n site logics and one aggregator logic, run for a fixed number of rounds."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fl_lightgbm.aggregator import Aggregator
from fl_lightgbm.binning import BinMapper
from fl_lightgbm.encoding import SPARSE, Encoding, FixedPoint
from fl_lightgbm.central import write_forced_bins
from fl_lightgbm.params import Params
from fl_lightgbm.rounds import RoundCountError
from fl_lightgbm.site import Site


@dataclass
class SimulationResult:
    model: str  # LightGBM text model
    report: str  # JSON training report
    forced_bins: str  # the agreed bin edges as LightGBM's forced-bins file (JSON)
    agreed_edges: list[BinMapper]
    rounds: int
    padding_rounds: int


def run(sites: list[Site], aggregator: Aggregator, rounds: int, secure_aggregation: bool = False) -> None:
    """Run exactly `rounds` rounds, as FL-Net does; each site and the aggregator must use up its budget exactly.
    With `secure_aggregation`, the aggregator receives from setup round 2 on only the element-wise sum of
    the sites' payloads, as FL-Net's SMPC delivers it."""
    encodings = [aggregator.encoding, *(site.encoding for site in sites)]
    if secure_aggregation and not all(isinstance(e, FixedPoint) for e in encodings):
        raise ValueError("secure aggregation needs the FixedPoint payload encoding, because SMPC sums numbers only")
    reply = None
    for round_number in range(1, rounds + 1):
        payloads = [site.payload(reply) for site in sites]
        if secure_aggregation and round_number > 1:
            payloads = [_sum_payloads(payloads, round_number)]
        reply = aggregator.reply(payloads)
    named: list[tuple[str, Site | Aggregator]] = [(f"site {i}", s) for i, s in enumerate(sites)]
    for name, logic in [*named, ("aggregator", aggregator)]:
        if logic.rounds_left:
            raise RoundCountError(f"{name} still expects {logic.rounds_left} round(s) after {rounds}")


def _sum_payloads(values: list, round_number: int, path: str = "payload") -> Any:
    """The sites' payloads summed over their JSON structure: numbers at the same position are added, and
    lists and dicts must have the same shape at every site. Anything else is what SMPC would choke on."""
    first = values[0]
    if all(isinstance(v, dict) and v.keys() == first.keys() for v in values):
        return {k: _sum_payloads([v[k] for v in values], round_number, f"{path}.{k}") for k in first}
    if all(isinstance(v, list) and len(v) == len(first) for v in values):
        return [_sum_payloads(list(column), round_number, f"{path}[{i}]") for i, column in enumerate(zip(*values))]
    if all(_is_number(v) for v in values):
        return sum(values)
    odd = [v for v in values if not isinstance(v, (dict, list)) and not _is_number(v)]
    problem = f"holds {odd[0]!r}, not a number" if odd else "differs in shape between sites"
    raise ValueError(f"round {round_number}: secure aggregation cannot sum {path}, which {problem}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def simulate(params: dict, tables: list[tuple[np.ndarray, np.ndarray]],
             bounds: list[tuple[float, float]] | None = None, encoding: Encoding = SPARSE,
             categorical: Sequence[int] = (), secure_aggregation: bool = False) -> SimulationResult:
    """Train on one (X, y) table per site; the round count comes from the parameters. `bounds`, if
    given, are per-feature (lo, hi) the sites use in place of sharing their min/max; `encoding` is how
    the sites send grid counts and histograms; `categorical` are the indices of the categorical features.
    `secure_aggregation` hands the aggregator only the sum of the sites' payloads from setup round 2 on;
    it needs a `FixedPoint` encoding."""
    p = Params.from_dict(params)
    names = [f"Column_{i}" for i in range(tables[0][0].shape[1])]
    sites = [Site(X, y, p, names, bounds, encoding, categorical) for X, y in tables]
    aggregator = Aggregator(p, bounds, encoding)
    run(sites, aggregator, p.round_budget, secure_aggregation)
    return SimulationResult(aggregator.model(), aggregator.report(), write_forced_bins(aggregator.bin_mappers),
                            aggregator.bin_mappers, p.round_budget, aggregator.padding_rounds)


# Site splitters: one dataset as the (X, y) tables of simulated sites. Every row lands at exactly one site,
# and each site keeps its rows in the dataset's order.

def split_evenly(X: np.ndarray, y: np.ndarray, num_sites: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rows shuffled and spread evenly over the sites."""
    return _tables(X, y, _even_rows(len(y), num_sites, seed))


def split_skewed(X: np.ndarray, y: np.ndarray, num_sites: int, large_share: float = 0.8,
                 seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """One large site with `large_share` of the rows, first, and several tiny ones with the rest. The tiny
    sites have labels from low to high: with the rows ranked by label, each tiny site draws its rows from
    its own band of ranks, the first from the lowest. The large site keeps about the overall balance."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    order = order[np.argsort(y[order], kind="stable")]  # by label, ties in random order
    tiny_size = round(len(y) * (1 - large_share) / (num_sites - 1))
    tiny = [rng.choice(band, tiny_size, replace=False) for band in np.array_split(order, num_sites - 1)]
    large = np.setdiff1d(np.arange(len(y)), np.concatenate(tiny))
    return _tables(X, y, [large, *tiny])


def split_by_column(X: np.ndarray, y: np.ndarray, column: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """One site per value of `column`, which holds one value per row (a feature, or for example each
    row's cohort name), in the order of the sorted values."""
    values, site_of_row = np.unique(column, return_inverse=True)
    return _tables(X, y, [np.flatnonzero(site_of_row == i) for i in range(len(values))])


def split_missing_category(X: np.ndarray, y: np.ndarray, num_sites: int, feature: int, category: float,
                           seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """An even split in which the first site lacks one category of a categorical feature entirely: its
    rows of that category go to the other sites."""
    rows = _even_rows(len(y), num_sites, seed)
    has_category = X[rows[0], feature] == category
    moved = rows[0][has_category]
    rows[0] = rows[0][~has_category]
    for i, extra in enumerate(np.array_split(moved, num_sites - 1), start=1):
        rows[i] = np.concatenate([rows[i], extra])
    return _tables(X, y, rows)


def _even_rows(n: int, num_sites: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(n), num_sites)


def _tables(X: np.ndarray, y: np.ndarray, rows: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    rows = [np.sort(r) for r in rows]
    return [(X[r], y[r]) for r in rows]
