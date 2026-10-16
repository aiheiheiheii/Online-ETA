"""Pre-generate reproducible, shared per-slot MEC service rates.

Compute rates are GHz (Gcycles/s); communication rates are MB/s.  Rates stay
constant within a slot.  The simulator converts a compute rate to GFLOPs of
service with ``rate_ghz * slot_seconds / cycles_per_flop``.

Static node/link heterogeneity and temporal variation are configurable
engineering assumptions.  A fresh trace is generated once per experiment
replicate and passed unchanged to all five assignment methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p, sqrt
from random import Random
from types import MappingProxyType
from typing import Mapping, Protocol

from simulator.network import LinkId, Network


class ResourceSettings(Protocol):
    """Configuration fields consumed by the resource generator."""

    num_slots: int
    mean_compute_ghz: float
    mean_bandwidth_mb_s: float
    compute_heterogeneity_cv: float
    bandwidth_heterogeneity_cv: float
    compute_temporal_cv: float
    bandwidth_temporal_cv: float
    min_resource_fraction: float


@dataclass(frozen=True, slots=True)
class ResourceTrace:
    """Immutable rates indexed by slot, then node or directed link ID."""

    compute_rates_ghz: tuple[tuple[float, ...], ...]
    link_rates_mb_s: tuple[Mapping[LinkId, float], ...]
    seed: int

    @property
    def compute_rates(self) -> tuple[tuple[float, ...], ...]:
        """Short alias for compute rates in GHz."""

        return self.compute_rates_ghz

    @property
    def link_rates(self) -> tuple[Mapping[LinkId, float], ...]:
        """Short alias for link rates in MB/s."""

        return self.link_rates_mb_s

    @property
    def num_slots(self) -> int:
        return len(self.compute_rates_ghz)


def _unit_mean_lognormal(rng: Random, coefficient_of_variation: float) -> float:
    """A positive factor whose population mean is one and CV is configured."""

    if coefficient_of_variation == 0:
        return 1.0
    sigma_sq = log1p(coefficient_of_variation**2)
    sigma = sqrt(sigma_sq)
    return exp(rng.gauss(-sigma_sq / 2, sigma))


def _normalized_static_factors(
    rng: Random, count: int, coefficient_of_variation: float
) -> tuple[float, ...]:
    """Heterogeneous factors with exact sample mean one for this topology."""

    if count == 0:
        return ()
    raw = tuple(_unit_mean_lognormal(rng, coefficient_of_variation) for _ in range(count))
    observed_mean = sum(raw) / count
    return tuple(value / observed_mean for value in raw)


def generate_resource_trace(
    config: ResourceSettings, network: Network, seed: int
) -> ResourceTrace:
    """Generate every M_i(t) and B_l(t) before running any method.

    The static node/link factors have sample mean one.  Independent positive
    temporal factors have population mean one.  A configurable lower bound
    prevents zero service capacity after a rare low random draw.
    """

    if config.num_slots <= 0:
        raise ValueError("num_slots must be positive")
    if not all(
        isfinite(value) and value > 0
        for value in (config.mean_compute_ghz, config.mean_bandwidth_mb_s)
    ):
        raise ValueError("Mean resource rates must be finite and positive")
    cvs = (
        config.compute_heterogeneity_cv,
        config.bandwidth_heterogeneity_cv,
        config.compute_temporal_cv,
        config.bandwidth_temporal_cv,
    )
    if any(not isfinite(value) or value < 0 for value in cvs):
        raise ValueError("Resource coefficients of variation must be nonnegative")
    if not isfinite(config.min_resource_fraction) or not 0 < config.min_resource_fraction <= 1:
        raise ValueError("min_resource_fraction must lie in (0, 1]")

    rng = Random(seed)
    nodes = network.nodes
    links = network.links
    if nodes != tuple(range(len(nodes))):
        raise ValueError("Network node IDs must be dense zero-based indices")
    compute_static = _normalized_static_factors(
        rng, len(nodes), config.compute_heterogeneity_cv
    )
    bandwidth_static = _normalized_static_factors(
        rng, len(links), config.bandwidth_heterogeneity_cv
    )

    compute_rows: list[tuple[float, ...]] = []
    link_rows: list[Mapping[LinkId, float]] = []
    for _slot in range(config.num_slots):
        compute_row = tuple(
            config.mean_compute_ghz
            * max(
                config.min_resource_fraction,
                compute_static[node]
                * _unit_mean_lognormal(rng, config.compute_temporal_cv),
            )
            for node in nodes
        )
        link_row = {
            link_id: config.mean_bandwidth_mb_s
            * max(
                config.min_resource_fraction,
                bandwidth_static[index]
                * _unit_mean_lognormal(rng, config.bandwidth_temporal_cv),
            )
            for index, link_id in enumerate(links)
        }
        if any(not isfinite(rate) or rate <= 0 for rate in compute_row):
            raise AssertionError("Generated an invalid compute rate")
        if any(not isfinite(rate) or rate <= 0 for rate in link_row.values()):
            raise AssertionError("Generated an invalid link rate")
        compute_rows.append(compute_row)
        link_rows.append(MappingProxyType(link_row))

    return ResourceTrace(tuple(compute_rows), tuple(link_rows), seed)
