"""Independent, inspectable request deadline generation.

The reference-latency formula here is an engineering default because the
research notes do not fix a unique deadline-generation rule.  GFLOPs times
cycles/FLOP are Gcycles; dividing by GHz gives seconds.  Activations are MB
and dividing by MB/s likewise gives seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence


class RequestWorkload(Protocol):
    """Minimum request fields needed for a reference deadline."""

    request_id: int
    gflops: tuple[float, ...]
    activation_mb: tuple[float, ...]


class DeadlineSettings(Protocol):
    """Configuration fields consumed by this module."""

    deadline_model: str
    theta_deadline: float
    cycles_per_flop: float
    mean_compute_ghz: float
    mean_bandwidth_mb_s: float
    deadline_reference_compute_ghz: float | None
    deadline_reference_bandwidth_mb_s: float | None
    deadline_min_seconds: float


@dataclass(frozen=True, slots=True)
class DeadlineRecord:
    """Auditable deadline calculation for one original dataset request."""

    request_id: int
    raw_reference_seconds: float
    base_seconds: float
    theta: float
    final_seconds: float

    @property
    def deadline_seconds(self) -> float:
        """Alias for simulator code that refers to the final deadline."""

        return self.final_seconds


def generate_deadlines(
    requests: Sequence[RequestWorkload], config: DeadlineSettings
) -> dict[int, DeadlineRecord]:
    """Build one deterministic reference-latency deadline per request.

    ``raw_reference_seconds = sum(g) * cycles_per_flop / M_ref
    + sum(activation_mb) / B_ref``.  A configurable minimum is applied to
    the *base* latency before multiplying by ``theta_deadline``.  Both the raw
    and floored values are kept so the assumption is visible in detailed CSV.
    """

    if config.deadline_model != "reference_latency_scaled":
        raise ValueError(f"Unsupported deadline_model: {config.deadline_model}")
    compute_rate = (
        config.mean_compute_ghz
        if config.deadline_reference_compute_ghz is None
        else config.deadline_reference_compute_ghz
    )
    bandwidth = (
        config.mean_bandwidth_mb_s
        if config.deadline_reference_bandwidth_mb_s is None
        else config.deadline_reference_bandwidth_mb_s
    )
    if not all(
        isfinite(value) and value > 0
        for value in (compute_rate, bandwidth, config.cycles_per_flop, config.theta_deadline)
    ):
        raise ValueError("Reference rates, cycles_per_flop, and theta must be positive")
    if not isfinite(config.deadline_min_seconds) or config.deadline_min_seconds < 0:
        raise ValueError("deadline_min_seconds must be finite and nonnegative")

    records: dict[int, DeadlineRecord] = {}
    for request in requests:
        if request.request_id in records:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        gflops = tuple(request.gflops)
        activation = tuple(request.activation_mb)
        if not gflops or len(activation) != len(gflops) - 1:
            raise ValueError(f"Invalid chain lengths for request {request.request_id}")
        if any(not isfinite(value) or value <= 0 for value in gflops):
            raise ValueError(f"Invalid GFLOPs in request {request.request_id}")
        if any(not isfinite(value) or value < 0 for value in activation):
            raise ValueError(f"Invalid activation MB in request {request.request_id}")

        raw_reference = (
            sum(gflops) * config.cycles_per_flop / compute_rate
            + sum(activation) / bandwidth
        )
        base = max(raw_reference, config.deadline_min_seconds)
        final = config.theta_deadline * base
        if not all(isfinite(value) and value > 0 for value in (raw_reference, base, final)):
            raise ValueError(f"Invalid generated deadline for request {request.request_id}")
        records[request.request_id] = DeadlineRecord(
            request_id=request.request_id,
            raw_reference_seconds=raw_reference,
            base_seconds=base,
            theta=config.theta_deadline,
            final_seconds=final,
        )
    return records
