"""Assign slots to the requests already present in a fixed Dataset_N file.

``fixed_total_uniform`` is a replaceable engineering default, not a published
arrival distribution.  This module never creates or removes a DNN request.
"""

from __future__ import annotations

from math import ceil, isfinite
from random import Random
from typing import Protocol, Sequence


class RequestWithId(Protocol):
    """Minimum request information needed by the arrival scheduler."""

    request_id: int


class ArrivalSettings(Protocol):
    """Configuration fields consumed by this module."""

    arrival_model: str
    arrival_start_slot: int
    arrival_window_fraction: float
    arrival_end_slot: int | None


def assign_arrival_slots(
    requests: Sequence[RequestWithId],
    num_slots: int,
    seed: int,
    config: ArrivalSettings,
) -> dict[int, int]:
    """Return one reproducible arrival slot for each input request ID.

    Uniform draws are independent *conditional on the fixed input request
    count*.  There is no arrival-count draw and no model-type resampling.
    Sorting IDs makes the mapping insensitive to input row ordering.
    """

    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    if not isfinite(config.arrival_window_fraction) or not 0 < config.arrival_window_fraction <= 1:
        raise ValueError("arrival_window_fraction must lie in (0, 1]")
    first = config.arrival_start_slot
    last = (
        ceil(num_slots * config.arrival_window_fraction) - 1
        if config.arrival_end_slot is None
        else config.arrival_end_slot
    )
    if not 0 <= first <= last < num_slots:
        raise ValueError("arrival interval must lie inside the study period")
    if config.arrival_model != "fixed_total_uniform":
        raise ValueError(f"Unsupported ARRIVAL_MODEL: {config.arrival_model}")

    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate request_id in fixed dataset")

    rng = Random(seed)
    schedule = {request_id: rng.randint(first, last) for request_id in sorted(request_ids)}
    if len(schedule) != len(requests) or any(
        not 0 <= slot < num_slots for slot in schedule.values()
    ):
        raise AssertionError("Arrival scheduler lost, duplicated, or misplaced a request")
    return schedule
