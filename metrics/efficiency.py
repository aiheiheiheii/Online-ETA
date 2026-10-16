"""Wall-clock assignment decision-time measures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

import numpy as np


def summarize_efficiency(slot_rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Return mean and P95 decision time across all recorded slots, in seconds."""

    times: list[float] = []
    seen_slots: set[int] = set()
    for row in slot_rows:
        if "slot" in row:
            slot = int(row["slot"])
            if slot in seen_slots:
                raise ValueError(f"duplicate slot record: {slot}")
            seen_slots.add(slot)
        try:
            decision_time = float(row["decision_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("slot record requires numeric decision_time") from exc
        if not isfinite(decision_time) or decision_time < 0:
            raise ValueError("decision_time must be finite and nonnegative")
        times.append(decision_time)
    if not times:
        raise ValueError("slot_rows must contain at least one simulated slot")
    return {
        "mean_decision_time": float(np.mean(times)),
        "p95_decision_time": float(np.percentile(times, 95)),
    }
