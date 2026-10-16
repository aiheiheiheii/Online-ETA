"""Effectiveness measures from recorded simulation outcomes.

The study's stated primary cost metric is total current-state execution cost
divided by the number of *assigned subtasks*.  Some prose calls it a
per-request metric; the equation and its explicit denominator take precedence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

import numpy as np


Row = Mapping[str, Any]


def _boolean(value: Any, field: str) -> bool:
    """Interpret booleans from in-memory records or CSV records."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _number(value: Any, field: str, *, allow_blank: bool = False) -> float | None:
    """Return one finite number, rejecting NaN and infinity."""

    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_blank:
            return None
        raise ValueError(f"{field} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _is_assigned(row: Row) -> bool:
    """Recognize an assigned subtask without counting unassigned tasks."""

    if "assignment_slot" in row:
        value = row["assignment_slot"]
        return value is not None and str(value).strip() != ""
    if "assigned_node" in row:
        value = row["assigned_node"]
        return value is not None and str(value).strip() != ""
    raise ValueError("subtask record lacks assignment_slot and assigned_node")


def summarize_effectiveness(
    request_rows: Iterable[Row], subtask_rows: Iterable[Row]
) -> dict[str, int | float | None]:
    """Compute the four effectiveness metrics and auditable auxiliary totals.

    ``average_execution_cost`` is ``J_C / assigned_subtasks``.  The optional
    ``average_execution_cost_per_request`` makes the source's inconsistent
    naming visible without changing its published equation.  Latency is
    undefined when no request finishes and is then returned as ``None``.
    """

    requests = list(request_rows)
    subtasks = list(subtask_rows)
    if not requests:
        raise ValueError("request_rows must include the fixed dataset requests")

    seen_ids: set[str] = set()
    completed_latencies: list[float] = []
    on_time_completed = 0
    for row in requests:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in seen_ids:
            raise ValueError(f"missing or duplicate request_id: {request_id!r}")
        seen_ids.add(request_id)
        finished = _boolean(row.get("finished", False), "finished")
        on_time = _boolean(row.get("on_time", False), "on_time")
        if on_time and not finished:
            raise ValueError(f"unfinished request {request_id} marked on time")
        if not finished:
            continue
        latency = _number(row.get("e2e_latency"), "e2e_latency")
        assert latency is not None
        if latency < 0:
            raise ValueError(f"request {request_id} has negative latency")
        deadline = _number(row.get("deadline"), "deadline", allow_blank=True)
        if deadline is not None:
            if deadline <= 0:
                raise ValueError(f"request {request_id} has nonpositive deadline")
            # Match the simulator's EPS tolerance for a slot-boundary equality.
            expected_on_time = latency <= deadline + 1e-9
            if on_time != expected_on_time:
                raise ValueError(f"request {request_id} has inconsistent on_time flag")
        completed_latencies.append(latency)
        on_time_completed += int(on_time)

    total_cost = 0.0
    assigned_subtasks = 0
    for row in subtasks:
        if not _is_assigned(row):
            continue
        cost = _number(row.get("execution_cost"), "execution_cost")
        assert cost is not None
        if cost < 0:
            raise ValueError("execution_cost must be nonnegative")
        total_cost += cost
        assigned_subtasks += 1

    if not isfinite(total_cost):
        raise ValueError("total_execution_cost overflowed")
    total_requests = len(requests)
    return {
        "on_time_completed": on_time_completed,
        "total_requests": total_requests,
        "finished_requests": len(completed_latencies),
        "completion_ratio": on_time_completed / total_requests,
        "assigned_subtasks": assigned_subtasks,
        "total_execution_cost": total_cost,
        "average_execution_cost": total_cost / assigned_subtasks if assigned_subtasks else None,
        "average_execution_cost_per_request": total_cost / total_requests,
        "average_e2e_latency": float(np.mean(completed_latencies)) if completed_latencies else None,
        "p95_e2e_latency": float(np.percentile(completed_latencies, 95)) if completed_latencies else None,
    }


def summarize(
    request_rows: Iterable[Row], subtask_rows: Iterable[Row], slot_rows: Iterable[Row]
) -> dict[str, int | float | None]:
    """Combine effectiveness and efficiency metrics for one replicate."""

    from .efficiency import summarize_efficiency

    return {**summarize_effectiveness(request_rows, subtask_rows), **summarize_efficiency(slot_rows)}
