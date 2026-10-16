"""FIFO, work-conserving queues in GFLOPs (compute) or MB (communication)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable


EPS = 1e-9


@dataclass
class QueueItem:
    key: Hashable
    remaining: float


class WorkQueue:
    """A single-server fluid-work queue with FIFO service within each slot."""

    def __init__(self) -> None:
        self._items: deque[QueueItem] = deque()
        self.backlog = 0.0

    def enqueue(self, key: Hashable, amount: float) -> None:
        if not 0.0 < amount < float("inf"):
            raise ValueError(f"Queue workload must be finite and positive: {amount}")
        self._items.append(QueueItem(key, float(amount)))
        self.backlog += amount

    def serve(self, capacity: float) -> tuple[list[Hashable], dict[Hashable, float]]:
        """Apply at most capacity service and return completed keys and service amounts."""
        if not 0.0 <= capacity < float("inf"):
            raise ValueError(f"Invalid slot service capacity: {capacity}")
        before = self.backlog
        budget = capacity
        completed: list[Hashable] = []
        served: dict[Hashable, float] = {}
        while self._items and budget > EPS:
            head = self._items[0]
            amount = min(head.remaining, budget)
            head.remaining -= amount
            self.backlog -= amount
            budget -= amount
            served[head.key] = served.get(head.key, 0.0) + amount
            if head.remaining <= EPS:
                completed.append(head.key)
                self._items.popleft()
        if self.backlog < EPS:
            self.backlog = 0.0
        expected = max(0.0, before - capacity)
        if abs(self.backlog - expected) > 1e-6:
            raise AssertionError("FIFO queue violated work-conserving workload update")
        return completed, served

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> tuple[Hashable, ...]:
        return tuple(item.key for item in self._items)
