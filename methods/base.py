"""Common policy interface and deterministic node tie breaking."""

from __future__ import annotations

from abc import ABC, abstractmethod

from simulator.decision import Assignment, DecisionContext, ReadyTask


class AssignmentPolicy(ABC):
    name: str

    @abstractmethod
    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        """Select at most one feasible node for each considered ready task."""


def minimum_candidate(candidates: list[tuple[float, int, object]]) -> object:
    """Use lower node ID for exact ties; never depend on hash iteration order."""
    if not candidates:
        raise ValueError("No feasible candidate node")
    return min(candidates, key=lambda entry: (entry[0], entry[1]))[2]
