"""Random feasible-node baseline with a policy-local reproducible RNG."""

from __future__ import annotations

from random import Random

from methods.base import AssignmentPolicy
from simulator.decision import Assignment, DecisionContext, ReadyTask


class RandomMethod(AssignmentPolicy):
    """Choose uniformly among topology-feasible nodes for each ready subtask.

    The choice uses no workload, rate, backlog, cost, or deadline information.
    The simulator evaluates the selected node after decision timing to record
    the same execution cost metric as the other policies.
    """

    name = "Random"

    def __init__(self, seed: int) -> None:
        self._rng = Random(seed)

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        for task in ready_tasks:
            feasible = context.feasible_nodes(task)
            if not feasible:
                raise ValueError(f"No feasible node for {task}")
            node = self._rng.choice(feasible)
            assignments.append(Assignment(task=task, node=node, candidate=None))
        return assignments
