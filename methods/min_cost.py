"""Current compute-plus-communication execution-cost baseline."""

from __future__ import annotations

from methods.base import AssignmentPolicy, minimum_candidate
from simulator.decision import Assignment, DecisionContext, ReadyTask


class MinCost(AssignmentPolicy):
    """Select argmin_i C_{r,p}^i(t) from actual slot-start W, U, M, and B.

    This baseline does not calculate urgency, dual prices, or intra-slot
    temporary competition updates.
    """

    name = "Min-Cost"

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        for task in ready_tasks:
            options = []
            for node in context.feasible_nodes(task):
                candidate = context.evaluate(task, node)
                options.append((candidate.total_cost_seconds, node, candidate))
            choice = minimum_candidate(options)
            assignments.append(Assignment(task=task, node=choice.node, candidate=choice))
        return assignments
