"""Urgency-ordered current execution-cost greedy baseline."""

from __future__ import annotations

from methods.base import AssignmentPolicy, minimum_candidate
from simulator.decision import Assignment, DecisionContext, ReadyTask


class UrgencyGreedy(AssignmentPolicy):
    """Sort ready tasks by descending eta, then select argmin current C.

    Unlike Proposed, this baseline has no dual-price terms and makes no
    intra-slot temporary resource update.  Its definition orders all ready
    subtasks; it does not add Proposed's unexpired-request filter.
    """

    name = "Urgency-Greedy"

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        ordered = [
            (context.urgency(task.request_id)[1], task)
            for task in ready_tasks
        ]
        ordered.sort(
            key=lambda item: (-item[0], item[1].request_id, item[1].subtask_index)
        )
        assignments: list[Assignment] = []
        for eta, task in ordered:
            options = []
            for node in context.feasible_nodes(task):
                candidate = context.evaluate(task, node)
                options.append((candidate.total_cost_seconds, node, candidate))
            choice = minimum_candidate(options)
            assignments.append(
                Assignment(task=task, node=choice.node, candidate=choice, urgency=eta)
            )
        return assignments
