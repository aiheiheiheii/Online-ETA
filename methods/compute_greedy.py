"""Current computing-state greedy baseline."""

from __future__ import annotations

from methods.base import AssignmentPolicy, minimum_candidate
from simulator.decision import Assignment, DecisionContext, ReadyTask


class ComputeGreedy(AssignmentPolicy):
    """Select argmin_i (W_i(t) + g_{r,p}) / M_i(t).

    All decisions in a slot observe the same actual slot-start compute state.
    Neither communications nor temporary intra-slot load updates enter the
    selection score.  The simulator evaluates the selected node after decision
    timing to record execution cost.
    """

    name = "Compute-Greedy"

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        for task in ready_tasks:
            workload = context.requests[task.request_id].spec.gflops[task.subtask_index]
            options = [
                (
                    (context.compute_backlog_gflops[node] + workload)
                    / context.compute_rates_ghz[node],
                    node,
                    node,
                )
                for node in context.feasible_nodes(task)
            ]
            node = minimum_candidate(options)
            assignments.append(Assignment(task=task, node=node, candidate=None))
        return assignments
