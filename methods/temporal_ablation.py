"""Single-component ablations for the Performance over Time experiment."""
from __future__ import annotations

from methods.base import AssignmentPolicy, minimum_candidate
from methods.proposed import gamma_for_candidate
from simulator.decision import Assignment, DecisionContext, ReadyTask


class _TemporalAblation(AssignmentPolicy):
    """The simulator supplies the common unexpired ready set."""
    use_urgency: bool
    use_prices: bool

    def select_assignments(self, context: DecisionContext,
                           ready_tasks: list[ReadyTask]) -> list[Assignment]:
        eta_by_request = {}
        if self.use_urgency:
            eta_by_request = {
                rid: context.urgency(rid)[1]
                for rid, request in context.requests.items()
                if request.arrival_slot <= context.slot and request.finish_slot is None
            }
            ordered = sorted(ready_tasks, key=lambda task: (
                -eta_by_request[task.request_id], task.request_id, task.subtask_index))
        else:
            def fifo_key(task: ReadyTask) -> tuple[int, int, int]:
                ready_slot = context.requests[task.request_id].subtasks[task.subtask_index].ready_slot
                if ready_slot is None:
                    raise AssertionError("FIFO requires the first ready timestamp")
                return ready_slot, task.request_id, task.subtask_index
            ordered = sorted(ready_tasks, key=fifo_key)

        temporary_compute = list(context.compute_backlog_gflops)
        temporary_link = dict(context.link_backlog_mb)
        assignments = []
        for task in ordered:
            request = context.requests[task.request_id]
            options = []
            for node in context.feasible_nodes(task):
                candidate = context.evaluate(task, node)
                if self.use_prices:
                    gamma, lambda_i, communication_dual = gamma_for_candidate(
                        context, task, candidate, temporary_compute, temporary_link)
                    selection_cost = gamma
                else:
                    lambda_i = communication_dual = 0.0
                    gamma = candidate.total_cost_seconds / request.deadline_seconds
                    selection_cost = candidate.total_cost_seconds
                options.append((selection_cost, node, Assignment(
                    task=task, node=node, candidate=candidate,
                    urgency=eta_by_request.get(task.request_id),
                    lambda_price=lambda_i, mu_price_sum=communication_dual, gamma=gamma)))
            choice = minimum_candidate(options)
            assignments.append(choice)
            # Retained in both ablations; only price terms use these copies.
            temporary_compute[choice.node] += request.spec.gflops[task.subtask_index]
            for predecessor, path in choice.candidate.paths.items():
                amount = request.activation_mb(predecessor, task.subtask_index)
                for link in path:
                    temporary_link[link] += amount
        return assignments


class WithoutUrgency(_TemporalAblation):
    name = "w/o Urgency"
    use_urgency = False
    use_prices = True


class WithoutResourcePrice(_TemporalAblation):
    name = "w/o Resource Price"
    use_urgency = True
    use_prices = False
