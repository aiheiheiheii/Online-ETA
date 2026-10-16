"""The paper's urgency ordered primal-dual online assignment, Eq. (14)-(27)."""

from __future__ import annotations

from simulator.decision import Assignment, Candidate, DecisionContext, LinkId, ReadyTask
from methods.base import AssignmentPolicy, minimum_candidate


def compute_lambda(workload_gflops: float, rate_ghz: float, delta: float, chi: float) -> float:
    """Eq. (19): temporary compute backlog / one-slot compute service."""
    return workload_gflops / (rate_ghz * delta / chi)


def compute_mu(backlog_mb: float, bandwidth_mb_s: float, delta: float) -> float:
    """Eq. (20): temporary link backlog / one-slot link service."""
    return backlog_mb / (bandwidth_mb_s * delta)


def gamma_for_candidate(
    context: DecisionContext,
    task: ReadyTask,
    candidate: Candidate,
    temporary_compute: list[float],
    temporary_link: dict[LinkId, float],
) -> tuple[float, float, float]:
    """Eq. (21), with actual-state C and temporary-state dual prices.

    Returns (Gamma, lambda_i, communication dual term).  The routes in
    candidate.paths were chosen using actual U/B by Eq. (10)-(11).
    """
    request = context.requests[task.request_id]
    node = candidate.node
    delta = context.config.slot_seconds
    chi = context.config.cycles_per_flop
    rate = context.compute_rates_ghz[node]
    g = request.spec.gflops[task.subtask_index]
    lambda_i = compute_lambda(temporary_compute[node], rate, delta, chi)
    compute_dual = lambda_i * g / (rate * delta / chi)
    communication_dual = 0.0
    for predecessor, path in candidate.paths.items():
        amount = request.activation_mb(predecessor, task.subtask_index)
        for link in path:
            bandwidth = context.link_rates_mb_s[link]
            mu = compute_mu(temporary_link[link], bandwidth, delta)
            communication_dual += mu * amount / (bandwidth * delta)
    gamma = (
        candidate.total_cost_seconds / request.deadline_seconds
        + compute_dual
        + communication_dual
    )
    return gamma, lambda_i, communication_dual


class Proposed(AssignmentPolicy):
    name = "Proposed"

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]:
        """Sort by urgency, price all feasible nodes, then update tilde loads."""
        active_urgency = {
            request_id: context.urgency(request_id)
            for request_id, request in context.requests.items()
            if request.arrival_slot <= context.slot and request.finish_slot is None
        }
        ordered: list[tuple[float, ReadyTask]] = []
        for task in ready_tasks:
            tau, eta = active_urgency[task.request_id]
            if tau >= 0:
                ordered.append((eta, task))
        ordered.sort(key=lambda entry: (-entry[0], entry[1].request_id, entry[1].subtask_index))
        temporary_compute = list(context.compute_backlog_gflops)
        temporary_link = dict(context.link_backlog_mb)
        assignments: list[Assignment] = []
        for eta, task in ordered:
            options: list[tuple[float, int, Assignment]] = []
            for node in context.feasible_nodes(task):
                candidate = context.evaluate(task, node)
                gamma, lambda_i, communication_dual = gamma_for_candidate(
                    context, task, candidate, temporary_compute, temporary_link
                )
                options.append((
                    gamma,
                    node,
                    Assignment(
                        task=task,
                        node=node,
                        candidate=candidate,
                        urgency=eta,
                        lambda_price=lambda_i,
                        mu_price_sum=communication_dual,
                        gamma=gamma,
                    ),
                ))
            choice = minimum_candidate(options)
            assignments.append(choice)
            # Eq. (26)-(27): update immediately for the next task in this slot.
            request = context.requests[task.request_id]
            temporary_compute[choice.node] += request.spec.gflops[task.subtask_index]
            for predecessor, path in choice.candidate.paths.items():
                amount = request.activation_mb(predecessor, task.subtask_index)
                for link in path:
                    temporary_link[link] += amount
        return assignments
