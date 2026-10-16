"""Current-slot observations and the paper's execution-cost equations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from config import ExperimentConfig
from simulator.subtask import RequestState

LinkId = tuple[int, int]


@dataclass(frozen=True, order=True)
class ReadyTask:
    request_id: int
    subtask_index: int


@dataclass
class Candidate:
    node: int
    compute_cost_seconds: float
    communication_cost_seconds: float
    total_cost_seconds: float
    paths: dict[int, tuple[LinkId, ...]] = field(default_factory=dict)


@dataclass
class Assignment:
    task: ReadyTask
    node: int
    candidate: Candidate | None
    urgency: float | None = None
    lambda_price: float | None = None
    mu_price_sum: float | None = None
    gamma: float | None = None


@dataclass
class DecisionContext:
    slot: int
    config: ExperimentConfig
    requests: Mapping[int, RequestState]
    network: object
    compute_backlog_gflops: tuple[float, ...]
    link_backlog_mb: Mapping[LinkId, float]
    compute_rates_ghz: tuple[float, ...]
    link_rates_mb_s: Mapping[LinkId, float]

    def urgency(self, request_id: int) -> tuple[float, float]:
        request = self.requests[request_id]
        tau = request.deadline_seconds - (
            self.slot - request.arrival_slot
        ) * self.config.slot_seconds
        eta = request.deadline_seconds / max(tau, self.config.slot_seconds)
        return tau, eta

    def feasible_nodes(self, task: ReadyTask) -> tuple[int, ...]:
        request = self.requests[task.request_id]
        nodes = []
        for node in range(self.config.num_nodes):
            if self.compute_rates_ghz[node] <= 0:
                continue
            if all(
                self.network.has_path(request.subtasks[p].assigned_node, node)
                for p in request.predecessors(task.subtask_index)
            ):
                nodes.append(node)
        return tuple(nodes)

    def evaluate(self, task: ReadyTask, node: int) -> Candidate:
        """Equations (9)-(12), using actual slot-start W and U throughout."""
        request = self.requests[task.request_id]
        g = request.spec.gflops[task.subtask_index]
        rate = self.compute_rates_ghz[node]
        if rate <= 0:
            raise ValueError(f"Node {node} has no current compute service")
        # GFLOPs / GHz is seconds when chi is cycles/FLOP.
        compute = self.config.cycles_per_flop * (
            self.compute_backlog_gflops[node] + g
        ) / rate
        communication = 0.0
        paths: dict[int, tuple[LinkId, ...]] = {}
        for predecessor in request.predecessors(task.subtask_index):
            source = request.subtasks[predecessor].assigned_node
            if source is None:
                raise AssertionError("A ready successor has an unassigned predecessor")
            if source == node:
                continue
            amount = request.activation_mb(predecessor, task.subtask_index)
            if amount == 0:
                continue
            path, path_cost = self.network.best_path(
                source,
                node,
                amount,
                self.link_backlog_mb,
                self.link_rates_mb_s,
            )
            paths[predecessor] = tuple(path)
            communication += path_cost
        return Candidate(node, compute, communication, compute + communication, paths)
