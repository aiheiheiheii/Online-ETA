"""Discrete-slot MEC simulation shared unchanged by every assignment method."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from config import ExperimentConfig
from simulator.decision import Assignment, DecisionContext, LinkId, ReadyTask
from simulator.queues import EPS, WorkQueue
from simulator.request import RequestSpec
from simulator.subtask import RequestState
from simulator.arrival import assign_arrival_slots
from simulator.deadline import generate_deadlines
from simulator.network import build_network
from simulator.resource_trace import generate_resource_trace
from data.dataset_loader import load_dataset


class AssignmentMethod(Protocol):
    name: str

    def select_assignments(
        self, context: DecisionContext, ready_tasks: list[ReadyTask]
    ) -> list[Assignment]: ...


@dataclass(frozen=True)
class Scenario:
    """One immutable environment, reused by all five policies in a replicate."""

    config: ExperimentConfig
    requests: tuple[RequestSpec, ...]
    arrivals: Mapping[int, int]
    deadlines: Mapping[int, object]
    network: object
    resource_trace: object
    seed: int

    def validate(self) -> None:
        ids = {r.request_id for r in self.requests}
        if len(ids) != len(self.requests):
            raise AssertionError("Duplicate request IDs in scenario")
        if len(self.requests) != self.config.num_requests:
            raise AssertionError("Scenario request count differs from its fixed dataset scale")
        if ids != set(self.arrivals) or ids != set(self.deadlines):
            raise AssertionError("Arrival/deadline mapping does not match dataset")
        if self.network.nodes != tuple(range(self.config.num_nodes)):
            raise AssertionError("Network nodes differ from configured feasible nodes")
        if not all(0 <= t < self.config.num_slots for t in self.arrivals.values()):
            raise AssertionError("Arrival slot outside the research horizon")
        if len(self.resource_trace.compute_rates_ghz) != self.config.num_slots:
            raise AssertionError("Invalid compute trace length")
        if len(self.resource_trace.link_rates_mb_s) != self.config.num_slots:
            raise AssertionError("Invalid link trace length")


def create_scenario(config: ExperimentConfig, seed: int) -> Scenario:
    """Build one fixed request/environment realization for all five methods."""
    config.validate()
    requests = tuple(load_dataset(config.num_requests, config))
    network = build_network(config, seed + 10_000)
    # Populate the topology-only route cache before policy decision timing.
    # This also gives every method the same set of available paths.
    for source in network.nodes:
        for destination in network.nodes:
            if source != destination:
                network.candidate_paths(source, destination)
    arrivals = assign_arrival_slots(requests, config.num_slots, seed + 20_000, config)
    deadlines = generate_deadlines(requests, config)
    trace = generate_resource_trace(config, network, seed + 30_000)
    scenario = Scenario(config, requests, arrivals, deadlines, network, trace, seed)
    scenario.validate()
    return scenario


@dataclass
class TransmissionState:
    key: tuple[int, int, int]
    amount_mb: float
    path: tuple[LinkId, ...]
    hop_index: int = 0
    remaining_mb: float = 0.0

    def __post_init__(self) -> None:
        self.remaining_mb = self.amount_mb


@dataclass
class SimulationResult:
    method: str
    scenario: Scenario
    requests: dict[int, RequestState]
    slot_rows: list[dict[str, object]] = field(default_factory=list)
    assignment_history: list[Assignment] = field(default_factory=list)

    def request_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for request in self.requests.values():
            latency = request.latency_seconds(self.scenario.config.slot_seconds)
            rows.append(
                {
                    "request_id": request.spec.request_id,
                    "model": request.spec.model,
                    "source_dataset": request.spec.source_dataset,
                    "arrival_slot": request.arrival_slot,
                    "base_deadline": request.base_deadline_seconds,
                    "theta_deadline": request.theta_deadline,
                    "deadline": request.deadline_seconds,
                    "finish_slot": request.finish_slot,
                    "finished": request.finish_slot is not None,
                    "on_time": bool(
                        latency is not None and latency <= request.deadline_seconds + EPS
                    ),
                    "e2e_latency": latency,
                    "num_subtasks": request.spec.num_subtasks,
                    "total_gflops": request.spec.raw_total_gflops,
                    "total_activation": request.spec.raw_total_activation,
                }
            )
        return rows

    def subtask_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for request in self.requests.values():
            for state in request.subtasks:
                rows.append(
                    {
                        "request_id": request.spec.request_id,
                        "subtask_id": state.index + 1,
                        "ready_slot": state.ready_slot,
                        "assignment_slot": state.assignment_slot,
                        "assigned_node": state.assigned_node,
                        "compute_workload": request.spec.gflops[state.index],
                        "predecessor_node": state.predecessor_node,
                        "communication_required": state.communication_required,
                        "communication_data": state.communication_data_mb,
                        "communication_path": state.communication_path,
                        "activation_arrival_slot": state.activation_arrival_slot,
                        "compute_start": state.compute_start_slot,
                        "compute_finish": state.compute_finish_slot,
                        "execution_cost": state.execution_cost_seconds,
                        "urgency": state.urgency,
                        "lambda": state.lambda_price,
                        "mu": state.mu_price_sum,
                        "Gamma": state.gamma,
                    }
                )
        return rows


class MECSimulator:
    """Run FIFO communication and computation once per slot, after decisions."""

    def __init__(self, scenario: Scenario, method: AssignmentMethod, *,
                 unexpired_only: bool = False) -> None:
        self.unexpired_only = unexpired_only
        scenario.validate()
        self.scenario = scenario
        self.method = method
        self.config = scenario.config
        self.requests = {
            spec.request_id: RequestState(
                spec=spec,
                arrival_slot=scenario.arrivals[spec.request_id],
                base_deadline_seconds=scenario.deadlines[spec.request_id].base_seconds,
                theta_deadline=scenario.deadlines[spec.request_id].theta,
                deadline_seconds=scenario.deadlines[spec.request_id].final_seconds,
            )
            for spec in scenario.requests
        }
        self.compute_queues = [WorkQueue() for _ in scenario.network.nodes]
        self.link_queues = {link: WorkQueue() for link in scenario.network.links}
        self.transmissions: dict[tuple[int, int, int], TransmissionState] = {}
        self.pending_links: list[tuple[LinkId, tuple[int, int, int], float]] = []
        self.pending_compute: list[tuple[int, int]] = []

    def _flush_boundary_arrivals(self) -> None:
        """Relay/activation completion in slot t becomes eligible in t+1."""
        for link, key, amount in self.pending_links:
            self.link_queues[link].enqueue(key, amount)
        self.pending_links.clear()
        for request_id, subtask_index in self.pending_compute:
            state = self.requests[request_id].subtasks[subtask_index]
            node = state.assigned_node
            assert node is not None
            self.compute_queues[node].enqueue(
                (request_id, subtask_index), state.remaining_gflops
            )
        self.pending_compute.clear()

    def _ready_tasks(self, slot: int) -> list[ReadyTask]:
        ready: list[ReadyTask] = []
        for request in self.requests.values():
            if slot < request.arrival_slot or request.finish_slot is not None:
                continue
            for state in request.subtasks:
                if state.assignment_slot is not None:
                    continue
                if all(
                    request.subtasks[p].compute_finish_slot is not None
                    and request.subtasks[p].compute_finish_slot <= slot
                    for p in request.predecessors(state.index)
                ):
                    if state.ready_slot is None:
                        state.ready_slot = slot
                    ready.append(ReadyTask(request.spec.request_id, state.index))
        return ready

    def _make_context(self, slot: int) -> DecisionContext:
        compute_rates = tuple(
            float(x) for x in self.scenario.resource_trace.compute_rates_ghz[slot]
        )
        link_rates = self.scenario.resource_trace.link_rates_mb_s[slot]
        return DecisionContext(
            slot=slot,
            config=self.config,
            requests=self.requests,
            network=self.scenario.network,
            compute_backlog_gflops=tuple(q.backlog for q in self.compute_queues),
            link_backlog_mb={link: q.backlog for link, q in self.link_queues.items()},
            compute_rates_ghz=compute_rates,
            link_rates_mb_s=link_rates,
        )

    def _apply_assignment(
        self, assignment: Assignment, context: DecisionContext, slot: int
    ) -> None:
        task = assignment.task
        request = self.requests[task.request_id]
        state = request.subtasks[task.subtask_index]
        if slot < request.arrival_slot:
            raise AssertionError("Request executed before arrival")
        if state.assignment_slot is not None:
            raise AssertionError("Subtask assigned twice")
        if assignment.node not in context.feasible_nodes(task):
            raise AssertionError("Assignment selected an infeasible node")
        if assignment.candidate is None:
            raise AssertionError("Missing post-decision execution-cost evaluation")
        if assignment.candidate.node != assignment.node:
            raise AssertionError("Candidate node and assignment node differ")
        if not math.isfinite(assignment.candidate.total_cost_seconds):
            raise AssertionError("Nonfinite assignment cost")
        for p in request.predecessors(task.subtask_index):
            finish = request.subtasks[p].compute_finish_slot
            if finish is None or finish > slot:
                raise AssertionError("Successor assigned before predecessor finished")
        state.assignment_slot = slot
        state.assigned_node = assignment.node
        state.execution_cost_seconds = assignment.candidate.total_cost_seconds
        state.urgency = assignment.urgency
        state.lambda_price = assignment.lambda_price
        state.mu_price_sum = assignment.mu_price_sum
        state.gamma = assignment.gamma
        path_strings = []
        for predecessor in request.predecessors(task.subtask_index):
            source = request.subtasks[predecessor].assigned_node
            assert source is not None
            state.predecessor_node = source
            if source == assignment.node:
                continue
            amount = request.activation_mb(predecessor, task.subtask_index)
            if amount == 0:
                continue
            path = assignment.candidate.paths[predecessor]
            if not path:
                raise AssertionError("Cross-node task has no communication path")
            key = (task.request_id, task.subtask_index, predecessor)
            self.transmissions[key] = TransmissionState(key, amount, path)
            self.link_queues[path[0]].enqueue(key, amount)
            state.pending_transmissions += 1
            state.communication_required = True
            state.communication_data_mb += amount
            path_strings.append(f"{predecessor + 1}:" + ">".join(
                f"{a}-{b}" for a, b in path
            ))
        state.communication_path = "|".join(path_strings)
        if state.pending_transmissions == 0:
            state.activation_arrival_slot = slot
            self.compute_queues[assignment.node].enqueue(
                (task.request_id, task.subtask_index), state.remaining_gflops
            )

    def _serve_links(self, slot: int) -> float:
        total = 0.0
        rates = self.scenario.resource_trace.link_rates_mb_s[slot]
        for link in self.scenario.network.links:
            completed, served = self.link_queues[link].serve(
                rates[link] * self.config.slot_seconds
            )
            total += sum(served.values())
            for key, amount in served.items():
                transmission = self.transmissions[key]
                transmission.remaining_mb -= amount
            for key in completed:
                transmission = self.transmissions[key]
                transmission.hop_index += 1
                if transmission.hop_index < len(transmission.path):
                    transmission.remaining_mb = transmission.amount_mb
                    next_link = transmission.path[transmission.hop_index]
                    self.pending_links.append((next_link, key, transmission.amount_mb))
                else:
                    request_id, subtask_index, _ = key
                    state = self.requests[request_id].subtasks[subtask_index]
                    state.pending_transmissions -= 1
                    if state.pending_transmissions < 0:
                        raise AssertionError("Negative pending-transmission count")
                    if state.pending_transmissions == 0:
                        state.activation_arrival_slot = slot + 1
                        self.pending_compute.append((request_id, subtask_index))
                    del self.transmissions[key]
        return total

    def _serve_compute(self, slot: int) -> float:
        total = 0.0
        rates = self.scenario.resource_trace.compute_rates_ghz[slot]
        for node, queue in enumerate(self.compute_queues):
            capacity_gflops = (
                rates[node] * self.config.slot_seconds / self.config.cycles_per_flop
            )
            completed, served = queue.serve(capacity_gflops)
            total += sum(served.values())
            for key, amount in served.items():
                request_id, subtask_index = key
                state = self.requests[request_id].subtasks[subtask_index]
                if state.compute_start_slot is None:
                    state.compute_start_slot = slot
                state.remaining_gflops = max(0.0, state.remaining_gflops - amount)
            for request_id, subtask_index in completed:
                request = self.requests[request_id]
                state = request.subtasks[subtask_index]
                state.compute_finish_slot = slot + 1
                state.remaining_gflops = 0.0
                if subtask_index == request.spec.num_subtasks - 1:
                    request.finish_slot = slot + 1
        return total

    def _validate_result(self, result: SimulationResult) -> None:
        """Fail immediately on impossible timing, state, or numeric outcomes."""
        if len(result.slot_rows) != self.config.num_slots:
            raise AssertionError("Missing slot-level records")
        for row in result.slot_rows:
            for field_name in (
                "decision_time",
                "total_compute_backlog",
                "total_communication_backlog",
                "compute_backlog_end",
                "communication_backlog_end",
                "compute_service_gflops",
                "communication_service_mb",
            ):
                value = float(row[field_name])
                if not math.isfinite(value) or value < -EPS:
                    raise AssertionError(f"Invalid slot metric {field_name}: {value}")
        for request in self.requests.values():
            if not 0 <= request.arrival_slot < self.config.num_slots:
                raise AssertionError("Invalid request arrival")
            for state in request.subtasks:
                if state.assignment_slot is None:
                    if state.assigned_node is not None or state.compute_start_slot is not None:
                        raise AssertionError("Unassigned subtask has execution state")
                    continue
                if state.assigned_node is None or not 0 <= state.assigned_node < self.config.num_nodes:
                    raise AssertionError("Invalid assigned node")
                if state.assignment_slot < request.arrival_slot:
                    raise AssertionError("Subtask assigned before arrival")
                if state.ready_slot is None or state.ready_slot > state.assignment_slot:
                    raise AssertionError("Subtask assigned before ready")
                for predecessor in request.predecessors(state.index):
                    finish = request.subtasks[predecessor].compute_finish_slot
                    if finish is None or finish > state.assignment_slot:
                        raise AssertionError("Successor assigned before predecessor completion")
                if state.compute_start_slot is not None:
                    if state.activation_arrival_slot is None:
                        raise AssertionError("Computation began without activation")
                    if state.compute_start_slot < state.activation_arrival_slot:
                        raise AssertionError("Computation preceded activation arrival")
                    if state.compute_start_slot < state.assignment_slot:
                        raise AssertionError("Computation preceded assignment")
                if state.compute_finish_slot is not None:
                    if state.compute_start_slot is None or state.compute_finish_slot <= state.compute_start_slot:
                        raise AssertionError("Invalid compute completion boundary")
            terminal_finish = request.subtasks[-1].compute_finish_slot
            if request.finish_slot != terminal_finish:
                raise AssertionError("Request finish must equal terminal subtask finish")
        if len(result.assignment_history) != len({
            (assignment.task.request_id, assignment.task.subtask_index)
            for assignment in result.assignment_history
        }):
            raise AssertionError("A subtask appears more than once in assignment history")

    def run(self) -> SimulationResult:
        result = SimulationResult(self.method.name, self.scenario, self.requests)
        for slot in range(self.config.num_slots):
            self._flush_boundary_arrivals()
            num_arrivals = sum(r.arrival_slot == slot for r in self.requests.values())
            active = sum(
                r.arrival_slot <= slot and r.finish_slot is None
                for r in self.requests.values()
            )
            start = time.perf_counter()
            ready_before_filter = self._ready_tasks(slot)
            ready = ready_before_filter
            if self.unexpired_only:
                ready = [task for task in ready if
                         self.requests[task.request_id].deadline_seconds -
                         (slot - self.requests[task.request_id].arrival_slot) *
                         self.config.slot_seconds >= 0]
            context = self._make_context(slot)
            assignments = self.method.select_assignments(context, ready)
            decision_time = time.perf_counter() - start
            if decision_time < 0:
                raise AssertionError("Negative decision time")
            if len({a.task for a in assignments}) != len(assignments):
                raise AssertionError("Multiple assignments for one ready subtask")
            ready_set = set(ready)
            for assignment in assignments:
                if assignment.task not in ready_set:
                    raise AssertionError("Assignment outside current ready set")
                if assignment.candidate is None:
                    # Random/Compute-Greedy do not evaluate C to choose a node.
                    # Compute their accounting cost after the decision timer.
                    assignment.candidate = context.evaluate(
                        assignment.task, assignment.node
                    )
                self._apply_assignment(assignment, context, slot)
                result.assignment_history.append(assignment)
            link_backlog_start = sum(context.link_backlog_mb.values())
            compute_backlog_start = sum(context.compute_backlog_gflops)
            link_service = self._serve_links(slot)
            compute_service = self._serve_compute(slot)
            compute_backlog_end = sum(q.backlog for q in self.compute_queues)
            link_backlog_end = sum(q.backlog for q in self.link_queues.values())
            if min(compute_backlog_end, link_backlog_end) < -EPS:
                raise AssertionError("Negative queue backlog")
            result.slot_rows.append(
                {
                    "slot": slot,
                    "num_arrivals": num_arrivals,
                    "num_active_requests": active,
                    "num_ready_subtasks": len(ready),
                    "num_ready_before_filter": len(ready_before_filter),
                    "num_expired_ready_subtasks": len(ready_before_filter) - len(ready),
                    "num_assignments": len(assignments),
                    "decision_time": decision_time,
                    "total_compute_backlog": compute_backlog_start,
                    "total_communication_backlog": link_backlog_start,
                    "compute_backlog_end": compute_backlog_end,
                    "communication_backlog_end": link_backlog_end,
                    "compute_service_gflops": compute_service,
                    "communication_service_mb": link_service,
                }
            )
        self._validate_result(result)
        return result
