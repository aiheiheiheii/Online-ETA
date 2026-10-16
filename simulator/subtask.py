"""Mutable runtime state for one dataset request and its sequential DAG."""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.request import RequestSpec


@dataclass
class SubtaskState:
    index: int
    ready_slot: int | None = None
    assignment_slot: int | None = None
    assigned_node: int | None = None
    predecessor_node: int | None = None
    communication_required: bool = False
    communication_data_mb: float = 0.0
    communication_path: str = ""
    activation_arrival_slot: int | None = None
    compute_start_slot: int | None = None
    compute_finish_slot: int | None = None
    remaining_gflops: float = 0.0
    execution_cost_seconds: float | None = None
    urgency: float | None = None
    lambda_price: float | None = None
    mu_price_sum: float | None = None
    gamma: float | None = None
    pending_transmissions: int = 0

    @property
    def finished(self) -> bool:
        return self.compute_finish_slot is not None


@dataclass
class RequestState:
    spec: RequestSpec
    arrival_slot: int
    base_deadline_seconds: float
    theta_deadline: float
    deadline_seconds: float
    subtasks: list[SubtaskState] = field(default_factory=list)
    finish_slot: int | None = None

    def __post_init__(self) -> None:
        if not self.subtasks:
            self.subtasks = [
                SubtaskState(index=p, remaining_gflops=g)
                for p, g in enumerate(self.spec.gflops)
            ]

    def latency_seconds(self, delta: float) -> float | None:
        if self.finish_slot is None:
            return None
        return (self.finish_slot - self.arrival_slot) * delta

    def predecessors(self, subtask_index: int) -> tuple[int, ...]:
        """CSV rows encode a chain, so each non-source stage has one parent."""
        return () if subtask_index == 0 else (subtask_index - 1,)

    def activation_mb(self, predecessor: int, successor: int) -> float:
        if successor != predecessor + 1:
            raise ValueError("The supplied dataset has sequential DAG edges only")
        return self.spec.activation_mb[predecessor]
