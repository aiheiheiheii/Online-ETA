"""Immutable request input parsed from one original dataset row."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A fixed DNN request with a sequential DAG.

    ``gflops[p]`` is the workload of subtask ``p``; ``activation_mb[p]``
    is the intermediate activation on edge ``p -> p + 1``.  MB is the
    configurable engineering interpretation of the unitless CSV field.
    Dynamic arrival, deadline, assignment and execution state are maintained
    separately by the simulator.
    """

    request_id: int
    model: str
    gflops: tuple[float, ...]
    activation_mb: tuple[float, ...]
    raw_total_gflops: float
    raw_total_activation: float
    source_dataset: str

    @property
    def num_subtasks(self) -> int:
        """Number of stages in the source model's sequential DAG."""

        return len(self.gflops)
