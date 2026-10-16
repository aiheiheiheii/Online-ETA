"""Efficiency experiment: decision time versus the number of edge nodes."""

from .common import EFFICIENCY_METRICS, ExperimentSpec


SPEC = ExperimentSpec(
    "efficiency_edge_nodes", "edge_nodes", "num_nodes",
    (30, 40, 50, 60, 70, 80, 90, 100),
    "efficiency_edge_nodes", EFFICIENCY_METRICS,
)
