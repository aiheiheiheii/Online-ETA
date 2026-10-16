"""Set #1: vary the number of edge nodes."""

from .common import ExperimentSpec

SPEC = ExperimentSpec(
    "set1_edge_nodes", "edge_nodes", "num_nodes",
    (30, 40, 50, 60, 70, 80, 90, 100), "fig1_edge_nodes",
)
