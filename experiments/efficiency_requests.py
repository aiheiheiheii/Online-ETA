"""Efficiency experiment: decision time versus independent request datasets."""

from .common import EFFICIENCY_METRICS, ExperimentSpec

SPEC = ExperimentSpec(
    "efficiency_requests", "requests", "num_requests",
    (100, 200, 300, 400, 500, 600, 700, 800),
    "efficiency_requests", EFFICIENCY_METRICS,
)
