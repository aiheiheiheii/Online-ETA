"""Set #2: switch among eight independent fixed-total request datasets."""

from .common import ExperimentSpec

SPEC = ExperimentSpec(
    "set2_requests", "requests", "num_requests",
    (100, 200, 300, 400, 500, 600, 700, 800), "fig2_requests",
)
