"""Set #3: vary average computing service rate in GHz."""

from .common import ExperimentSpec

SPEC = ExperimentSpec(
    "set3_computing_rate", "computing_rate_ghz", "mean_compute_ghz",
    (7, 8, 9, 10, 11, 12), "fig3_computing_rate",
)
