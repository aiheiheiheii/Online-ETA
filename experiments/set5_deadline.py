"""Set #5: vary the deadline tightness multiplier."""

from .common import ExperimentSpec

SPEC = ExperimentSpec(
    "set5_deadline", "theta_deadline", "theta_deadline",
    (1.0, 1.2, 1.4, 1.6, 1.8), "fig5_deadline",
)
