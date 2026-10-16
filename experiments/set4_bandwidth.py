"""Set #4: vary average directed-link bandwidth in MB/s."""

from .common import ExperimentSpec

SPEC = ExperimentSpec(
    "set4_bandwidth", "bandwidth_mb_s", "mean_bandwidth_mb_s",
    (1.0, 1.2, 1.4, 1.6, 1.8, 2.0), "fig4_bandwidth",
)
