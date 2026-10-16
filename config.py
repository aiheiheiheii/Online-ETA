"""Central settings for the MEC experiments.

The fixed study settings and the configurable engineering assumptions are kept
in one place.  In particular, ``arrival_model`` is *not* a finalized paper
setting.  ``activation_unit='MB'`` is an explicit interpretation of an
otherwise unitless dataset column, not a claim about the source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
ARRIVAL_MODEL = "fixed_total_uniform"  # Replaceable engineering default.
ARRIVAL_WINDOW_FRACTION = 0.10  # Engineering default: first 10% of all slots.
NUM_REPEATS = 3


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, replaceable experiment configuration.

    Compute workloads are GFLOPs; compute rates are GHz with a normalized
    ``cycles_per_flop`` conversion.  Communication workloads and rates use MB
    and MB/s, respectively.  One slot lasts ``slot_seconds`` seconds.

    The arrival, deadline, topology, and resource variability choices below
    are engineering defaults pending a fully specified experimental protocol.
    """

    num_slots: int = 800
    slot_seconds: float = 2.0
    num_nodes: int = 60
    num_requests: int = 300
    mean_compute_ghz: float = 10.0
    mean_bandwidth_mb_s: float = 1.4
    theta_deadline: float = 1.4
    num_repeats: int = NUM_REPEATS
    base_seed: int = 0
    cycles_per_flop: float = 1.0

    dataset_dir: Path = field(
        default_factory=lambda: WORKSPACE_DIR / "DNN_task_datasets_ratio_3312"
    )
    output_dir: Path = field(default_factory=lambda: PROJECT_DIR / "outputs")

    # ARRIVAL_MODEL: replaceable engineering default; the paper does not fix
    # an arrival distribution for the already fixed Dataset_N request set.
    arrival_model: str = ARRIVAL_MODEL
    arrival_start_slot: int = 0
    arrival_window_fraction: float = ARRIVAL_WINDOW_FRACTION
    arrival_end_slot: int | None = None  # inclusive override of the window fraction

    # The CSV supplies numbers but no evidenced Activation_Size unit.  MB is
    # the explicit, configurable engineering interpretation used by the code.
    activation_unit: str = "MB"
    activation_mb_per_csv_unit: float = 1.0

    # The following are configurable environment assumptions, not source
    # claims.  A topology generator/resource-trace generator consumes them.
    topology_model: str = "ring_chord"
    ring_neighbors: int = 1
    extra_chords_per_node: int = 1
    max_path_hops: int | None = None
    max_candidate_paths: int = 3
    compute_heterogeneity_cv: float = 0.15
    bandwidth_heterogeneity_cv: float = 0.15
    compute_temporal_cv: float = 0.10
    bandwidth_temporal_cv: float = 0.10
    min_resource_fraction: float = 0.10

    # Reference-deadline construction is an engineering default unless the
    # experiment file supplies a unique formula.  Keep it independent of the
    # online assignment policy and record both reference and final deadline.
    deadline_model: str = "reference_latency_scaled"
    # Fixed reference rates avoid changing deadlines during the M/B sweeps.
    deadline_reference_compute_ghz: float | None = 10.0
    deadline_reference_bandwidth_mb_s: float | None = 1.4
    deadline_min_seconds: float = 0.0

    show_confidence_intervals: bool = False

    @property
    def arrival_last_slot(self) -> int:
        """Inclusive last slot in which a dataset request may arrive."""

        if self.arrival_end_slot is not None:
            return self.arrival_end_slot
        return ceil(self.num_slots * self.arrival_window_fraction) - 1

    def validate(self) -> None:
        """Reject malformed settings before any random state is generated."""

        if self.num_slots <= 0 or self.slot_seconds <= 0:
            raise ValueError("num_slots and slot_seconds must be positive")
        if self.num_nodes <= 0 or self.num_requests <= 0 or self.num_repeats <= 0:
            raise ValueError("num_nodes, num_requests, and num_repeats must be positive")
        if self.mean_compute_ghz <= 0 or self.mean_bandwidth_mb_s <= 0:
            raise ValueError("mean resource rates must be positive")
        if self.theta_deadline <= 0 or self.cycles_per_flop <= 0:
            raise ValueError("theta_deadline and cycles_per_flop must be positive")
        if not isfinite(self.arrival_window_fraction) or not 0 < self.arrival_window_fraction <= 1:
            raise ValueError("arrival_window_fraction must lie in (0, 1]")
        if not 0 <= self.arrival_start_slot <= self.arrival_last_slot < self.num_slots:
            raise ValueError("arrival slot interval must lie inside the study period")
        if not self.activation_unit.strip():
            raise ValueError("activation_unit must name the chosen input-unit assumption")
        if not isfinite(self.activation_mb_per_csv_unit) or self.activation_mb_per_csv_unit <= 0:
            raise ValueError("activation_mb_per_csv_unit must be finite and positive")
        if self.ring_neighbors < 0 or self.extra_chords_per_node < 0:
            raise ValueError("topology ring/chord counts must be nonnegative")
        if (self.max_path_hops is not None and self.max_path_hops <= 0) or self.max_candidate_paths <= 0:
            raise ValueError("path limits must be positive when specified")
        for name in (
            "compute_heterogeneity_cv",
            "bandwidth_heterogeneity_cv",
            "compute_temporal_cv",
            "bandwidth_temporal_cv",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.min_resource_fraction <= 1:
            raise ValueError("min_resource_fraction must lie in (0, 1]")
        if not isfinite(self.deadline_min_seconds) or self.deadline_min_seconds < 0:
            raise ValueError("deadline_min_seconds must be finite and nonnegative")
        if self.deadline_reference_compute_ghz is not None and self.deadline_reference_compute_ghz <= 0:
            raise ValueError("deadline_reference_compute_ghz must be positive")
        if self.deadline_reference_bandwidth_mb_s is not None and self.deadline_reference_bandwidth_mb_s <= 0:
            raise ValueError("deadline_reference_bandwidth_mb_s must be positive")
