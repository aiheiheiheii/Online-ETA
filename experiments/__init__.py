"""One-factor MEC experiment definitions and shared execution code."""

from .efficiency_edge_nodes import SPEC as EFFICIENCY_EDGE_NODES
from .efficiency_requests import SPEC as EFFICIENCY
from .set1_edge_nodes import SPEC as SET1
from .set2_requests import SPEC as SET2
from .set3_computing_rate import SPEC as SET3
from .set4_bandwidth import SPEC as SET4
from .set5_deadline import SPEC as SET5

EXPERIMENTS = {
    spec.name: spec for spec in (SET1, SET2, SET3, SET4, SET5, EFFICIENCY, EFFICIENCY_EDGE_NODES)
}

__all__ = [
    "EXPERIMENTS", "SET1", "SET2", "SET3", "SET4", "SET5",
    "EFFICIENCY", "EFFICIENCY_EDGE_NODES",
]
