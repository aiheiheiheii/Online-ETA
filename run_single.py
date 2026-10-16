"""Run one parameter point (all five policies) or a tiny integration smoke case.

Examples:
    python run_single.py --experiment set2_requests --value 300 --repeats 1
    python run_single.py --smoke --repeats 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import ExperimentConfig
from experiments import EXPERIMENTS
from experiments.common import EFFICIENCY_METRICS, ExperimentSpec, run_experiment
from plotting.plot_effectiveness import plot_effectiveness
from plotting.plot_efficiency import plot_efficiency


SMOKE_SPEC = ExperimentSpec(
    "smoke", "edge_nodes", "num_nodes", (6,), "smoke"
)


def main() -> None:
    """Parse CLI settings and execute the selected real simulation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="set2_requests")
    parser.add_argument("--value", type=float, help="One declared parameter value for the chosen experiment")
    parser.add_argument("--repeats", type=int, help="Replicates per point (default from config)")
    parser.add_argument("--seed", type=int, help="Base seed (default from config)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default experiment_project/outputs)")
    parser.add_argument("--smoke", action="store_true", help="20 slots, 6 nodes, Dataset_100, five methods")
    parser.add_argument("--slots", type=int, default=20, help="Smoke-only number of slots (default 20)")
    parser.add_argument("--nodes", type=int, default=6, help="Smoke-only number of nodes (default 6)")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV without rendering a figure")
    parser.add_argument(
        "--show-ci", action=argparse.BooleanOptionalAction, default=None,
        help="Show or hide 95%% CI bands (default from config)",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    updates: dict[str, object] = {}
    if args.repeats is not None:
        updates["num_repeats"] = args.repeats
    if args.seed is not None:
        updates["base_seed"] = args.seed
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.smoke:
        if args.value is not None:
            parser.error("--value is not used with --smoke")
        updates.update(num_slots=args.slots, num_nodes=args.nodes, num_requests=100)
        spec = replace(SMOKE_SPEC, values=(args.nodes,))
        selected_value: int | float = args.nodes
    else:
        if args.slots != 20 or args.nodes != 6:
            parser.error("--slots and --nodes are smoke-only settings")
        spec = EXPERIMENTS[args.experiment]
        if args.value is None:
            parser.error("--value is required unless --smoke is used")
        selected_value = int(args.value) if args.value.is_integer() else args.value
        if args.output_dir is None:
            value_label = str(selected_value).replace(".", "p")
            updates["output_dir"] = config.output_dir / "single_runs" / f"{spec.name}_{value_label}"
    config = replace(config, **updates)
    config.validate()
    replicate_csv, figure_csv = run_experiment(spec, config, (selected_value,))
    print(f"Replicates: {replicate_csv}")
    print(f"Figure data: {figure_csv}")
    if not args.no_plot and not args.smoke:
        figures_dir = config.output_dir / "figures"
        render = plot_efficiency if spec.metrics == EFFICIENCY_METRICS else plot_effectiveness
        show_ci = config.show_confidence_intervals if args.show_ci is None else args.show_ci
        for path in render(figure_csv, figures_dir, show_ci=show_ci):
            print(f"Figure: {path}")


if __name__ == "__main__":
    main()
