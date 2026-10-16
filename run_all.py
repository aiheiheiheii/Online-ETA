"""Run all five effectiveness sets and both efficiency sweeps.

The default is the full 800-slot, three-repeat experiment.  Use --smoke for
a quick end-to-end check before committing to the full run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import ExperimentConfig
from experiments import EXPERIMENTS
from experiments.common import EFFICIENCY_METRICS, run_experiment
from plotting.plot_effectiveness import plot_effectiveness
from plotting.plot_efficiency import plot_efficiency
from run_single import SMOKE_SPEC


def main() -> None:
    """Run selected groups, saving all intermediate CSVs before plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, help="Replicates per point (default from config)")
    parser.add_argument("--seed", type=int, help="Base seed (default from config)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--only", nargs="+", choices=tuple(EXPERIMENTS), help="Run only named groups")
    parser.add_argument("--smoke", action="store_true", help="Run one 20-slot, 6-node Dataset_100 case")
    parser.add_argument("--slots", type=int, default=20, help="Smoke-only number of slots (default 20)")
    parser.add_argument("--nodes", type=int, default=6, help="Smoke-only number of nodes (default 6)")
    parser.add_argument("--no-plot", action="store_true", help="Keep figure CSVs but skip rendering")
    parser.add_argument(
        "--show-ci", action=argparse.BooleanOptionalAction, default=None,
        help="Show or hide 95%% CI bands (default from config)",
    )
    args = parser.parse_args()

    updates: dict[str, object] = {}
    if args.repeats is not None:
        updates["num_repeats"] = args.repeats
    if args.seed is not None:
        updates["base_seed"] = args.seed
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.smoke:
        if args.only:
            parser.error("--only is not used with --smoke")
        updates.update(num_slots=args.slots, num_nodes=args.nodes, num_requests=100)
    elif args.slots != 20 or args.nodes != 6:
        parser.error("--slots and --nodes are smoke-only settings")
    config = replace(ExperimentConfig(), **updates)
    config.validate()

    if args.smoke:
        smoke_spec = replace(SMOKE_SPEC, values=(args.nodes,))
        replicate_csv, figure_csv = run_experiment(smoke_spec, config)
        print(f"Replicates: {replicate_csv}\nFigure data: {figure_csv}")
        return

    names = args.only or list(EXPERIMENTS)
    for name in names:
        spec = EXPERIMENTS[name]
        replicate_csv, figure_csv = run_experiment(spec, config)
        print(f"Replicates: {replicate_csv}\nFigure data: {figure_csv}")
        if not args.no_plot:
            render = plot_efficiency if spec.metrics == EFFICIENCY_METRICS else plot_effectiveness
            show_ci = config.show_confidence_intervals if args.show_ci is None else args.show_ci
            for path in render(figure_csv, config.output_dir / "figures", show_ci=show_ci):
                print(f"Figure: {path}")


if __name__ == "__main__":
    main()
