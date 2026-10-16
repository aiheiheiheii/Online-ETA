"""Regenerate every paper figure from already saved figure-data CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from plotting.plot_effectiveness import FIGURE_SPECS, plot_effectiveness
from plotting.plot_efficiency import plot_efficiency


def plot_all(
    figure_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    show_ci: bool | None = None,
) -> list[tuple[Path, Path]]:
    """Render saved effectiveness and efficiency CSVs without simulation.

    The added edge-node efficiency CSV is optional so older six-figure output
    directories remain independently reproducible.
    """

    config = ExperimentConfig()
    data_dir = Path(figure_data_dir) if figure_data_dir is not None else config.output_dir / "figure_data"
    figures_dir = Path(output_dir) if output_dir is not None else config.output_dir / "figures"
    use_ci = config.show_confidence_intervals if show_ci is None else show_ci
    output_paths = [
        plot_effectiveness(data_dir / f"{stem}.csv", figures_dir, show_ci=use_ci)
        for stem in FIGURE_SPECS
    ]
    output_paths.append(plot_efficiency(data_dir / "efficiency_requests.csv", figures_dir, show_ci=use_ci))
    edge_nodes_csv = data_dir / "efficiency_edge_nodes.csv"
    if edge_nodes_csv.is_file():
        output_paths.append(plot_efficiency(edge_nodes_csv, figures_dir, show_ci=use_ci))
    return output_paths


def main() -> None:
    """CLI entry point for CSV-only figure regeneration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-data-dir", type=Path, help="Directory holding the six figure CSV files")
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG and PDF figures")
    parser.add_argument(
        "--show-ci", action=argparse.BooleanOptionalAction, default=None,
        help="Show or hide 95%% CI bands (default from config)",
    )
    args = parser.parse_args()
    for png, pdf in plot_all(args.figure_data_dir, args.output_dir, show_ci=args.show_ci):
        print(f"{png}\n{pdf}")


if __name__ == "__main__":
    main()
