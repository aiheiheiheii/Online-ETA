"""Two-panel online decision-time figure from saved CSV summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ._common import METHODS, draw_metric, load_figure_rows, save_png_pdf


FIGURE_SPECS = {
    "efficiency_requests": ("Number of requests", "Decision time versus number of requests"),
    "efficiency_edge_nodes": ("Number of edge nodes", "Decision time versus edge nodes"),
}


def plot_efficiency(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    show_ci: bool = False,
) -> tuple[Path, Path]:
    """Render mean and P95 decision-time series from one figure-data CSV."""

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if csv_path.stem not in FIGURE_SPECS:
        raise ValueError(f"unknown efficiency figure: {csv_path.name}")
    xlabel, title = FIGURE_SPECS[csv_path.stem]
    rows = load_figure_rows(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7))
    draw_metric(axes[0], rows, "mean_decision_time", "(a) Mean decision time per slot (s)", xlabel, show_ci)
    draw_metric(axes[1], rows, "p95_decision_time", "(b) P95 decision time per slot (s)", xlabel, show_ci)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels != list(METHODS):
        raise ValueError(f"{csv_path} does not contain all five methods")
    fig.suptitle(title, fontsize=14)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return save_png_pdf(fig, output_dir, csv_path.stem)
