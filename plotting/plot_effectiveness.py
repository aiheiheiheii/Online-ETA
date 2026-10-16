"""The five four-panel effectiveness figures from saved CSV summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ._common import METHODS, draw_metric, load_figure_rows, save_png_pdf


FIGURE_SPECS: dict[str, tuple[str, str]] = {
    "fig1_edge_nodes": ("Number of edge nodes", "Effectiveness versus edge nodes"),
    "fig2_requests": ("Number of requests", "Effectiveness versus requests"),
    "fig3_computing_rate": ("Average computing rate (GHz)", "Effectiveness versus computing rate"),
    "fig4_bandwidth": ("Average bandwidth (MB/s)", "Effectiveness versus bandwidth"),
    "fig5_deadline": ("Deadline tightness factor", "Effectiveness versus deadline tightness"),
}

PANELS = (
    ("on_time_completed", "(a) On-time completed requests"),
    ("average_execution_cost", "(b) Average execution cost (s / assigned subtask)"),
    ("average_e2e_latency", "(c) Average end-to-end latency (s)"),
    ("p95_e2e_latency", "(d) P95 end-to-end latency (s)"),
)


def plot_effectiveness(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    show_ci: bool = False,
) -> tuple[Path, Path]:
    """Render one complete four-panel figure using only its figure-data CSV."""

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if csv_path.stem not in FIGURE_SPECS:
        raise ValueError(f"unknown effectiveness figure: {csv_path.name}")
    xlabel, title = FIGURE_SPECS[csv_path.stem]
    rows = load_figure_rows(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for ax, (metric, ylabel) in zip(axes.flat, PANELS, strict=True):
        draw_metric(ax, rows, metric, ylabel, xlabel, show_ci)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if labels != list(METHODS):
        raise ValueError(f"{csv_path} does not contain all five methods")
    fig.suptitle(title, fontsize=14)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return save_png_pdf(fig, output_dir, csv_path.stem)
