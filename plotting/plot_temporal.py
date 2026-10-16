"""Fig. 8 from the complete saved slot-statistics grid; no smoothing."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.temporal import METHODS, METRICS

STYLES = {
    "Proposed": ("#d62728", "-", 1.45, 4),
    "w/o Urgency": ("#1f77b4", "--", 1.05, 2),
    "w/o Resource Price": ("#2ca02c", ":", 1.05, 3),
}
YLABELS = ("Number of Requests Completed\nwithin Deadline",
           "Average Current-State\nExecution Cost (s)", "Online Decision Time (ms)")


def plot_temporal(csv_path: Path, output_dir: Path):
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7200
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.7, "legend.frameon": False})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.13, hspace=0.67)
    handles = []
    for index, (ax, metric, ylabel) in enumerate(zip(axes, METRICS, YLABELS)):
        for method in METHODS:
            group = sorted((r for r in rows if r["metric"] == metric and r["method"] == method),
                           key=lambda r: int(r["slot"]))
            assert [int(r["slot"]) for r in group] == list(range(800))
            x = np.array([int(r["display_slot"]) for r in group])
            y = np.array([float(r["mean"]) if r["mean"] else np.nan for r in group])
            low = np.array([float(r["ci95_low"]) if r["ci95_low"] else np.nan for r in group])
            high = np.array([float(r["ci95_high"]) if r["ci95_high"] else np.nan for r in group])
            color, linestyle, width, zorder = STYLES[method]
            line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=width,
                            label=method, zorder=zorder)
            ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=1)
            if index == 0:
                handles.append(line)
        ax.set_xlim(1, 800)
        ax.set_xticks([1, 200, 400, 600, 800])
        ax.set_xlabel("Time Slot (t)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        ax.text(-0.14, 1.05, chr(ord("a") + index), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
    fig.suptitle("Fig. 8 | Performance over Time", fontsize=11, y=0.98)
    fig.legend(handles, METHODS, loc="upper center", bbox_to_anchor=(0.58, 0.95),
               ncol=3, fontsize=8, handlelength=2.8)
    fig.text(0.18, 0.025,
             "Mean and pointwise 95% t intervals across 10 independent seeds.\n"
             "Cost: valid assignment replicates only; undefined slots remain gaps.",
             fontsize=7.5, va="bottom")
    fig.canvas.draw()
    stem = output_dir / "fig8_performance_over_time"
    # A portable plot-area check; the installed figure audit also checks labels/gutters.
    boxes = [ax.get_position().bounds for ax in axes]
    assert max(b[0] for b in boxes) - min(b[0] for b in boxes) < 1e-8
    assert max(b[2] for b in boxes) - min(b[2] for b in boxes) < 1e-8
    assert max(b[3] for b in boxes) - min(b[3] for b in boxes) < 1e-8
    alignment_script = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts" / "audit_panel_alignment.py"
    if alignment_script.exists():
        spec = importlib.util.spec_from_file_location("_temporal_panel_audit", alignment_script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.require_matplotlib_panel_alignment(
            fig, json_out=stem.with_suffix(".alignment.json"),
            overlay_svg=stem.with_suffix(".alignment.svg"), tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    else:
        stem.with_suffix(".alignment.json").write_text(json.dumps(
            {"status": "plot_area_check_only", "axes_bounds": boxes}), encoding="utf-8")
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")
