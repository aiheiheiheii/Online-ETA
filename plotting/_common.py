"""CSV validation and shared figure styling."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("Random", "Compute-Greedy", "Min-Cost", "Urgency-Greedy", "Proposed")
COLORS = ("#777777", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")
MARKERS = ("o", "s", "^", "D", "*")
REQUIRED_COLUMNS = frozenset(
    {"experiment", "parameter_name", "parameter_value", "method", "metric", "mean", "std", "ci95", "n_repeats"}
)


def load_figure_rows(csv_path: Path) -> list[dict[str, object]]:
    """Load one long-format figure CSV and reject malformed numeric data."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        records: list[dict[str, object]] = []
        seen: set[tuple[str, float, str, str]] = set()
        for line_no, raw in enumerate(reader, start=2):
            try:
                x = float(raw["parameter_value"])
                mean = float(raw["mean"])
                std = float(raw["std"])
                ci95 = float(raw["ci95"])
                n_repeats = int(raw["n_repeats"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_no} has invalid numeric figure data") from exc
            if not all(isfinite(value) for value in (x, mean, std, ci95)):
                raise ValueError(f"{csv_path}:{line_no} contains NaN or infinity")
            if std < 0 or ci95 < 0 or n_repeats < 1:
                raise ValueError(f"{csv_path}:{line_no} has invalid uncertainty/count")
            method = raw["method"].strip()
            metric = raw["metric"].strip()
            experiment = raw["experiment"].strip()
            if method not in METHODS:
                raise ValueError(f"{csv_path}:{line_no} has unknown method {method!r}")
            key = (experiment, x, method, metric)
            if key in seen:
                raise ValueError(f"{csv_path}:{line_no} duplicates {key}")
            seen.add(key)
            records.append(
                {
                    "experiment": experiment,
                    "parameter_name": raw["parameter_name"].strip(),
                    "parameter_value": x,
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95": ci95,
                    "n_repeats": n_repeats,
                }
            )
    if not records:
        raise ValueError(f"{csv_path} contains no figure data")
    return records


def draw_metric(
    ax: plt.Axes,
    rows: Sequence[dict[str, object]],
    metric: str,
    ylabel: str,
    xlabel: str,
    show_ci: bool,
) -> None:
    """Draw all five policy series for one saved metric."""

    all_x = sorted({float(row["parameter_value"]) for row in rows if row["metric"] == metric})
    if not all_x:
        raise ValueError(f"figure CSV lacks metric {metric!r}")
    for method, color, marker in zip(METHODS, COLORS, MARKERS, strict=True):
        method_rows = sorted(
            (row for row in rows if row["metric"] == metric and row["method"] == method),
            key=lambda row: float(row["parameter_value"]),
        )
        if [float(row["parameter_value"]) for row in method_rows] != all_x:
            raise ValueError(f"metric {metric!r} does not have a complete {method} series")
        x = [float(row["parameter_value"]) for row in method_rows]
        y = [float(row["mean"]) for row in method_rows]
        ax.plot(x, y, marker=marker, linewidth=1.8, markersize=5, color=color, label=method)
        if show_ci:
            ci = [float(row["ci95"]) for row in method_rows]
            ax.fill_between(x, [v - c for v, c in zip(y, ci, strict=True)],
                            [v + c for v, c in zip(y, ci, strict=True)], color=color, alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(all_x)
    if all(value.is_integer() for value in all_x):
        ax.set_xticklabels([str(int(value)) for value in all_x])
    ax.grid(alpha=0.25)


def save_png_pdf(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write matching raster and vector figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
