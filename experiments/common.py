"""Fair replicate execution and auditable CSV export for every experiment.

One Scenario is created per (experiment, parameter value, repeat) and passed
unchanged to all five policies.  All result data derive from MECSimulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from config import ExperimentConfig
from methods.compute_greedy import ComputeGreedy
from methods.min_cost import MinCost
from methods.proposed import Proposed
from methods.random_method import RandomMethod
from methods.urgency_greedy import UrgencyGreedy
from metrics.effectiveness import summarize
from simulator.environment import MECSimulator, Scenario, create_scenario


EFFECTIVENESS_METRICS = (
    "on_time_completed",
    "average_execution_cost",
    "average_e2e_latency",
    "p95_e2e_latency",
)
EFFICIENCY_METRICS = ("mean_decision_time", "p95_decision_time")
ALL_SUMMARY_METRICS = (
    "on_time_completed", "completion_ratio", "finished_requests",
    "assigned_subtasks", "total_execution_cost", "average_execution_cost",
    "average_execution_cost_per_request", "average_e2e_latency", "p95_e2e_latency",
    "mean_decision_time", "p95_decision_time",
)
META_COLUMNS = ("experiment", "parameter_name", "parameter_value", "repeat", "seed", "method")
FIGURE_COLUMNS = (
    "experiment", "parameter_name", "parameter_value", "method", "metric",
    "mean", "std", "ci95", "n_repeats",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """The single parameter varied by one paper figure."""

    name: str
    parameter_name: str
    config_field: str
    values: tuple[int | float, ...]
    figure_stem: str
    metrics: tuple[str, ...] = EFFECTIVENESS_METRICS

    def config_for(self, base: ExperimentConfig, value: int | float) -> ExperimentConfig:
        if value not in self.values:
            raise ValueError(f"{value} is not a declared {self.name} value")
        config = replace(base, **{self.config_field: value})
        config.validate()
        return config


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Iterable[str] | None = None) -> None:
    """Save real records with a stable header and no implicit index column."""

    if columns is None:
        records = list(rows)
        columns = tuple(dict.fromkeys(key for row in records for key in row))
        rows = records
    fieldnames = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_dir(config: ExperimentConfig, spec: ExperimentSpec, value: int | float, repeat: int) -> Path:
    value_label = str(value).replace(".", "p")
    return config.output_dir / "raw" / spec.name / f"{spec.parameter_name}_{value_label}" / f"repeat_{repeat}"


def save_scenario(scenario: Scenario, case_dir: Path) -> None:
    """Record exact request schedule, deadlines, topology, and resource trace."""

    config = scenario.config
    case_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config.dataset_dir / f"Dataset_{config.num_requests}.csv"
    manifest = {
        "seed": scenario.seed,
        "dataset_file": dataset_path.name,
        "dataset_sha256": _dataset_hash(dataset_path),
        "config": asdict(config),
        "resource_trace_seed": scenario.resource_trace.seed,
        "num_requests": len(scenario.requests),
        "num_nodes": len(scenario.network.nodes),
        "num_links": len(scenario.network.links),
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    request_rows = []
    for request in scenario.requests:
        deadline = scenario.deadlines[request.request_id]
        request_rows.append({
            "request_id": request.request_id,
            "source_dataset": request.source_dataset,
            "model": request.model,
            "arrival_slot": scenario.arrivals[request.request_id],
            "raw_reference_seconds": deadline.raw_reference_seconds,
            "base_deadline": deadline.base_seconds,
            "theta_deadline": deadline.theta,
            "deadline": deadline.final_seconds,
            "num_subtasks": request.num_subtasks,
            "subtask_gflops": request.gflops,
            "activation_mb": request.activation_mb,
            "total_gflops": request.raw_total_gflops,
            "total_activation": request.raw_total_activation,
        })
    write_csv(case_dir / "request_schedule.csv", request_rows)
    write_csv(
        case_dir / "topology.csv",
        ({"source_node": source, "destination_node": destination}
         for source, destination in scenario.network.links),
        ("source_node", "destination_node"),
    )
    write_csv(
        case_dir / "compute_trace.csv",
        ({"slot": slot, "node": node, "compute_rate_ghz": rate}
         for slot, rates in enumerate(scenario.resource_trace.compute_rates_ghz)
         for node, rate in enumerate(rates)),
        ("slot", "node", "compute_rate_ghz"),
    )
    write_csv(
        case_dir / "link_trace.csv",
        ({"slot": slot, "source_node": link[0], "destination_node": link[1],
          "bandwidth_mb_s": rate}
         for slot, rates in enumerate(scenario.resource_trace.link_rates_mb_s)
         for link, rate in sorted(rates.items())),
        ("slot", "source_node", "destination_node", "bandwidth_mb_s"),
    )


def _annotate(rows: Iterable[Mapping[str, object]], meta: Mapping[str, object]) -> list[dict[str, object]]:
    return [{**meta, **row} for row in rows]


def _check_result(request_rows: list[dict[str, object]], subtask_rows: list[dict[str, object]],
                  slot_rows: list[dict[str, object]], scenario: Scenario) -> None:
    """Reject malformed or missing rows before they enter paper CSVs."""

    if len(request_rows) != len(scenario.requests):
        raise AssertionError("A dataset request was dropped or duplicated")
    if len(slot_rows) != scenario.config.num_slots:
        raise AssertionError("A slot record was dropped or duplicated")
    expected_subtasks = sum(request.num_subtasks for request in scenario.requests)
    if len(subtask_rows) != expected_subtasks:
        raise AssertionError("A DAG subtask record was dropped or duplicated")
    for row in subtask_rows:
        node = row["assigned_node"]
        if node is not None and node not in scenario.network.nodes:
            raise AssertionError("Assignment node is outside the topology")
        assignment_slot = row["assignment_slot"]
        if assignment_slot is not None and not 0 <= assignment_slot < scenario.config.num_slots:
            raise AssertionError("Assignment slot is outside the horizon")
        if assignment_slot is not None and assignment_slot < scenario.arrivals[row["request_id"]]:
            raise AssertionError("Subtask assigned before request arrival")
    for row in slot_rows:
        for name in ("decision_time", "total_compute_backlog", "total_communication_backlog"):
            number = float(row[name])
            if not isfinite(number) or number < 0:
                raise AssertionError(f"Invalid slot {name}: {number}")


def run_case(
    spec: ExperimentSpec, value: int | float, repeat: int, base_config: ExperimentConfig
) -> list[dict[str, object]]:
    """Run five methods against one immutable, pre-generated environment."""

    config = spec.config_for(base_config, value)
    seed = config.base_seed + repeat
    scenario = create_scenario(config, seed)
    case_dir = _case_dir(config, spec, value, repeat)
    save_scenario(scenario, case_dir)
    summaries: list[dict[str, object]] = []
    methods = (
        RandomMethod(seed + 40_000), ComputeGreedy(), MinCost(),
        UrgencyGreedy(), Proposed(),
    )
    for method in methods:
        result = MECSimulator(scenario, method).run()
        request_rows = result.request_rows()
        for row in request_rows:
            row["raw_reference_seconds"] = scenario.deadlines[row["request_id"]].raw_reference_seconds
        subtask_rows = result.subtask_rows()
        slot_rows = result.slot_rows
        _check_result(request_rows, subtask_rows, slot_rows, scenario)
        meta = {
            "experiment": spec.name,
            "parameter_name": spec.parameter_name,
            "parameter_value": value,
            "repeat": repeat,
            "seed": seed,
            "method": method.name,
        }
        detailed_dir = config.output_dir / "detailed" / spec.name / case_dir.parent.name / case_dir.name
        write_csv(detailed_dir / f"{method.name}_requests.csv", _annotate(request_rows, meta))
        write_csv(detailed_dir / f"{method.name}_subtasks.csv", _annotate(subtask_rows, meta))
        write_csv(detailed_dir / f"{method.name}_slots.csv", _annotate(slot_rows, meta))
        summary = {**meta, **summarize(request_rows, subtask_rows, slot_rows)}
        if not 0 <= summary["on_time_completed"] <= summary["total_requests"]:
            raise AssertionError("On-time completion count outside valid range")
        summaries.append(summary)
        write_csv(case_dir / f"{method.name}_summary.csv", [summary])
        LOGGER.info(
            "%s value=%s repeat=%s method=%s on_time=%s/%s cost=%s latency=%s",
            spec.name, value, repeat, method.name,
            summary["on_time_completed"], summary["total_requests"],
            summary["average_execution_cost"], summary["average_e2e_latency"],
        )
    return summaries


def aggregate_summary(
    spec: ExperimentSpec, rows: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Compute mean, sample SD, and normal-approximation 95% CI from replicates."""

    records = list(rows)
    aggregate: list[dict[str, object]] = []
    for value in spec.values:
        for method in ("Random", "Compute-Greedy", "Min-Cost", "Urgency-Greedy", "Proposed"):
            group = [row for row in records if row["parameter_value"] == value and row["method"] == method]
            if not group:
                continue
            for metric in ALL_SUMMARY_METRICS:
                values = [float(row[metric]) for row in group if row[metric] is not None]
                if not values:
                    # No request completed: latency is undefined, never made up.
                    continue
                sd = stdev(values) if len(values) > 1 else 0.0
                aggregate.append({
                    "experiment": spec.name,
                    "parameter_name": spec.parameter_name,
                    "parameter_value": value,
                    "method": method,
                    "metric": metric,
                    "mean": mean(values),
                    "std": sd,
                    "ci95": 1.96 * sd / sqrt(len(values)),
                    "n_repeats": len(values),
                })
    return aggregate


def run_experiment(
    spec: ExperimentSpec, base_config: ExperimentConfig,
    values: Iterable[int | float] | None = None,
) -> tuple[Path, Path]:
    """Run one experiment group and save replicate, summary, figure CSVs."""

    selected = tuple(spec.values if values is None else values)
    if not selected:
        raise ValueError("At least one experiment value is required")
    base_config.validate()
    for subdir in ("raw", "detailed", "summary", "figure_data", "figures", "logs"):
        (base_config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    log_dir = base_config.output_dir / "logs"
    handler = logging.FileHandler(log_dir / f"{spec.name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    try:
        rows: list[dict[str, object]] = []
        replicate_path = base_config.output_dir / "summary" / f"{spec.name}_replicates.csv"
        for value in selected:
            for repeat in range(base_config.num_repeats):
                rows.extend(run_case(spec, value, repeat, base_config))
                write_csv(replicate_path, rows)
        aggregate = aggregate_summary(spec, rows)
        summary_path = base_config.output_dir / "summary" / f"{spec.name}_summary.csv"
        write_csv(summary_path, aggregate, FIGURE_COLUMNS)
        figure_rows = [row for row in aggregate if row["metric"] in spec.metrics]
        figure_path = base_config.output_dir / "figure_data" / f"{spec.figure_stem}.csv"
        write_csv(figure_path, figure_rows, FIGURE_COLUMNS)
        return replicate_path, figure_path
    except Exception:
        LOGGER.exception("Experiment %s failed", spec.name)
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
