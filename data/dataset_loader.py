"""Load one fixed Dataset_N CSV without altering or augmenting requests."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from config import ExperimentConfig
from simulator.request import RequestSpec


EXPECTED_FIELDS = (
    "Task_ID",
    "Model",
    "Num_Subtasks",
    "Subtask_GFLOPs",
    "Activation_Size",
    "Total_GFLOPs",
    "Total_Activation",
)
TOTAL_TOLERANCE = 0.002  # Source numbers are printed to three decimals.


def _parse_finite_float(value: str, field: str, row_number: int, *, allow_zero: bool = False) -> float:
    """Parse a positive finite source value with a useful row error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"row {row_number}: {field} must be finite and {qualifier}")
    return result


def _parse_sequence(value: str, field: str, row_number: int, *, allow_zero: bool = False) -> tuple[float, ...]:
    """Parse source semicolon-separated workloads without resampling them."""

    if not value:
        raise ValueError(f"row {row_number}: {field} is empty")
    return tuple(
        _parse_finite_float(item.strip(), field, row_number, allow_zero=allow_zero)
        for item in value.split(";")
    )


def load_dataset(n: int, config: ExperimentConfig) -> list[RequestSpec]:
    """Return exactly the ``n`` requests in the actual ``Dataset_n.csv``.

    The dataset determines *which* requests exist.  An independent arrival
    scheduler assigns slots later; this function neither samples nor creates
    requests.  CSV files remain read-only.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    dataset_path = Path(config.dataset_dir) / f"Dataset_{n}.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing independent request dataset: {dataset_path}")

    requests: list[RequestSpec] = []
    seen_ids: set[int] = set()
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(
                f"{dataset_path.name}: expected fields {EXPECTED_FIELDS}, found {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError(f"{dataset_path.name} row {row_number}: missing or extra CSV field")
            try:
                request_id = int(row["Task_ID"])
                num_subtasks = int(row["Num_Subtasks"])
            except ValueError as exc:
                raise ValueError(f"{dataset_path.name} row {row_number}: invalid integer field") from exc
            if request_id <= 0 or request_id in seen_ids:
                raise ValueError(f"{dataset_path.name} row {row_number}: invalid or repeated Task_ID")
            if num_subtasks <= 0:
                raise ValueError(f"{dataset_path.name} row {row_number}: Num_Subtasks must be positive")
            seen_ids.add(request_id)

            gflops = _parse_sequence(row["Subtask_GFLOPs"], "Subtask_GFLOPs", row_number)
            activation = _parse_sequence(
                row["Activation_Size"], "Activation_Size", row_number, allow_zero=True
            )
            if len(gflops) != num_subtasks:
                raise ValueError(f"{dataset_path.name} row {row_number}: GFLOPs count differs from Num_Subtasks")
            if len(activation) != num_subtasks - 1:
                raise ValueError(f"{dataset_path.name} row {row_number}: activation count differs from DAG edge count")

            total_gflops = _parse_finite_float(row["Total_GFLOPs"], "Total_GFLOPs", row_number)
            total_activation = _parse_finite_float(
                row["Total_Activation"], "Total_Activation", row_number, allow_zero=True
            )
            if not math.isclose(sum(gflops), total_gflops, rel_tol=0, abs_tol=TOTAL_TOLERANCE):
                raise ValueError(f"{dataset_path.name} row {row_number}: Total_GFLOPs does not match stage sum")
            if not math.isclose(sum(activation), total_activation, rel_tol=0, abs_tol=TOTAL_TOLERANCE):
                raise ValueError(f"{dataset_path.name} row {row_number}: Total_Activation does not match edge sum")

            requests.append(
                RequestSpec(
                    request_id=request_id,
                    model=row["Model"].strip(),
                    gflops=gflops,
                    activation_mb=tuple(
                        value * config.activation_mb_per_csv_unit for value in activation
                    ),
                    raw_total_gflops=total_gflops,
                    raw_total_activation=total_activation,
                    source_dataset=dataset_path.name,
                )
            )

    if len(requests) != n:
        raise ValueError(f"{dataset_path.name}: expected {n} rows, found {len(requests)}")
    if seen_ids != set(range(1, n + 1)):
        raise ValueError(f"{dataset_path.name}: Task_ID must be exactly 1..{n}")
    return requests
