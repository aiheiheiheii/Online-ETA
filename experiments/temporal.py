"""Fixed-scenario temporal ablation, raw records, t intervals and audits."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

from config import ExperimentConfig
from experiments.common import save_scenario, write_csv
from methods.proposed import Proposed
from methods.temporal_ablation import WithoutResourcePrice, WithoutUrgency
from metrics.effectiveness import summarize
from simulator.environment import MECSimulator, create_scenario
from simulator.queues import EPS

EXPERIMENT = "time_slots_temporal_performance"
METHODS = ("Proposed", "w/o Urgency", "w/o Resource Price")
SLUGS = dict(zip(METHODS, ("proposed", "without_urgency", "without_resource_price")))
METRICS = ("on_time_completed_in_slot", "average_current_state_cost_seconds", "decision_time_ms")
# Student t(.975, df=1..9), covering every n_valid in ten replicates.
T975 = {1: 12.7062047364321, 2: 4.30265272969614, 3: 3.18244630528426,
        4: 2.77644510519780, 5: 2.57058183563631, 6: 2.44691185114497,
        7: 2.36462425159278, 8: 2.30600413520417, 9: 2.26215716285410}
STATS_COLUMNS = ("experiment", "method", "slot", "display_slot", "metric",
                 "mean", "std", "ci95_half_width", "ci95_low", "ci95_high", "n_valid")


def temporal_config(output_dir: Path) -> ExperimentConfig:
    return replace(ExperimentConfig(), num_nodes=60, num_requests=300,
                   mean_compute_ghz=10.0, mean_bandwidth_mb_s=1.4,
                   theta_deadline=1.4, slot_seconds=2.0, num_slots=800,
                   num_repeats=10, base_seed=0, arrival_model="fixed_total_uniform",
                   arrival_start_slot=0, arrival_window_fraction=1.0,
                   arrival_end_slot=799, output_dir=output_dir)


def observation_status(row: dict, horizon_seconds: float, delta: float) -> str:
    if row["finish_slot"] is not None:
        return "on_time_finished" if row["on_time"] else "late_finished"
    absolute_deadline = int(row["arrival_slot"]) * delta + float(row["deadline"])
    return "right_censored" if absolute_deadline > horizon_seconds + EPS else "unfinished_deadline_failed"


def _fingerprint(scenario) -> str:
    h = hashlib.sha256()
    pieces = (
        tuple((r.request_id, r.model, r.gflops, r.activation_mb) for r in scenario.requests),
        tuple(sorted(scenario.arrivals.items())),
        tuple((rid, repr(deadline)) for rid, deadline in sorted(scenario.deadlines.items())),
        scenario.network.nodes, scenario.network.links,
        scenario.resource_trace.compute_rates_ghz,
        tuple(tuple(sorted(row.items())) for row in scenario.resource_trace.link_rates_mb_s),
        tuple((source, dest, scenario.network.candidate_paths(source, dest))
              for source in scenario.network.nodes for dest in scenario.network.nodes if source != dest),
    )
    for piece in pieces:
        h.update(repr(piece).encode("utf-8"))
    return h.hexdigest()


def extract_temporal_rows(result, repeat: int, fingerprint: str):
    config = result.scenario.config
    request_rows = result.request_rows()
    subtask_rows = result.subtask_rows()
    completed = Counter(int(row["finish_slot"]) - 1 for row in request_rows if row["on_time"])
    costs = defaultdict(list)
    for row in subtask_rows:
        if row["assignment_slot"] is not None:
            costs[int(row["assignment_slot"])].append(float(row["execution_cost"]))
    rows = []
    cumulative = 0
    for slot_row in result.slot_rows:
        slot = int(slot_row["slot"])
        assigned = len(costs[slot])
        if assigned != slot_row["num_assignments"]:
            raise AssertionError("Slot assignment count disagrees with subtask records")
        cumulative += completed[slot]
        rows.append({
            "experiment": EXPERIMENT, "repeat": repeat, "seed": result.scenario.seed,
            "method": result.method, "scenario_fingerprint": fingerprint,
            **slot_row, "display_slot": slot + 1,
            "execution_cost_sum_seconds": math.fsum(costs[slot]),
            "average_current_state_cost_seconds": math.fsum(costs[slot]) / assigned if assigned else None,
            "on_time_completed_in_slot": completed[slot],
            "cumulative_on_time_completed": cumulative,
            "decision_time_ms": float(slot_row["decision_time"]) * 1000,
        })
    for row in request_rows:
        row["observation_status"] = observation_status(row, config.num_slots * config.slot_seconds,
                                                      config.slot_seconds)
    return rows, request_rows, subtask_rows


def validate_records(rows, requests, subtasks, scenario):
    assert len(rows) == scenario.config.num_slots
    assert [r["slot"] for r in rows] == list(range(scenario.config.num_slots))
    assert len(requests) == scenario.config.num_requests
    assert len({r["request_id"] for r in requests}) == len(requests)
    assert [r["display_slot"] for r in rows] == list(range(1, scenario.config.num_slots + 1))
    assert sum(r["num_arrivals"] for r in rows) == scenario.config.num_requests
    assert all(0 <= slot <= 799 for slot in scenario.arrivals.values())
    assert sum(r["on_time_completed_in_slot"] for r in rows) == sum(r["on_time"] for r in requests)
    assigned = [r for r in subtasks if r["assignment_slot"] is not None]
    assert sum(r["num_assignments"] for r in rows) == len(assigned)
    assert math.isclose(math.fsum(r["execution_cost_sum_seconds"] for r in rows),
                        math.fsum(float(r["execution_cost"]) for r in assigned), rel_tol=1e-12, abs_tol=1e-9)
    finishes = Counter(int(r["finish_slot"]) - 1 for r in requests if r["on_time"])
    for r in rows:
        assert r["on_time_completed_in_slot"] == finishes[r["slot"]]
        assert r["decision_time_ms"] >= 0 and math.isfinite(r["decision_time_ms"])
        assert r["num_assignments"] <= r["num_ready_subtasks"]
        if r["num_assignments"] == 0:
            assert r["average_current_state_cost_seconds"] is None
        else:
            assert math.isclose(r["average_current_state_cost_seconds"],
                                r["execution_cost_sum_seconds"] / r["num_assignments"], rel_tol=1e-12)
    for r in assigned:
        elapsed = (int(r["assignment_slot"]) - scenario.arrivals[r["request_id"]]) * scenario.config.slot_seconds
        assert scenario.deadlines[r["request_id"]].final_seconds - elapsed >= 0
    for r in requests:
        if r["finish_slot"] is not None:
            assert r["e2e_latency"] == (r["finish_slot"] - r["arrival_slot"]) * scenario.config.slot_seconds
        assert r["observation_status"] == observation_status(
            r, scenario.config.num_slots * scenario.config.slot_seconds, scenario.config.slot_seconds)


def aggregate_slots(records, seeds):
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("This protocol requires ten distinct seeds")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], int(row["slot"]))].append(row)
    stats = []
    for method in METHODS:
        for slot in range(800):
            group = grouped[method, slot]
            assert len(group) == 10 and {int(r["seed"]) for r in group} == set(seeds)
            for metric in METRICS:
                values = [float(r[metric]) for r in group if r[metric] is not None]
                n = len(values)
                average = mean(values) if n else None
                sd = stdev(values) if n > 1 else None
                ci = T975[n - 1] * sd / math.sqrt(n) if n > 1 else None
                stats.append({
                    "experiment": EXPERIMENT, "method": method, "slot": slot, "display_slot": slot + 1,
                    "metric": metric, "mean": average, "std": sd, "ci95_half_width": ci,
                    "ci95_low": average - ci if ci is not None else None,
                    "ci95_high": average + ci if ci is not None else None, "n_valid": n,
                })
    return stats


def read_slot_records(output_dir):
    records = []
    for path in sorted((output_dir / "detailed").glob("repeat_*/*_slots.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                for name in METRICS:
                    raw[name] = float(raw[name]) if raw[name] else None
                for name in ("repeat", "seed", "slot", "display_slot", "num_arrivals",
                             "num_assignments", "num_ready_subtasks", "cumulative_on_time_completed"):
                    raw[name] = int(raw[name])
                raw["execution_cost_sum_seconds"] = float(raw["execution_cost_sum_seconds"])
                records.append(raw)
    return records


def refresh_statistics(output_dir: Path):
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    records = read_slot_records(output_dir)
    stats = aggregate_slots(records, manifest["seeds"])
    write_csv(output_dir / "summary" / "temporal_slot_statistics.csv", stats, STATS_COLUMNS)
    write_csv(output_dir / "figure_data" / "fig8_performance_over_time.csv", stats, STATS_COLUMNS)
    return records, stats


def run_temporal(output_dir: Path, seeds):
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Provide ten distinct integer seeds")
    if (output_dir / "experiment_manifest.json").exists():
        raise FileExistsError("Results already exist; use --plot-only or a new --output-dir")
    config = temporal_config(output_dir)
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": EXPERIMENT, "seeds": list(seeds), "config": asdict(config),
        "methods": METHODS, "arrival_definition": "300 unique requests, iid uniform integer slots 0..799",
        "expiry_rule": "tau<0: no new assignment; tau=0 allowed; queued work continues",
        "time_axis": "internal slot 0..799, display slot 1..800; finish boundary slot+1",
        "statistics": "pointwise mean +/- Student t(.975,n_valid-1)*sample SD/sqrt(n_valid)",
        "resource_price_ablation": "zero prices; retained temporary updates do not enter actual-state C",
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "processor": platform.processor(), "machine": platform.machine()},
        "dataset_dag": "sequential DAG supplied by the dataset",
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summaries = []
    for repeat, seed in enumerate(seeds):
        scenario = create_scenario(config, seed)
        fingerprint = _fingerprint(scenario)
        common_dir = output_dir / "raw" / f"repeat_{repeat:02d}"
        save_scenario(scenario, common_dir)
        candidate_paths = [
            {"source": src, "destination": dst, "paths": scenario.network.candidate_paths(src, dst)}
            for src in scenario.network.nodes for dst in scenario.network.nodes if src != dst
        ]
        (common_dir / "candidate_paths.json").write_text(json.dumps(candidate_paths), encoding="utf-8")
        scenario_files = sorted(p for p in common_dir.iterdir() if p.is_file())
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scenario_files}
        (common_dir / "scenario_hashes.json").write_text(
            json.dumps({"fingerprint": fingerprint, "files": hashes}, indent=2), encoding="utf-8")
        write_csv(common_dir / "arrivals_by_slot.csv",
                  ({"slot": slot, "display_slot": slot + 1,
                    "num_arrivals": sum(t == slot for t in scenario.arrivals.values())}
                   for slot in range(config.num_slots)))
        for method in (Proposed(), WithoutUrgency(), WithoutResourcePrice()):
            print(f"repeat={repeat} seed={seed} method={method.name}", flush=True)
            result = MECSimulator(scenario, method, unexpired_only=True).run()
            assert result.scenario is scenario and _fingerprint(scenario) == fingerprint
            rows, requests, subtasks = extract_temporal_rows(result, repeat, fingerprint)
            validate_records(rows, requests, subtasks, scenario)
            meta = {"experiment": EXPERIMENT, "repeat": repeat, "seed": seed,
                    "method": method.name, "scenario_fingerprint": fingerprint}
            detailed = output_dir / "detailed" / f"repeat_{repeat:02d}"
            slug = SLUGS[method.name]
            write_csv(detailed / f"{slug}_slots.csv", rows)
            write_csv(detailed / f"{slug}_requests.csv", ({**meta, **r} for r in requests))
            write_csv(detailed / f"{slug}_subtasks.csv", ({**meta, **r} for r in subtasks))
            summary = {**meta, **summarize(requests, subtasks, result.slot_rows)}
            statuses = Counter(r["observation_status"] for r in requests)
            summary.update({status: statuses[status] for status in (
                "on_time_finished", "late_finished", "unfinished_deadline_failed", "right_censored")})
            assert sum(statuses.values()) == 300
            summary["observed_deadline_failures"] = statuses["late_finished"] + statuses["unfinished_deadline_failed"]
            assert summary["on_time_completed"] == sum(r["on_time_completed_in_slot"] for r in rows)
            summaries.append(summary)
            write_csv(output_dir / "summary" / "temporal_replicates.csv", summaries)
            print(f"  on_time={summary['on_time_completed']} assigned={summary['assigned_subtasks']}"
                  f" right_censored={summary['right_censored']}", flush=True)
    records, stats = refresh_statistics(output_dir)
    write_report(output_dir, summaries)
    verify_saved(output_dir)
    return output_dir


def verify_saved(output_dir: Path):
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    records = read_slot_records(output_dir)
    recomputed = aggregate_slots(records, manifest["seeds"])
    with (output_dir / "figure_data" / "fig8_performance_over_time.csv").open(encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(recomputed) == 7200
    for actual, expected in zip(saved, recomputed):
        for key in STATS_COLUMNS:
            if expected[key] is None:
                assert actual[key] == ""
            elif isinstance(expected[key], (int, float)):
                assert math.isclose(float(actual[key]), expected[key], rel_tol=1e-12, abs_tol=1e-12)
            else:
                assert actual[key] == expected[key]
    with (output_dir / "summary" / "temporal_replicates.csv").open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 30
    for repeat in range(10):
        common = output_dir / "raw" / f"repeat_{repeat:02d}"
        hashes = json.loads((common / "scenario_hashes.json").read_text(encoding="utf-8"))
        for filename, digest in hashes["files"].items():
            assert hashlib.sha256((common / filename).read_bytes()).hexdigest() == digest
        with (common / "request_schedule.csv").open(encoding="utf-8") as handle:
            schedule = list(csv.DictReader(handle))
        assert len(schedule) == 300 and len({r["request_id"] for r in schedule}) == 300
        assert all(0 <= int(r["arrival_slot"]) < 800 for r in schedule)
        arrivals = Counter(int(r["arrival_slot"]) for r in schedule)
        schedule_by_id = {r["request_id"]: r for r in schedule}
        for method in METHODS:
            group = [r for r in records if r["repeat"] == repeat and r["method"] == method]
            assert len(group) == 800
            assert {r["scenario_fingerprint"] for r in group} == {hashes["fingerprint"]}
            assert all(r["num_arrivals"] == arrivals[r["slot"]] for r in group)
            summary = next(r for r in summary_rows if int(r["repeat"]) == repeat and r["method"] == method)
            detailed = output_dir / "detailed" / f"repeat_{repeat:02d}"
            with (detailed / f"{SLUGS[method]}_requests.csv").open(encoding="utf-8") as handle:
                requests = list(csv.DictReader(handle))
            with (detailed / f"{SLUGS[method]}_subtasks.csv").open(encoding="utf-8") as handle:
                subtasks = list(csv.DictReader(handle))
            assert len(requests) == 300 and {r['request_id'] for r in requests} == set(schedule_by_id)
            subtask_by_id = {(r['request_id'], int(r['subtask_id'])): r for r in subtasks}
            assert len(subtask_by_id) == len(subtasks) == sum(int(r['num_subtasks']) for r in requests)
            completions, assignments = Counter(), Counter()
            costs = defaultdict(list)
            for r in requests:
                arrival, deadline = int(r['arrival_slot']), float(r['deadline'])
                assert arrival == int(schedule_by_id[r['request_id']]['arrival_slot'])
                assert deadline == float(schedule_by_id[r['request_id']]['deadline'])
                terminal = subtask_by_id[r['request_id'], int(r['num_subtasks'])]['compute_finish']
                assert r['finish_slot'] == terminal
                finished = int(terminal) if terminal else None
                latency = (finished - arrival) * 2.0 if finished is not None else None
                on_time = latency is not None and latency <= deadline + EPS
                assert (r['on_time'] == 'True') == on_time
                if finished is not None:
                    assert 1 <= finished <= 800 and float(r['e2e_latency']) == latency
                    assert ((finished + 1) - (arrival + 1)) * 2.0 == latency
                    assert all(subtask_by_id[r['request_id'], k]['compute_finish']
                               for k in range(1, int(r['num_subtasks']) + 1))
                assert r['observation_status'] == observation_status(
                    {'arrival_slot': arrival, 'deadline': deadline,
                     'finish_slot': finished, 'on_time': on_time}, 1600.0, 2.0)
                if on_time:
                    completions[finished - 1] += 1
            for r in subtasks:
                if not r['assignment_slot']:
                    continue
                slot = int(r['assignment_slot'])
                assert 0 <= slot < 800 and int(r['ready_slot']) <= slot
                source = schedule_by_id[r['request_id']]
                assert float(source['deadline']) - (slot - int(source['arrival_slot'])) * 2.0 >= 0
                assignments[slot] += 1
                costs[slot].append(float(r['execution_cost']))
                if method == 'w/o Urgency':
                    assert r['urgency'] == ''
                if method == 'w/o Resource Price':
                    assert float(r['lambda']) == float(r['mu']) == 0.0
            for r in group:
                assert r['display_slot'] == r['slot'] + 1
                assert r['on_time_completed_in_slot'] == completions[r['slot']]
                assert r['num_assignments'] == assignments[r['slot']]
                assert math.isclose(r['execution_cost_sum_seconds'], math.fsum(costs[r['slot']]),
                                    rel_tol=1e-12, abs_tol=1e-9)
                if not assignments[r['slot']]:
                    assert r['average_current_state_cost_seconds'] is None
                else:
                    assert math.isclose(r['average_current_state_cost_seconds'],
                                        math.fsum(costs[r['slot']]) / assignments[r['slot']], rel_tol=1e-12)
            assert sum(r["on_time_completed_in_slot"] for r in group) == int(summary["on_time_completed"])
            assert sum(r["num_assignments"] for r in group) == int(summary["assigned_subtasks"])
            assert math.isclose(math.fsum(r["execution_cost_sum_seconds"] for r in group),
                                float(summary["total_execution_cost"]), rel_tol=1e-12, abs_tol=1e-9)
            assert sum(int(summary[k]) for k in (
                "on_time_finished", "late_finished", "unfinished_deadline_failed", "right_censored")) == 300
    audit = {"status": "PASS", "seeds": manifest["seeds"], "replicates": 30,
             "slot_rows": len(records), "statistics_rows": len(saved),
             "checks": ["shared scenario fingerprints and saved input hashes", "unique full-horizon arrivals",
                        "800 slots per policy/seed", "completion/assignment/cost reconciliation",
                        "recomputed pointwise t intervals and missing values", "mutually exclusive observation statuses",
                        "terminal subtask completion and unchanged displayed-time latency",
                        "shared expiry rule and removed urgency/zero price records"]}
    (output_dir / "validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def refresh_report(output_dir: Path):
    with (output_dir / "summary" / "temporal_replicates.csv").open(encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    for r in summaries:
        for key in ("repeat", "seed"):
            r[key] = int(r[key])
        for key in ("on_time_completed", "observed_deadline_failures", "right_censored",
                    "average_execution_cost", "mean_decision_time"):
            r[key] = float(r[key])
    write_report(output_dir, summaries)


def write_report(output_dir, summaries):
    lines = ["# 4.3 Performance over Time", "",
             "本组使用300个唯一请求，在全部800槽内均匀随机到达；每槽2秒，"
             "60节点，平均10 GHz/1.4 MB/s，deadline factor=1.4。"
             "三个方法共享同一随机场景，每个方法执行10个种子。",
             "", "## 周期结果（跨种子均值）", "",
             "| 方法 | 按时完成 | 已知deadline失败 | 观察截止 | 已分配子任务平均代价(s) | 每槽决策时间(ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    groups = {m: [r for r in summaries if r["method"] == m] for m in METHODS}
    for method, group in groups.items():
        lines.append(f"| {method} | {mean(r['on_time_completed'] for r in group):.2f} | "
                     f"{mean(r['observed_deadline_failures'] for r in group):.2f} | "
                     f"{mean(r['right_censored'] for r in group):.2f} | "
                     f"{mean(r['average_execution_cost'] for r in group):.4f} | "
                     f"{mean(r['mean_decision_time'] for r in group)*1000:.4f} |")
    lines.extend(["", "## 消融比较（同种子配对差值：Proposed减消融）", ""])
    paired = []
    for variant in METHODS[1:]:
        for metric in ("on_time_completed", "average_execution_cost", "mean_decision_time"):
            diffs = []
            for seed in range(10):
                full = groups["Proposed"][seed]
                other = groups[variant][seed]
                assert full["seed"] == other["seed"]
                diffs.append(float(full[metric]) - float(other[metric]))
            factor = 1000 if metric == "mean_decision_time" else 1
            diffs = [x * factor for x in diffs]
            avg = mean(diffs)
            sd = stdev(diffs)
            half = T975[9] * sd / math.sqrt(10)
            paired.append({"comparison": f"Proposed - {variant}", "metric": metric,
                           "mean_difference": avg, "ci95_low": avg-half,
                           "ci95_high": avg+half, "n_pairs": 10})
            lines.append(f"- {variant}，{metric}：平均差值 {avg:.6g}，"
                         f"95%配对t区间 [{avg-half:.6g}, {avg+half:.6g}]。"
                         + (" 决策时间差值单位为ms。" if metric == "mean_decision_time" else ""))
    write_csv(output_dir / "summary" / "paired_differences.csv", paired)
    lines.extend(["", "## 指标与解读", "",
                  "- 图(a)为逐槽新完成且满足deadline的完整请求数；周期按时总数由800槽相加得到。",
                  "- 图(b)使用真实槽初C，按本槽分配数归一化；无分配时保留缺口，"
                  "跨重复均值仅使用该槽实际分配的重复，n_valid随槽变化。",
                  "- 图(c)为实际决策时间；场景生成、统计、文件写入和物理服务不计入。"
                  "运行时间受当前硬件与系统负载影响。",
                  "- 价格消融保留临时更新，但其节点评分只用真实C；无紧迫度版本保留价格与逐次更新。",
                  "- 三方法均采用共同过期规则；已入队工作继续服务。",
                  "- 按时数量相同或相近时再结合执行代价评价第二目标，不能仅凭较低代价判定整体优越。",
                  "- 结论限于该固定到达强度、资源波动及800槽窗口。"
                  "临近观察终点且deadline尚未到达的未完成请求单独标为观察截止。",
                  "- 当前数据为链式DAG，结果不扩展到未经测试的分支/汇合结构。", ""])
    for variant in METHODS[1:]:
        same = sum(
            next(r for r in summaries if r["method"] == "Proposed" and r["repeat"] == rep)["on_time_completed"]
            == next(r for r in summaries if r["method"] == variant and r["repeat"] == rep)["on_time_completed"]
            for rep in range(10))
        lines.append(f"Proposed与{variant}在{same}/10个种子的周期按时完成数量相同。")
    lines.extend(["", "## 基于本次数据的机制比较", ""])
    full = groups["Proposed"]
    urgency = groups["w/o Urgency"]
    price = groups["w/o Resource Price"]
    def difference(variant, metric):
        return next(r for r in paired if r["comparison"] == f"Proposed - {variant}" and r["metric"] == metric)
    u_count = difference("w/o Urgency", "on_time_completed")
    u_cost = difference("w/o Urgency", "average_execution_cost")
    p_count = difference("w/o Resource Price", "on_time_completed")
    p_cost = difference("w/o Resource Price", "average_execution_cost")
    lines.append(f"**紧迫度排序。** Proposed每个周期平均多按时完成{u_count['mean_difference']:.2f}个请求，"
                 f"但配对95%区间[{u_count['ci95_low']:.3f}, {u_count['ci95_high']:.3f}]包含0，"
                 "本组未给出稳定提升按时完成数量的证据。"
                 f"已分配子任务的周期平均当前态代价降低{-u_cost['mean_difference']:.4f} s，"
                 f"对应差值区间[{u_cost['ci95_low']:.4f}, {u_cost['ci95_high']:.4f}]。"
                 "周期代价按各次运行的全部已分配子任务求平均，与图(b)逐槽条件均值的平均不同。")
    if mean(r['observed_deadline_failures'] for r in full + urgency) == 0:
        lines.append("两者均没有已确定的deadline失败；按时完成数量的差别对应观察终点的未完成请求数，"
                     "不能将观察截止请求解释为超时失败。")
    lines.append(f"**资源价格。** 相比w/o Resource Price，Proposed平均多按时完成"
                 f"{p_count['mean_difference']:.2f}个请求，配对95%区间"
                 f"[{p_count['ci95_low']:.3f}, {p_count['ci95_high']:.3f}]。"
                 f"已知deadline失败由平均{mean(r['observed_deadline_failures'] for r in price):.2f}个"
                 f"变为{mean(r['observed_deadline_failures'] for r in full):.2f}个。"
                 f"平均代价差为{p_cost['mean_difference']:.4f} s，区间"
                 f"[{p_cost['ci95_low']:.4f}, {p_cost['ci95_high']:.4f}]包含0。"
                 "本次数据支持其按时完成能力的有限改善，没有证据表明资源价格降低了平均执行代价。")
    u_time = difference("w/o Urgency", "mean_decision_time")
    p_time = difference("w/o Resource Price", "mean_decision_time")
    lines.append(f"**在线开销。** 三方法每槽平均决策时间分别为"
                 f"{mean(r['mean_decision_time'] for r in full)*1000:.4f}、"
                 f"{mean(r['mean_decision_time'] for r in urgency)*1000:.4f}、"
                 f"{mean(r['mean_decision_time'] for r in price)*1000:.4f} ms。"
                 f"去除紧迫度和资源价格后的平均时间差分别为{u_time['mean_difference']:.4f}、"
                 f"{p_time['mean_difference']:.4f} ms；紧迫度时间差区间包含0，"
                 "价格计算的时间差区间为正。该时间来自本机顺序运行的实际测量，"
                 "保留空槽开销及偶发运行时间峰值，不作平滑或异常值删除。")
    lines.append("平均到达强度为300/800=0.375请求/槽（0.1875请求/秒）。"
                 "持续运行结论仅适用于该到达强度、当前资源波动设置和1600秒观察窗口。")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
