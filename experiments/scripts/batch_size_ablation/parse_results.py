#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


BATCHES = [16, 24, 32, 48]
LOG_FREQ = 50
WARMUP_STEPS = 500
TOTAL_STEPS = 1500


METRIC_RE = re.compile(
    r"INFO (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"epch:(?P<epoch>[0-9.]+)\s+"
    r"loss:(?P<loss>[0-9.]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.]+)\s+"
    r"data_s:(?P<data_s>[0-9.]+)"
)


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def percentile(values, q):
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(" MiB", "").replace(" W", "").replace("%", "")
    if text in {"", "[Not Supported]", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_dt(value):
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def parse_train_log(run_dir, batch_size):
    log_path = run_dir / "train.log"
    rows = []
    if not log_path.exists():
        return rows

    metric_idx = 0
    for line in log_path.read_text(errors="ignore").replace("\r", "\n").splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        metric_idx += 1
        exact_step = metric_idx * LOG_FREQ
        data_s = float(match.group("data_s"))
        update_s = float(match.group("update_s"))
        step_s = data_s + update_s
        rows.append(
            {
                "timestamp": match.group("ts"),
                "exact_step": exact_step,
                "measurement": exact_step > WARMUP_STEPS,
                "epoch": float(match.group("epoch")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "learning_rate": float(match.group("lr")),
                "update_s": update_s,
                "data_s": data_s,
                "step_s": step_s,
                "samples_s": batch_size / step_s if step_s > 0 else None,
            }
        )
    return rows


def parse_gpu_csv(run_dir, start_dt=None, end_dt=None):
    gpu_path = run_dir / "gpu.csv"
    if not gpu_path.exists():
        return []

    rows = []
    with gpu_path.open(newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            normalized = {k.strip(): v for k, v in raw.items() if k is not None}
            ts_key = next((k for k in normalized if k.startswith("timestamp")), None)
            util_key = next((k for k in normalized if "utilization.gpu" in k), None)
            mem_used_key = next((k for k in normalized if "memory.used" in k), None)
            mem_total_key = next((k for k in normalized if "memory.total" in k), None)
            power_key = next((k for k in normalized if "power.draw" in k), None)
            ts = parse_dt(normalized.get(ts_key)) if ts_key else None
            if start_dt and ts and ts < start_dt:
                continue
            if end_dt and ts and ts > end_dt:
                continue
            rows.append(
                {
                    "timestamp": normalized.get(ts_key) if ts_key else "",
                    "gpu_util": parse_float(normalized.get(util_key)) if util_key else None,
                    "memory_used_mib": parse_float(normalized.get(mem_used_key)) if mem_used_key else None,
                    "memory_total_mib": parse_float(normalized.get(mem_total_key)) if mem_total_key else None,
                    "power_w": parse_float(normalized.get(power_key)) if power_key else None,
                }
            )
    return rows


def read_run_meta(run_dir):
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    summary_rows = []

    for batch_size in BATCHES:
        run_dir = exp_dir / f"bs{batch_size}"
        metric_rows = parse_train_log(run_dir, batch_size)
        if metric_rows:
            write_csv(
                run_dir / "metrics.csv",
                metric_rows,
                [
                    "timestamp",
                    "exact_step",
                    "measurement",
                    "epoch",
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "update_s",
                    "data_s",
                    "step_s",
                    "samples_s",
                ],
            )

        measurement_rows = [r for r in metric_rows if r["measurement"]]
        start_dt = parse_dt(measurement_rows[0]["timestamp"]) if measurement_rows else None
        end_dt = parse_dt(measurement_rows[-1]["timestamp"]) if measurement_rows else None
        gpu_rows = parse_gpu_csv(run_dir, start_dt=start_dt, end_dt=end_dt)
        meta = read_run_meta(run_dir)

        wall_seconds = meta.get("wall_seconds")
        wall_samples_s = None
        if isinstance(wall_seconds, (int, float)) and wall_seconds > 0:
            wall_samples_s = batch_size * TOTAL_STEPS / wall_seconds

        status = meta.get("status", "missing")
        if status == "failed" and (run_dir / "train.log").exists():
            text = (run_dir / "train.log").read_text(errors="ignore")
            if "out of memory" in text.lower():
                status = "oom"

        summary_rows.append(
            {
                "batch_size": batch_size,
                "status": status,
                "return_code": meta.get("return_code", ""),
                "total_steps": TOTAL_STEPS,
                "warmup_steps": WARMUP_STEPS,
                "measurement_steps": max(0, TOTAL_STEPS - WARMUP_STEPS),
                "metric_points": len(metric_rows),
                "measurement_points": len(measurement_rows),
                "wall_seconds": wall_seconds if wall_seconds is not None else "",
                "wall_samples_s": wall_samples_s if wall_samples_s is not None else "",
                "mean_step_s": mean(r["step_s"] for r in measurement_rows),
                "mean_data_s": mean(r["data_s"] for r in measurement_rows),
                "mean_update_s": mean(r["update_s"] for r in measurement_rows),
                "log_samples_s": mean(r["samples_s"] for r in measurement_rows),
                "final_loss": metric_rows[-1]["loss"] if metric_rows else "",
                "final_grad_norm": metric_rows[-1]["grad_norm"] if metric_rows else "",
                "final_lr": metric_rows[-1]["learning_rate"] if metric_rows else "",
                "mean_gpu_util": mean(r["gpu_util"] for r in gpu_rows),
                "p95_gpu_util": percentile([r["gpu_util"] for r in gpu_rows], 0.95),
                "peak_memory_mib": max([r["memory_used_mib"] for r in gpu_rows if r["memory_used_mib"] is not None], default=None),
                "mean_power_w": mean(r["power_w"] for r in gpu_rows),
                "gpu_samples": len(gpu_rows),
            }
        )

    write_csv(
        exp_dir / "summary.csv",
        summary_rows,
        [
            "batch_size",
            "status",
            "return_code",
            "total_steps",
            "warmup_steps",
            "measurement_steps",
            "metric_points",
            "measurement_points",
            "wall_seconds",
            "wall_samples_s",
            "mean_step_s",
            "mean_data_s",
            "mean_update_s",
            "log_samples_s",
            "final_loss",
            "final_grad_norm",
            "final_lr",
            "mean_gpu_util",
            "p95_gpu_util",
            "peak_memory_mib",
            "mean_power_w",
            "gpu_samples",
        ],
    )
    print(exp_dir / "summary.csv")


if __name__ == "__main__":
    main()
