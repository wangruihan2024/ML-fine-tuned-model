#!/usr/bin/env python3
import csv
from contextlib import suppress
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
SUMMARY = EXP_DIR / "summary.csv"
OUT = EXP_DIR / "libero_bs_ablation_summary.png"


def read_rows():
    with SUMMARY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"status"}:
                continue
            if value == "":
                row[key] = None
                continue
            with suppress(ValueError):
                row[key] = float(value)
    return rows


def vals(rows, key, completed_only=False):
    data = []
    for row in rows:
        if completed_only and row["status"] != "completed":
            continue
        data.append(row[key])
    return data


def main():
    rows = read_rows()
    completed = [r for r in rows if r["status"] == "completed"]
    all_bs = [int(r["batch_size"]) for r in rows]
    completed_bs = [int(r["batch_size"]) for r in completed]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("LIBERO SmolVLA Batch Size Ablation (1500 steps; last 1000 measured)", fontsize=14)

    ax = axes[0, 0]
    ax.plot(completed_bs, vals(completed, "log_samples_s"), marker="o", label="log samples/s")
    ax.plot(completed_bs, vals(completed, "wall_samples_s"), marker="s", label="end-to-end samples/s")
    ax.set_title("Throughput")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("samples/s")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(completed_bs, vals(completed, "mean_step_s"), marker="o", color="#ff7f0e")
    ax.set_title("Mean Step Time")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("seconds/step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    peak_mem = vals(rows, "peak_memory_mib")
    colors = ["#2ca02c" if r["status"] == "completed" else "#d62728" for r in rows]
    ax.bar([str(b) for b in all_bs], peak_mem, color=colors)
    ax.axhline(24564, linestyle="--", linewidth=1, color="black", alpha=0.7, label="RTX 4090 total")
    for i, row in enumerate(rows):
        label = row["status"]
        ax.text(i, (row["peak_memory_mib"] or 0) + 350, label, ha="center", fontsize=9)
    ax.set_title("Peak VRAM")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("MiB")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    data_s = vals(completed, "mean_data_s")
    update_s = vals(completed, "mean_update_s")
    ax.bar([str(b) for b in completed_bs], data_s, label="data_s")
    ax.bar([str(b) for b in completed_bs], update_s, bottom=data_s, label="updt_s")
    ax.set_title("Logged Step Components")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("seconds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
