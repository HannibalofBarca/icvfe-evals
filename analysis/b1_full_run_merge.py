"""
Merges every b1_full_results.shard*.csv in output/b1_full_run/ (whatever
shard-naming scheme produced them -- plain indices, or offloaded-task
suffixes like "1b") into the same outputs b1_full_run.py itself would have
produced: b1_full_results.csv, b1_method_summary.csv, b1_task_method_summary.csv.

Usage:
    python b1_full_run_merge.py
"""
from __future__ import annotations

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "b1_full_run")

METRIC_COLS = ["ttc_error", "video_chunk_error", "bc_error", "naive_dino_error"]


def main():
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "b1_full_results.shard*.csv")))
    if not paths:
        raise SystemExit(f"no shard CSVs found in {OUTPUT_DIR}")

    parts = []
    for path in paths:
        part = pd.read_csv(path)
        parts.append(part)
        print(f"{os.path.basename(path)}: {len(part)} rows, tasks={sorted(part['task'].unique())}")

    df = pd.concat(parts, ignore_index=True)
    n_tasks = df["task"].nunique()
    print(f"\n{n_tasks} distinct tasks across {len(paths)} shard files, {len(df)} rows total")

    out_path = os.path.join(OUTPUT_DIR, "b1_full_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    summary = df.groupby("method")[METRIC_COLS].mean()
    print("\nMean metric by method (all tasks pooled):")
    print(summary.to_string())
    summary_path = os.path.join(OUTPUT_DIR, "b1_method_summary.csv")
    summary.to_csv(summary_path)

    task_method_summary = df.groupby(["task", "method"])[METRIC_COLS].mean()
    task_method_path = os.path.join(OUTPUT_DIR, "b1_task_method_summary.csv")
    task_method_summary.to_csv(task_method_path)
    print(f"Wrote {summary_path} and {task_method_path}")


if __name__ == "__main__":
    main()
