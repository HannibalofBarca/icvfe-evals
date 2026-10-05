"""
Runs b1_full_run's run_task() over an explicit list of task names (rather
than a round-robin shard slice) -- used to re-offload the leftover/unfinished
tasks from a previous sharded run onto fresh, finer-grained shards.

Usage:
    python b1_full_run_tasklist.py <output_suffix> <task name> [<task name> ...]
    # writes output/b1_full_run/b1_full_results.shard<output_suffix>.csv
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

from b_metrics import DinoIndex
from b1_retrieval import load_manual
from b1_value_sources import load_all_value_sources
from b1_full_run import run_task

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "b1_full_run")


def main():
    suffix = sys.argv[1]
    my_tasks = sys.argv[2:]

    manual = load_manual()
    dino = DinoIndex()
    value_sources = load_all_value_sources()
    print(f"shard {suffix}: {len(my_tasks)} tasks: {my_tasks}", flush=True)

    all_rows = []
    t0 = time.time()
    for i, task in enumerate(my_tasks):
        rows = run_task(task, manual, dino, value_sources)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"[shard {suffix}] [{i+1}/{len(my_tasks)}] {task!r}: {len(rows)} rows "
              f"(total {len(all_rows)}, {elapsed:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"b1_full_results.shard{suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"shard {suffix}: wrote {out_path} ({len(df)} rows), {time.time()-t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
