"""
Sharded variant of b1_full_run.py -- splits the 27-task list into N shards so
multiple shards can run as separate processes in parallel, each writing its
own partial CSVs. Run b1_full_run_merge.py afterward to combine them and
regenerate the summary CSVs.

Usage:
    python b1_full_run_shard.py <shard_id> <num_shards>
    # e.g. six shards: shard ids 0..5
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
    shard_id = int(sys.argv[1])
    num_shards = int(sys.argv[2])

    manual = load_manual()
    dino = DinoIndex()
    value_sources = load_all_value_sources()
    tasks = sorted({e["task"] for e in manual.values()})
    my_tasks = tasks[shard_id::num_shards]
    print(f"shard {shard_id}/{num_shards}: {len(my_tasks)} tasks: {my_tasks}", flush=True)

    all_rows = []
    t0 = time.time()
    for i, task in enumerate(my_tasks):
        rows = run_task(task, manual, dino, value_sources)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"[shard {shard_id}] [{i+1}/{len(my_tasks)}] {task!r}: {len(rows)} rows "
              f"(total {len(all_rows)}, {elapsed:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"b1_full_results.shard{shard_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"shard {shard_id}: wrote {out_path} ({len(df)} rows), {time.time()-t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
