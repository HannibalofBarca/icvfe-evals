"""
B2: lambda sweep on the vision-value retrieval mixture.

Per Eval_Research_Summary.md's ablation table:
  variants: lambda in {0, 0.25, 0.5, 0.75, 1.0} over the same online-eligible
            pool from B1 (no RoboDopamine)
  metrics:  videoDTW error, TTC, DINO error, BC error on the retrieved chunk
  goal:     "find the best mixing weight" -- same metric suite as B1, applied
            across the sweep

Retrieval score for a given lambda, value source, query frame, and ep1
candidate start frame:
    combined = lambda * value_dist + (1 - lambda) * vision_dist
where value_dist = |query_value - candidate_value| (raw scale, same source
used by that value method) and vision_dist = Euclidean DINO embedding
distance -- both RAW, unweighted (no z-scoring), matching the "naive"
unweighted-sum convention already used for B1's naive_vision_value_* methods
(lambda=0.5 here reproduces those exactly; lambda=1.0 reproduces B1's plain
value_<source> retrieval; lambda=0.0 is plain vision-only retrieval, but
restricted to the same per-source query coverage subset used at the other
lambdas, for a fair sweep -- NOT the same population as B1's pooled "vision"
row, which spans all 27 tasks).

RoboDopamine is excluded from the value-source roster per the doc ("no
RoboDopamine") -- it's reused elsewhere in B1 purely as an upper-ceiling
value-correlation reference, not as an online-eligible retrieval key.

Value-source roster restricted to icvfe_8800 and icvfe_ema_0.5 per user
direction (RoboMeter FT online and RECAP FT, also online-eligible per B1,
are excluded from this sweep).

Usage:
    python b2_lambda_sweep.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from b_metrics import DinoIndex, CHUNK_HORIZON, _episode_length, ttc_error, video_chunk_error, bc_error, naive_dino_error
from b1_retrieval import ep1_for_task, query_episodes_for_task, load_manual
from b1_value_sources import load_all_value_sources
from b1_full_run import dense_embeddings
from seen_unseen_split import SEEN_TASKS, UNSEEN_TASKS, dense_values

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "B2-vision value lambda sweep")

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_task(task: str, manual: dict, dino: DinoIndex, value_sources: dict[str, dict]) -> list[dict]:
    ep1_uid = ep1_for_task(task)
    queries = query_episodes_for_task(task, manual, ep1_uid)
    ep1_len = _episode_length(ep1_uid)
    n_cand = max(ep1_len - CHUNK_HORIZON, 0) + 1
    if n_cand <= 0 or not queries:
        return []
    starts = np.arange(n_cand)

    ep1_emb_dense = dense_embeddings(dino, ep1_uid)
    cand_emb = ep1_emb_dense[starts]
    cand_norm_sq = np.sum(cand_emb ** 2, axis=1)

    cand_val_by_source: dict[str, np.ndarray] = {}
    for name, curve in value_sources.items():
        dv = dense_values(curve, ep1_uid)
        if dv is not None:
            cand_val_by_source[name] = dv[starts]

    metric_cache: dict[tuple, dict] = {}  # (q_uid, qf, rf) -> metrics, reused across sources/lambdas

    def metrics_for(q_uid: str, qf: int, rf: int) -> dict:
        key = (q_uid, qf, rf)
        if key not in metric_cache:
            metric_cache[key] = {
                "ttc_error": ttc_error(q_uid, qf, ep1_uid, rf),
                "video_chunk_error": video_chunk_error(dino, q_uid, qf, ep1_uid, rf),
                "bc_error": bc_error(q_uid, qf, ep1_uid, rf),
                "naive_dino_error": naive_dino_error(dino, q_uid, qf, ep1_uid, rf),
            }
        return metric_cache[key]

    rows = []
    for q_uid in queries:
        q_len = _episode_length(q_uid)
        q_frames = np.arange(0, max(q_len - CHUNK_HORIZON, 0) + 1)
        if len(q_frames) == 0:
            continue

        q_emb_dense = dense_embeddings(dino, q_uid)
        q_emb = q_emb_dense[q_frames]
        q_norm_sq = np.sum(q_emb ** 2, axis=1)

        dot = q_emb @ cand_emb.T
        dist_sq = q_norm_sq[:, None] + cand_norm_sq[None, :] - 2 * dot
        np.maximum(dist_sq, 0, out=dist_sq)
        vision_dist = np.sqrt(dist_sq)

        for name, cand_val in cand_val_by_source.items():
            q_val_dense = dense_values(value_sources[name], q_uid)
            if q_val_dense is None:
                continue
            q_val = q_val_dense[q_frames]
            value_dist = np.abs(q_val[:, None] - cand_val[None, :])

            for lam in LAMBDAS:
                combined = lam * value_dist + (1 - lam) * vision_dist
                idx = np.argmin(combined, axis=1)
                retrieved_frames = starts[idx]
                for qf, rf in zip(q_frames.tolist(), retrieved_frames.tolist()):
                    row = {"task": task, "query_uid": q_uid, "query_frame": qf,
                           "value_source": name, "lambda": lam,
                           "ep1_uid": ep1_uid, "retrieved_frame": rf}
                    row.update(metrics_for(q_uid, qf, rf))
                    rows.append(row)
    return rows


def write_summaries(df: pd.DataFrame) -> None:
    metric_cols = ["ttc_error", "video_chunk_error", "bc_error", "naive_dino_error"]

    print("\nMean metric by (value_source, lambda):")
    summary = df.groupby(["value_source", "lambda"])[metric_cols].mean()
    print(summary.to_string())
    summary_path = os.path.join(OUTPUT_DIR, "b2_source_lambda_summary.csv")
    summary.to_csv(summary_path)

    print("\nMean metric by lambda (pooled over value sources):")
    pooled = df.groupby("lambda")[metric_cols].mean()
    print(pooled.to_string())
    pooled_path = os.path.join(OUTPUT_DIR, "b2_lambda_summary.csv")
    pooled.to_csv(pooled_path)

    task_summary = df.groupby(["task", "value_source", "lambda"])[metric_cols].mean()
    task_summary_path = os.path.join(OUTPUT_DIR, "b2_task_source_lambda_summary.csv")
    task_summary.to_csv(task_summary_path)

    print(f"\nWrote {summary_path}, {pooled_path}, {task_summary_path}")

    # in-domain (seen) / out-of-domain (unseen) split -- same task-novelty
    # partition as seen_unseen_split.py, applied to B1/B2 retrieval queries
    # by the QUERY episode's task (not ep1's task, which is fixed per task
    # anyway since ep1/queries always share one task).
    domain = df["task"].map(lambda t: "in_domain" if t in SEEN_TASKS else "out_of_domain" if t in UNSEEN_TASKS else None)
    assert domain.notna().all(), "task(s) outside SEEN_TASKS/UNSEEN_TASKS found in B2 results"
    df = df.assign(domain=domain)

    print("\nMean metric by (value_source, lambda, domain):")
    domain_summary = df.groupby(["value_source", "lambda", "domain"])[metric_cols].mean()
    print(domain_summary.to_string())
    domain_summary_path = os.path.join(OUTPUT_DIR, "b2_domain_split_summary.csv")
    domain_summary.to_csv(domain_summary_path)

    print("\nMean metric by (lambda, domain), pooled over value sources:")
    domain_pooled = df.groupby(["lambda", "domain"])[metric_cols].mean()
    print(domain_pooled.to_string())
    domain_pooled_path = os.path.join(OUTPUT_DIR, "b2_lambda_domain_summary.csv")
    domain_pooled.to_csv(domain_pooled_path)

    print(f"Wrote {domain_summary_path}, {domain_pooled_path}")


def merge_shards(num_shards: int) -> None:
    paths = [os.path.join(OUTPUT_DIR, f"b2_full_results_shard{i}.csv") for i in range(num_shards)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing shard output(s): {missing}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "b2_full_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Merged {len(df)} rows from {num_shards} shards -> {out_path}")
    write_summaries(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="This process's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (tasks split round-robin)")
    parser.add_argument("--merge", action="store_true", help="Merge all shard CSVs already on disk and write summaries")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.merge:
        merge_shards(args.num_shards)
        return

    manual = load_manual()
    dino = DinoIndex()
    all_sources = load_all_value_sources()
    value_sources = {k: v for k, v in all_sources.items() if k in ("icvfe_8800", "icvfe_ema_0.5")}
    tasks = sorted({e["task"] for e in manual.values()})
    my_tasks = tasks[args.shard::args.num_shards] if args.num_shards > 1 else tasks
    print(f"shard {args.shard}/{args.num_shards}: {len(my_tasks)}/{len(tasks)} tasks, "
          f"{len(value_sources)} value sources: {list(value_sources)}, lambdas={LAMBDAS}", flush=True)

    all_rows = []
    t0 = time.time()
    for i, task in enumerate(my_tasks):
        rows = run_task(task, manual, dino, value_sources)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(my_tasks)}] {task!r}: {len(rows)} rows "
              f"(total {len(all_rows)}, {elapsed:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    out_path = os.path.join(OUTPUT_DIR, f"b2_full_results{suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows), total time {time.time()-t0:.1f}s", flush=True)

    if args.num_shards == 1:
        write_summaries(df)


if __name__ == "__main__":
    main()
