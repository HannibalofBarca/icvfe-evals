"""
B1 full run: all 27 tasks, every query frame (no stride).

Methods:
  - value_<source>_vs_human           for source in {robometer_ft, recap_ft,
                                        icvfe_8800, icvfe_ema_0.5, robodopamine_zs}
  - vision                            (DINO nearest-neighbor)
  - naive_vision_value_<source>       for the same 5 sources (unweighted sum
                                       of raw value error + raw DINO distance)

robodopamine_zs is included as an upper-ceiling reference per user direction
(the doc excludes it from B1 proper since it needs full-episode/goal-frame
context and isn't usable online -- but we're just reusing its precomputed
progress curve as a retrieval key here, not re-running it online).

recap_ft/icvfe_8800/icvfe_ema_0.5 only cover a 240-episode subset (see
b1_value_sources.py) -- a task is skipped entirely for one of these sources
if its own ep1 lacks coverage; individual query episodes lacking coverage
are skipped one at a time otherwise.

Retrieval search is vectorized per (task, value_source) via matmul-based
pairwise distance; the four metrics themselves reuse b_metrics.py's
per-pair functions.

Usage:
    python b1_full_run.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from b_metrics import DinoIndex, CHUNK_HORIZON, _episode_length, ttc_error, video_chunk_error, bc_error, naive_dino_error
from b1_retrieval import ep1_for_task, query_episodes_for_task, load_manual
from b1_value_sources import load_all_value_sources

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "b1_full_run")


def dense_embeddings(dino: DinoIndex, episode_uid: str) -> np.ndarray:
    length = _episode_length(episode_uid)
    frames = dino._frames[episode_uid]
    emb = dino._embeddings[episode_uid]
    target = np.clip(np.arange(length), frames[0], frames[-1])
    idx = np.clip(np.searchsorted(frames, target, side="right") - 1, 0, len(frames) - 2)
    f0, f1 = frames[idx], frames[idx + 1]
    t = np.divide(target - f0, f1 - f0, out=np.zeros_like(target, dtype=np.float64), where=(f1 != f0))
    return emb[idx] + t[:, None] * (emb[idx + 1] - emb[idx])


def dense_values(value_curve: dict, episode_uid: str) -> np.ndarray | None:
    entry = value_curve.get(episode_uid)
    if entry is None:
        return None
    length = _episode_length(episode_uid)
    fidx, val = entry["frame_index"], entry["value"]
    target = np.clip(np.arange(length), fidx[0], fidx[-1])
    return np.interp(target, fidx, val)


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

    # per-source candidate values in ep1 (None if this source doesn't cover ep1 -> skip source for whole task)
    cand_val_by_source: dict[str, np.ndarray] = {}
    for name, curve in value_sources.items():
        dv = dense_values(curve, ep1_uid)
        if dv is not None:
            cand_val_by_source[name] = dv[starts]

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
        vision_idx = np.argmin(vision_dist, axis=1)
        vision_frames = starts[vision_idx]

        method_frames = {"vision": vision_frames}

        for name, cand_val in cand_val_by_source.items():
            q_val_dense = dense_values(value_sources[name], q_uid)
            if q_val_dense is None:
                continue  # this query episode isn't covered by this source
            q_val = q_val_dense[q_frames]
            value_dist = np.abs(q_val[:, None] - cand_val[None, :])
            value_idx = np.argmin(value_dist, axis=1)
            method_frames[f"value_{name}"] = starts[value_idx]

            combined = value_dist + vision_dist
            naive_idx = np.argmin(combined, axis=1)
            method_frames[f"naive_vision_value_{name}"] = starts[naive_idx]

        for method, retrieved_frames in method_frames.items():
            for qf, rf in zip(q_frames.tolist(), retrieved_frames.tolist()):
                rows.append({
                    "task": task, "query_uid": q_uid, "query_frame": qf,
                    "method": method, "ep1_uid": ep1_uid, "retrieved_frame": rf,
                    "ttc_error": ttc_error(q_uid, qf, ep1_uid, rf),
                    "video_chunk_error": video_chunk_error(dino, q_uid, qf, ep1_uid, rf),
                    "bc_error": bc_error(q_uid, qf, ep1_uid, rf),
                    "naive_dino_error": naive_dino_error(dino, q_uid, qf, ep1_uid, rf),
                })
    return rows


def main():
    manual = load_manual()
    dino = DinoIndex()
    value_sources = load_all_value_sources()
    tasks = sorted({e["task"] for e in manual.values()})
    print(f"{len(tasks)} tasks, {len(value_sources)} value sources: {list(value_sources)}", flush=True)

    all_rows = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        rows = run_task(task, manual, dino, value_sources)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        methods_seen = sorted({r["method"] for r in rows})
        print(f"[{i+1}/{len(tasks)}] {task!r}: {len(rows)} rows, methods={methods_seen} "
              f"(total {len(all_rows)}, {elapsed:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "b1_full_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows), total time {time.time()-t0:.1f}s", flush=True)

    print("\nMean metric by method (all tasks pooled):")
    summary = df.groupby("method")[["ttc_error", "video_chunk_error", "bc_error", "naive_dino_error"]].mean()
    print(summary.to_string())

    summary_path = os.path.join(OUTPUT_DIR, "b1_method_summary.csv")
    summary.to_csv(summary_path)

    task_method_summary = df.groupby(["task", "method"])[["ttc_error", "video_chunk_error", "bc_error", "naive_dino_error"]].mean()
    task_method_path = os.path.join(OUTPUT_DIR, "b1_task_method_summary.csv")
    task_method_summary.to_csv(task_method_path)
    print(f"Wrote {summary_path} and {task_method_path}")


if __name__ == "__main__":
    main()
