"""
VEP-based retrieval: nearest-neighbor by VEP embedding distance, as a second
single-embedding retrieval signal alongside DINO ("vision"/naive_dino_error
in b1_full_run.py).

Per the b1_retrieval.py docstring, VEP was flagged as excluded "for now --
might be added later"; user has since copied the VEP embedding export over
(data/.../vep/vep_icl_demo_dataset_embeddings.zip -- 128-dim, one .npz per
episode, sparse every-10th-frame + tail, same sampling convention as DINO).

Retrieval mechanism is identical to "vision" in b1_full_run.py (same ep1
demo bank, same candidate/query framing), just swapping DinoIndex for
VepIndex as the nearest-neighbor key. Reported metrics stay TTC error, BC
error, and videoDTW error -- videoDTW is still computed over DINO embedding
sequences (not VEP) for both "vision" and "vep", since it's meant as a fixed
visual-similarity yardstick to judge retrieval quality by, independent of
which signal did the retrieving (same reasoning value-based retrieval is
also judged by DINO-based videoDTW, not by its own value curve).

Usage:
    python b1_vep_retrieval.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from b_metrics import DinoIndex, VepIndex, CHUNK_HORIZON, _episode_length, ttc_error, video_chunk_error, bc_error
from b1_retrieval import ep1_for_task, query_episodes_for_task, load_manual
from b1_full_run import dense_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "b1_vep_retrieval")


def run_task(task: str, manual: dict, dino: DinoIndex, vep: VepIndex) -> list[dict]:
    ep1_uid = ep1_for_task(task)
    queries = query_episodes_for_task(task, manual, ep1_uid)
    if ep1_uid not in vep._frames:
        return []  # this task's ep1 episode isn't covered by the VEP export

    ep1_len = _episode_length(ep1_uid)
    n_cand = max(ep1_len - CHUNK_HORIZON, 0) + 1
    if n_cand <= 0 or not queries:
        return []
    starts = np.arange(n_cand)

    cand_emb = dense_embeddings(vep, ep1_uid)[starts]
    cand_norm_sq = np.sum(cand_emb ** 2, axis=1)

    rows = []
    for q_uid in queries:
        if q_uid not in vep._frames:
            continue  # this query episode isn't covered by the VEP export
        q_len = _episode_length(q_uid)
        q_frames = np.arange(0, max(q_len - CHUNK_HORIZON, 0) + 1)
        if len(q_frames) == 0:
            continue

        q_emb = dense_embeddings(vep, q_uid)[q_frames]
        q_norm_sq = np.sum(q_emb ** 2, axis=1)

        dot = q_emb @ cand_emb.T
        dist_sq = q_norm_sq[:, None] + cand_norm_sq[None, :] - 2 * dot
        np.maximum(dist_sq, 0, out=dist_sq)
        vep_dist = np.sqrt(dist_sq)
        idx = np.argmin(vep_dist, axis=1)
        retrieved_frames = starts[idx]

        for qf, rf in zip(q_frames.tolist(), retrieved_frames.tolist()):
            rows.append({
                "task": task, "query_uid": q_uid, "query_frame": qf,
                "method": "vep", "ep1_uid": ep1_uid, "retrieved_frame": rf,
                "ttc_error": ttc_error(q_uid, qf, ep1_uid, rf),
                "video_chunk_error": video_chunk_error(dino, q_uid, qf, ep1_uid, rf),
                "bc_error": bc_error(q_uid, qf, ep1_uid, rf),
            })
    return rows


def main():
    manual = load_manual()
    dino = DinoIndex()
    vep = VepIndex()
    tasks = sorted({e["task"] for e in manual.values()})
    print(f"{len(tasks)} tasks, {len(vep._frames)} VEP-covered episodes", flush=True)

    all_rows = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        rows = run_task(task, manual, dino, vep)
        all_rows.extend(rows)
        print(f"[{i+1}/{len(tasks)}] {task!r}: {len(rows)} rows "
              f"(total {len(all_rows)}, {time.time()-t0:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "b1_vep_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows), total time {time.time()-t0:.1f}s", flush=True)

    print("\nMean metric (vep method, all covered tasks pooled):")
    print(df[["ttc_error", "video_chunk_error", "bc_error"]].mean().to_string())


if __name__ == "__main__":
    main()
