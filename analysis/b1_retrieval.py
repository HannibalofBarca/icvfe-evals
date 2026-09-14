"""
B1 retrieval methods, built on top of b_metrics.py's metric functions.

Demo bank / query setup (per user direction):
  - Per task, "ep1" = the single fixed demo-bank episode = the episode with
    the lowest dataset-wide episode_index for that task (plain first-episode
    reading, not tied to ICVFE's own "eval_episode_NNNNNN" numbering, which
    turned out to be a GLOBAL index across the whole dataset, not per-task --
    see conversation).
  - Candidate retrieval positions = every start frame in ep1 from 0 to
    len(ep1) - CHUNK_HORIZON (sliding, stride 1).
  - Queries = every other episode of that task in the 243-episode manual
    (human-annotated) set.

Retrieval methods (per user direction):
  1. value-based: retrieve the ep1 candidate whose value (from some scorer)
     at its start frame is closest to the query frame's value under the same
     scorer. Generic over any scorer with a per-(episode,frame) value lookup
     -- instantiated below for robometer_ft (online) and recap_ft, the two
     online-eligible flat/near-flat value sources already loaded elsewhere
     in this repo. (ICVFE FT-only is deferred -- its in-context format needs
     picking/averaging over context replicates to get an intrinsic value
     curve for ep1, which is extra plumbing; flagging rather than guessing.)
  2. VEP -- excluded per user direction, may be added later.
  3. vision-based: retrieve the ep1 candidate whose DINO embedding at its
     start frame is closest (Euclidean) to the query frame's DINO embedding.
  4. naive vision+value: retrieve the candidate minimizing the *unweighted
     sum* of the raw value error and the raw DINO distance -- "naive"
     literally, no rescaling to a common range (per doc: "simple average").

This is a single-task proof of concept (task "hit the eggplant with the
mallet") to validate the mechanism before scaling to all 27 tasks / the full
value-source roster.
"""
from __future__ import annotations

import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

from b_metrics import (
    DinoIndex, CHUNK_HORIZON, _episode_length, _episodes_meta,
    ttc_error, video_dtw_error, bc_error, naive_dino_error,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "data", "demo_set_annotations", "demo_set_annotations")
)
CACHE_DIR = os.path.join(HERE, "cache", "b1_retrieval")


# --------------------------------------------------------------------------
# ep1 / query-set resolution
# --------------------------------------------------------------------------

def ep1_for_task(task: str) -> str:
    meta = _episodes_meta()
    candidates = [e for e in meta if e["tasks"] and e["tasks"][0] == task]
    return min(candidates, key=lambda e: e["episode_index"])["episode_uid"]


def query_episodes_for_task(task: str, manual: dict, ep1_uid: str) -> list[str]:
    return sorted(uid for uid, e in manual.items() if e["task"] == task and uid != ep1_uid)


# --------------------------------------------------------------------------
# Flat value-curve sources (robometer_ft_online, recap_ft) -- reused loaders
# --------------------------------------------------------------------------

def _extract_zip(name: str, subdir: str, filename: str) -> str:
    dest = os.path.join(CACHE_DIR, name)
    if not (os.path.isdir(dest) and os.listdir(dest)):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(os.path.join(DATA_ROOT, subdir, filename)) as zf:
            zf.extractall(dest)
    matches = glob.glob(os.path.join(dest, "**", "*.json"), recursive=True)
    return os.path.dirname(matches[0])


def load_flat_value_curve(name: str, subdir: str, filename: str) -> dict:
    """Returns {episode_uid: {"task": str, "frame_index": np.ndarray, "value": np.ndarray}} (0-1 scale)."""
    json_dir = _extract_zip(name, subdir, filename)
    out = {}
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        frame_index = np.array([p["frame_index"] for p in d["points"]], dtype=np.int64)
        progress = np.array([p["progress"] for p in d["points"]], dtype=np.float64) / 100.0
        out[d["episode_uid"]] = {"task": d["task"], "frame_index": frame_index, "value": progress}
    return out


def load_manual() -> dict:
    return load_flat_value_curve("manual", "manual", "manual_icl_demo_dataset_continuous.zip")


def value_at(curve_set: dict, episode_uid: str, frame: int) -> float:
    entry = curve_set[episode_uid]
    fidx, val = entry["frame_index"], entry["value"]
    frame = min(max(frame, int(fidx[0])), int(fidx[-1]))
    return float(np.interp(frame, fidx, val))


# --------------------------------------------------------------------------
# Retrieval methods
# --------------------------------------------------------------------------

def candidate_starts(ep1_uid: str) -> np.ndarray:
    length = _episode_length(ep1_uid)
    return np.arange(0, max(length - CHUNK_HORIZON, 0) + 1)


def retrieve_value(value_curve: dict, query_uid: str, query_frame: int, ep1_uid: str) -> int:
    starts = candidate_starts(ep1_uid)
    q_val = value_at(value_curve, query_uid, query_frame)
    cand_vals = np.array([value_at(value_curve, ep1_uid, int(s)) for s in starts])
    return int(starts[np.argmin(np.abs(cand_vals - q_val))])


def retrieve_vision(dino: DinoIndex, query_uid: str, query_frame: int, ep1_uid: str) -> int:
    starts = candidate_starts(ep1_uid)
    q_emb = dino.interp(query_uid, query_frame)
    cand_emb = np.stack([dino.interp(ep1_uid, int(s)) for s in starts])
    dists = np.linalg.norm(cand_emb - q_emb[None, :], axis=1)
    return int(starts[np.argmin(dists)])


def retrieve_naive_vision_value(dino: DinoIndex, value_curve: dict, query_uid: str, query_frame: int, ep1_uid: str) -> int:
    starts = candidate_starts(ep1_uid)
    q_val = value_at(value_curve, query_uid, query_frame)
    q_emb = dino.interp(query_uid, query_frame)
    cand_vals = np.array([value_at(value_curve, ep1_uid, int(s)) for s in starts])
    cand_emb = np.stack([dino.interp(ep1_uid, int(s)) for s in starts])
    value_err = np.abs(cand_vals - q_val)
    vision_err = np.linalg.norm(cand_emb - q_emb[None, :], axis=1)
    combined = value_err + vision_err  # naive: unweighted sum (== proportional to the average)
    return int(starts[np.argmin(combined)])


# --------------------------------------------------------------------------
# One-task demo
# --------------------------------------------------------------------------

def main():
    task = "hit the eggplant with the mallet"
    manual = load_manual()
    robometer_ft = load_flat_value_curve(
        "robometer_ft_online", "robometer", "finetuned_robometer_online_icl_demo_dataset_continuous.zip"
    )
    dino = DinoIndex()

    ep1_uid = ep1_for_task(task)
    queries = query_episodes_for_task(task, manual, ep1_uid)
    print(f"task={task!r}  ep1={ep1_uid}  n_candidates={len(candidate_starts(ep1_uid))}  n_queries={len(queries)}")

    rows = []
    for q_uid in queries:
        q_len = _episode_length(q_uid)
        for q_frame in range(0, max(q_len - CHUNK_HORIZON, 0) + 1, 20):  # stride 20 for a fast demo
            methods = {
                "value_robometer_ft": retrieve_value(robometer_ft, q_uid, q_frame, ep1_uid),
                "vision": retrieve_vision(dino, q_uid, q_frame, ep1_uid),
                "naive_vision_value": retrieve_naive_vision_value(dino, robometer_ft, q_uid, q_frame, ep1_uid),
            }
            for method, r_frame in methods.items():
                rows.append({
                    "query_uid": q_uid, "query_frame": q_frame, "method": method, "retrieved_frame": r_frame,
                    "ttc_error": ttc_error(q_uid, q_frame, ep1_uid, r_frame),
                    "video_dtw_error": video_dtw_error(dino, q_uid, q_frame, ep1_uid, r_frame),
                    "bc_error": bc_error(q_uid, q_frame, ep1_uid, r_frame),
                    "naive_dino_error": naive_dino_error(dino, q_uid, q_frame, ep1_uid, r_frame),
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "output", "b1_demo"), exist_ok=True)
    out_path = os.path.join(HERE, "output", "b1_demo", "one_task_demo.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")

    print("\nMean metric by method:")
    print(df.groupby("method")[["ttc_error", "video_dtw_error", "bc_error", "naive_dino_error"]].mean().to_string())


if __name__ == "__main__":
    main()
