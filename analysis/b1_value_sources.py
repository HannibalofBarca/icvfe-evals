"""
Value-curve sources for B1 retrieval: every scorer usable as a "value based
retrieval" key, in the same {episode_uid: {"task", "frame_index", "value"}}
shape so b1_full_run.py can treat them uniformly.

  - robometer_ft   : finetuned_robometer_online (flat json, dense, all tasks)
  - recap_ft       : recap FT-only npz (single no-context pass per episode,
                     240/285-episode coverage)
  - icvfe_8800     : ICVFE FT-only npz, checkpoint 8800 (in-context, 9
                     replicates per query episode -- averaged pointwise here
                     since timesteps are identical across replicates,
                     verified; 240/285-episode coverage)
  - icvfe_ema_0.5  : same, EMA-smoothed checkpoint
  - robodopamine_zs: zero-shot RoboDopamine (flat json, dense, all tasks) --
                     included as an upper-ceiling reference per user
                     direction, despite the doc excluding it from B1 proper
                     (needs full-episode/goal-frame context, not usable
                     online) -- here we're just reusing its precomputed
                     progress curve as a retrieval key, not re-running it
                     online, so that constraint doesn't block this use.

icvfe/recap only cover the 240-episode "keep"-eligible subset (see study-1
README) -- 9/27 tasks have their own ep1 (lowest episode_index) fall outside
that set, so those tasks are skipped entirely for icvfe/recap value methods
(per user direction). Individual *query* episodes lacking coverage are
skipped one at a time rather than dropping the whole task.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

import a3_pipeline as a3
from b1_retrieval import load_flat_value_curve


def load_robometer_ft() -> dict:
    return load_flat_value_curve(
        "robometer_ft_online", "robometer", "finetuned_robometer_online_icl_demo_dataset_continuous.zip"
    )


def load_robodopamine_zs() -> dict:
    return load_flat_value_curve(
        "robodopamine_zs", "robo_dopamine", "zs_robodopamine_icl_demo_dataset_continuous.zip"
    )


def _pred_progress(npz) -> np.ndarray:
    if "prediction_progress" in npz:
        return np.asarray(npz["prediction_progress"], dtype=np.float64)
    # icvfe_ema_0.5-style export: only ships the signed "prediction" key
    # (prediction_progress - 1), verified exactly against icvfe_8800 in
    # a3_pipeline.py's _traj_aligned_vs_manual.
    return np.asarray(npz["prediction"], dtype=np.float64) + 1.0


def load_recap_ft() -> dict:
    a3.extract_archives()
    trajectories = a3.load_recap_trajectories("recap_ft")
    out = {}
    for t in trajectories:
        with np.load(t["npz_path"]) as d:
            pred = _pred_progress(d)
            timesteps = np.asarray(d["timesteps"], dtype=np.int64)
        out[t["episode_uid"]] = {"task": t["task"], "frame_index": timesteps, "value": pred}
    return out


def load_icvfe(name: str) -> dict:
    a3.extract_archives()
    trajectories = a3.load_icvfe_trajectories(name)
    by_episode: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    task_by_episode: dict[str, str] = {}
    for t in trajectories:
        with np.load(t["npz_path"]) as d:
            pred = _pred_progress(d)
            timesteps = np.asarray(d["timesteps"], dtype=np.int64)
        by_episode[t["episode_uid"]].append((timesteps, pred))
        task_by_episode[t["episode_uid"]] = t["task"]

    out = {}
    for uid, reps in by_episode.items():
        base_ts = reps[0][0]
        matching = [p for ts, p in reps if len(ts) == len(base_ts) and np.array_equal(ts, base_ts)]
        avg_pred = np.mean(np.stack(matching), axis=0) if matching else reps[0][1]
        out[uid] = {"task": task_by_episode[uid], "frame_index": base_ts, "value": avg_pred}
    return out


def load_all_value_sources() -> dict[str, dict]:
    return {
        "robometer_ft": load_robometer_ft(),
        "recap_ft": load_recap_ft(),
        "icvfe_8800": load_icvfe("icvfe_8800"),
        "icvfe_ema_0.5": load_icvfe("icvfe_ema_0.5"),
        "robodopamine_zs": load_robodopamine_zs(),
    }
