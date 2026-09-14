"""
ICVFE Evals correlation/agreement pipeline.

Computes Pearson r, Kendall tau-b, and MAE between four evaluator-pairs,
at episode / task / total granularity, from the raw archives in ../data.

Pairs:
  - robodopamine vs human   (zs_robodopamine_icl_demo_dataset_continuous vs manual_icl_demo_dataset_continuous)
  - robometer    vs human   (finetuned_robometer_icl_demo_dataset_continuous vs manual)
  - icvfe        vs human   (eval_on_vfe_icl_keep_true_annotation's prediction curve vs the manual curve,
                             aligned via the npz's own frame indices -- NOT the npz's `target_progress`,
                             which is actually robodopamine's curve; see build_icvfe_rows())
  - robometer    vs robodopamine (cross-correlation between the two automatic scorers)
  - icvfe        vs robodopamine (icvfe's prediction curve vs its own npz `target_progress`, which is
                             robodopamine's curve for that episode -- this is what ICVFE was scored
                             against natively)

See ../README.md for full methodology notes.

Usage:
    python pipeline.py            # extract (if needed) + compute + write CSVs to output/
    python pipeline.py --no-cache # force re-extraction of archives
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tarfile
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
CACHE_DIR = os.path.join(HERE, "cache")
OUTPUT_DIR = os.path.join(HERE, "output")

MANUAL_ARCHIVE = "manual_icl_demo_dataset_continuous.zip"
ROBOMETER_ARCHIVE = "finetuned_robometer_icl_demo_dataset_continuous.zip"
ROBODOPAMINE_ARCHIVE = "zs_robodopamine_icl_demo_dataset_continuous.zip"
ICVFE_ARCHIVE = "eval_on_vfe_icl_keep_true_annotation.tar.gz"
RECAP_ARCHIVE = "eval_recap_20000.tar.gz"

PAIR_RD_H = "robodopamine_vs_human"
PAIR_RM_H = "robometer_vs_human"
PAIR_ICVFE_H = "icvfe_vs_human"
PAIR_RECAP_H = "recap_vs_human"
PAIR_RM_RD = "robometer_vs_robodopamine"
PAIR_ICVFE_RD = "icvfe_vs_robodopamine"
PAIR_RECAP_RD = "recap_vs_robodopamine"

METRIC_PEARSON = "pearson"
METRIC_KENDALL = "kendall_tau_b"
METRIC_MAE = "mae"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_archives(force: bool = False) -> None:
    if force and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    zips = {
        "manual": MANUAL_ARCHIVE,
        "finetuned_robometer": ROBOMETER_ARCHIVE,
        "zs_robodopamine": ROBODOPAMINE_ARCHIVE,
    }
    for name, archive in zips.items():
        dest = os.path.join(CACHE_DIR, name)
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(os.path.join(DATA_DIR, archive)) as zf:
            zf.extractall(dest)

    dest = os.path.join(CACHE_DIR, "eval_on_vfe_icl_keep_true_annotation")
    if not (os.path.isdir(dest) and os.listdir(dest)):
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(os.path.join(DATA_DIR, ICVFE_ARCHIVE)) as tf:
            tf.extractall(dest, filter="data")

    dest = os.path.join(CACHE_DIR, "eval_recap_20000")
    if not (os.path.isdir(dest) and os.listdir(dest)):
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(os.path.join(DATA_DIR, RECAP_ARCHIVE)) as tf:
            tf.extractall(dest, filter="data")


# --------------------------------------------------------------------------
# Loading raw progress curves (human / robodopamine / robometer)
# --------------------------------------------------------------------------

def _find_json_dir(root: str) -> str:
    matches = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No json files found under {root}")
    return os.path.dirname(matches[0])


def load_curve_set(cache_subdir: str) -> dict:
    """Returns {episode_uid: {"task": str, "progress": np.ndarray}}"""
    root = os.path.join(CACHE_DIR, cache_subdir)
    json_dir = _find_json_dir(root)
    out = {}
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        progress = np.array([p["progress"] for p in d["points"]], dtype=np.float64)
        out[d["episode_uid"]] = {"task": d["task"], "progress": progress}
    return out


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(kendalltau(x, y, variant="b")[0])


def safe_mae(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def compute_all_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        METRIC_PEARSON: safe_pearson(x, y),
        METRIC_KENDALL: safe_kendall(x, y),
        METRIC_MAE: safe_mae(x, y),
    }


# --------------------------------------------------------------------------
# Raw-curve pairwise analysis (robodopamine/robometer/human)
# --------------------------------------------------------------------------

def build_raw_curve_rows(manual: dict, robometer: dict, robodop: dict) -> list:
    """Episode-level rows for the three raw-curve pairs."""
    common = sorted(set(manual) & set(robometer) & set(robodop))
    rows = []
    pooled_frames = defaultdict(lambda: defaultdict(list))  # task -> pair -> (x list, y list)
    total_frames = defaultdict(lambda: ([], []))  # pair -> (x, y)

    pair_sources = {
        PAIR_RD_H: (robodop, manual),
        PAIR_RM_H: (robometer, manual),
        PAIR_RM_RD: (robometer, robodop),
    }

    for uid in common:
        task = manual[uid]["task"]
        for pair, (src_a, src_b) in pair_sources.items():
            a = src_a[uid]["progress"]
            b = src_b[uid]["progress"]
            n = min(len(a), len(b))
            # progress ships as 0-100; rescaling it to "points out of 100" for
            # MAE is itself just another normalization, not a move away from
            # one -- report MAE in the same 0-1 fraction-of-task-progress
            # units the underlying annotation is defined in.
            a, b = a[:n] / 100.0, b[:n] / 100.0
            m = compute_all_metrics(a, b)
            for metric_name, value in m.items():
                rows.append({
                    "level": "episode",
                    "task": task,
                    "group_id": uid,
                    "pair": pair,
                    "metric": metric_name,
                    "value": value,
                    "n": n,
                })
            pooled_frames[task][pair] = pooled_frames[task].get(pair, ([], []))
            pooled_frames[task][pair][0].extend(a.tolist())
            pooled_frames[task][pair][1].extend(b.tolist())
            total_frames[pair][0].extend(a.tolist())
            total_frames[pair][1].extend(b.tolist())

    # task level (pooled frames across all episodes of the task)
    for task, pairs in pooled_frames.items():
        n_eps = len({uid for uid in common if manual[uid]["task"] == task})
        for pair, (xs, ys) in pairs.items():
            x = np.array(xs)
            y = np.array(ys)
            m = compute_all_metrics(x, y)
            for metric_name, value in m.items():
                rows.append({
                    "level": "task",
                    "task": task,
                    "group_id": task,
                    "pair": pair,
                    "metric": metric_name,
                    "value": value,
                    "n": len(x),
                    "n_episodes": n_eps,
                })

    # total level (pooled frames across everything)
    for pair, (xs, ys) in total_frames.items():
        x = np.array(xs)
        y = np.array(ys)
        m = compute_all_metrics(x, y)
        for metric_name, value in m.items():
            rows.append({
                "level": "total",
                "task": "ALL",
                "group_id": "ALL",
                "pair": pair,
                "metric": metric_name,
                "value": value,
                "n": len(x),
                "n_episodes": len(common),
            })

    return rows


# --------------------------------------------------------------------------
# ICVFE vs human analysis
# --------------------------------------------------------------------------

def _icvfe_root() -> str:
    root = os.path.join(CACHE_DIR, "eval_on_vfe_icl_keep_true_annotation")
    # the tarball contains a single checkpoint-id directory, e.g. "8800"
    entries = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint dir under {root}, found {entries}")
    return os.path.join(root, entries[0])


def _aggregate_icvfe_pair(tdf: pd.DataFrame, pair: str, pearson_col: str, kendall_col: str, mae_col: str) -> list:
    """Episode/task/total rows for one ICVFE pair, from a per-trajectory dataframe.

    ICVFE scores each query episode once per in-context example drawn from its
    task (multiple "context replicates" per episode). Episode level averages
    those replicates; task/total average the resulting episode-level numbers
    (not frame-pooling, which would repeat the same target/comparison curve
    once per replicate and inflate the effective sample size).
    """
    rows = []

    ep_group = tdf.groupby(["task", "episode_uid"], as_index=False)[
        [pearson_col, kendall_col, mae_col]
    ].mean()
    n_reps = tdf.groupby(["task", "episode_uid"]).size().rename("n").reset_index()
    ep_group = ep_group.merge(n_reps, on=["task", "episode_uid"])

    metric_cols = {METRIC_PEARSON: pearson_col, METRIC_KENDALL: kendall_col, METRIC_MAE: mae_col}

    for _, r in ep_group.iterrows():
        for metric_name, col in metric_cols.items():
            rows.append({
                "level": "episode",
                "task": r["task"],
                "group_id": r["episode_uid"],
                "pair": pair,
                "metric": metric_name,
                "value": r[col],
                "n": int(r["n"]),
            })

    task_group = ep_group.groupby("task", as_index=False)[[pearson_col, kendall_col, mae_col]].mean()
    n_ep_per_task = ep_group.groupby("task").size().rename("n_episodes").reset_index()
    task_group = task_group.merge(n_ep_per_task, on="task")

    for _, r in task_group.iterrows():
        for metric_name, col in metric_cols.items():
            rows.append({
                "level": "task",
                "task": r["task"],
                "group_id": r["task"],
                "pair": pair,
                "metric": metric_name,
                "value": r[col],
                "n": int(r["n_episodes"]),
                "n_episodes": int(r["n_episodes"]),
            })

    for metric_name, col in metric_cols.items():
        rows.append({
            "level": "total",
            "task": "ALL",
            "group_id": "ALL",
            "pair": pair,
            "metric": metric_name,
            "value": float(ep_group[col].mean()),
            "n": len(ep_group),
            "n_episodes": len(ep_group),
        })

    return rows


def build_icvfe_rows(manual: dict) -> list:
    """
    IMPORTANT: the npz `target_progress` curve in this archive is NOT the
    manual/human annotation. Its metadata_json's `annotation_path` points at
    `zs_robodopamine_icl_demo_dataset_continuous` for every trajectory sampled
    (verified across tasks), and its values match the `zs_robodopamine`
    curve for the same episode to ~1e-6 (float32 rounding only). So:
      - `target_progress` vs `prediction_progress`  -> icvfe vs robodopamine
      - `prediction_progress` vs the `manual` curve  -> icvfe vs human (we
        align it ourselves via the npz's `timesteps` index)
    """
    ckpt_root = _icvfe_root()
    with open(os.path.join(ckpt_root, "metrics.json")) as f:
        meta = json.load(f)

    trajectories = meta["trajectories"]

    traj_metrics = []
    for t in trajectories:
        task_dir = f"task_{t['task_id']:03d}+{t['task_name'].replace(' ', '_')}"
        npz_path = os.path.join(
            ckpt_root, task_dir,
            f"context_{t['context_demo_id']}",
            "curves",
            f"query_{t['demo_id']}.npz",
        )
        with np.load(npz_path) as d:
            # native 0-1 fraction-of-task-progress scale -- left as-is rather
            # than rescaled to "points out of 100" (see build_raw_curve_rows)
            target = np.asarray(d["target_progress"], dtype=np.float64)  # == robodopamine
            pred = np.asarray(d["prediction_progress"], dtype=np.float64)
            timesteps = np.asarray(d["timesteps"], dtype=np.int64)

        m_rd = compute_all_metrics(pred, target)

        row = {
            "task": t["task_name"],
            "episode_uid": t["episode_uid"],
            "demo_id": t["demo_id"],
            "context_demo_id": t["context_demo_id"],
            "pearson_rd": m_rd[METRIC_PEARSON],
            "kendall_rd": m_rd[METRIC_KENDALL],
            "mae_rd": m_rd[METRIC_MAE],
        }

        manual_entry = manual.get(t["episode_uid"])
        if manual_entry is not None:
            manual_progress = manual_entry["progress"] / 100.0  # ships as 0-100 -> match pred's 0-1 scale
            valid = timesteps < len(manual_progress)
            manual_aligned = manual_progress[timesteps[valid]]
            pred_aligned = pred[valid]
            m_human = compute_all_metrics(pred_aligned, manual_aligned)
        else:
            m_human = {METRIC_PEARSON: float("nan"), METRIC_KENDALL: float("nan"), METRIC_MAE: float("nan")}
        row["pearson_h"] = m_human[METRIC_PEARSON]
        row["kendall_h"] = m_human[METRIC_KENDALL]
        row["mae_h"] = m_human[METRIC_MAE]

        traj_metrics.append(row)

    tdf = pd.DataFrame(traj_metrics)

    rows = []
    rows += _aggregate_icvfe_pair(tdf, PAIR_ICVFE_H, "pearson_h", "kendall_h", "mae_h")
    rows += _aggregate_icvfe_pair(tdf, PAIR_ICVFE_RD, "pearson_rd", "kendall_rd", "mae_rd")
    return rows


# --------------------------------------------------------------------------
# pi0.6-recap vs human / robodopamine
# --------------------------------------------------------------------------

def _recap_root() -> str:
    root = os.path.join(CACHE_DIR, "eval_recap_20000")
    # the tarball contains a single checkpoint-step dir, e.g. "20000"
    entries = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint dir under {root}, found {entries}")
    return os.path.join(root, entries[0])


def build_recap_rows(manual: dict) -> list:
    """
    pi0.6-recap (checkpoint step 20000) eval, split across "seen"/"unseen"
    task-novelty subdirs. Same target-curve caveat as ICVFE (see
    build_icvfe_rows): each npz's `target_progress` is robodopamine's curve
    for that episode, not the human one (verified the same way -- metadata's
    `annotation_path` points at zs_robodopamine_icl_demo_dataset_continuous).

    Unlike ICVFE this eval is single-pass / no in-context replicates (one
    query pass per episode, context_demo_id == ""), so there's nothing to
    average per episode -- but we still route it through
    _aggregate_icvfe_pair for identical task/total aggregation semantics
    (averaging episode-level numbers, not frame-pooling) so recap's numbers
    are aggregated the same way as ICVFE's and comparable in derivation.
    """
    ckpt_root = _recap_root()

    traj_metrics = []
    for split in ("seen", "unseen"):
        split_dir = os.path.join(ckpt_root, split)
        with open(os.path.join(split_dir, "metrics.json")) as f:
            meta = json.load(f)

        for t in meta["trajectories"]:
            npz_path = os.path.join(
                split_dir, "curves", f"task_{t['task_id']:03d}_{t['demo_id']}.npz"
            )
            with np.load(npz_path) as d:
                target = np.asarray(d["target_progress"], dtype=np.float64)  # == robodopamine
                pred = np.asarray(d["prediction_progress"], dtype=np.float64)
                timesteps = np.asarray(d["timesteps"], dtype=np.int64)

            m_rd = compute_all_metrics(pred, target)

            row = {
                "task": t["task_name"],
                "episode_uid": t["episode_uid"],
                "demo_id": t["demo_id"],
                "task_novelty": t.get("task_novelty", split),
                "pearson_rd": m_rd[METRIC_PEARSON],
                "kendall_rd": m_rd[METRIC_KENDALL],
                "mae_rd": m_rd[METRIC_MAE],
            }

            manual_entry = manual.get(t["episode_uid"])
            if manual_entry is not None:
                manual_progress = manual_entry["progress"] / 100.0  # ships as 0-100 -> match pred's 0-1 scale
                valid = timesteps < len(manual_progress)
                manual_aligned = manual_progress[timesteps[valid]]
                pred_aligned = pred[valid]
                m_human = compute_all_metrics(pred_aligned, manual_aligned)
            else:
                m_human = {METRIC_PEARSON: float("nan"), METRIC_KENDALL: float("nan"), METRIC_MAE: float("nan")}
            row["pearson_h"] = m_human[METRIC_PEARSON]
            row["kendall_h"] = m_human[METRIC_KENDALL]
            row["mae_h"] = m_human[METRIC_MAE]

            traj_metrics.append(row)

    tdf = pd.DataFrame(traj_metrics)

    rows = []
    rows += _aggregate_icvfe_pair(tdf, PAIR_RECAP_H, "pearson_h", "kendall_h", "mae_h")
    rows += _aggregate_icvfe_pair(tdf, PAIR_RECAP_RD, "pearson_rd", "kendall_rd", "mae_rd")
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Force re-extraction of archives")
    args = parser.parse_args()

    extract_archives(force=args.no_cache)

    manual = load_curve_set("manual")
    robometer = load_curve_set("finetuned_robometer")
    robodop = load_curve_set("zs_robodopamine")

    print(f"Loaded {len(manual)} manual, {len(robometer)} robometer, {len(robodop)} robodopamine episodes")

    raw_rows = build_raw_curve_rows(manual, robometer, robodop)
    icvfe_rows = build_icvfe_rows(manual)
    recap_rows = build_recap_rows(manual)

    all_rows = raw_rows + icvfe_rows + recap_rows
    long_df = pd.DataFrame(all_rows)
    long_df = long_df[["level", "task", "group_id", "pair", "metric", "value", "n"] +
                       (["n_episodes"] if "n_episodes" in long_df.columns else [])]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_path = os.path.join(OUTPUT_DIR, "metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    # ---- wide convenience tables ----
    def make_wide(level: str, index_cols: list) -> pd.DataFrame:
        sub = long_df[long_df["level"] == level].copy()
        sub["col"] = sub["pair"] + "__" + sub["metric"]
        wide = sub.pivot_table(index=index_cols, columns="col", values="value", aggfunc="first")
        n_col = sub.groupby(index_cols)["n"].max()
        wide.insert(0, "n_frames_or_reps", n_col)
        wide = wide.reset_index()
        return wide

    episode_wide = make_wide("episode", ["task", "group_id"]).rename(columns={"group_id": "episode_uid"})
    episode_wide.to_csv(os.path.join(OUTPUT_DIR, "episode_level_wide.csv"), index=False)

    task_wide = make_wide("task", ["task"])
    n_ep_map = long_df[long_df["level"] == "task"].groupby("task")["n_episodes"].max()
    task_wide["n_episodes"] = task_wide["task"].map(n_ep_map)
    task_wide.to_csv(os.path.join(OUTPUT_DIR, "task_level_wide.csv"), index=False)

    total_sub = long_df[long_df["level"] == "total"].copy()
    total_wide = total_sub.pivot_table(index="pair", columns="metric", values="value", aggfunc="first")
    n_map = total_sub.groupby("pair")["n"].max()
    ne_map = total_sub.groupby("pair")["n_episodes"].max()
    total_wide.insert(0, "n_episodes", ne_map)
    total_wide.insert(0, "n_frames_or_reps", n_map)
    total_wide = total_wide.reset_index()
    total_wide.to_csv(os.path.join(OUTPUT_DIR, "total_level_wide.csv"), index=False)

    print("Wrote episode_level_wide.csv, task_level_wide.csv, total_level_wide.csv")
    print("\nTotal-level summary:")
    print(total_wide.to_string(index=False))


if __name__ == "__main__":
    main()
