"""
A3 ablation pipeline: online-only FT models vs. human (baseline changed
from RoboDopamine ZS to manual annotations per user direction).

Per Eval_Research_Summary.md, ablation A3 ("FT-only models vs. RoboDopamine
ZS reference" in the doc, but scored against manual here instead), restricted
to the online-eligible variant of each model (per B1's "online-eligible
retrieval keys" framing -- these don't need full-episode/goal-frame context,
unlike RoboDopamine ZS/FT, so they're usable online):
  variants: RoboMeter ZS (online), RoboMeter FT (online), recap (FT-only),
  ICVFE (FT-only)
  scored against: manual (human) annotations
  metrics: Pearson r, Kendall tau-b, MAE -- at episode / task / total granularity
  notes: A2 will likely show FT > ZS in aggregate, but FT is known to
  produce degenerate results on some episodes -- this checks per-episode
  against a dense reference to catch that.

RoboMeter here uses the *online* fine-tuned export
(finetuned_robometer_online_icl_demo_dataset_continuous.zip), not the base
one A2 used -- this ablation is scoped to online-only models, and the base
variant isn't one.

"FT-only" here describes the model/checkpoint (these evaluators only exist
as fine-tuned checkpoints -- there's no separate zero-shot variant to
compare against, unlike robometer/robodopamine in A2), not the eval
methodology: ICVFE's own export is still a genuine leave-one-out in-context
eval (real per-episode context_demo_id values, 9 context replicates per
query episode), same design as the original study-1 ICVFE analysis. recap
is separately, natively no-context (context_demo_id == "" always).

Two ICVFE checkpoint exports are both included as separate roster entries
per user direction:
  - icvfe_8800    -- same checkpoint used in the original study-1 pipeline
  - icvfe_ema_0.5 -- same base checkpoint, EMA-smoothed predictions (alpha=0.5)

Evaluator formats in this roster are NOT uniform:
  - robometer_zs_online, robometer_ft_online: plain per-frame JSON (points/frame_index/progress),
    same shape as manual -- aligned via a1/a2's frame_index +
    linear-interpolation method (build_rows/align_to_reference).
  - recap_ft, icvfe_8800, icvfe_ema_0.5: per-trajectory .npz curve files
    (prediction_progress/timesteps), one .npz per (context, query) pair.
    Their embedded `target_progress` is NOT the human curve -- per study 1's
    README, it's actually robodopamine's curve for that episode (verified
    via metadata + ~1e-6 curve match) -- so it is ignored entirely here;
    `prediction_progress` is instead aligned against the `manual` curve via
    the npz's own `timesteps` array. Aggregation for these three follows
    pipeline.py's `_aggregate_icvfe_pair`: episode level averages a query
    episode's context replicates (n = replicate count, not frame count).
    Task/total level, for every source, is the mean of the per-episode
    metrics (aggregation.py); frames are never pooled across episodes.

Usage:
    python a3_pipeline.py            # extract (if needed) + compute + write CSVs to output/
    python a3_pipeline.py --no-cache # force re-extraction of archives
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

from aggregation import aggregate_episode_rows

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "data", "demo_set_annotations", "demo_set_annotations")
)
CACHE_DIR = os.path.join(HERE, "cache", "A3-online ft models vs human")
OUTPUT_DIR = os.path.join(HERE, "output", "A3-online ft models vs human")

REFERENCE = "manual"

# name -> (subdir under DATA_ROOT, archive filename)
ARCHIVES = {
    "manual": ("manual", "manual_icl_demo_dataset_continuous.zip"),
    "robometer_zs_online": ("robometer", "zeroshot_robometer_online_icl_demo_dataset_continuous.zip"),
    "robometer_ft_online": ("robometer", "finetuned_robometer_online_icl_demo_dataset_continuous.zip"),
    "recap_ft": ("RECAP", "recap_20000_icl_demo_dataset_continuous.zip"),
    "icvfe_8800": ("ICVFE", "icvfe_8800_icl_demo_dataset_continuous.zip"),
    "icvfe_ema_0.5": ("ICVFE", "icvfe_ema_0.5_icl_demo_dataset_continuous.zip"),
}

FLAT_ROSTER = ["robometer_zs_online", "robometer_ft_online"]   # plain per-frame JSON
NPZ_ROSTER = ["recap_ft", "icvfe_8800", "icvfe_ema_0.5"]  # per-trajectory npz

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

    for name, (subdir, filename) in ARCHIVES.items():
        dest = os.path.join(CACHE_DIR, name)
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        os.makedirs(dest, exist_ok=True)
        src = os.path.join(DATA_ROOT, subdir, filename)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest)


def _single_subdir_root(root: str) -> str:
    """Archives extract into one wrapping folder (e.g. icvfe_8800_icl_demo_dataset_continuous/) -- descend into it."""
    entries = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one subdir under {root}, found {entries}")
    return os.path.join(root, entries[0])


# --------------------------------------------------------------------------
# Loading plain per-frame progress curves (manual, robometer_ft)
# --------------------------------------------------------------------------

def _find_json_dir(root: str) -> str:
    matches = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No json files found under {root}")
    return os.path.dirname(matches[0])


def load_curve_set(name: str) -> dict:
    """Returns {episode_uid: {"task": str, "frame_index": np.ndarray, "progress": np.ndarray}}"""
    root = os.path.join(CACHE_DIR, name)
    json_dir = _find_json_dir(root)
    out = {}
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        frame_index = np.array([p["frame_index"] for p in d["points"]], dtype=np.int64)
        progress = np.array([p["progress"] for p in d["points"]], dtype=np.float64)
        out[d["episode_uid"]] = {
            "task": d["task"],
            "frame_index": frame_index,
            "progress": progress,
        }
    return out


def align_to_reference(entry: dict, ref_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate `entry`'s curve onto every integer frame index it
    spans, then look up `ref_entry`'s (dense) curve at those same frames
    (by frame_index, not list position).
    """
    ref_lookup = np.full(int(ref_entry["frame_index"].max()) + 1, np.nan, dtype=np.float64)
    ref_lookup[ref_entry["frame_index"]] = ref_entry["progress"]

    fidx = entry["frame_index"]
    lo, hi = int(fidx.min()), min(int(fidx.max()), len(ref_lookup) - 1)
    target = np.arange(lo, hi + 1)
    entry_progress = np.interp(target, fidx, entry["progress"])
    ref_progress = ref_lookup[target]

    valid = ~np.isnan(ref_progress)
    return entry_progress[valid], ref_progress[valid]


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    # Undefined for a constant (or single-frame) curve; scored 0 rather than NaN
    # so these episodes count instead of being skipped (matches evaluate_ref._corr).
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y)[0])


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    # Undefined for a constant (or single-frame) curve; scored 0 rather than NaN
    # so these episodes count instead of being skipped (matches evaluate_ref._corr).
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
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
# Flat evaluators vs. human -- robometer_zs/ft_online
# --------------------------------------------------------------------------

def build_flat_rows(manual: dict, flat: dict) -> list:
    """
    Episode-level rows only, computed over each episode's own aligned frames;
    task and total level are averaged from these in main() (aggregation.py).
    """
    common = sorted(set(manual).intersection(*[set(c) for c in flat.values()]))

    rows = []
    for uid in common:
        task = manual[uid]["task"]
        for name, curve_set in flat.items():
            entry = curve_set[uid]
            a, b = align_to_reference(entry, manual[uid])
            a, b = a / 100.0, b / 100.0
            n = len(a)
            pair = f"{name}_vs_human"
            m = compute_all_metrics(a, b)
            for metric_name, value in m.items():
                rows.append({
                    "level": "episode", "task": task, "group_id": uid,
                    "pair": pair, "metric": metric_name, "value": value, "n": n,
                })

    return rows


# --------------------------------------------------------------------------
# NPZ (per-trajectory) evaluators vs. human --
# recap_ft, icvfe_8800, icvfe_ema_0.5
# --------------------------------------------------------------------------

def _traj_aligned_vs_manual(npz_path: str, manual: dict, episode_uid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(prediction, manual) aligned on the npz's own timesteps, both on a 0-1 scale."""
    with np.load(npz_path) as d:
        if "prediction_progress" in d:
            pred = np.asarray(d["prediction_progress"], dtype=np.float64)  # native 0-1 scale
        else:
            # icvfe_ema_0.5's export only ships the signed "prediction" key
            # (prediction_progress - 1, i.e. distance-to-completion in
            # [-1, 0]) -- verified exactly against icvfe_8800, which ships
            # both keys and satisfies prediction == prediction_progress - 1
            # to float32 rounding. Undo the shift to recover the 0-1 scale.
            pred = np.asarray(d["prediction"], dtype=np.float64) + 1.0
        timesteps = np.asarray(d["timesteps"], dtype=np.int64)
        # NOTE: d["target_progress"]/d["target"] are deliberately ignored --
        # they're robodopamine's curve, not human (see study-1 README).

    manual_entry = manual.get(episode_uid)
    if manual_entry is None:
        return None
    manual_progress = manual_entry["progress"] / 100.0  # 0-100 -> 0-1, matches pred's scale
    valid = timesteps < len(manual_progress)
    return pred[valid], manual_progress[timesteps[valid]]


def load_icvfe_trajectories(name: str) -> list:
    ckpt_root = _single_subdir_root(os.path.join(CACHE_DIR, name))
    with open(os.path.join(ckpt_root, "metrics.json")) as f:
        meta = json.load(f)

    out = []
    for t in meta["trajectories"]:
        if t.get("source_curve"):
            npz_path = os.path.join(ckpt_root, t["source_curve"])
        else:
            task_dir = f"task_{t['task_id']:03d}+{t['task_name'].replace(' ', '_')}"
            npz_path = os.path.join(
                ckpt_root, task_dir, f"context_{t['context_demo_id']}",
                "curves", f"query_{t['demo_id']}.npz",
            )
        out.append({"task": t["task_name"], "episode_uid": t["episode_uid"], "npz_path": npz_path})
    return out


def load_recap_trajectories(name: str) -> list:
    ckpt_root = _single_subdir_root(os.path.join(CACHE_DIR, name))
    out = []
    for split in ("seen", "unseen"):
        split_dir = os.path.join(ckpt_root, split)
        with open(os.path.join(split_dir, "metrics.json")) as f:
            meta = json.load(f)
        for t in meta["trajectories"]:
            npz_path = os.path.join(split_dir, "curves", f"task_{t['task_id']:03d}_{t['demo_id']}.npz")
            out.append({"task": t["task_name"], "episode_uid": t["episode_uid"], "npz_path": npz_path})
    return out


NPZ_LOADERS = {
    "recap_ft": load_recap_trajectories,
    "icvfe_8800": load_icvfe_trajectories,
    "icvfe_ema_0.5": load_icvfe_trajectories,
}


def build_npz_rows(manual: dict, name: str) -> list:
    """
    Episode-level rows for one npz-based evaluator vs. human: metrics are
    computed per trajectory (context replicate), then averaged over a query
    episode's replicates (n = replicate count), so each episode counts once
    in the task/total averages computed in main().
    """
    trajectories = NPZ_LOADERS[name](name)
    pair = f"{name}_vs_human"
    metric_names = (METRIC_PEARSON, METRIC_KENDALL, METRIC_MAE)

    traj_rows = []
    for t in trajectories:
        aligned = _traj_aligned_vs_manual(t["npz_path"], manual, t["episode_uid"])
        if aligned is None:
            continue
        x, y = aligned
        traj_rows.append({"task": t["task"], "episode_uid": t["episode_uid"], **compute_all_metrics(x, y)})

    tdf = pd.DataFrame(traj_rows)
    rows = []

    ep_group = tdf.groupby(["task", "episode_uid"], as_index=False)[list(metric_names)].mean()
    n_reps = tdf.groupby(["task", "episode_uid"]).size().rename("n").reset_index()
    ep_group = ep_group.merge(n_reps, on=["task", "episode_uid"])
    for _, r in ep_group.iterrows():
        for metric_name in metric_names:
            rows.append({
                "level": "episode", "task": r["task"], "group_id": r["episode_uid"],
                "pair": pair, "metric": metric_name, "value": r[metric_name], "n": int(r["n"]),
            })

    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Force re-extraction of archives")
    args = parser.parse_args()

    extract_archives(force=args.no_cache)

    manual = load_curve_set(REFERENCE)
    flat = {name: load_curve_set(name) for name in FLAT_ROSTER}

    print(f"Loaded manual={len(manual)}, " + ", ".join(f"{n}={len(c)}" for n, c in flat.items()))

    rows = build_flat_rows(manual, flat)
    for name in NPZ_ROSTER:
        npz_rows = build_npz_rows(manual, name)
        n_eps = len({r["group_id"] for r in npz_rows if r["level"] == "episode"})
        print(f"Loaded {name}: {n_eps} episodes (npz, in-context/no-context per its own design)")
        rows += npz_rows
    rows += aggregate_episode_rows(rows)

    long_df = pd.DataFrame(rows)
    long_df = long_df[["level", "task", "group_id", "pair", "metric", "value", "n"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_path = os.path.join(OUTPUT_DIR, "metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

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

    task_wide = make_wide("task", ["task"]).rename(columns={"n_frames_or_reps": "n_episodes"})
    task_wide.to_csv(os.path.join(OUTPUT_DIR, "task_level_wide.csv"), index=False)

    total_sub = long_df[long_df["level"] == "total"].copy()
    total_wide = total_sub.pivot_table(index="pair", columns="metric", values="value", aggfunc="first")
    total_wide.insert(0, "n_episodes", total_sub.groupby("pair")["n"].max())
    total_wide = total_wide.reset_index()
    total_wide.to_csv(os.path.join(OUTPUT_DIR, "total_level_wide.csv"), index=False)

    print("Wrote episode_level_wide.csv, task_level_wide.csv, total_level_wide.csv")
    print("\nTotal-level summary:")
    print(total_wide.to_string(index=False))


if __name__ == "__main__":
    main()
