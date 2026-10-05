"""
A1 ablation pipeline: reward model roster vs. human.

Per Eval_Research_Summary.md, ablation A1 ("Reward model roster vs. human"):
  variants: TOPReward, GVL, RoboMeter ZS, RoboDopamine ZS
  scored against: manual (human) annotations
  metrics: Pearson r, Kendall tau-b, MAE -- at episode / task / total granularity

(ProcVLM and LIV were dropped from this ablation's roster per user direction --
ProcVLM's export hasn't been produced yet, and LIV wasn't distinguished from
plain DINO in the data drop.)

IMPORTANT -- alignment is by frame_index, with linear interpolation, not
list position: unlike the study-1 sources (manual/robodopamine/robometer,
all dense per-frame arrays of identical length), topreward samples every
10th frame and gvl samples an irregular, non-uniform subset of frames.
Zipping progress arrays positionally and truncating to min length (as
pipeline.py does) would silently pair each sparse evaluator's point i
against manual's frame i, not against the frame that point i actually
describes. Instead, each evaluator's curve is linearly interpolated onto
every integer frame index it spans, then looked up against the (dense)
human curve at each of those frames -- so sparse evaluators are compared
at full per-frame resolution instead of only at their handful of raw
sample points.

Usage:
    python a1_pipeline.py            # extract (if needed) + compute + write CSVs to output/A1/
    python a1_pipeline.py --no-cache # force re-extraction of archives
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "data", "demo_set_annotations", "demo_set_annotations")
)
CACHE_DIR = os.path.join(HERE, "cache", "A1-reward model ablation")
OUTPUT_DIR = os.path.join(HERE, "output", "A1-reward model ablation")

REFERENCE = "manual"

# name -> (subdir under DATA_ROOT, archive filename)
ARCHIVES = {
    "manual": ("manual", "manual_icl_demo_dataset_continuous.zip"),
    "topreward": ("topreward", "zs_topreward_icl_demo_dataset_continuous.zip"),
    "gvl": ("gvl", "zs_gvl_icl_demo_dataset_continuous.zip"),
    "robometer_zs": ("robometer", "zeroshot_robometer_icl_demo_dataset_continuous.zip"),
    "robodopamine_zs": ("robo_dopamine", "zs_robodopamine_icl_demo_dataset_continuous.zip"),
}
ROSTER = ["topreward", "gvl", "robometer_zs", "robodopamine_zs"]

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


# --------------------------------------------------------------------------
# Loading raw progress curves
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
    spans, then look up `ref_entry`'s (dense) curve at those same frames.
    This is by frame_index, not list position -- entry may sample a
    sparser/irregular subset of frames than the reference (e.g.
    topreward/gvl vs. the dense manual curve), so interpolating first means
    sparse evaluators get compared at full per-frame resolution rather than
    only at their handful of raw sample points.

    Returns (entry_progress_aligned, ref_progress_aligned), restricted to
    frame indices entry's span actually has that also exist in the reference.
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
# Roster-vs-human analysis
# --------------------------------------------------------------------------

def build_rows(manual: dict, roster: dict) -> list:
    """Episode/task/total rows for every roster evaluator vs. manual (human).

    Episodes are restricted to those common to manual and every roster
    source, so all pairs are computed over the same episode set (consistent
    with the study-1 pipeline's rationale for doing the same).

    Nested aggregation throughout: episode level is computed over that
    episode's own (aligned) frames -- frames within an episode are highly
    autocorrelated, so this is the smallest unit where pooling raw points is
    still appropriate. Task/total level average the resulting episode-level
    numbers instead of pooling frames across episodes -- pooling would treat
    every frame as an independent observation when the episode, not the
    frame, is the actual independent sampling unit (longer episodes would
    otherwise dominate the number just by contributing more non-independent
    points). This matches the aggregation `a3_pipeline.build_npz_rows`
    already uses for the npz sources.
    """
    common = sorted(set(manual).intersection(*[set(c) for c in roster.values()]))

    rows = []
    episode_rows = defaultdict(list)  # name -> [{"task": ..., pearson, kendall_tau_b, mae}, ...]

    for uid in common:
        task = manual[uid]["task"]
        for name, curve_set in roster.items():
            entry = curve_set[uid]
            # progress ships as 0-100; rescaling to a 0-1 fraction of task
            # progress rather than reporting MAE in "points out of 100",
            # matching pipeline.py's convention (see its README section).
            a, b = align_to_reference(entry, manual[uid])
            a, b = a / 100.0, b / 100.0
            n = len(a)
            pair = f"{name}_vs_human"
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
            episode_rows[name].append({"task": task, **m})

    for name, ep_rows in episode_rows.items():
        edf = pd.DataFrame(ep_rows)
        pair = f"{name}_vs_human"

        task_group = edf.groupby("task", as_index=False)[[METRIC_PEARSON, METRIC_KENDALL, METRIC_MAE]].mean()
        n_ep_per_task = edf.groupby("task").size().rename("n").reset_index()
        task_group = task_group.merge(n_ep_per_task, on="task")
        for _, r in task_group.iterrows():
            for metric_name in (METRIC_PEARSON, METRIC_KENDALL, METRIC_MAE):
                rows.append({
                    "level": "task",
                    "task": r["task"],
                    "group_id": r["task"],
                    "pair": pair,
                    "metric": metric_name,
                    "value": r[metric_name],
                    "n": int(r["n"]),
                })

        for metric_name in (METRIC_PEARSON, METRIC_KENDALL, METRIC_MAE):
            rows.append({
                "level": "total",
                "task": "ALL",
                "group_id": "ALL",
                "pair": pair,
                "metric": metric_name,
                "value": float(edf[metric_name].mean()),
                "n": len(edf),
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
    roster = {name: load_curve_set(name) for name in ROSTER}

    counts = ", ".join(f"{name}={len(c)}" for name, c in roster.items())
    print(f"Loaded manual={len(manual)}, {counts}")

    rows = build_rows(manual, roster)
    long_df = pd.DataFrame(rows)
    long_df = long_df[["level", "task", "group_id", "pair", "metric", "value", "n"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_path = os.path.join(OUTPUT_DIR, "metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    def make_wide(level: str, index_cols: list, n_label: str) -> pd.DataFrame:
        sub = long_df[long_df["level"] == level].copy()
        sub["col"] = sub["pair"] + "__" + sub["metric"]
        wide = sub.pivot_table(index=index_cols, columns="col", values="value", aggfunc="first")
        n_col = sub.groupby(index_cols)["n"].max()
        wide.insert(0, n_label, n_col)
        wide = wide.reset_index()
        return wide

    # "n" means frame count at episode level, but episode count (the number
    # of episode-level numbers averaged) at task/total level -- see build_rows.
    episode_wide = make_wide("episode", ["task", "group_id"], "n_frames").rename(columns={"group_id": "episode_uid"})
    episode_wide.to_csv(os.path.join(OUTPUT_DIR, "episode_level_wide.csv"), index=False)

    task_wide = make_wide("task", ["task"], "n_episodes")
    task_wide.to_csv(os.path.join(OUTPUT_DIR, "task_level_wide.csv"), index=False)

    total_sub = long_df[long_df["level"] == "total"].copy()
    total_wide = total_sub.pivot_table(index="pair", columns="metric", values="value", aggfunc="first")
    n_map = total_sub.groupby("pair")["n"].max()
    total_wide.insert(0, "n_episodes", n_map)
    total_wide = total_wide.reset_index()
    total_wide.to_csv(os.path.join(OUTPUT_DIR, "total_level_wide.csv"), index=False)

    print("Wrote episode_level_wide.csv, task_level_wide.csv, total_level_wide.csv")
    print("\nTotal-level summary:")
    print(total_wide.to_string(index=False))


if __name__ == "__main__":
    main()
