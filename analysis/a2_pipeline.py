"""
A2 ablation pipeline: fine-tuned vs. zero-shot.

Per Eval_Research_Summary.md, ablation A2 ("Fine-tuned vs. zero-shot"):
  variants: RoboMeter FT vs. ZS; RoboDopamine FT vs. ZS
  scored against: manual (human) annotations
  metrics: Pearson r, Kendall tau-b, MAE -- at episode / task / total granularity
  tests whether adaptation helps at all, per model family

Both robometer variants here are the "base" (non-online) exports, matching
A1's robometer_zs choice -- the "online" exports are held back for the
online-eligible retrieval ablations (B1/B2) per user direction.

Same frame_index + linear-interpolation alignment as a1_pipeline.py: the
finetuned_robodopamine export samples every 10th frame (sparse, like
topreward in A1), so it's interpolated onto the dense per-frame grid before
comparing against manual, rather than compared only at its raw sample points.

Usage:
    python a2_pipeline.py            # extract (if needed) + compute + write CSVs to output/A2/
    python a2_pipeline.py --no-cache # force re-extraction of archives
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
CACHE_DIR = os.path.join(HERE, "cache", "A2-finetuned vs zeroshot")
OUTPUT_DIR = os.path.join(HERE, "output", "A2-finetuned vs zeroshot")

REFERENCE = "manual"

# name -> (subdir under DATA_ROOT, archive filename)
ARCHIVES = {
    "manual": ("manual", "manual_icl_demo_dataset_continuous.zip"),
    "robometer_zs": ("robometer", "zeroshot_robometer_icl_demo_dataset_continuous.zip"),
    "robometer_ft": ("robometer", "finetuned_robometer_icl_demo_dataset_continuous.zip"),
    "robodopamine_zs": ("robo_dopamine", "zs_robodopamine_icl_demo_dataset_continuous.zip"),
    "robodopamine_ft": ("robo_dopamine", "finetuned_robodopamine_icl_demo_dataset_continuous.zip"),
}
ROSTER = ["robometer_zs", "robometer_ft", "robodopamine_zs", "robodopamine_ft"]

# model family -> (zs name, ft name), for the FT-vs-ZS delta summary
MODEL_FAMILIES = {
    "robometer": ("robometer_zs", "robometer_ft"),
    "robodopamine": ("robodopamine_zs", "robodopamine_ft"),
}

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
    sparser/irregular subset of frames than the reference (e.g. the
    finetuned robodopamine export vs. the dense manual curve), so
    interpolating first means sparse evaluators get compared at full
    per-frame resolution rather than only at their handful of raw sample
    points.

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
# Roster-vs-human analysis
# --------------------------------------------------------------------------

def build_rows(manual: dict, roster: dict) -> list:
    """Episode/task/total rows for every roster evaluator vs. manual (human).

    Episodes are restricted to those common to manual and every roster
    source, so all pairs are computed over the same episode set (consistent
    with the study-1 pipeline's rationale for doing the same). Task and total
    level are the mean of the per-episode metrics (see aggregation.py).
    """
    common = sorted(set(manual).intersection(*[set(c) for c in roster.values()]))

    rows = []
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

    return rows + aggregate_episode_rows(rows)


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
    long_df = long_df[["level", "task", "group_id", "pair", "metric", "value", "n"] +
                       (["n_episodes"] if "n_episodes" in long_df.columns else [])]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_path = os.path.join(OUTPUT_DIR, "metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    def make_wide(level: str, index_cols: list) -> pd.DataFrame:
        sub = long_df[long_df["level"] == level].copy()
        sub["col"] = sub["pair"] + "__" + sub["metric"]
        wide = sub.pivot_table(index=index_cols, columns="col", values="value", aggfunc="first")
        n_col = sub.groupby(index_cols)["n"].max()
        wide.insert(0, "n_frames", n_col)
        wide = wide.reset_index()
        return wide

    episode_wide = make_wide("episode", ["task", "group_id"]).rename(columns={"group_id": "episode_uid"})
    episode_wide.to_csv(os.path.join(OUTPUT_DIR, "episode_level_wide.csv"), index=False)

    task_wide = make_wide("task", ["task"]).rename(columns={"n_frames": "n_episodes"})
    task_wide.to_csv(os.path.join(OUTPUT_DIR, "task_level_wide.csv"), index=False)

    total_sub = long_df[long_df["level"] == "total"].copy()
    total_wide = total_sub.pivot_table(index="pair", columns="metric", values="value", aggfunc="first")
    total_wide.insert(0, "n_episodes", total_sub.groupby("pair")["n_episodes"].max())
    total_wide = total_wide.reset_index()
    total_wide.to_csv(os.path.join(OUTPUT_DIR, "total_level_wide.csv"), index=False)

    print("Wrote episode_level_wide.csv, task_level_wide.csv, total_level_wide.csv")
    print("\nTotal-level summary:")
    print(total_wide.to_string(index=False))

    # FT-vs-ZS delta summary, per model family, at total level (does adaptation help?)
    print("\nFT - ZS deltas (total level, positive = fine-tuning helped):")
    total_by_pair = total_wide.set_index("pair")
    for family, (zs_name, ft_name) in MODEL_FAMILIES.items():
        zs_row = total_by_pair.loc[f"{zs_name}_vs_human"]
        ft_row = total_by_pair.loc[f"{ft_name}_vs_human"]
        d_pearson = ft_row[METRIC_PEARSON] - zs_row[METRIC_PEARSON]
        d_kendall = ft_row[METRIC_KENDALL] - zs_row[METRIC_KENDALL]
        d_mae = ft_row[METRIC_MAE] - zs_row[METRIC_MAE]  # negative = FT lowered error = helped
        print(f"  {family}: pearson {d_pearson:+.3f}, kendall_tau_b {d_kendall:+.3f}, mae {d_mae:+.3f}")


if __name__ == "__main__":
    main()
