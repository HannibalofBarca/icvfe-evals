"""
Actual overlaid progress curves (not just agreement stats).

The pipeline/plotting/baseline_reports scripts all answer "how well does X
track Y" as a single correlation/MAE number. This script instead plots what
the curves themselves look like, at three levels of granularity:

  1. Mean curves (pooled + per-task) -- each episode's progress trace is
     resampled onto a common normalized-time grid (0-100% of episode length)
     so curves of different frame counts can be averaged, then plotted as one
     line per evaluator. `*__global.png`.
  2. Per-episode curves -- every single episode gets its own actual (raw
     frame-index, not resampled) curve plot, organized into one subdirectory
     per task. `*__per_task/<task_slug>/<episode_uid>.png`.
  3. Per-task representative curves -- one episode per task (the longest
     episode -- every episode in this dataset is annotated "success", so
     "longest success" reduces to "longest") as a single representative
     plot per task. `*__representative/<task_slug>.png`.

for two comparisons:

  (a) human_vs_robodopamine -- human curve vs robodopamine curve
  (b) all_evaluators        -- robodopamine / robometer / icvfe / recap
                                curves together (their raw progress traces,
                                not agreement with each other)

Usage:
    python progress_curves.py   # run after pipeline.py (reuses its cache/)
"""
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output", "progress_curves")

N_GRID = 101
GRID = np.linspace(0.0, 100.0, N_GRID)  # x-axis for mean curves: % of episode length

EVALUATOR_COLORS = {
    "human": "#0b0b0b",
    # warm -> cold: recap warmest, then icvfe, then robometer, then
    # robodopamine coldest.
    "recap": "#d7263d",
    "icvfe": "#f4a259",
    "robometer": "#5b9bd5",
    "robodopamine": "#1f3c88",
}
EVALUATOR_LABELS = {
    "human": "Human",
    "robodopamine": "RoboDopamine",
    "robometer": "RoboMeter",
    "icvfe": "ICVFE",
    "recap": "pi0.6-recap",
}
# icvfe/recap are visually noisy (frame-to-frame jitter) and were drowning
# out robodopamine/robometer/human when drawn on top of them at full
# opacity -- draw them behind (lower zorder) and at 75% transparency
# (alpha 0.25) so the smoother curves stay legible.
EVALUATOR_ALPHA = {"human": 1.0, "robodopamine": 1.0, "robometer": 1.0, "icvfe": 0.75, "recap": 0.75}
EVALUATOR_ZORDER = {"human": 3, "robodopamine": 3, "robometer": 3, "icvfe": 1, "recap": 1}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "font.size": 9,
    "font.family": "sans-serif",
})


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def safe_uid(uid: str) -> str:
    return uid.replace(":", "__").replace("/", "_")


# --------------------------------------------------------------------------
# Resampling (for mean curves)
# --------------------------------------------------------------------------

def resample(progress: np.ndarray) -> np.ndarray:
    """Linearly resample a 0-1 scale progress curve onto the common GRID."""
    progress = np.asarray(progress, dtype=np.float64)
    L = len(progress)
    if L == 0:
        return np.full(N_GRID, np.nan)
    if L == 1:
        return np.full(N_GRID, progress[0])
    x = np.linspace(0.0, 100.0, L)
    return np.interp(GRID, x, progress)


def build_raw_resampled(curve_set: dict) -> dict:
    """{episode_uid: {"task": str, "curve": ndarray[N_GRID]}} from a raw 0-100 curve_set."""
    out = {}
    for uid, d in curve_set.items():
        out[uid] = {"task": d["task"], "curve": resample(d["progress"] / 100.0)}
    return out


# --------------------------------------------------------------------------
# ICVFE / recap loaders -- raw per-episode (timesteps + averaged prediction,
# 0-1 scale, NOT resampled) plus a resampled view derived from the same read.
# --------------------------------------------------------------------------

def build_icvfe_episode_curves() -> dict:
    """Averages icvfe's prediction across its in-context replicates per episode
    (timesteps are identical across replicates for the same query episode, so
    the raw pred arrays are averaged directly, no resampling here)."""
    ckpt_root = pl._icvfe_root()
    with open(os.path.join(ckpt_root, "metrics.json")) as f:
        meta = json.load(f)

    preds_by_uid = defaultdict(list)
    ts_by_uid = {}
    task_by_uid = {}
    for t in meta["trajectories"]:
        task_dir = f"task_{t['task_id']:03d}+{t['task_name'].replace(' ', '_')}"
        npz_path = os.path.join(
            ckpt_root, task_dir, f"context_{t['context_demo_id']}", "curves", f"query_{t['demo_id']}.npz"
        )
        with np.load(npz_path) as d:
            pred = np.asarray(d["prediction_progress"], dtype=np.float64)
            ts = np.asarray(d["timesteps"], dtype=np.int64)
        preds_by_uid[t["episode_uid"]].append(pred)
        ts_by_uid[t["episode_uid"]] = ts
        task_by_uid[t["episode_uid"]] = t["task_name"]

    out = {}
    for uid, preds in preds_by_uid.items():
        min_len = min(len(p) for p in preds)
        avg = np.mean([p[:min_len] for p in preds], axis=0)
        out[uid] = {"task": task_by_uid[uid], "timesteps": ts_by_uid[uid][:min_len], "pred": avg}
    return out


def build_recap_episode_curves() -> dict:
    ckpt_root = pl._recap_root()
    out = {}
    for split in ("seen", "unseen"):
        split_dir = os.path.join(ckpt_root, split)
        with open(os.path.join(split_dir, "metrics.json")) as f:
            meta = json.load(f)
        for t in meta["trajectories"]:
            npz_path = os.path.join(split_dir, "curves", f"task_{t['task_id']:03d}_{t['demo_id']}.npz")
            with np.load(npz_path) as d:
                pred = np.asarray(d["prediction_progress"], dtype=np.float64)
                ts = np.asarray(d["timesteps"], dtype=np.int64)
            out[t["episode_uid"]] = {"task": t["task_name"], "timesteps": ts, "pred": pred}
    return out


def resampled_view(episode_curves: dict) -> dict:
    """{uid: {"task":..., "curve": ndarray[N_GRID]}} from an icvfe/recap raw
    episode_curves dict (0-1 scale prediction, arbitrary timesteps)."""
    out = {}
    for uid, d in episode_curves.items():
        out[uid] = {"task": d["task"], "curve": resample(d["pred"])}
    return out


# --------------------------------------------------------------------------
# Mean-curve aggregation
# --------------------------------------------------------------------------

def mean_curve(resampled: dict, uids=None):
    keys = list(resampled.keys()) if uids is None else [u for u in uids if u in resampled]
    arrs = [resampled[k]["curve"] for k in keys]
    if not arrs:
        return None, None, 0
    stacked = np.vstack(arrs)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), len(arrs)


def per_task_means(resampled: dict) -> dict:
    by_task = defaultdict(list)
    for uid, d in resampled.items():
        by_task[d["task"]].append(uid)
    return {task: mean_curve(resampled, uids) for task, uids in by_task.items()}


def build_task_curves(series: dict, names: list) -> dict:
    """series: {evaluator_name: resampled_dict}. Returns {task: {evaluator_name: (mean, std, n)}}."""
    tasks = set()
    for name in names:
        for d in series[name].values():
            tasks.add(d["task"])
    out = {t: {} for t in tasks}
    for name in names:
        for task, (m, sd, n) in per_task_means(series[name]).items():
            out[task][name] = (m, sd, n)
    return out


def plot_global(series: dict, names: list, filename: str, title: str, band: bool = True):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name in names:
        m, sd, n = mean_curve(series[name])
        if m is None:
            continue
        ax.plot(GRID, m * 100, color=EVALUATOR_COLORS[name], linewidth=2.2,
                alpha=EVALUATOR_ALPHA.get(name, 1.0), zorder=EVALUATOR_ZORDER.get(name, 2),
                label=f"{EVALUATOR_LABELS[name]}  (n={n})")
        if band:
            ax.fill_between(GRID, (m - sd) * 100, (m + sd) * 100,
                             color=EVALUATOR_COLORS[name], alpha=0.12, linewidth=0,
                             zorder=EVALUATOR_ZORDER.get(name, 2) - 0.5)
    ax.set_xlabel("Episode progress (% of frames)")
    ax.set_ylabel("Annotated task progress (0-100)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close(fig)


def curves_to_df(series: dict, names: list) -> pd.DataFrame:
    rows = []
    for name in names:
        m, sd, n = mean_curve(series[name])
        if m is None:
            continue
        for pct, mv, sv in zip(GRID, m, sd):
            rows.append({"level": "global", "task": "ALL", "series": name,
                         "pct_progress": pct, "mean": mv, "std": sv, "n_episodes": n})
    task_curves = build_task_curves(series, names)
    for task, sdict in task_curves.items():
        for name, (m, sd, n) in sdict.items():
            if m is None:
                continue
            for pct, mv, sv in zip(GRID, m, sd):
                rows.append({"level": "task", "task": task, "series": name,
                             "pct_progress": pct, "mean": mv, "std": sv, "n_episodes": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per-episode (actual, non-resampled) curves
# --------------------------------------------------------------------------

def plot_episode(ax, curves: dict, names: list, title: str):
    """curves: {evaluator_name: (x_frames, y_0to100)}."""
    ax.clear()
    for name in names:
        entry = curves.get(name)
        if entry is None:
            continue
        x, y = entry
        ax.plot(x, y, color=EVALUATOR_COLORS[name], linewidth=1.6,
                alpha=EVALUATOR_ALPHA.get(name, 1.0), zorder=EVALUATOR_ZORDER.get(name, 2),
                label=EVALUATOR_LABELS[name])
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_ylabel("Progress (0-100)", fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left", wrap=True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def write_episode_tree(episode_curves: dict, task_of: dict, names: list, out_root: str) -> int:
    """One PNG per episode, under out_root/<task_slug>/<episode_uid>.png."""
    os.makedirs(out_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    n_written = 0
    for uid, curves in episode_curves.items():
        task = task_of[uid]
        task_dir = os.path.join(out_root, slugify(task))
        os.makedirs(task_dir, exist_ok=True)
        plot_episode(ax, curves, names, f"{task}\n{uid}")
        fig.tight_layout()
        fig.savefig(os.path.join(task_dir, f"{safe_uid(uid)}.png"), dpi=130)
        n_written += 1
    plt.close(fig)
    return n_written


def pick_longest_per_task(length_of: dict, task_of: dict, uids) -> dict:
    """Returns {task: uid} picking, within `uids`, the episode with the most
    frames per task (every episode in this dataset is annotated "success",
    so "longest success episode" reduces to "longest episode")."""
    best = {}  # task -> (length, uid)
    for uid in uids:
        task = task_of[uid]
        L = length_of[uid]
        if task not in best or L > best[task][0]:
            best[task] = (L, uid)
    return {task: uid for task, (L, uid) in best.items()}


def write_representative(episode_curves: dict, task_of: dict, names: list,
                          longest_by_task: dict, out_root: str) -> int:
    """One PNG per task, under out_root/<task_slug>.png -- the longest episode."""
    os.makedirs(out_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    n_written = 0
    for task, uid in sorted(longest_by_task.items()):
        curves = episode_curves[uid]
        plot_episode(ax, curves, names, f"{task}\n{uid}  (representative: longest episode)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, f"{slugify(task)}.png"), dpi=150)
        n_written += 1
    plt.close(fig)
    return n_written


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    pl.extract_archives()
    os.makedirs(OUT_DIR, exist_ok=True)

    manual_raw = pl.load_curve_set("manual")
    robometer_raw = pl.load_curve_set("finetuned_robometer")
    robodop_raw = pl.load_curve_set("zs_robodopamine")
    icvfe_raw = build_icvfe_episode_curves()
    recap_raw = build_recap_episode_curves()

    human_r = build_raw_resampled(manual_raw)
    robodop_r = build_raw_resampled(robodop_raw)
    robometer_r = build_raw_resampled(robometer_raw)
    icvfe_r = resampled_view(icvfe_raw)
    recap_r = resampled_view(recap_raw)

    print(f"Loaded: human={len(human_r)} robodopamine={len(robodop_r)} "
          f"robometer={len(robometer_r)} icvfe={len(icvfe_r)} recap={len(recap_r)}")

    # Match pipeline.py's convention: restrict the three raw curves to the
    # 243 episodes common to all three (26 robodopamine/robometer episodes
    # have no human annotation and would otherwise skew the mean curve).
    common_a = sorted(set(human_r) & set(robodop_r) & set(robometer_r))
    human_common_r = {k: human_r[k] for k in common_a}
    robodop_common_r = {k: robodop_r[k] for k in common_a}
    print(f"human_vs_robodopamine: {len(common_a)} episodes common to human/robodopamine/robometer")

    # (b) restricted to the episodes all four evaluators cover (icvfe/recap's
    # ~240-episode eval set), so the comparison isn't skewed by
    # robodopamine/robometer's extra episodes.
    common_b = sorted(set(robodop_r) & set(robometer_r) & set(icvfe_r) & set(recap_r))
    print(f"all_evaluators: {len(common_b)} episodes common to all four")

    # ---- (a) mean curves ----
    series_a = {"human": human_common_r, "robodopamine": robodop_common_r}
    names_a = ["human", "robodopamine"]
    plot_global(series_a, names_a, "human_vs_robodopamine__global.png",
                "Mean progress curve — Human vs RoboDopamine (pooled, all episodes)", band=True)
    curves_to_df(series_a, names_a).to_csv(
        os.path.join(OUT_DIR, "curve_data_human_vs_robodopamine.csv"), index=False)

    # ---- (b) mean curves ----
    series_b = {
        "robodopamine": {k: robodop_r[k] for k in common_b},
        "robometer": {k: robometer_r[k] for k in common_b},
        "icvfe": {k: icvfe_r[k] for k in common_b},
        "recap": {k: recap_r[k] for k in common_b},
    }
    names_b = ["robodopamine", "robometer", "icvfe", "recap"]
    plot_global(series_b, names_b, "all_evaluators__global.png",
                "Mean progress curve — RoboDopamine vs RoboMeter vs ICVFE vs pi0.6-recap", band=False)
    curves_to_df(series_b, names_b).to_csv(
        os.path.join(OUT_DIR, "curve_data_all_evaluators.csv"), index=False)

    # ---- (a) per-episode actual curves (one tree, one representative dir) ----
    task_of_a = {uid: manual_raw[uid]["task"] for uid in common_a}
    episode_curves_a = {
        uid: {
            "human": (np.arange(len(manual_raw[uid]["progress"])), manual_raw[uid]["progress"]),
            "robodopamine": (np.arange(len(robodop_raw[uid]["progress"])), robodop_raw[uid]["progress"]),
        }
        for uid in common_a
    }
    n = write_episode_tree(episode_curves_a, task_of_a, names_a,
                            os.path.join(OUT_DIR, "human_vs_robodopamine__per_task"))
    print(f"Wrote {n} per-episode human_vs_robodopamine plots")

    length_of_a = {uid: len(manual_raw[uid]["progress"]) for uid in common_a}
    longest_a = pick_longest_per_task(length_of_a, task_of_a, common_a)
    n = write_representative(episode_curves_a, task_of_a, names_a, longest_a,
                              os.path.join(OUT_DIR, "human_vs_robodopamine__representative"))
    print(f"Wrote {n} representative human_vs_robodopamine plots")

    # ---- (b) per-episode actual curves ----
    task_of_b = {uid: robodop_raw[uid]["task"] for uid in common_b}
    episode_curves_b = {
        uid: {
            "robodopamine": (np.arange(len(robodop_raw[uid]["progress"])), robodop_raw[uid]["progress"]),
            "robometer": (np.arange(len(robometer_raw[uid]["progress"])), robometer_raw[uid]["progress"]),
            "icvfe": (icvfe_raw[uid]["timesteps"], icvfe_raw[uid]["pred"] * 100.0),
            "recap": (recap_raw[uid]["timesteps"], recap_raw[uid]["pred"] * 100.0),
        }
        for uid in common_b
    }
    n = write_episode_tree(episode_curves_b, task_of_b, names_b,
                            os.path.join(OUT_DIR, "all_evaluators__per_task"))
    print(f"Wrote {n} per-episode all_evaluators plots")

    length_of_b = {uid: len(robodop_raw[uid]["progress"]) for uid in common_b}
    longest_b = pick_longest_per_task(length_of_b, task_of_b, common_b)
    n = write_representative(episode_curves_b, task_of_b, names_b, longest_b,
                              os.path.join(OUT_DIR, "all_evaluators__representative"))
    print(f"Wrote {n} representative all_evaluators plots")

    print(f"Done. All outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
