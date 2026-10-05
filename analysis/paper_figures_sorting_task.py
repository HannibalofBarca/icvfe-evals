"""
Per-episode progress-curve figures for the long-horizon sorting task
("sort the items into their containers"), for the paper.

Five curves per episode, on the same (frame-index) x-axis:
  - ICVFE raw          (icvfe_8800, averaged across its 9 in-context
                         replicates per query episode -- same convention as
                         progress_curves.build_icvfe_episode_curves)
  - ICVFE EMA          (alpha=0.05 EMA computed here from ICVFE's own raw
                         curve -- NOT the precomputed icvfe_ema_0.5 export,
                         which is fixed at alpha=0.5 and too light to read as
                         smooth; computing it ourselves lets both EMA curves
                         use the same, stronger smoothing constant)
  - RECAP raw          (recap_ft, single pass per episode, no replicates)
  - RECAP EMA          (alpha=0.05 EMA computed here from RECAP's raw curve --
                         no precomputed RECAP EMA export exists in the data)
  - Manual (human)     (reference signal -- dense per-frame curve from the
                         manual_icl_demo_dataset_continuous archive, i.e. the
                         actual human-annotated ground truth, NOT the npz's
                         own `target_progress` field -- see README's "ICVFE's
                         target is not the human curve" section for why that
                         field is actually RoboDopamine's curve, not this one.
                         Plotted at lowered alpha so it reads as a reference
                         band rather than competing with the raw/EMA lines.)

Usage:
    python paper_figures_sorting_task.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import a1_pipeline as a1
import a3_pipeline as a3

TASK = "stack the three cubes into a tower"
ICVFE_EMA_ALPHA = 0.05
RECAP_EMA_ALPHA = 0.05

REPO_ROOT = os.path.normpath(os.path.join(a3.HERE, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "paper_figures")

COLORS = {
    "icvfe_raw": "#f4a259",
    "icvfe_ema": "#c96f1e",
    "recap_raw": "#d7263d",
    "recap_ema": "#8c1626",
    "manual": "#1f3c88",
}
LABELS = {
    "icvfe_raw": "IC-VFE (raw)",
    "icvfe_ema": "IC-VFE (EMA, $\\alpha$=0.05)",
    "recap_raw": "$\\pi_{0.6}$ RECAP (raw)",
    "recap_ema": "$\\pi_{0.6}$ RECAP (EMA, $\\alpha$=0.05)",
    "manual": "Human annotation (reference)",
}
LINESTYLES = {
    "icvfe_raw": "-", "icvfe_ema": "-",
    "recap_raw": "-", "recap_ema": "-",
    "manual": "-",
}
# Raw curves are dense per-frame noise that drowns out the EMA trend at
# full-episode width -- thin + faded raw, thick + opaque EMA on top, matching
# the reference repo's own rescore_vfe_ema.py plot_episode() convention.
LINEWIDTHS = {"icvfe_raw": 0.8, "icvfe_ema": 2.0, "recap_raw": 0.8, "recap_ema": 2.0, "manual": 2.0}
ALPHAS = {"icvfe_raw": 0.45, "icvfe_ema": 1.0, "recap_raw": 0.45, "recap_ema": 1.0, "manual": 0.5}
ZORDER = {"icvfe_raw": 1, "icvfe_ema": 2, "recap_raw": 1, "recap_ema": 2, "manual": 3}

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


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Standard exponential moving average: y[0]=x[0], y[t]=alpha*x[t]+(1-alpha)*y[t-1]."""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _get_curves(d) -> tuple[np.ndarray, np.ndarray]:
    """(target, prediction) 0-1 scale, handling icvfe_ema_0.5's signed-only export
    (see a3_pipeline._traj_aligned_vs_manual for the same fallback)."""
    if "prediction_progress" in d:
        pred = np.asarray(d["prediction_progress"], dtype=np.float64)
    else:
        pred = np.asarray(d["prediction"], dtype=np.float64) + 1.0
    return pred


def build_averaged_episode_curves(name: str, task: str) -> dict:
    """{episode_uid: {"timesteps": ndarray, "pred": ndarray}} -- averages
    prediction across in-context replicates per query episode (timesteps are
    identical across replicates for the same query episode)."""
    trajectories = [t for t in a3.load_icvfe_trajectories(name) if t["task"] == task]
    preds_by_uid, ts_by_uid = {}, {}
    for t in trajectories:
        with np.load(t["npz_path"]) as d:
            pred = _get_curves(d)
            ts = np.asarray(d["timesteps"], dtype=np.int64)
        preds_by_uid.setdefault(t["episode_uid"], []).append(pred)
        ts_by_uid[t["episode_uid"]] = ts

    out = {}
    for uid, preds in preds_by_uid.items():
        min_len = min(len(p) for p in preds)
        avg = np.mean([p[:min_len] for p in preds], axis=0)
        out[uid] = {"timesteps": ts_by_uid[uid][:min_len], "pred": avg}
    return out


def build_recap_episode_curves(task: str) -> dict:
    """{episode_uid: {"timesteps": ndarray, "pred": ndarray}} -- single pass per episode."""
    out = {}
    for t in a3.load_recap_trajectories("recap_ft"):
        if t["task"] != task:
            continue
        with np.load(t["npz_path"]) as d:
            pred = _get_curves(d)
            ts = np.asarray(d["timesteps"], dtype=np.int64)
        out[t["episode_uid"]] = {"timesteps": ts, "pred": pred}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    icvfe_raw = build_averaged_episode_curves("icvfe_8800", TASK)
    recap_raw = build_recap_episode_curves(TASK)

    a1.extract_archives()
    manual_dense = a1.load_curve_set("manual")

    common = sorted(set(icvfe_raw) & set(recap_raw))
    print(f"Found {len(common)} episodes for task {TASK!r}: {common}")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n_written = 0
    for uid in common:
        ax.clear()

        recap_ts, recap_pred = recap_raw[uid]["timesteps"], recap_raw[uid]["pred"]
        recap_ema_pred = ema(recap_pred, RECAP_EMA_ALPHA)

        icvfe_ts, icvfe_pred = icvfe_raw[uid]["timesteps"], icvfe_raw[uid]["pred"]
        icvfe_ema_pred = ema(icvfe_pred, ICVFE_EMA_ALPHA)

        manual_entry = manual_dense.get(uid)
        series = {
            "manual": (manual_entry["frame_index"], manual_entry["progress"]) if manual_entry else None,
            "icvfe_raw": (icvfe_ts, icvfe_pred * 100.0),
            "icvfe_ema": (icvfe_ts, icvfe_ema_pred * 100.0),
            "recap_raw": (recap_ts, recap_pred * 100.0),
            "recap_ema": (recap_ts, recap_ema_pred * 100.0),
        }
        for name, entry in series.items():
            if entry is None:
                continue
            x, y = entry
            ax.plot(x, y, color=COLORS[name], linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
                     alpha=ALPHAS[name], zorder=ZORDER[name], label=LABELS[name])

        ax.set_xlabel("Frame")
        ax.set_ylabel("Progress (0-100)")
        ax.set_ylim(-5, 105)
        ax.set_title(f"{TASK}\n{uid}", fontsize=9, fontweight="bold", loc="left", wrap=True)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        safe_uid = uid.replace(":", "__").replace("/", "_")
        out_path = os.path.join(OUT_DIR, f"{safe_uid}.png")
        fig.savefig(out_path, dpi=160)
        n_written += 1

    plt.close(fig)
    print(f"Wrote {n_written} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
