"""
Single-episode progress-curve figure for episode 4 of "sort the food items
into the cups", for the paper.

Five series, on the same (frame-index) x-axis:
  - IC-VFE (8800):     mean prediction across this query episode's 9
                        in-context replicates. The line is EMA-smoothed
                        (alpha=ICVFE_EMA_ALPHA); the faint background band is
                        a rolling-window std (window=ROLLING_STD_WINDOW
                        frames) of the *un-smoothed* mean curve -- the band
                        is deliberately not EMA-smoothed, so it reflects the
                        raw local variability the smoothed line is hiding.
  - RECAP (ft, 20000): same treatment (EMA-smoothed line + un-smoothed
                        rolling-std band), computed on RECAP's own raw curve
                        -- it has no in-context replicates (context_demo_id
                        is always "" -- single pass per episode, see repo
                        README), so rolling std (not cross-replicate std) is
                        the only option here, and IC-VFE uses the same
                        method for consistency.
  - RoboMeter (online finetuned): plotted as-is -- raw curve, no smoothing.
  - Human (manual annotation):    raw curve, black, reference signal.
  - RoboDopamine (zero-shot):     raw curve, dark grey.

Milestones (from manual_icl_demo_dataset_milestone.zip's per-episode
key_events) are marked with a dotted vertical guideline down to the x-axis
at that event's frame, topped with a caret marker (no shaft/arrowhead line,
just the "^"/"v" glyph) -- green "^" where the milestone's value increased
over the previous key event, red "v" where it dropped (e.g. "drop the
eggplant" mid-episode). The very first event ("start") has no prior value to
compare against, so it gets no marker.

Draw order (front to back): IC-VFE > RECAP > RoboMeter > RoboDopamine >
Human (manual is drawn behind everything else).

Colors: manual=black, robodopamine=dark grey (both fixed per spec); IC-VFE /
RECAP / RoboMeter online each take one color from PALETTE.

Usage:
    python paper_figures_sorting_cups_ep4.py
"""
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import a1_pipeline as a1
import a3_pipeline as a3

TASK = "sort the food items into the cups"
EPISODE_UID = "sorting_food_items_into_cups_20260803_134551:000004"

ICVFE_EMA_ALPHA = 0.075
RECAP_EMA_ALPHA = 0.075
ROLLING_STD_WINDOW = 15  # frames; local-noise proxy band for both IC-VFE and RECAP

MILESTONE_ARCHIVE = ("manual", "manual_icl_demo_dataset_milestone.zip")
MILESTONE_CACHE_DIR = os.path.join(a3.HERE, "cache", "milestone_manual")
MILESTONE_UP_COLOR = "#1a9850"
MILESTONE_DOWN_COLOR = "#d73027"

REPO_ROOT = os.path.normpath(os.path.join(a3.HERE, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "paper_figures")

# Saturation (HSV) doubled from the original brief's palette
# (["#FFC5C5", "#C7EFE0", "#B4E5A2", "#C0E5EF", "#FFE89F"]), hue/value held fixed.
# PALETTE[1] (cyan/mint, used for RoboMeter) additionally boosted 1.5x on top of that.
PALETTE = ["#FF8B8B", "#77EFC2", "#83E55F", "#91DBEF", "#FFD13F"]

COLORS = {
    "icvfe": PALETTE[0],
    "robometer_online": PALETTE[1],
    "recap": PALETTE[2],
    "manual": "#000000",
    # Same HSV saturation as the PALETTE average and the hue of PALETTE's
    # blue swatch (#91DBEF), at low value so it still reads as "dark grey"
    # next to black -- not a flat 0-saturation grey, so it belongs to the
    # same family as everything else. Saturation doubled along with PALETTE.
    "robodopamine": "#396773",
}
LABELS = {
    "icvfe": "IC-VFE",
    "recap": "$\\pi_{0.6}$ RECAP",
    "robometer_online": "RoboMeter",
    "manual": "Human annotation",
    "robodopamine": "RoboDopamine",
}
LINEWIDTHS = {"icvfe": 1.0, "recap": 1.0, "robometer_online": 1.0, "manual": 1.0, "robodopamine": 1.0}
# Draw order (front to back): icvfe > recap > robometer > robodopamine > manual
ZORDER = {"icvfe": 5, "recap": 4, "robometer_online": 3, "robodopamine": 2, "manual": 1}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": False,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.5,
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


def load_icvfe_replicates(uid: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(timesteps, preds) where preds is (n_replicates, n_frames), 0-1 scale."""
    trajectories = [t for t in a3.load_icvfe_trajectories("icvfe_8800")
                     if t["episode_uid"] == uid and t["task"] == task]
    preds, ts = [], None
    for t in trajectories:
        with np.load(t["npz_path"]) as d:
            pred = np.asarray(d["prediction_progress"], dtype=np.float64)
            cur_ts = np.asarray(d["timesteps"], dtype=np.int64)
        if ts is None:
            ts = cur_ts
        preds.append(pred)
    min_len = min(len(p) for p in preds)
    preds = np.stack([p[:min_len] for p in preds], axis=0)
    return ts[:min_len], preds


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(window, center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy()
    )


def load_milestone_events(uid: str) -> list[dict]:
    """key_events list for one episode (frame/value/subtask), 0-1 value scale."""
    if not (os.path.isdir(MILESTONE_CACHE_DIR) and os.listdir(MILESTONE_CACHE_DIR)):
        os.makedirs(MILESTONE_CACHE_DIR, exist_ok=True)
        subdir, filename = MILESTONE_ARCHIVE
        src = os.path.join(a1.DATA_ROOT, subdir, filename)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(MILESTONE_CACHE_DIR)

    json_dir = a1._find_json_dir(MILESTONE_CACHE_DIR)
    safe_uid = uid.replace(":", "__")
    with open(os.path.join(json_dir, f"{safe_uid}.json")) as f:
        d = json.load(f)
    return d["key_events"]


def load_recap_raw(uid: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(timesteps, pred) 0-1 scale, single pass."""
    for t in a3.load_recap_trajectories("recap_ft"):
        if t["episode_uid"] == uid and t["task"] == task:
            with np.load(t["npz_path"]) as d:
                pred = np.asarray(d["prediction_progress"], dtype=np.float64)
                ts = np.asarray(d["timesteps"], dtype=np.int64)
            return ts, pred
    raise KeyError(f"No RECAP trajectory found for {uid!r} / {task!r}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    icvfe_ts, icvfe_preds = load_icvfe_replicates(EPISODE_UID, TASK)
    icvfe_mean = icvfe_preds.mean(axis=0)
    icvfe_mean_ema = ema(icvfe_mean, ICVFE_EMA_ALPHA)
    # icvfe_std = rolling_std(icvfe_mean, ROLLING_STD_WINDOW)  # rolling std band removed

    recap_ts, recap_pred = load_recap_raw(EPISODE_UID, TASK)
    recap_mean_ema = ema(recap_pred, RECAP_EMA_ALPHA)
    # recap_std = rolling_std(recap_pred, ROLLING_STD_WINDOW)  # rolling std band removed

    a1.extract_archives()
    manual_entry = a1.load_curve_set("manual")[EPISODE_UID]
    robodop_entry = a1.load_curve_set("robodopamine_zs")[EPISODE_UID]
    robometer_entry = a3.load_curve_set("robometer_ft_online")[EPISODE_UID]

    for entry in (manual_entry, robodop_entry, robometer_entry):
        progress = np.asarray(entry["progress"], dtype=np.float64)
        entry["progress"] = progress / 100.0

    fig, ax = plt.subplots(figsize=(8.5, 2.0))

    # ax.fill_between(icvfe_ts, icvfe_mean_ema - icvfe_std, icvfe_mean_ema + icvfe_std,
    #                  color=COLORS["icvfe"], alpha=0.15, linewidth=0, zorder=0)
    # ax.fill_between(recap_ts, recap_mean_ema - recap_std, recap_mean_ema + recap_std,
    #                  color=COLORS["recap"], alpha=0.15, linewidth=0, zorder=0)

    ax.plot(manual_entry["frame_index"], manual_entry["progress"],
            color=COLORS["manual"], linewidth=LINEWIDTHS["manual"],
            zorder=ZORDER["manual"], label=LABELS["manual"], alpha=0.9)
    ax.plot(robodop_entry["frame_index"], robodop_entry["progress"],
            color=COLORS["robodopamine"], linewidth=LINEWIDTHS["robodopamine"],
            zorder=ZORDER["robodopamine"], label=LABELS["robodopamine"], alpha=0.75)
    ax.plot(robometer_entry["frame_index"], robometer_entry["progress"],
            color=COLORS["robometer_online"], linewidth=LINEWIDTHS["robometer_online"],
            zorder=ZORDER["robometer_online"], label=LABELS["robometer_online"], alpha=0.75)
    ax.plot(recap_ts, recap_mean_ema,
            color=COLORS["recap"], linewidth=LINEWIDTHS["recap"],
            zorder=ZORDER["recap"], label=LABELS["recap"], alpha=0.75)
    ax.plot(icvfe_ts, icvfe_mean_ema,
            color=COLORS["icvfe"], linewidth=LINEWIDTHS["icvfe"],
            zorder=ZORDER["icvfe"], label=LABELS["icvfe"], alpha=0.75)

    milestone_events = load_milestone_events(EPISODE_UID)
    for prev_event, event in zip(milestone_events, milestone_events[1:]):
        frame, value, delta = event["frame"], event["value"], event["value"] - prev_event["value"]
        if delta == 0:
            continue
        color = MILESTONE_UP_COLOR if delta > 0 else MILESTONE_DOWN_COLOR
        marker = "^" if delta > 0 else "v"
        ax.plot([frame, frame], [0.0, value], linestyle=":", linewidth=0.9,
                 color=color, zorder=8)
        ax.plot([frame], [value], marker=marker, markersize=7, color=color,
                 markeredgewidth=0, linestyle="none", zorder=9)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Value (0-1)")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "sorting_food_items_into_cups__episode_4.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
