"""
Graphing pipeline for the evaluator-agreement tables produced by pipeline.py.

Reads analysis/output/*.csv and writes plain matplotlib PNG figures to
analysis/output/figures/. No interactive/JS component - just a reproducible
Python charting step, run after pipeline.py.

Usage:
    python plotting.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")

PAIR_ORDER = [
    "robodopamine_vs_human",
    "robometer_vs_human",
    "icvfe_vs_human",
    "recap_vs_human",
    "robometer_vs_robodopamine",
    "icvfe_vs_robodopamine",
    "recap_vs_robodopamine",
]
PAIR_LABELS = {
    "robodopamine_vs_human": "RoboDopamine vs Human",
    "robometer_vs_human": "RoboMeter vs Human",
    "icvfe_vs_human": "ICVFE vs Human",
    "recap_vs_human": "pi0.6-recap vs Human",
    "robometer_vs_robodopamine": "RoboMeter vs RoboDopamine",
    "icvfe_vs_robodopamine": "ICVFE vs RoboDopamine",
    "recap_vs_robodopamine": "pi0.6-recap vs RoboDopamine",
}
PAIR_COLORS = {
    "robodopamine_vs_human": "#2a78d6",
    "robometer_vs_human": "#eb6834",
    "icvfe_vs_human": "#1baf7a",
    "recap_vs_human": "#8e44ad",
    "robometer_vs_robodopamine": "#eda100",
    "icvfe_vs_robodopamine": "#e87ba4",
    "recap_vs_robodopamine": "#5a4fcf",
}

METRIC_LABELS = {
    "pearson": "Pearson r",
    "kendall_tau_b": "Kendall tau-b",
    "mae": "MAE",
}

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


def _bar_positions(n_groups: int, n_series: int, group_gap: float = 1.0, total_width: float = 0.86):
    """Returns (group_centers, offsets, bar_width) for a grouped bar chart."""
    bar_width = total_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_width
    centers = np.arange(n_groups) * group_gap
    return centers, offsets, bar_width


def _wrap_label(text: str, width: int = 26, max_lines: int = 2) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1].rstrip() + "…"
    return "\n".join(kept)


def plot_task_level(task_wide: pd.DataFrame, metric: str, filename: str,
                     pair_order=None, pair_labels=None, pair_colors=None, fig_dir=None):
    pair_order = pair_order or PAIR_ORDER
    pair_labels = pair_labels or PAIR_LABELS
    pair_colors = pair_colors or PAIR_COLORS
    fig_dir = fig_dir or FIG_DIR

    col = lambda pair: f"{pair}__{metric}"
    sort_col = col(pair_order[0])
    df = task_wide.sort_values(sort_col, ascending=True).reset_index(drop=True)

    n = len(df)
    centers, offsets, bw = _bar_positions(n, len(pair_order), group_gap=1.0)

    fig_h = max(6, n * 0.44)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    for i, pair in enumerate(pair_order):
        vals = df[col(pair)].values
        ax.barh(centers + offsets[i], vals, height=bw, color=pair_colors[pair],
                label=pair_labels[pair], zorder=3)

    ax.set_yticks(centers)
    ax.set_yticklabels([_wrap_label(t) for t in df["task"]], fontsize=8, linespacing=0.9)
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_title(f"Per-task agreement — {METRIC_LABELS[metric]}", fontsize=11, fontweight="bold", loc="left")
    if metric in ("pearson", "kendall_tau_b"):
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, df[[col(p) for p in pair_order]].to_numpy(dtype=float).max() * 1.1)
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), dpi=160)
    plt.close(fig)


def plot_total_level(total_wide: pd.DataFrame, filename: str,
                      pair_order=None, pair_labels=None, pair_colors=None, fig_dir=None):
    pair_order = pair_order or PAIR_ORDER
    pair_labels = pair_labels or PAIR_LABELS
    pair_colors = pair_colors or PAIR_COLORS
    fig_dir = fig_dir or FIG_DIR

    metrics = ["pearson", "kendall_tau_b", "mae"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

    for ax, metric in zip(axes, metrics):
        vals, colors, labels = [], [], []
        for pair in pair_order:
            row = total_wide[total_wide["pair"] == pair].iloc[0]
            vals.append(row[metric])
            colors.append(pair_colors[pair])
            labels.append(pair_labels[pair])
        x = np.arange(len(pair_order)) * 1.3
        ax.bar(x, vals, color=colors, width=0.8, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.015 if metric != "mae" else v * 0.02), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8, rotation=12, ha="right", rotation_mode="anchor")
        ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        if metric in ("pearson", "kendall_tau_b"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, max(vals) * 1.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Dataset-wide agreement (pooled over all tasks)", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(fig_dir, filename), dpi=160)
    plt.close(fig)


def plot_episode_distribution(episode_wide: pd.DataFrame, metric: str, filename: str,
                               pair_order=None, pair_labels=None, pair_colors=None, fig_dir=None):
    pair_order = pair_order or PAIR_ORDER
    pair_labels = pair_labels or PAIR_LABELS
    pair_colors = pair_colors or PAIR_COLORS
    fig_dir = fig_dir or FIG_DIR

    col_key = "kendall_tau_b" if metric == "kendall" else metric
    data = []
    labels = []
    colors = []
    for pair in pair_order:
        col = f"{pair}__{col_key}"
        vals = episode_wide[col].dropna().values
        data.append(vals)
        labels.append(pair_labels[pair])
        colors.append(pair_colors[pair])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                     medianprops=dict(color="#0b0b0b", linewidth=1.4),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel(METRIC_LABELS[col_key])
    ax.set_title(f"Per-episode distribution — {METRIC_LABELS[col_key]}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    task_wide = pd.read_csv(os.path.join(OUT_DIR, "task_level_wide.csv"))
    total_wide = pd.read_csv(os.path.join(OUT_DIR, "total_level_wide.csv"))
    episode_wide = pd.read_csv(os.path.join(OUT_DIR, "episode_level_wide.csv"))

    for metric, fname in [
        ("pearson", "task_level_pearson.png"),
        ("kendall_tau_b", "task_level_kendall.png"),
        ("mae", "task_level_mae.png"),
    ]:
        plot_task_level(task_wide, metric, fname)
        print(f"Wrote figures/{fname}")

    plot_total_level(total_wide, "total_level_summary.png")
    print("Wrote figures/total_level_summary.png")

    for metric, fname in [
        ("pearson", "episode_distribution_pearson.png"),
        ("kendall", "episode_distribution_kendall.png"),
        ("mae", "episode_distribution_mae.png"),
    ]:
        plot_episode_distribution(episode_wide, metric, fname)
        print(f"Wrote figures/{fname}")


if __name__ == "__main__":
    main()
