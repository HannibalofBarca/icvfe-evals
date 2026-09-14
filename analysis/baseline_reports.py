"""
Baseline-specific 3-way (4-way for the human baseline) comparison reports.

Splits the master metrics_long.csv / *_wide.csv (from pipeline.py) into two
self-contained subdirectories, each scoped to one baseline:

  output/vs_robodopamine/  -- robometer, icvfe, recap, each vs robodopamine
  output/vs_human/         -- robodopamine, robometer, icvfe, recap, each vs human

("robodopamine vs human" is included in vs_human/ too -- robodopamine is the
baseline in vs_robodopamine/, but it's just another candidate evaluator once
human is the baseline, so all four numbers belong together there.)

Each subdir gets its own filtered task/total CSVs and the same bar-chart
figures plotting.py produces for the master report, scoped to just its pairs.

Usage:
    python baseline_reports.py   # run after pipeline.py
"""
import os

import pandas as pd

import plotting as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

BASELINES = {
    "vs_robodopamine": {
        "pairs": ["robometer_vs_robodopamine", "icvfe_vs_robodopamine", "recap_vs_robodopamine"],
        "title": "vs RoboDopamine baseline",
    },
    "vs_human": {
        "pairs": ["robodopamine_vs_human", "robometer_vs_human", "icvfe_vs_human", "recap_vs_human"],
        "title": "vs Human baseline",
    },
}


def build_baseline(name: str, pairs: list) -> None:
    sub_dir = os.path.join(OUT_DIR, name)
    fig_dir = os.path.join(sub_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    long_df = pd.read_csv(os.path.join(OUT_DIR, "metrics_long.csv"))
    long_df = long_df[long_df["pair"].isin(pairs)].copy()
    long_df.to_csv(os.path.join(sub_dir, "metrics_long.csv"), index=False)

    def cols_for(wide: pd.DataFrame, index_cols: list) -> pd.DataFrame:
        keep = list(index_cols)
        if "n_frames_or_reps" in wide.columns:
            keep.append("n_frames_or_reps")
        if "n_episodes" in wide.columns and "n_episodes" not in keep:
            keep.append("n_episodes")
        keep += [c for c in wide.columns if c.split("__")[0] in pairs]
        return wide[keep]

    task_wide = pd.read_csv(os.path.join(OUT_DIR, "task_level_wide.csv"))
    task_wide = cols_for(task_wide, ["task"])
    task_wide.to_csv(os.path.join(sub_dir, "task_level_wide.csv"), index=False)

    episode_wide = pd.read_csv(os.path.join(OUT_DIR, "episode_level_wide.csv"))
    episode_wide = cols_for(episode_wide, ["task", "episode_uid"])
    episode_wide.to_csv(os.path.join(sub_dir, "episode_level_wide.csv"), index=False)

    total_wide_full = pd.read_csv(os.path.join(OUT_DIR, "total_level_wide.csv"))
    total_wide = total_wide_full[total_wide_full["pair"].isin(pairs)].copy()
    total_wide.to_csv(os.path.join(sub_dir, "total_level_wide.csv"), index=False)

    pair_order = pairs
    pair_labels = {p: pl.PAIR_LABELS[p] for p in pairs}
    pair_colors = {p: pl.PAIR_COLORS[p] for p in pairs}

    for metric, fname in [
        ("pearson", "task_level_pearson.png"),
        ("kendall_tau_b", "task_level_kendall.png"),
        ("mae", "task_level_mae.png"),
    ]:
        pl.plot_task_level(task_wide, metric, fname,
                            pair_order=pair_order, pair_labels=pair_labels,
                            pair_colors=pair_colors, fig_dir=fig_dir)

    pl.plot_total_level(total_wide, "total_level_summary.png",
                         pair_order=pair_order, pair_labels=pair_labels,
                         pair_colors=pair_colors, fig_dir=fig_dir)

    for metric, fname in [
        ("pearson", "episode_distribution_pearson.png"),
        ("kendall", "episode_distribution_kendall.png"),
        ("mae", "episode_distribution_mae.png"),
    ]:
        pl.plot_episode_distribution(episode_wide, metric, fname,
                                      pair_order=pair_order, pair_labels=pair_labels,
                                      pair_colors=pair_colors, fig_dir=fig_dir)

    # ---- short markdown summary ----
    total_rows = []
    for pair in pairs:
        r = total_wide[total_wide["pair"] == pair].iloc[0]
        total_rows.append({
            "Pair": pair_labels[pair],
            "Pearson r": f"{r['pearson']:.3f}",
            "Kendall tau-b": f"{r['kendall_tau_b']:.3f}",
            "MAE": f"{r['mae']:.3f}",
            "n episodes": int(r["n_episodes"]),
        })
    total_table_df = pd.DataFrame(total_rows)
    header = "| " + " | ".join(total_table_df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in total_table_df.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r.values) + " |" for _, r in total_table_df.iterrows())
    total_table_md = "\n".join([header, sep, body])

    md = f"""# {BASELINES_TITLE[name]}

Dataset-wide (total-level, pooled) agreement for this baseline. Full
per-task numbers: [`task_level_wide.csv`](task_level_wide.csv); per-episode:
[`episode_level_wide.csv`](episode_level_wide.csv); long format (filter for
anything): [`metrics_long.csv`](metrics_long.csv). Methodology and the
ICVFE/recap target-curve caveat: [`../../README.md`](../../README.md).

{total_table_md}

![Total-level summary](figures/total_level_summary.png)

![Task-level Pearson r](figures/task_level_pearson.png)

![Task-level Kendall tau-b](figures/task_level_kendall.png)

![Task-level MAE](figures/task_level_mae.png)

![Episode-level Pearson r distribution](figures/episode_distribution_pearson.png)

![Episode-level Kendall tau-b distribution](figures/episode_distribution_kendall.png)

![Episode-level MAE distribution](figures/episode_distribution_mae.png)
"""
    with open(os.path.join(sub_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote {sub_dir}/ ({len(pairs)} pairs)")


BASELINES_TITLE = {
    "vs_robodopamine": "RoboMeter / ICVFE / pi0.6-recap vs RoboDopamine",
    "vs_human": "RoboDopamine / RoboMeter / ICVFE / pi0.6-recap vs Human",
}


def main():
    for name, spec in BASELINES.items():
        build_baseline(name, spec["pairs"])


if __name__ == "__main__":
    main()
