"""
Splits every method's total-level metric (vs. human) into "seen" vs.
"unseen" task subsets, for the LaTeX summary table.

The seen/unseen task-novelty partition is not something we defined -- it's
lifted directly from RECAP's own data (data/.../RECAP/recap_20000_icl_demo_
dataset_continuous.zip ships two independent seen/ and unseen/ subdirs,
i.e. which tasks were in-distribution vs. held-out for RECAP's fine-tuning
run). Extracted via each split's metrics.json (see SEEN_TASKS/UNSEEN_TASKS
below) -- 15 seen + 12 unseen = all 27 eval tasks, so this same partition is
applied uniformly to every OTHER method's rows too (not just RECAP's own),
on the assumption that "seen"/"unseen" is a property of the task/icl_dataset
split itself, not of which model happens to report it.

Both source types now use the same nested-aggregation convention (episode
level over that episode's own frames, then average episode-level numbers --
see a1_pipeline.build_rows / a3_pipeline.build_npz_rows for the rationale):
  - "flat" sources (topreward, gvl, robometer_zs, robodopamine_zs,
    robometer_ft, robodopamine_ft, robometer_ft_online): recomputed here
    from scratch -- per-episode metrics from aligned frames, restricted to
    episodes whose task falls in the given split, then averaged.
  - "npz" sources (recap_ft, icvfe_8800, icvfe_ema_0.5): each source's
    existing episode_level_wide.csv already has per-episode metrics (see
    a3_pipeline.build_npz_rows), so here we just filter to episodes whose
    task is in the split and average the metric columns.

Usage:
    python seen_unseen_split.py
"""
from __future__ import annotations

import os

import pandas as pd

import a1_pipeline as a1
import a2_pipeline as a2
import a3_pipeline as a3

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "seen-unseen split")

SEEN_TASKS = {
    "hit the yellow cube with the mallet using the left arm",
    "open the gatorade bottle",
    "open the notebook",
    "pass the salt shaker from the left arm to the right arm",
    "pass the salt shaker from the right arm to the left arm",
    "pick up the red chilli and place it on the plate with the left arm",
    "pick up the red chilli and place it on the plate with the right arm",
    "put the circle on the peg",
    "put the eggplant into the box with the left arm",
    "put the eggplant into the box with the right arm",
    "put the green pepper into the grocery bag",
    "sort the items into their containers",
    "stack the colored octagons",
    "stack the three cups into a tower",
    "uncap the red marker",
}
UNSEEN_TASKS = {
    "hit the eggplant with the mallet",
    "open the doctor pepper bottle",
    "open the notebook and place the cube on it",
    "pass the gusset to the plate",
    "pass the mustard to the plate",
    "put the eggplant into the bag",
    "put the mustard into the box",
    "put the square on the peg",
    "sort the food items into the cups",
    "stack the cubes and hit them with the mallet",
    "stack the three cubes into a tower",
    "uncap the black marker",
}
assert not (SEEN_TASKS & UNSEEN_TASKS)


def flat_split_metrics(manual: dict, curve_set: dict, align_fn, compute_fn) -> dict:
    """Episode-averaged {split: {pearson, kendall_tau_b, mae, n}} for one evaluator vs. manual.

    Per-episode metrics come from that episode's own aligned frames; the
    split number is the mean of those episode-level numbers (nested
    aggregation), not a pool of raw frames across episodes -- see module
    docstring.
    """
    common = sorted(set(manual) & set(curve_set))
    episode_rows = {"seen": [], "unseen": []}
    for uid in common:
        task = manual[uid]["task"]
        split = "seen" if task in SEEN_TASKS else "unseen" if task in UNSEEN_TASKS else None
        if split is None:
            continue
        a_arr, b_arr = align_fn(curve_set[uid], manual[uid])
        a_arr, b_arr = a_arr / 100.0, b_arr / 100.0
        episode_rows[split].append(compute_fn(a_arr, b_arr))
    out = {}
    for split, metrics_list in episode_rows.items():
        edf = pd.DataFrame(metrics_list)
        out[split] = {
            "pearson": float(edf["pearson"].mean()) if len(edf) else float("nan"),
            "kendall_tau_b": float(edf["kendall_tau_b"].mean()) if len(edf) else float("nan"),
            "mae": float(edf["mae"].mean()) if len(edf) else float("nan"),
            "n": len(edf),
        }
    return out


def npz_split_metrics(episode_wide_path: str, col_prefix: str) -> dict:
    """Episode-averaged {split: {pearson, kendall_tau_b, mae, n}}, from an existing episode_level_wide.csv."""
    df = pd.read_csv(episode_wide_path)
    out = {}
    for split, tasks in (("seen", SEEN_TASKS), ("unseen", UNSEEN_TASKS)):
        sub = df[df["task"].isin(tasks)]
        sub = sub.dropna(subset=[f"{col_prefix}__pearson"])
        out[split] = {
            "pearson": float(sub[f"{col_prefix}__pearson"].mean()),
            "kendall_tau_b": float(sub[f"{col_prefix}__kendall_tau_b"].mean()),
            "mae": float(sub[f"{col_prefix}__mae"].mean()),
            "n": len(sub),
        }
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    # -- A1 flat sources (topreward, gvl, robometer_zs, robodopamine_zs) --
    a1.extract_archives()
    manual_a1 = a1.load_curve_set(a1.REFERENCE)
    for name in a1.ROSTER:
        curve_set = a1.load_curve_set(name)
        split = flat_split_metrics(manual_a1, curve_set, a1.align_to_reference, a1.compute_all_metrics)
        for s, m in split.items():
            rows.append({"section": "A1", "method": name, "split": s, **m})

    # -- A2 flat sources (robometer_ft, robodopamine_ft; zs already covered by A1) --
    a2.extract_archives()
    manual_a2 = a2.load_curve_set(a2.REFERENCE)
    for name in ("robometer_ft", "robodopamine_ft"):
        curve_set = a2.load_curve_set(name)
        split = flat_split_metrics(manual_a2, curve_set, a2.align_to_reference, a2.compute_all_metrics)
        for s, m in split.items():
            rows.append({"section": "A2", "method": name, "split": s, **m})

    # -- A3 flat source (robometer_ft_online) --
    a3.extract_archives()
    manual_a3 = a3.load_curve_set(a3.REFERENCE)
    curve_set = a3.load_curve_set("robometer_ft_online")
    split = flat_split_metrics(manual_a3, curve_set, a3.align_to_reference, a3.compute_all_metrics)
    for s, m in split.items():
        rows.append({"section": "A3", "method": "robometer_ft_online", "split": s, **m})

    # -- A3 npz sources (recap_ft, icvfe_8800, icvfe_ema_0.5) -- reuse existing episode_level_wide.csv
    episode_wide_path = os.path.join(HERE, "output", "A3-online ft models vs human", "episode_level_wide.csv")
    for name in ("recap_ft", "icvfe_8800", "icvfe_ema_0.5"):
        split = npz_split_metrics(episode_wide_path, f"{name}_vs_human")
        for s, m in split.items():
            rows.append({"section": "A3", "method": name, "split": s, **m})

    df = pd.DataFrame(rows)[["section", "method", "split", "n", "pearson", "kendall_tau_b", "mae"]]
    out_path = os.path.join(OUTPUT_DIR, "seen_unseen_split.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
