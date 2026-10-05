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

All sources are frame-pooled within each split: every aligned (prediction,
human) frame from every episode whose task falls in the split is pooled into
one correlation (see a1_pipeline.build_rows / a3_pipeline.build_npz_rows):
  - "flat" sources (topreward, gvl, robometer_zs, robodopamine_zs,
    robometer_ft, robodopamine_ft, robometer_zs_online, robometer_ft_online):
    per-frame JSON curves, aligned by frame_index.
  - "npz" sources (recap_ft, icvfe_8800, icvfe_ema_0.5): per-trajectory npz
    predictions, aligned on their own timesteps; every context replicate's
    frames are pooled.

Usage:
    python seen_unseen_split.py
"""
from __future__ import annotations

import os

import numpy as np
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
    """Frame-pooled {split: {pearson, kendall_tau_b, mae, n}} for one evaluator vs. manual."""
    common = sorted(set(manual) & set(curve_set))
    pooled = {"seen": ([], []), "unseen": ([], [])}
    for uid in common:
        task = manual[uid]["task"]
        split = "seen" if task in SEEN_TASKS else "unseen" if task in UNSEEN_TASKS else None
        if split is None:
            continue
        a_arr, b_arr = align_fn(curve_set[uid], manual[uid])
        a_arr, b_arr = a_arr / 100.0, b_arr / 100.0
        xs, ys = pooled[split]
        xs.extend(a_arr.tolist())
        ys.extend(b_arr.tolist())
    out = {}
    for split, (xs, ys) in pooled.items():
        m = compute_fn(np.array(xs), np.array(ys))
        m["n"] = len(xs)
        out[split] = m
    return out


def npz_split_metrics(manual: dict, name: str) -> dict:
    """Frame-pooled {split: {pearson, kendall_tau_b, mae, n}} over every replicate of an npz source."""
    pooled = {"seen": ([], []), "unseen": ([], [])}
    for t in a3.NPZ_LOADERS[name](name):
        split = "seen" if t["task"] in SEEN_TASKS else "unseen" if t["task"] in UNSEEN_TASKS else None
        if split is None:
            continue
        aligned = a3._traj_aligned_vs_manual(t["npz_path"], manual, t["episode_uid"])
        if aligned is None:
            continue
        xs, ys = pooled[split]
        xs.extend(aligned[0].tolist())
        ys.extend(aligned[1].tolist())
    out = {}
    for split, (xs, ys) in pooled.items():
        m = a3.compute_all_metrics(np.array(xs), np.array(ys))
        m["n"] = len(xs)
        out[split] = m
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

    # -- A3 flat sources (robometer_zs_online, robometer_ft_online) --
    a3.extract_archives()
    manual_a3 = a3.load_curve_set(a3.REFERENCE)
    for name in a3.FLAT_ROSTER:
        curve_set = a3.load_curve_set(name)
        split = flat_split_metrics(manual_a3, curve_set, a3.align_to_reference, a3.compute_all_metrics)
        for s, m in split.items():
            rows.append({"section": "A3", "method": name, "split": s, **m})

    # -- A3 npz sources (recap_ft, icvfe_8800, icvfe_ema_0.5) --
    for name in a3.NPZ_ROSTER:
        split = npz_split_metrics(manual_a3, name)
        for s, m in split.items():
            rows.append({"section": "A3", "method": name, "split": s, **m})

    df = pd.DataFrame(rows)[["section", "method", "split", "n", "pearson", "kendall_tau_b", "mae"]]
    out_path = os.path.join(OUTPUT_DIR, "seen_unseen_split.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
