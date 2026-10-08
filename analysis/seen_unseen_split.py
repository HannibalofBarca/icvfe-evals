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

Each source is reported under two aggregations (the `aggregation` column),
over every episode it covers whose task falls in the split:
  - "episode": metrics are computed per episode over that episode's own
    aligned frames, then averaged over episodes (each episode counts once).
    This is what the A1-A3 pipelines report.
  - "frame": every aligned (prediction, human) frame from every episode in the
    split is pooled into one correlation. Reported for comparison only.
  - "flat" sources (topreward, gvl, robometer_zs, robodopamine_zs,
    robometer_ft, robodopamine_ft, robometer_zs_online, robometer_ft_online):
    per-frame JSON curves, aligned by frame_index.
  - "npz" sources (recap_ft, icvfe_8800, icvfe_ema_0.5): per-trajectory npz
    predictions, aligned on their own timesteps. Episode aggregation averages
    a query episode's context replicates into one episode value first; frame
    aggregation pools every replicate's frames.

Usage:
    python seen_unseen_split.py
"""
from __future__ import annotations

import os
from collections import defaultdict

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


METRICS = ("pearson", "kendall_tau_b", "mae")


def _split_of(task: str) -> str | None:
    return "seen" if task in SEEN_TASKS else "unseen" if task in UNSEEN_TASKS else None


def _split_metrics(per_episode: dict, compute_fn) -> dict:
    """{split: [(episode replicate metric dicts, pooled pred, pooled human)]} ->
    {aggregation: {split: {metric: value, n: episode or frame count}}}."""
    out = {"episode": {}, "frame": {}}
    for split in ("seen", "unseen"):
        eps = per_episode.get(split, [])
        ep_vals = [{k: float(np.mean([r[k] for r in reps])) for k in METRICS} for reps, _, _ in eps]
        out["episode"][split] = {k: float(np.mean([m[k] for m in ep_vals])) for k in METRICS}
        out["episode"][split]["n"] = len(eps)
        xs = np.concatenate([x for _, x, _ in eps])
        ys = np.concatenate([y for _, _, y in eps])
        out["frame"][split] = compute_fn(xs, ys)
        out["frame"][split]["n"] = len(xs)
    return out


def flat_split_metrics(manual: dict, curve_set: dict, align_fn, compute_fn) -> dict:
    """Episode-averaged and frame-pooled metrics for one evaluator vs. manual."""
    per_episode = defaultdict(list)
    for uid in sorted(set(manual) & set(curve_set)):
        split = _split_of(manual[uid]["task"])
        if split is None:
            continue
        a_arr, b_arr = align_fn(curve_set[uid], manual[uid])
        a_arr, b_arr = a_arr / 100.0, b_arr / 100.0
        per_episode[split].append(([compute_fn(a_arr, b_arr)], a_arr, b_arr))
    return _split_metrics(per_episode, compute_fn)


def npz_split_metrics(manual: dict, name: str) -> dict:
    """Episode-averaged and frame-pooled metrics for an npz source.

    Episode aggregation computes metrics per trajectory (context replicate) and
    averages a query episode's replicates first; frame aggregation pools every
    replicate's frames.
    """
    by_episode = defaultdict(list)
    for t in a3.NPZ_LOADERS[name](name):
        split = _split_of(t["task"])
        if split is None:
            continue
        aligned = a3._traj_aligned_vs_manual(t["npz_path"], manual, t["episode_uid"])
        if aligned is None:
            continue
        by_episode[(split, t["episode_uid"])].append(aligned)
    per_episode = defaultdict(list)
    for (split, _), reps in by_episode.items():
        per_episode[split].append((
            [a3.compute_all_metrics(*r) for r in reps],
            np.concatenate([r[0] for r in reps]),
            np.concatenate([r[1] for r in reps]),
        ))
    return _split_metrics(per_episode, a3.compute_all_metrics)


def _rows(section: str, method: str, metrics: dict) -> list[dict]:
    return [{"section": section, "method": method, "aggregation": agg, "split": s, **m}
            for agg, splits in metrics.items() for s, m in splits.items()]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    # -- A1 flat sources (topreward, gvl, robometer_zs, robodopamine_zs) --
    a1.extract_archives()
    manual_a1 = a1.load_curve_set(a1.REFERENCE)
    for name in a1.ROSTER:
        curve_set = a1.load_curve_set(name)
        split = flat_split_metrics(manual_a1, curve_set, a1.align_to_reference, a1.compute_all_metrics)
        rows += _rows("A1", name, split)

    # -- A2 flat sources (robometer_ft, robodopamine_ft; zs already covered by A1) --
    a2.extract_archives()
    manual_a2 = a2.load_curve_set(a2.REFERENCE)
    for name in ("robometer_ft", "robodopamine_ft"):
        curve_set = a2.load_curve_set(name)
        split = flat_split_metrics(manual_a2, curve_set, a2.align_to_reference, a2.compute_all_metrics)
        rows += _rows("A2", name, split)

    # -- A3 flat sources (robometer_zs_online, robometer_ft_online) --
    a3.extract_archives()
    manual_a3 = a3.load_curve_set(a3.REFERENCE)
    for name in a3.FLAT_ROSTER:
        curve_set = a3.load_curve_set(name)
        split = flat_split_metrics(manual_a3, curve_set, a3.align_to_reference, a3.compute_all_metrics)
        rows += _rows("A3", name, split)

    # -- A3 npz sources (recap_ft, icvfe_8800, icvfe_ema_0.5) --
    for name in a3.NPZ_ROSTER:
        split = npz_split_metrics(manual_a3, name)
        rows += _rows("A3", name, split)

    df = pd.DataFrame(rows)[["section", "method", "aggregation", "split", "n", "pearson", "kendall_tau_b", "mae"]]
    out_path = os.path.join(OUTPUT_DIR, "seen_unseen_split.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
