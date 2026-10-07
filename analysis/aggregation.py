"""
Episode-averaged aggregation shared by the A1-A3 pipelines and the seen/unseen split.

Every metric is computed per episode first (one correlation over that
episode's own aligned frames); task- and total-level values are the plain
mean of those per-episode values, so each episode counts equally regardless
of its length. Frames are never pooled across episodes.
"""
from __future__ import annotations

import pandas as pd


def aggregate_episode_rows(episode_rows: list[dict]) -> list[dict]:
    """Task- and total-level rows as the mean of `episode_rows` values.

    `episode_rows` are the pipelines' long-format episode rows (keys: task,
    group_id, pair, metric, value). In the returned rows, `n` and
    `n_episodes` are the number of episodes averaged.
    """
    df = pd.DataFrame(episode_rows)
    rows = []
    for (task, pair, metric), g in df.groupby(["task", "pair", "metric"]):
        rows.append({
            "level": "task", "task": task, "group_id": task, "pair": pair,
            "metric": metric, "value": float(g["value"].mean()),
            "n": len(g), "n_episodes": len(g),
        })
    for (pair, metric), g in df.groupby(["pair", "metric"]):
        rows.append({
            "level": "total", "task": "ALL", "group_id": "ALL", "pair": pair,
            "metric": metric, "value": float(g["value"].mean()),
            "n": len(g), "n_episodes": len(g),
        })
    return rows
