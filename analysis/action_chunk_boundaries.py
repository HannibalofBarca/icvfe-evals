"""
Action chunk boundaries for the yor-icl-canonical FAST tokenizer.

Detour before B1/B2: those ablations' BC error metric needs to compare a
"retrieved" action chunk against the "needed" one (see Eval_Research_Summary.md,
BC error definition). This script computes which frame ranges of each episode
constitute one action chunk, i.e. the boundaries a retrieval index would need
to look up -- WITHOUT yet extracting/flattening/tokenizing the actual action
values. See module-level rationale below for why tokenization is deferred.

Tokenizer: https://huggingface.co/lair-nyu/yor-icl-canonical-fast-tokenizer
  - time_horizon = 30   (fixed chunk length, confirmed via processor_config.json)
  - action_dim = 20     (14 q_target + 2 gripper + 3 base_vel + 1 lift_cmd,
                         per user: the last 4 -- base_vel/lift_cmd -- are dead/
                         unused for this stationary-base manipulation setup,
                         16 of the 20 dims are meaningful)
  - The tokenizer itself does no windowing (confirmed by reading
    processing_action_tokenizer.py -- it DCT-quantizes whatever fixed-length
    chunk it's handed); windowing into 30-frame chunks happens upstream, and
    that convention isn't recorded in the tokenizer repo. Per user direction:
    non-overlapping chunks, remainder dropped (chunk i covers frames
    [i*30, i*30+30), stopping at the last full chunk).

Chunk *extraction* (concatenating action.q_target/gripper/base_vel/lift_cmd
into the actual 20-dim vectors, and running them through the tokenizer) is
deliberately NOT done here -- the exact column concatenation order used to
train this tokenizer isn't confirmed from the HF repo (dimension-matching
proves the dims exist, not their order), and getting that wrong would
silently corrupt every downstream token. This script only produces the frame
boundaries, which depend solely on episode length + time_horizon (both
confirmed), so they carry no such risk. Extraction/tokenization is a
follow-up step once the column order is confirmed.

Data source: local LeRobot-format parquet dataset already cached at
../../obs_fix/output/icl-demo-dataset-fixed-action (sibling repo to this
one) -- verified against progress-curve data: episode
"3_cube_tower_stack_20260803_114230:000000" has 1290 frames here, matching
the frame count already seen in manual/robodopamine/robometer's continuous
archives for the same episode_uid. Falls back to downloading
https://huggingface.co/datasets/adityx23/icl-demo-dataset if the local
cache is missing.

Usage:
    python action_chunk_boundaries.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ACTION_DATA_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "..", "obs_fix", "output", "icl-demo-dataset-fixed-action")
)
HF_FALLBACK_DATASET = "adityx23/icl-demo-dataset"
OUTPUT_DIR = os.path.join(HERE, "output", "action_chunk_boundaries")

TIME_HORIZON = 30  # tokenizer's processor_config.json: time_horizon


def _episodes_meta() -> list[dict]:
    meta_path = os.path.join(LOCAL_ACTION_DATA_ROOT, "meta", "episodes.jsonl")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"No local action dataset at {LOCAL_ACTION_DATA_ROOT}. "
            f"Fall back to downloading the '{HF_FALLBACK_DATASET}' HF dataset instead."
        )
    with open(meta_path) as f:
        return [json.loads(line) for line in f]


def build_boundaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = _episodes_meta()

    chunk_rows = []
    episode_rows = []
    for ep in episodes:
        uid = ep["episode_uid"]
        task = ep["tasks"][0] if ep["tasks"] else None
        length = ep["length"]
        n_chunks = length // TIME_HORIZON
        dropped = length - n_chunks * TIME_HORIZON

        for i in range(n_chunks):
            start = i * TIME_HORIZON
            end = start + TIME_HORIZON
            chunk_rows.append({
                "episode_uid": uid, "task": task, "chunk_index": i,
                "start_frame": start, "end_frame": end,
            })

        episode_rows.append({
            "episode_uid": uid, "task": task, "length": length,
            "n_chunks": n_chunks, "dropped_frames": dropped,
            "success": ep.get("success"), "valid": ep.get("valid"), "keep": ep.get("keep"),
        })

    return pd.DataFrame(chunk_rows), pd.DataFrame(episode_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunks_df, episodes_df = build_boundaries()

    chunks_path = os.path.join(OUTPUT_DIR, "chunk_boundaries_long.csv")
    episodes_path = os.path.join(OUTPUT_DIR, "episode_chunk_summary.csv")
    chunks_df.to_csv(chunks_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)

    print(f"Wrote {chunks_path} ({len(chunks_df)} chunks across {len(episodes_df)} episodes)")
    print(f"Wrote {episodes_path}")
    print(f"\ntime_horizon={TIME_HORIZON}, total chunks={len(chunks_df)}, "
          f"total dropped frames={episodes_df['dropped_frames'].sum()}")
    print(episodes_df["n_chunks"].describe().to_string())


if __name__ == "__main__":
    main()
