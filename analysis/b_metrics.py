"""
B1/B2 retrieval metrics pipeline (TTC, videoDTW, BC error, naive DINO error).

This module implements the four metric functions per user direction, decoupled
from any particular retrieval method -- each takes a (query episode/frame,
retrieved episode/frame) pair and returns an error value. Actual retrieval
methods (DINO nearest-neighbor, RoboMeter-guided, etc. -- B1's roster)
are a separate, later step; this is metrics-only infrastructure so they can be
plugged in once decided.

Metric definitions (as given, superseding the DRAQ-based ratio-vs-random
framing in Eval_Research_Summary.md's "Metrics To Calculate" table):

1. TTC error -- |time-left-to-completion(retrieved) - time-left-to-completion(query)|,
   as a fraction of each episode's own length (dimensionless, comparable
   across episodes of different lengths). "Will not have this information in
   an online rollout" (the query episode's true remaining length is unknown
   during a real rollout) "but we have a ground truth we can use to
   evaluate" -- i.e. this is an eval-only metric, using each episode's known
   total length after the fact.

2. videoDTW error -- path-normalized DTW distance (per the doc's existing
   videoDTW definition) between two 30-frame DINO-embedding sequences: the
   "current chunk" (query_frame, query_frame+1, ..., query_frame+29) and the
   "retrieved chunk" (same window anchored at retrieved_frame in the
   retrieved episode).

3. BC error -- identical windowing to videoDTW, but a direct mean absolute
   error (L1) over raw action vectors instead of DINO embeddings
   (action.q_target + action.gripper + action.base_vel + action.lift_cmd,
   concatenated in a fixed, self-consistent order -- order doesn't need to
   match the FAST tokenizer's training convention here, since this is a raw
   continuous-space distance, not tokenization). L1 rather than DTW because
   both chunks are the same fixed horizon anchored at a known frame
   correspondence -- no warping needed -- matching the standard
   action-chunking behavior-cloning loss (e.g. ACT, Zhao et al. 2023) rather
   than this project's earlier (superseded) DTW formulation.

4. naive DINO error -- single-frame embedding distance between the query
   frame's DINO embedding and the retrieved chunk's *first* frame (i.e.
   DINO(query_frame) vs DINO(retrieved_frame), no sequence/DTW involved).

Data sources:
  - DINO embeddings: ../../Reward Based Retrieval/dino_embeddings_demo.zip
    (sibling repo; one .npz per task, all containing that task's episodes'
    embeddings sampled every 10th frame + a final tail frame -- sparse, like
    topreward/gvl in A1). Verified: covers all 27 tasks.
  - Raw actions + episode lengths: ../../obs_fix/output/icl-demo-dataset-fixed-action
    (same source as action_chunk_boundaries.py).

Frame lookup for sparse DINO embeddings uses per-dimension linear
interpolation between the two bracketing sampled frames, not nearest. With a
10-frame chunk horizon and embeddings sampled every 10 frames, a chunk
typically brackets only one native sample -- nearest-frame lookup would
collapse most chunks into a near-constant repeated vector, giving DTW almost
no signal. Interpolation instead produces a smoothly-varying 10-point
sequence over the same single 10-frame gap the samples were taken at, which
is a reasonable local approximation despite the embedding manifold being
non-linear in general. This, the Euclidean local-cost function for DTW, and
the action-column concatenation order are all flagged as revisitable design
choices once we discuss methods -- everything here is internally consistent
but none of it has been validated against a reference implementation.
"""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache", "b_metrics")

DINO_ARCHIVE = os.path.normpath(
    os.path.join(HERE, "..", "..", "Reward Based Retrieval", "dino_embeddings_demo.zip")
)
ACTION_DATA_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "..", "obs_fix", "output", "icl-demo-dataset-fixed-action")
)

ACTION_COLUMNS = ["action.q_target", "action.gripper", "action.base_vel", "action.lift_cmd"]
CHUNK_HORIZON = 10  # retrieval chunk length (not the FAST tokenizer's time_horizon=30)


# --------------------------------------------------------------------------
# DINO embeddings (sparse, every-10th-frame + tail frame, per task)
# --------------------------------------------------------------------------

def _extract_dino_archive() -> str:
    dest = os.path.join(CACHE_DIR, "dino_embeddings_demo")
    if not (os.path.isdir(dest) and os.listdir(dest)):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(DINO_ARCHIVE) as zf:
            zf.extractall(dest)
    # zip contains one wrapping folder; descend into it (both on a fresh
    # extract and on a cache hit from a previous run)
    entries = [e for e in os.listdir(dest) if os.path.isdir(os.path.join(dest, e))]
    return os.path.join(dest, entries[0]) if len(entries) == 1 else dest


class EmbeddingIndex:
    """episode_uid -> sorted (frames, embeddings) for nearest-frame lookup.

    Base class for DinoIndex -- a sparse
    (every-10th-frame + tail), per-episode embedding set; subclasses differ only in
    where/how they're loaded from disk. Subclasses populate self._frames /
    self._embeddings in __init__.
    """

    def interp(self, episode_uid: str, frame: int) -> np.ndarray:
        """Per-dimension linear interpolation between the two bracketing sampled frames (clamped at episode ends)."""
        frames = self._frames[episode_uid]
        emb = self._embeddings[episode_uid]
        if frame <= frames[0]:
            return emb[0]
        if frame >= frames[-1]:
            return emb[-1]
        j = int(np.searchsorted(frames, frame, side="right")) - 1
        t = (frame - frames[j]) / (frames[j + 1] - frames[j])
        return emb[j] + t * (emb[j + 1] - emb[j])

    def sequence(self, episode_uid: str, start_frame: int, horizon: int = CHUNK_HORIZON) -> np.ndarray:
        length = _episode_length(episode_uid)
        end_frame = min(start_frame + horizon, length)
        return np.stack([self.interp(episode_uid, f) for f in range(start_frame, end_frame)])


class DinoIndex(EmbeddingIndex):
    """DINO ViT embeddings, one .npz per TASK (multiple episodes concatenated inside)."""

    def __init__(self):
        root = _extract_dino_archive()
        by_episode: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for fname in os.listdir(root):
            if not fname.endswith(".npz"):
                continue
            with np.load(os.path.join(root, fname), allow_pickle=True) as d:
                uids, frames, emb = d["episode_uid"], d["frame"], d["key_embeddings"]
                for uid in np.unique(uids):
                    mask = uids == uid
                    by_episode.setdefault(str(uid), []).append((frames[mask], emb[mask]))

        self._frames: dict[str, np.ndarray] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        for uid, parts in by_episode.items():
            frames = np.concatenate([p[0] for p in parts])
            emb = np.concatenate([p[1] for p in parts])
            order = np.argsort(frames)
            self._frames[uid] = frames[order]
            self._embeddings[uid] = emb[order]


# --------------------------------------------------------------------------
# Raw actions + episode lengths
# --------------------------------------------------------------------------

_EPISODES_META_CACHE: list[dict] | None = None


def _episodes_meta() -> list[dict]:
    global _EPISODES_META_CACHE
    if _EPISODES_META_CACHE is None:
        with open(os.path.join(ACTION_DATA_ROOT, "meta", "episodes.jsonl")) as f:
            _EPISODES_META_CACHE = [json.loads(line) for line in f]
    return _EPISODES_META_CACHE


_EPISODE_LENGTH_INDEX: dict[str, int] | None = None
_EPISODE_INDEX_LOOKUP: dict[str, int] | None = None


def _episode_length(episode_uid: str) -> int:
    global _EPISODE_LENGTH_INDEX
    if _EPISODE_LENGTH_INDEX is None:
        _EPISODE_LENGTH_INDEX = {e["episode_uid"]: e["length"] for e in _episodes_meta()}
    return _EPISODE_LENGTH_INDEX[episode_uid]


def _episode_parquet_path(episode_uid: str) -> str:
    global _EPISODE_INDEX_LOOKUP
    if _EPISODE_INDEX_LOOKUP is None:
        _EPISODE_INDEX_LOOKUP = {e["episode_uid"]: e["episode_index"] for e in _episodes_meta()}
    ep_idx = _EPISODE_INDEX_LOOKUP[episode_uid]
    chunk = ep_idx // 100  # chunks_size=100 per info.json
    return os.path.join(ACTION_DATA_ROOT, "data", f"chunk-{chunk:03d}", f"episode_{ep_idx:06d}.parquet")


_ACTION_CACHE: dict[str, np.ndarray] = {}


def _action_array(episode_uid: str) -> np.ndarray:
    """Returns [n_frames, 20] raw action vector (14 q_target + 2 gripper + 3 base_vel + 1 lift_cmd)."""
    if episode_uid not in _ACTION_CACHE:
        df = pd.read_parquet(_episode_parquet_path(episode_uid), columns=ACTION_COLUMNS)
        cols = [np.stack(df[c].to_numpy()) for c in ACTION_COLUMNS]
        _ACTION_CACHE[episode_uid] = np.concatenate(cols, axis=1)
    return _ACTION_CACHE[episode_uid]


def action_sequence(episode_uid: str, start_frame: int, horizon: int = CHUNK_HORIZON) -> np.ndarray:
    arr = _action_array(episode_uid)
    end_frame = min(start_frame + horizon, len(arr))
    return arr[start_frame:end_frame]


# --------------------------------------------------------------------------
# DTW (path-normalized), per Eval_Research_Summary.md's videoDTW definition
# --------------------------------------------------------------------------

def dtw_path_normalized(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    gamma(i,j) = d(a_i,b_j) + min(gamma(i-1,j), gamma(i,j-1), gamma(i-1,j-1))
    boundary: gamma(0,0)=0; gamma(i,0)=gamma(0,j)=inf for i,j>0
    returns gamma(n,m) / |path|  (path length = n + m - 1 for a full DTW alignment)
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("nan")

    # local Euclidean cost matrix
    diff = seq_a[:, None, :] - seq_b[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=-1))

    gamma = np.full((n + 1, m + 1), np.inf)
    gamma[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            gamma[i, j] = cost[i - 1, j - 1] + min(gamma[i - 1, j], gamma[i, j - 1], gamma[i - 1, j - 1])

    path_length = n + m - 1  # length of the shortest-possible monotone alignment path
    return float(gamma[n, m] / path_length)


# --------------------------------------------------------------------------
# The four metrics
# --------------------------------------------------------------------------

def ttc_error(query_uid: str, query_frame: int, retrieved_uid: str, retrieved_frame: int) -> float:
    ttc_query = (_episode_length(query_uid) - query_frame) / _episode_length(query_uid)
    ttc_retrieved = (_episode_length(retrieved_uid) - retrieved_frame) / _episode_length(retrieved_uid)
    return abs(ttc_retrieved - ttc_query)


def video_chunk_error(embeddings: EmbeddingIndex, query_uid: str, query_frame: int,
                       retrieved_uid: str, retrieved_frame: int, horizon: int = CHUNK_HORIZON) -> float:
    """Mean absolute error (L1) between the retrieved and reference frame-embedding
    chunks -- same formulation as bc_error, pass a DinoIndex. No DTW
    alignment: both chunks are the same fixed horizon anchored at a known frame
    correspondence. Chunks are truncated to the shorter length if either runs
    past its episode's end."""
    current_chunk = embeddings.sequence(query_uid, query_frame, horizon)
    retrieved_chunk = embeddings.sequence(retrieved_uid, retrieved_frame, horizon)
    n = min(len(current_chunk), len(retrieved_chunk))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(retrieved_chunk[:n] - current_chunk[:n])))


def bc_error(query_uid: str, query_frame: int, retrieved_uid: str, retrieved_frame: int,
             horizon: int = CHUNK_HORIZON) -> float:
    """Mean absolute error (L1) between the retrieved and needed action chunks,
    matching the standard action-chunking behavior-cloning loss (e.g. ACT,
    Zhao et al. 2023) rather than a DTW alignment -- both chunks are the same
    fixed horizon anchored at a known frame correspondence, so no temporal
    warping is needed. Chunks are truncated to the shorter length if either
    runs past its episode's end."""
    needed = action_sequence(query_uid, query_frame, horizon)
    retrieved = action_sequence(retrieved_uid, retrieved_frame, horizon)
    n = min(len(needed), len(retrieved))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(retrieved[:n] - needed[:n])))


def naive_dino_error(embeddings: EmbeddingIndex, query_uid: str, query_frame: int,
                      retrieved_uid: str, retrieved_frame: int) -> float:
    """Single-frame embedding distance -- pass a DinoIndex (naming kept for backward compat)."""
    query_emb = embeddings.interp(query_uid, query_frame)
    retrieved_emb = embeddings.interp(retrieved_uid, retrieved_frame)
    return float(np.linalg.norm(query_emb - retrieved_emb))


# --------------------------------------------------------------------------
# Self-test: sanity-check on real data with trivial + arbitrary pairs
# --------------------------------------------------------------------------

def main():
    dino = DinoIndex()
    meta = _episodes_meta()
    ep_a = next(e for e in meta if e["episode_uid"].startswith("hit_the_eggplant") and e["length"] > 60)
    uid_a = ep_a["episode_uid"]
    ep_b = next(e for e in meta if e["episode_uid"].startswith("hit_the_eggplant") and e["episode_uid"] != uid_a and e["length"] > 60)
    uid_b = ep_b["episode_uid"]

    print(f"episode A: {uid_a} (length={ep_a['length']})")
    print(f"episode B: {uid_b} (length={ep_b['length']})")

    print("\n-- identity retrieval (query==retrieved), expect ~0 error on all four --")
    q_frame = 20
    print("ttc_error       :", ttc_error(uid_a, q_frame, uid_a, q_frame))
    print("video_chunk_error:", video_chunk_error(dino, uid_a, q_frame, uid_a, q_frame))
    print("bc_error        :", bc_error(uid_a, q_frame, uid_a, q_frame))
    print("naive_dino_error:", naive_dino_error(dino, uid_a, q_frame, uid_a, q_frame))

    print("\n-- cross-episode retrieval (arbitrary, same task), expect nonzero --")
    r_frame = 15
    print("ttc_error       :", ttc_error(uid_a, q_frame, uid_b, r_frame))
    print("video_chunk_error:", video_chunk_error(dino, uid_a, q_frame, uid_b, r_frame))
    print("bc_error        :", bc_error(uid_a, q_frame, uid_b, r_frame))
    print("naive_dino_error:", naive_dino_error(dino, uid_a, q_frame, uid_b, r_frame))


if __name__ == "__main__":
    main()
