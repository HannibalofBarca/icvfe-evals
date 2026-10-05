from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .data import FrameRef, VfeFrameDataset, derive_context_max_timesteps
from .data_splits import load_preset_preprocess_config, resolve_data_split, resolve_processed_dir
from .evaluate import (
    EvaluationContextTokenCache,
    aggregate_context_cache_summaries,
    aggregate_metric_dicts,
    distributed_barrier,
    initialize_distributed_evaluation,
    load_model,
    predict_trajectory,
    shard_trajectory_work_items,
    trajectory_metrics,
)
from .labels import (
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    annotation_progress_values,
    discretize_values,
)
from .model import (
    CONTEXT_BACKBONES,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
    ValueFunctionEstimator,
    VfeModelConfig,
)
from .observation_history import normalize_query_history_offsets
from .visualize import plot_error_curve, plot_value_curve
from .vfe_presets import DEFAULT_VFE_PRESET_PATH, load_vfe_preset, resolve_vfe_backbone


DEFAULT_EXPERIMENT = (
    "vfe_icl_keep_true_gemma270m_dualres_stride30_3cam_bs160_4gpu_lr1e5_step30000"
)
DEFAULT_CHECKPOINT = f"outputs/{DEFAULT_EXPERIMENT}/checkpoints/checkpoint_step_30000.pt"
DEFAULT_OUTPUT_DIR = f"outputs/{DEFAULT_EXPERIMENT}/eval/manual_annotation_step_stride10"
DEFAULT_DATASET_ROOT = "data/lerobot/adityx23/icl-dataset"
DEFAULT_EVAL_ROOT = "data/lerobot/test/data/generated/eval_set_combined"
DEFAULT_VFE_PRESET = "gemma3_270m_native_dualres_context177_stride30_3cam_freeze_vision"
MANUAL_GROUND_TRUTH_VERSION = "manual_key_events_step_v1"
CONTINUOUS_ANNOTATION_TYPE = "continuous_per_frame"


def task_output_dir_name(task_id: int, task_name: str) -> str:
    """Return a readable, filesystem-safe directory name for one task."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_name).strip()).strip("._-")
    if not slug:
        slug = "unnamed_task"
    return f"task_{int(task_id):03d}+{slug}"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def manual_annotation_progress(
    annotation: dict[str, Any],
    demo_length: int,
) -> np.ndarray:
    """Expand sparse manual key events into a dense, piecewise-constant progress target.

    A key event changes the progress beginning at its annotated frame. The annotation
    tool can place a terminal event at ``frame == demo_length``; that boundary event is
    applied to the final observable frame so the terminal outcome is represented.
    """

    if demo_length <= 0:
        raise ValueError(f"demo_length must be positive, got {demo_length}")
    events = annotation.get("key_events")
    if not isinstance(events, list) or not events:
        raise ValueError("manual annotation must contain a non-empty key_events list")

    frames: list[int] = []
    values: list[float] = []
    for event_index, event in enumerate(events):
        frame = event.get("frame")
        value = event.get("value")
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise TypeError(f"key event {event_index} has a non-integer frame: {frame!r}")
        if frame < 0 or frame > demo_length:
            raise IndexError(
                f"key event {event_index} frame {frame} is outside [0, {demo_length}]"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise TypeError(f"key event {event_index} has a non-finite numeric value: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"key event {event_index} value {value} is outside [0, 1]")
        frames.append(frame)
        values.append(float(value))

    if frames[0] != 0:
        raise ValueError(f"the first key event must start at frame 0, got frame {frames[0]}")
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("manual key event frames must be strictly increasing")

    progress = np.full(demo_length, values[0], dtype=np.float32)
    for frame, value in zip(frames[1:], values[1:]):
        progress[min(frame, demo_length - 1) :] = value
    return progress


def choose_context_demo_id(
    context_pool: list[str],
    *,
    local_eval_index: int,
    task_id: int,
    seed: int,
    strategy: str,
    query_episode_uid: str,
    context_uid_by_demo_id: dict[str, str],
) -> str:
    """Choose one fixed context per eval trajectory, excluding exact self-context."""

    if not context_pool:
        raise ValueError(f"Task {task_id} has no context demonstrations")
    if strategy == "round_robin":
        start = local_eval_index % len(context_pool)
    elif strategy == "seeded":
        rng = random.Random(seed + int(task_id) * 10_007 + int(local_eval_index) * 101)
        start = rng.randrange(len(context_pool))
    else:
        raise ValueError(f"Unsupported ICL eval context selection: {strategy!r}")

    for offset in range(len(context_pool)):
        candidate = context_pool[(start + offset) % len(context_pool)]
        if context_uid_by_demo_id.get(candidate) != query_episode_uid:
            return candidate
    raise ValueError(
        f"Task {task_id} has no context other than query episode_uid={query_episode_uid!r}"
    )


def _annotation_paths_by_episode_index(eval_root: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted((eval_root / "episodes").glob("*/*/manual_annotation.json")):
        payload = _read_json(path)
        episode_index = int(payload["episode_index"])
        if episode_index in paths:
            raise ValueError(
                f"Duplicate manual annotations for episode {episode_index}: "
                f"{paths[episode_index]} and {path}"
            )
        paths[episode_index] = path.resolve()
    return paths


def load_eval_episode_entries(
    *,
    dataset_root: str | Path,
    eval_root: str | Path,
    value_min: float,
    value_max: float,
    num_bins: int,
    max_tasks: int = 0,
    trajectories_per_task: int = 0,
) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, str], dict[str, np.ndarray]], dict[str, Any]]:
    """Load and strictly align the eval list, full trajectories, and manual annotations."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    eval_root = Path(eval_root).expanduser().resolve()
    info = _read_json(dataset_root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 at {dataset_root}")
    task_rows = _read_jsonl(dataset_root / "meta/tasks.jsonl")
    episode_rows = _read_jsonl(dataset_root / "meta/episodes.jsonl")
    task_id_by_name = {str(row["task"]): int(row["task_index"]) for row in task_rows}
    episode_by_index = {int(row["episode_index"]): row for row in episode_rows}
    annotation_paths = _annotation_paths_by_episode_index(eval_root)
    eval_spec = _read_json(eval_root / "eval_set.json")
    task_specs = list(eval_spec["tasks"])
    if max_tasks > 0:
        task_specs = task_specs[:max_tasks]

    chunks_size = int(info.get("chunks_size", 100))
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    entries_by_task: dict[int, list[dict[str, Any]]] = {}
    labels: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    seen_indices: set[int] = set()
    seen_uids: set[str] = set()
    terminal_boundary_events = 0

    for task_spec in task_specs:
        task_name = str(task_spec["task"])
        if task_name not in task_id_by_name:
            raise KeyError(f"Eval list references unknown dataset task {task_name!r}")
        task_id = task_id_by_name[task_name]
        positions = list(task_spec["positions"])
        episode_indices = list(task_spec["episode_indices"])
        if len(positions) != len(episode_indices):
            raise ValueError(f"Task {task_name!r} has mismatched positions and episode_indices")
        pairs = list(zip(positions, episode_indices))
        if trajectories_per_task > 0:
            pairs = pairs[:trajectories_per_task]

        task_entries: list[dict[str, Any]] = []
        for task_position, raw_episode_index in pairs:
            episode_index = int(raw_episode_index)
            if episode_index in seen_indices:
                raise ValueError(f"Eval episode index {episode_index} occurs more than once")
            if episode_index not in episode_by_index:
                raise KeyError(f"Eval episode index {episode_index} is absent from {dataset_root}")
            if episode_index not in annotation_paths:
                raise FileNotFoundError(f"Missing manual annotation for episode {episode_index}")
            episode = episode_by_index[episode_index]
            annotation_path = annotation_paths[episode_index]
            annotation = _read_json(annotation_path)
            metadata_path = annotation_path.with_name("metadata.json")
            metadata = _read_json(metadata_path)
            episode_uid = str(episode["episode_uid"])
            demo_length = int(episode["length"])

            expected = {
                "dataset task": (episode.get("tasks"), [task_name]),
                "metadata episode_index": (metadata.get("episode_index"), episode_index),
                "annotation episode_index": (annotation.get("episode_index"), episode_index),
                "metadata episode_uid": (metadata.get("episode_uid"), episode_uid),
                "annotation episode_uid": (annotation.get("episode_uid"), episode_uid),
                "metadata num_frames": (metadata.get("num_frames"), demo_length),
                "annotation num_frames": (annotation.get("num_frames"), demo_length),
                "metadata task_position": (metadata.get("task_position"), int(task_position)),
            }
            mismatches = [
                f"{name}: got {actual!r}, expected {wanted!r}"
                for name, (actual, wanted) in expected.items()
                if actual != wanted
            ]
            annotation_task = str(annotation.get("task", "")).split(" — ", 1)[0]
            if annotation_task != task_name:
                mismatches.append(
                    f"annotation task: got {annotation.get('task')!r}, expected {task_name!r}"
                )
            if mismatches:
                raise ValueError(f"Eval episode {episode_index} is misaligned: {'; '.join(mismatches)}")
            if episode_uid in seen_uids:
                raise ValueError(f"Eval episode_uid {episode_uid!r} occurs more than once")

            progress = manual_annotation_progress(annotation, demo_length)
            values = annotation_progress_values(
                progress * 100.0,
                value_min=value_min,
                value_max=value_max,
            )
            bins = discretize_values(
                values,
                num_bins=num_bins,
                value_min=value_min,
                value_max=value_max,
            )
            chunk = episode_index // chunks_size
            source_parquet = (
                dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            )
            source_videos = {
                key: str(
                    dataset_root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / key
                    / f"episode_{episode_index:06d}.mp4"
                )
                for key in video_keys
            }
            if not source_parquet.is_file():
                raise FileNotFoundError(source_parquet)
            missing_videos = [path for path in source_videos.values() if not Path(path).is_file()]
            if missing_videos:
                raise FileNotFoundError(missing_videos[0])

            demo_id = f"eval_episode_{episode_index:06d}"
            dataset_success = "success" if bool(episode["success"]) else "fail"
            annotation_dataset_success = str(annotation.get("dataset_success"))
            if annotation_dataset_success != dataset_success:
                raise ValueError(
                    f"Eval episode {episode_index} annotation dataset_success="
                    f"{annotation_dataset_success!r}, expected {dataset_success!r}"
                )
            outcome_override = annotation.get("outcome_override")
            effective_success = str(outcome_override or dataset_success)
            event_frames = np.asarray(
                [min(int(event["frame"]), demo_length - 1) for event in annotation["key_events"]],
                dtype=np.int64,
            )
            event_values = np.asarray(
                [float(event["value"]) for event in annotation["key_events"]],
                dtype=np.float32,
            )
            terminal_boundary_events += sum(
                int(event["frame"] == demo_length) for event in annotation["key_events"]
            )
            entry = {
                "demo_id": demo_id,
                "demo_role": "eval",
                "demo_split": "eval",
                "demo_length": demo_length,
                "source_format": "lerobot_v21",
                "source_episode_index": episode_index,
                "source_episode_uid": episode_uid,
                "source_parquet": str(source_parquet),
                "source_videos": source_videos,
                "annotation_path": str(annotation_path),
                "eval_metadata_path": str(metadata_path.resolve()),
                "task_position": int(task_position),
                "task_name": task_name,
                "dataset_success": dataset_success,
                "effective_success": effective_success,
                "outcome_override": outcome_override,
                "event_frames": event_frames,
                "event_values": event_values,
            }
            labels[(task_id, demo_id)] = {
                "value_continuous": values,
                "value_bin": bins,
                "annotation_progress": progress,
            }
            task_entries.append(entry)
            seen_indices.add(episode_index)
            seen_uids.add(episode_uid)
        entries_by_task[task_id] = task_entries

    summary = {
        "dataset_root": str(dataset_root),
        "eval_root": str(eval_root),
        "dataset_revision": eval_spec.get("dataset_revision"),
        "num_tasks": len(entries_by_task),
        "num_trajectories": sum(len(entries) for entries in entries_by_task.values()),
        "num_frames": sum(entry["demo_length"] for entries in entries_by_task.values() for entry in entries),
        "terminal_boundary_events_clamped": terminal_boundary_events,
        "ground_truth_version": MANUAL_GROUND_TRUTH_VERSION,
    }
    return entries_by_task, labels, summary


def _dataset_success_label(value: Any) -> str:
    return "success" if bool(value) else "fail"


def load_continuous_eval_episode_entries(
    *,
    dataset_root: str | Path,
    annotation_root: str | Path,
    training_task_names: set[str],
    value_min: float,
    value_max: float,
    num_bins: int,
    task_split: str = "all",
    max_tasks: int = 0,
    trajectories_per_task: int = 0,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[tuple[int, str], dict[str, np.ndarray]],
    dict[str, Any],
]:
    """Load dense annotations for the companion ICL demo dataset.

    Every annotated keep=true episode is an evaluation query and a possible
    same-task context demonstration. Tasks are classified as seen/unseen by
    exact instruction-string membership in the checkpoint training dataset.
    """

    if task_split not in {"all", "seen", "unseen"}:
        raise ValueError(f"Unsupported task_split={task_split!r}")
    dataset_root = Path(dataset_root).expanduser().resolve()
    annotation_root = Path(annotation_root).expanduser().resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(annotation_root)

    info = _read_json(dataset_root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 at {dataset_root}")
    dataset_fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 100))
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    task_rows = sorted(
        _read_jsonl(dataset_root / "meta/tasks.jsonl"),
        key=lambda row: int(row["task_index"]),
    )
    episode_rows = _read_jsonl(dataset_root / "meta/episodes.jsonl")
    episode_by_uid: dict[str, dict[str, Any]] = {}
    for episode in episode_rows:
        uid = str(episode["episode_uid"])
        if uid in episode_by_uid:
            raise ValueError(f"Duplicate dataset episode_uid {uid!r}")
        episode_by_uid[uid] = episode

    annotation_by_uid: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(annotation_root.glob("*.json")):
        payload = _read_json(path)
        uid = str(payload.get("episode_uid", ""))
        if not uid:
            raise KeyError(f"Annotation has no episode_uid: {path}")
        if uid in annotation_by_uid:
            raise ValueError(
                f"Duplicate annotations for {uid!r}: {annotation_by_uid[uid][0]} and {path}"
            )
        annotation_by_uid[uid] = (path.resolve(), payload)

    dataset_uids = set(episode_by_uid)
    annotated_uids = set(annotation_by_uid)
    extra_uids = sorted(annotated_uids - dataset_uids)
    keep_true_uids = {
        uid for uid, episode in episode_by_uid.items() if bool(episode.get("keep"))
    }
    missing_keep_true = sorted(keep_true_uids - annotated_uids)
    annotated_keep_false = sorted(
        uid
        for uid in annotated_uids & dataset_uids
        if not bool(episode_by_uid[uid].get("keep"))
    )
    if missing_keep_true:
        raise ValueError(
            "Continuous annotations must cover every dataset keep=true episode; "
            f"missing_keep_true={missing_keep_true[:5]}"
        )

    episodes_by_task: dict[int, list[dict[str, Any]]] = {
        int(row["task_index"]): [] for row in task_rows
    }
    task_name_by_id = {
        int(row["task_index"]): str(row["task"]) for row in task_rows
    }
    task_id_by_name = {name: task_id for task_id, name in task_name_by_id.items()}
    for episode in episode_rows:
        names = episode.get("tasks")
        if not isinstance(names, list) or len(names) != 1:
            raise ValueError(
                f"Episode {episode.get('episode_index')} must have exactly one task"
            )
        task_name = str(names[0])
        if task_name not in task_id_by_name:
            raise KeyError(f"Episode references unknown task {task_name!r}")
        if bool(episode.get("keep")) and str(episode["episode_uid"]) in annotated_uids:
            episodes_by_task[task_id_by_name[task_name]].append(episode)

    selected_task_rows = []
    for row in task_rows:
        task_name = str(row["task"])
        novelty = "seen" if task_name in training_task_names else "unseen"
        if task_split == "all" or novelty == task_split:
            selected_task_rows.append(row)
    if max_tasks > 0:
        selected_task_rows = selected_task_rows[:max_tasks]

    entries_by_task: dict[int, list[dict[str, Any]]] = {}
    labels: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    seen_indices: set[int] = set()
    seen_uids: set[str] = set()
    novelty_counts: Counter[str] = Counter()

    for task_row in selected_task_rows:
        task_id = int(task_row["task_index"])
        task_name = str(task_row["task"])
        task_novelty = "seen" if task_name in training_task_names else "unseen"
        episodes = sorted(
            episodes_by_task[task_id], key=lambda row: int(row["episode_index"])
        )
        query_uids = {
            str(episode["episode_uid"])
            for episode in (
                episodes[:trajectories_per_task]
                if trajectories_per_task > 0
                else episodes
            )
        }
        task_entries: list[dict[str, Any]] = []

        for task_position, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            source_episode_index = int(
                episode.get("source_episode_index", episode_index)
            )
            episode_uid = str(episode["episode_uid"])
            demo_length = int(episode["length"])
            annotation_path, annotation = annotation_by_uid[episode_uid]
            expected_filename = f"{episode_uid.replace(':', '__')}.json"
            expected = {
                "annotation filename": (annotation_path.name, expected_filename),
                "annotation episode_index": (
                    annotation.get("episode_index"),
                    source_episode_index,
                ),
                "annotation episode_uid": (annotation.get("episode_uid"), episode_uid),
                "annotation task": (annotation.get("task"), task_name),
                "annotation num_frames": (annotation.get("num_frames"), demo_length),
                "annotation fps": (float(annotation.get("fps", -1)), dataset_fps),
            }
            mismatches = [
                f"{name}: got {actual!r}, expected {wanted!r}"
                for name, (actual, wanted) in expected.items()
                if actual != wanted
            ]
            dataset_success = _dataset_success_label(episode.get("success"))
            if str(annotation.get("success")) != dataset_success:
                mismatches.append(
                    f"annotation success: got {annotation.get('success')!r}, "
                    f"expected {dataset_success!r}"
                )
            if mismatches:
                raise ValueError(
                    f"Eval episode {episode_index} is misaligned: {'; '.join(mismatches)}"
                )

            points = annotation.get("points")
            if not isinstance(points, list) or len(points) != demo_length:
                point_count = len(points) if isinstance(points, list) else None
                raise ValueError(
                    f"Annotation {annotation_path} has {point_count} points; "
                    f"expected {demo_length}"
                )
            frame_indices = [point.get("frame_index") for point in points]
            if frame_indices != list(range(demo_length)):
                raise ValueError(
                    f"Annotation {annotation_path} frame indices are not 0..{demo_length - 1}"
                )
            timestamp_mismatches = []
            for index, point in enumerate(points):
                timestamp = float(point.get("timestamp", float("nan")))
                if not np.isfinite(timestamp) or abs(
                    timestamp - round(index / dataset_fps, 4)
                ) > 5e-5:
                    timestamp_mismatches.append(index)
            if timestamp_mismatches:
                raise ValueError(
                    f"Annotation {annotation_path} has misaligned timestamps at frames "
                    f"{timestamp_mismatches[:5]}"
                )
            try:
                progress = np.asarray(
                    [point["progress"] for point in points], dtype=np.float32
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Annotation {annotation_path} has invalid progress values"
                ) from exc
            if not np.isfinite(progress).all():
                raise ValueError(f"Annotation {annotation_path} has non-finite progress")
            values = annotation_progress_values(
                progress, value_min=value_min, value_max=value_max
            )
            bins = discretize_values(
                values, num_bins=num_bins, value_min=value_min, value_max=value_max
            )

            chunk = episode_index // chunks_size
            source_parquet = (
                dataset_root
                / "data"
                / f"chunk-{chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            source_videos = {
                key: str(
                    dataset_root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / key
                    / f"episode_{episode_index:06d}.mp4"
                )
                for key in video_keys
            }
            if not source_parquet.is_file():
                raise FileNotFoundError(source_parquet)
            missing_videos = [path for path in source_videos.values() if not Path(path).is_file()]
            if missing_videos:
                raise FileNotFoundError(missing_videos[0])
            if episode_index in seen_indices or episode_uid in seen_uids:
                raise ValueError(f"Duplicate selected eval episode {episode_uid!r}")

            demo_id = f"eval_episode_{episode_index:06d}"
            entry = {
                "demo_id": demo_id,
                "demo_role": "eval",
                "demo_split": "eval",
                "demo_length": demo_length,
                "source_format": "lerobot_v21",
                "dataset_episode_index": episode_index,
                "source_episode_index": source_episode_index,
                "source_episode_uid": episode_uid,
                "source_parquet": str(source_parquet),
                "source_videos": source_videos,
                "annotation_path": str(annotation_path),
                "task_position": task_position,
                "task_name": task_name,
                "task_novelty": task_novelty,
                "evaluate": episode_uid in query_uids,
                "dataset_success": dataset_success,
                "effective_success": dataset_success,
                "outcome_override": None,
                "event_frames": np.empty(0, dtype=np.int64),
                "event_values": np.empty(0, dtype=np.float32),
            }
            labels[(task_id, demo_id)] = {
                "value_continuous": values,
                "value_bin": bins,
                "annotation_progress": np.clip(progress, 0.0, 100.0) / 100.0,
            }
            task_entries.append(entry)
            seen_indices.add(episode_index)
            seen_uids.add(episode_uid)
            if entry["evaluate"]:
                novelty_counts[task_novelty] += 1
        if task_entries:
            entries_by_task[task_id] = task_entries

    summary = {
        "annotation_type": CONTINUOUS_ANNOTATION_TYPE,
        "dataset_root": str(dataset_root),
        "annotation_root": str(annotation_root),
        "dataset_total_episodes": len(episode_rows),
        "dataset_keep_true_episodes": len(keep_true_uids),
        "annotation_files": len(annotation_by_uid),
        "annotation_files_matching_dataset": len(annotated_uids & dataset_uids),
        "annotation_files_outside_dataset": len(extra_uids),
        "annotation_files_for_keep_false": len(annotated_keep_false),
        "task_split": task_split,
        "num_tasks": len(entries_by_task),
        "num_trajectories": sum(
            int(entry["evaluate"])
            for entries in entries_by_task.values()
            for entry in entries
        ),
        "num_frames": sum(
            entry["demo_length"]
            for entries in entries_by_task.values()
            for entry in entries
            if entry["evaluate"]
        ),
        "task_novelty_trajectories": dict(sorted(novelty_counts.items())),
        "ground_truth_version": ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    }
    return entries_by_task, labels, summary


class IclManualEvalDataset(VfeFrameDataset):
    """Use full-dataset eval queries with keep-true context demonstrations."""

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        data_split_preset: str,
        context_split: dict[str, Any],
        entries_by_task: dict[int, list[dict[str, Any]]],
        eval_labels: dict[tuple[int, str], dict[str, np.ndarray]],
        obs_key: str,
        proprio_key: str,
        image_size: int,
        context_obs_key: str,
        context_image_size: int,
        context_max_timesteps: int,
        context_stride: int,
        context_max_original_timesteps: int,
        context_progress_enabled: bool = False,
        query_history_offsets: tuple[int, ...] | list[int] = (0,),
        context_native_feature_dir: str | Path | None,
        native_vision_model_name: str,
        native_vision_model_revision: str | None,
        context_feature_model_name: str | None,
        context_feature_model_revision: str | None,
        context_feature_tokens_per_image: int,
        context_native_feature_stage: str | None,
        context_native_feature_dtype: str | None,
        context_selection: str,
        context_seed: int,
    ) -> None:
        task_ids = sorted(entries_by_task)
        placeholder_queries = {
            task_id: [context_split["query_demo_ids_by_task"][task_id][0]]
            for task_id in task_ids
        }
        context_ids = {
            task_id: list(context_split["context_demo_ids_by_task"][task_id])
            for task_id in task_ids
        }
        super().__init__(
            processed_dir,
            data_split_preset=data_split_preset,
            split="all",
            obs_key=obs_key,
            proprio_key=proprio_key,
            image_size=image_size,
            query_history_offsets=query_history_offsets,
            seed=context_seed,
            task_id_filter=set(task_ids),
            query_demo_ids_by_task=placeholder_queries,
            context_demo_ids_by_task=context_ids,
            context_enabled=True,
            context_progress_enabled=context_progress_enabled,
            context_selection="eval_fixed",
            context_seed=context_seed,
            context_obs_key=context_obs_key,
            context_image_size=context_image_size,
            context_max_timesteps=context_max_timesteps,
            context_stride=context_stride,
            context_max_original_timesteps=context_max_original_timesteps,
            context_native_feature_dir=context_native_feature_dir,
            native_vision_model_name=native_vision_model_name,
            native_vision_model_revision=native_vision_model_revision,
            context_feature_model_name=context_feature_model_name,
            context_feature_model_revision=context_feature_model_revision,
            context_feature_tokens_per_image=context_feature_tokens_per_image,
            context_native_feature_stage=context_native_feature_stage,
            context_native_feature_dtype=context_native_feature_dtype,
        )
        self._eval_labels = eval_labels
        self.icl_eval_context_selection = context_selection
        self.context_selection = f"icl_eval_{context_selection}"

        keep_true_uids_by_task: dict[int, dict[str, str]] = {}
        uid_roles: dict[str, str] = {}
        keep_true_root = Path(self.summary["dataset_root"])
        keep_true_episodes = {
            int(row["episode_index"]): str(row["episode_uid"])
            for row in _read_jsonl(keep_true_root / "meta/episodes.jsonl")
        }
        for task_id in task_ids:
            demo_uids: dict[str, str] = {}
            query_ids = set(context_split["query_demo_ids_by_task"][task_id])
            context_pool = set(context_ids[task_id])
            for demo_id, demo in self.all_demo_by_task[task_id].items():
                source_index = int(demo["source_episode_index"])
                uid = keep_true_episodes[source_index]
                demo_uids[demo_id] = uid
                if demo_id in query_ids:
                    uid_roles[uid] = "train_query"
                elif demo_id in context_pool:
                    uid_roles[uid] = "train_context"
            keep_true_uids_by_task[task_id] = demo_uids
        self.context_uid_by_demo_id = keep_true_uids_by_task

        self.demo_by_task = {}
        self.query_demo_local_index_by_task = {}
        for task_id, entries in entries_by_task.items():
            task_name = str(self.tasks[task_id]["task_name"])
            if any(entry["task_name"] != task_name for entry in entries):
                raise ValueError(f"Eval and context task names disagree for task {task_id}")
            for entry in entries:
                entry["training_overlap"] = uid_roles.get(
                    entry["source_episode_uid"], "not_keep_true"
                )
                self.all_demo_by_task[task_id][entry["demo_id"]] = entry
            self.demo_by_task[task_id] = {entry["demo_id"]: entry for entry in entries}
            self.query_demo_local_index_by_task[task_id] = {
                entry["demo_id"]: index for index, entry in enumerate(entries)
            }

    def load_label(self, task_id: int, demo_id: str) -> dict[str, np.ndarray]:
        key = (int(task_id), str(demo_id))
        if key in self._eval_labels:
            return self._eval_labels[key]
        return super().load_label(task_id, demo_id)

    def select_context_demo_id(
        self,
        task_id: int,
        query_demo_id: str,
        *,
        frame_idx: int,
        sample_index: int,
        epoch: int,
    ) -> str:
        del frame_idx, sample_index, epoch
        task_id = int(task_id)
        entry = self.demo_entry(task_id, query_demo_id)
        return choose_context_demo_id(
            list(self.context_demo_ids_by_task[task_id]),
            local_eval_index=self.query_demo_local_index_by_task[task_id][query_demo_id],
            task_id=task_id,
            seed=self.context_seed,
            strategy=self.icl_eval_context_selection,
            query_episode_uid=str(entry["source_episode_uid"]),
            context_uid_by_demo_id=self.context_uid_by_demo_id[task_id],
        )


class IclContinuousEvalDataset(VfeFrameDataset):
    """Evaluate dense annotations with same-task contexts from the demo dataset."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        entries_by_task: dict[int, list[dict[str, Any]]],
        eval_labels: dict[tuple[int, str], dict[str, np.ndarray]],
        obs_key: str,
        proprio_key: str,
        image_size: int,
        context_obs_key: str,
        context_image_size: int,
        context_max_timesteps: int,
        context_stride: int,
        context_max_original_timesteps: int,
        context_progress_enabled: bool = False,
        query_history_offsets: tuple[int, ...] | list[int] = (0,),
        context_selection: str,
        context_seed: int,
        contexts_per_task: int,
        all_contexts: bool,
        proprio_dim: int,
    ) -> None:
        if contexts_per_task < 2:
            raise ValueError("contexts_per_task must be at least 2 for self-excluding eval")
        self.processed_dir = Path(dataset_root).expanduser().resolve()
        self.summary = {"dataset_root": str(self.processed_dir)}
        self.data_split_preset = "icl_demo_continuous_eval"
        self.split = "all"
        self.obs_key = obs_key
        self.obs_keys = [key.strip() for key in obs_key.split(",") if key.strip()]
        self.proprio_keys = [key.strip() for key in proprio_key.split(",") if key.strip()]
        self.proprio_key = ",".join(self.proprio_keys)
        self.image_size = int(image_size)
        self.query_history_offsets = normalize_query_history_offsets(query_history_offsets)
        self.rng = random.Random(context_seed)
        self.context_enabled = True
        self.context_progress_enabled = bool(context_progress_enabled)
        self.icl_eval_context_selection = context_selection
        self.context_selection = f"icl_demo_{context_selection}"
        self.context_seed = int(context_seed)
        self.all_contexts = bool(all_contexts)
        self._fixed_context_demo_id_by_task: dict[int, str] = {}
        self.context_obs_keys = [
            key.strip() for key in (context_obs_key or obs_key).split(",") if key.strip()
        ]
        self.context_image_size = int(context_image_size)
        self.context_stride = int(context_stride)
        self.context_max_original_timesteps = int(context_max_original_timesteps)
        self.context_max_timesteps = int(context_max_timesteps)
        # The training native-feature cache contains a different dataset. Demo
        # contexts are intentionally decoded and encoded from their own videos.
        self.context_native_feature_dir = None
        self.context_feature_store = None
        self._proprio_dim = int(proprio_dim)
        self._eval_labels = eval_labels

        self._h5_cache = {}
        self._video_cache = OrderedDict()
        self._video_sequential_state = {}
        self._parquet_column_cache = {}
        self._label_cache = {}
        self.tasks = {}
        self.all_demo_by_task = {}
        self.demo_by_task = {}
        self.query_demo_local_index_by_task = {}
        self.query_demo_ids_by_task = {}
        self.context_demo_ids_by_task = {}
        self.context_uid_by_demo_id: dict[int, dict[str, str]] = {}
        self.frames = []
        self.by_task_indices = {}

        for task_id, entries in sorted(entries_by_task.items()):
            if len(entries) < 2:
                raise ValueError(
                    f"Task {task_id} needs at least two annotated episodes for self-excluding context"
                )
            task_name = str(entries[0]["task_name"])
            if any(str(entry["task_name"]) != task_name for entry in entries):
                raise ValueError(f"Task {task_id} contains multiple task names")
            demos = {str(entry["demo_id"]): entry for entry in entries}
            query_ids = [
                str(entry["demo_id"])
                for entry in entries
                if bool(entry.get("evaluate", True))
            ]
            if not query_ids:
                continue
            context_ids = (
                list(demos)
                if self.all_contexts
                else list(demos)[: min(contexts_per_task, len(demos))]
            )
            self.tasks[int(task_id)] = {
                "task_id": int(task_id),
                "task_name": task_name,
                "language_instruction": task_name,
                "obs_keys": list(self.obs_keys),
            }
            self.all_demo_by_task[int(task_id)] = demos
            self.demo_by_task[int(task_id)] = {
                demo_id: demos[demo_id] for demo_id in query_ids
            }
            self.query_demo_ids_by_task[int(task_id)] = query_ids
            self.context_demo_ids_by_task[int(task_id)] = context_ids
            self.query_demo_local_index_by_task[int(task_id)] = {
                demo_id: index for index, demo_id in enumerate(query_ids)
            }
            self.context_uid_by_demo_id[int(task_id)] = {
                demo_id: str(demos[demo_id]["source_episode_uid"])
                for demo_id in context_ids
            }
            self.by_task_indices[int(task_id)] = []
            for demo_id in query_ids:
                entry = demos[demo_id]
                for frame_idx in range(int(entry["demo_length"])):
                    self.by_task_indices[int(task_id)].append(len(self.frames))
                    self.frames.append(
                        FrameRef(
                            task_id=int(task_id), demo_id=demo_id, frame_idx=frame_idx
                        )
                    )
        if not self.frames:
            raise ValueError("No annotated demo-dataset frames selected for evaluation")

    def infer_proprio_dim(self) -> int:
        return self._proprio_dim

    def load_label(self, task_id: int, demo_id: str) -> dict[str, np.ndarray]:
        return self._eval_labels[(int(task_id), str(demo_id))]

    def set_fixed_context(self, task_id: int, context_demo_id: str) -> None:
        """Force one context for every frame of the next query trajectory."""

        task_id = int(task_id)
        context_demo_id = str(context_demo_id)
        if context_demo_id not in self.context_demo_ids_by_task[task_id]:
            raise KeyError(
                f"Context {context_demo_id!r} is not available for task {task_id}"
            )
        self._fixed_context_demo_id_by_task[task_id] = context_demo_id

    def select_context_demo_id(
        self,
        task_id: int,
        query_demo_id: str,
        *,
        frame_idx: int,
        sample_index: int,
        epoch: int,
    ) -> str:
        del frame_idx, sample_index, epoch
        task_id = int(task_id)
        entry = self.demo_entry(task_id, query_demo_id)
        fixed_context = self._fixed_context_demo_id_by_task.get(task_id)
        if fixed_context is not None:
            context_uid = self.context_uid_by_demo_id[task_id][fixed_context]
            if context_uid == str(entry["source_episode_uid"]):
                raise ValueError(
                    f"Task {task_id} query {query_demo_id!r} cannot use itself as context"
                )
            return fixed_context
        return choose_context_demo_id(
            list(self.context_demo_ids_by_task[task_id]),
            local_eval_index=self.query_demo_local_index_by_task[task_id][query_demo_id],
            task_id=task_id,
            seed=self.context_seed,
            strategy=self.icl_eval_context_selection,
            query_episode_uid=str(entry["source_episode_uid"]),
            context_uid_by_demo_id=self.context_uid_by_demo_id[task_id],
        )


def build_all_context_eval_work_items(
    dataset: IclContinuousEvalDataset,
) -> tuple[list[tuple[int, int, int, str, str]], int]:
    """Build ordered leave-one-out (context, query) pairs, grouped by context."""

    work_items: list[tuple[int, int, int, str, str]] = []
    group_index = 0
    trajectory_index = 0
    for task_id in dataset.task_ids:
        query_ids = list(dataset.demo_by_task[task_id])
        for context_demo_id in dataset.context_demo_ids_by_task[task_id]:
            context_uid = dataset.context_uid_by_demo_id[task_id][context_demo_id]
            for query_demo_id in query_ids:
                query_entry = dataset.demo_entry(task_id, query_demo_id)
                if str(query_entry["source_episode_uid"]) == context_uid:
                    continue
                work_items.append(
                    (
                        trajectory_index,
                        group_index,
                        int(task_id),
                        str(context_demo_id),
                        str(query_demo_id),
                    )
                )
                trajectory_index += 1
            group_index += 1
    return work_items, group_index


def shard_all_context_eval_work_items(
    work_items: list[tuple[int, int, int, str, str]],
    dataset: IclContinuousEvalDataset,
    *,
    rank: int,
    world_size: int,
) -> tuple[list[tuple[int, int, int, str, str]], list[int]]:
    """Assign complete context groups using estimated query-frame cost.

    A context group must stay on one rank so its encoded context tokens are
    reused for all queries. Round-robin group assignment can nevertheless be
    badly imbalanced because episode lengths vary. Longest-processing-time
    scheduling keeps groups intact while balancing their total query frames.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    groups: dict[int, list[tuple[int, int, int, str, str]]] = {}
    for item in work_items:
        groups.setdefault(int(item[1]), []).append(item)

    group_costs = {
        group_index: sum(
            int(dataset.demo_entry(task_id, query_demo_id)["demo_length"])
            for _, _, task_id, _, query_demo_id in items
        )
        for group_index, items in groups.items()
    }
    rank_costs = [0 for _ in range(world_size)]
    assigned_rank_by_group: dict[int, int] = {}
    for group_index, cost in sorted(
        group_costs.items(), key=lambda item: (-item[1], item[0])
    ):
        assigned_rank = min(range(world_size), key=lambda value: (rank_costs[value], value))
        assigned_rank_by_group[group_index] = assigned_rank
        rank_costs[assigned_rank] += int(cost)

    local_items = [
        item
        for item in work_items
        if assigned_rank_by_group[int(item[1])] == rank
    ]
    return local_items, rank_costs


def resolve_record_context_demo_id(
    context_demo_ids: list[str] | np.ndarray,
    *,
    forced_context_demo_id: str,
    model_context_enabled: bool,
    query_demo_id: str,
) -> str:
    """Resolve the context identity recorded for one evaluation result.

    Context models must report exactly the context they consumed. No-context
    models report no consumed context; for an all-contexts comparison, retain
    the enumerated context identity so artifacts keep the same directory and
    aggregation layout as the context-model evaluation.
    """

    observed = sorted(set(str(value) for value in context_demo_ids))
    if model_context_enabled:
        if len(observed) != 1:
            raise RuntimeError(
                f"Eval trajectory {query_demo_id} used multiple contexts: {observed}"
            )
        if forced_context_demo_id and observed[0] != forced_context_demo_id:
            raise RuntimeError(
                f"Eval trajectory {query_demo_id} requested context {forced_context_demo_id}, "
                f"but used {observed[0]}"
            )
        return observed[0]
    if observed:
        raise RuntimeError(
            f"No-context eval trajectory {query_demo_id} unexpectedly reported contexts: {observed}"
        )
    return str(forced_context_demo_id)


_TRAJECTORY_RECORD_KEYS = (
    "trajectory_index",
    "task_id",
    "task_name",
    "task_novelty",
    "demo_id",
    "episode_index",
    "episode_uid",
    "dataset_success",
    "effective_success",
    "outcome_override",
    "training_overlap",
    "context_demo_id",
    "context_episode_index",
    "context_episode_uid",
    "num_source_frames",
    "num_evaluated_frames",
    "metrics",
)


def load_existing_trajectory_record(
    path: str | Path,
    *,
    trajectory_index: int,
    task_id: int,
    demo_id: str,
    episode_uid: str,
    context_demo_id: str,
    context_episode_uid: str,
    checkpoint: str | Path,
    ground_truth_version: str,
    vfe_preset: str,
) -> dict[str, Any]:
    """Load and strictly validate a completed trajectory artifact for resume."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"Existing trajectory artifact has no metadata_json: {path}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise TypeError(f"Existing trajectory metadata is not an object: {path}")

    expected = {
        "trajectory_index": int(trajectory_index),
        "task_id": int(task_id),
        "demo_id": str(demo_id),
        "episode_uid": str(episode_uid),
        "context_demo_id": str(context_demo_id),
        "context_episode_uid": str(context_episode_uid),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "ground_truth_version": str(ground_truth_version),
        "vfe_preset": str(vfe_preset),
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing trajectory artifact does not match this eval: {path}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    missing = [key for key in _TRAJECTORY_RECORD_KEYS if key not in metadata]
    if missing:
        raise KeyError(f"Existing trajectory artifact is missing record fields {missing}: {path}")
    return {key: metadata[key] for key in _TRAJECTORY_RECORD_KEYS}


def _load_checkpoint_sidecar(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    candidates = [
        checkpoint_path.parent.parent / "run_config.json",
        checkpoint_path.parent / "run_config.json",
    ]
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    raise FileNotFoundError(
        f"Could not find run_config.json next to checkpoint {checkpoint_path}; "
        "validation without loading the checkpoint is unavailable"
    )


def _runtime_checkpoint_config(
    checkpoint_path: str | Path,
    model: ValueFunctionEstimator | None,
) -> tuple[VfeModelConfig, str, str]:
    if model is not None:
        data_split_config = getattr(model, "checkpoint_data_split_config", None)
        data_split_preset = getattr(model, "checkpoint_data_split_preset", None)
        if not data_split_config or not data_split_preset:
            raise ValueError("Checkpoint must record data_split_config and data_split_preset")
        return model.config, str(data_split_config), str(data_split_preset)
    sidecar = _load_checkpoint_sidecar(checkpoint_path)
    if not sidecar.get("model_config"):
        raise KeyError("Checkpoint run_config.json has no model_config")
    return (
        VfeModelConfig.from_dict(sidecar["model_config"]),
        str(sidecar["data_split_config"]),
        str(sidecar["data_split_preset"]),
    )


def _metric_group(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_trajectories": len(entries),
        "aggregate": aggregate_metric_dicts([entry["metrics"] for entry in entries]),
    }


def _group_metrics(entries: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(entry[field]) for entry in entries})
    return {
        value: _metric_group([entry for entry in entries if str(entry[field]) == value])
        for value in values
    }


def _save_trajectory_result(
    path: Path,
    *,
    result: dict[str, np.ndarray],
    entry: dict[str, Any],
    metadata: dict[str, Any],
    value_min: float,
    value_max: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(value_max - value_min)
    np.savez_compressed(
        path,
        timesteps=np.asarray(result["timesteps"], dtype=np.int64),
        target=np.asarray(result["target"], dtype=np.float32),
        prediction=np.asarray(result["prediction"], dtype=np.float32),
        target_progress=(np.asarray(result["target"], dtype=np.float32) - value_min) / scale,
        prediction_progress=(np.asarray(result["prediction"], dtype=np.float32) - value_min) / scale,
        logits=np.asarray(result["logits"], dtype=np.float32),
        context_demo_ids=np.asarray(result["context_demo_ids"], dtype=str),
        annotation_event_frames=np.asarray(entry["event_frames"], dtype=np.int64),
        annotation_event_values=np.asarray(entry["event_values"], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _build_dataset(
    *,
    args: argparse.Namespace,
    model_config: VfeModelConfig,
    data_split_config: str,
    data_split_preset: str,
) -> tuple[VfeFrameDataset, dict[str, Any], dict[str, Any], dict[str, Any]]:
    context_split = resolve_data_split(
        preset_name=data_split_preset,
        phase="train",
        preset_path=data_split_config,
    )
    preprocess_config = load_preset_preprocess_config(data_split_preset, data_split_config)
    vfe_config = load_vfe_preset(args.vfe_preset, args.vfe_config)
    model_preset = vfe_config["model"]
    feature_preset = vfe_config["context_features"]
    expected_backbone = resolve_vfe_backbone(model_preset)
    if model_config.backbone != expected_backbone:
        raise ValueError(
            f"Checkpoint backbone {model_config.backbone!r} does not match VFE preset "
            f"{args.vfe_preset!r} backbone {expected_backbone!r}"
        )
    context_enabled = model_config.backbone in CONTEXT_BACKBONES
    if context_enabled != bool(model_preset["context_enabled"]):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_enabled does not match checkpoint "
            f"backbone {model_config.backbone!r}"
        )
    if args.cache_context_tokens and not context_enabled:
        raise ValueError("--cache-context-tokens requires a context-enabled checkpoint")
    if model_config.query_history_offsets != model_preset["query_history_offsets"]:
        raise ValueError("VFE preset query_history_offsets does not match checkpoint")
    if model_config.context_progress_enabled != bool(model_preset["context_progress_enabled"]):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_progress_enabled does not match "
            f"checkpoint value {model_config.context_progress_enabled}"
        )
    if model_config.context_slot_embedding_enabled != bool(
        model_preset["context_slot_embedding_enabled"]
    ):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_slot_embedding_enabled does not match "
            f"checkpoint value {model_config.context_slot_embedding_enabled}"
        )

    context_max_timesteps = derive_context_max_timesteps(
        int(model_preset["context_max_original_timesteps"]),
        int(model_preset["context_stride"]),
    )
    if context_enabled and context_max_timesteps != model_config.context_max_timesteps:
        raise ValueError(
            f"VFE preset derives {context_max_timesteps} context slots, but checkpoint expects "
            f"{model_config.context_max_timesteps}"
        )
    if args.annotation_root:
        training_dataset_root = Path(preprocess_config["dataset_root"]).expanduser().resolve()
        trained_task_ids = {int(task_id) for task_id in context_split["task_ids"]}
        training_task_names = {
            str(row["task"])
            for row in _read_jsonl(training_dataset_root / "meta/tasks.jsonl")
            if int(row["task_index"]) in trained_task_ids
        }
        entries_by_task, labels, eval_summary = load_continuous_eval_episode_entries(
            dataset_root=args.dataset_root,
            annotation_root=args.annotation_root,
            training_task_names=training_task_names,
            value_min=model_config.value_min,
            value_max=model_config.value_max,
            num_bins=model_config.num_bins,
            task_split=args.task_split,
            max_tasks=args.max_tasks,
            trajectories_per_task=args.trajectories_per_task,
        )
        dataset_info = _read_json(Path(args.dataset_root) / "meta/info.json")
        proprio_keys = [
            key.strip()
            for key in str(model_preset["proprio_key"]).split(",")
            if key.strip()
        ]
        proprio_dim = 0
        for key in proprio_keys:
            if key not in dataset_info["features"]:
                raise KeyError(f"Eval dataset has no proprio feature {key!r}")
            shape = dataset_info["features"][key].get("shape", [])
            feature_dim = int(np.prod(shape, dtype=np.int64))
            proprio_dim += feature_dim
        for entries in entries_by_task.values():
            for entry in entries:
                entry["training_overlap"] = "separate_dataset"
        dataset = IclContinuousEvalDataset(
            dataset_root=args.dataset_root,
            entries_by_task=entries_by_task,
            eval_labels=labels,
            obs_key=model_preset["obs_key"],
            proprio_key=model_preset["proprio_key"],
            image_size=int(model_preset["image_size"]),
            context_obs_key=model_preset["context_obs_key"],
            context_image_size=int(model_preset["context_image_size"]),
            context_max_timesteps=context_max_timesteps,
            context_stride=int(model_preset["context_stride"]),
            context_max_original_timesteps=int(model_preset["context_max_original_timesteps"]),
            context_progress_enabled=model_config.context_progress_enabled,
            query_history_offsets=model_config.query_history_offsets,
            context_selection=args.context_selection,
            context_seed=args.seed,
            contexts_per_task=args.contexts_per_task,
            all_contexts=args.all_contexts,
            proprio_dim=proprio_dim,
        )
        context_native_feature_dir = None
        processed_dir: Path | None = None
        eval_summary["model_context_enabled"] = context_enabled
        eval_summary["context_source"] = (
            "same_task_demo_dataset"
            if context_enabled
            else (
                "enumerated_for_no_context_comparison_only"
                if args.all_contexts
                else "none"
            )
        )
        eval_summary["contexts_per_task"] = (
            "all"
            if args.all_contexts
            else (args.contexts_per_task if context_enabled else 0)
        )
    else:
        processed_dir = resolve_processed_dir(
            preprocess_config["output_dir"], preprocess_config["ground_truth_version"]
        )
        entries_by_task, labels, eval_summary = load_eval_episode_entries(
            dataset_root=args.dataset_root,
            eval_root=args.eval_root,
            value_min=model_config.value_min,
            value_max=model_config.value_max,
            num_bins=model_config.num_bins,
            max_tasks=args.max_tasks,
            trajectories_per_task=args.trajectories_per_task,
        )
        context_native_feature_dir = (
            feature_preset["output_dir"]
            if bool(feature_preset["enabled"])
            and model_config.backbone == GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE
            and model_config.freeze_vision
            else None
        )
        dataset = IclManualEvalDataset(
            processed_dir=processed_dir,
            data_split_preset=data_split_preset,
            context_split=context_split,
            entries_by_task=entries_by_task,
            eval_labels=labels,
            obs_key=model_preset["obs_key"],
            proprio_key=model_preset["proprio_key"],
            image_size=int(model_preset["image_size"]),
            context_obs_key=model_preset["context_obs_key"],
            context_image_size=int(model_preset["context_image_size"]),
            context_max_timesteps=context_max_timesteps,
            context_stride=int(model_preset["context_stride"]),
            context_max_original_timesteps=int(model_preset["context_max_original_timesteps"]),
            context_progress_enabled=model_config.context_progress_enabled,
            query_history_offsets=model_config.query_history_offsets,
            context_native_feature_dir=context_native_feature_dir,
            native_vision_model_name=model_config.native_vision_model_name,
            native_vision_model_revision=model_config.native_vision_model_revision,
            context_feature_model_name=feature_preset.get("model_name"),
            context_feature_model_revision=feature_preset.get("model_revision"),
            context_feature_tokens_per_image=int(feature_preset.get("tokens_per_image", 256)),
            context_native_feature_stage=feature_preset["stage"],
            context_native_feature_dtype=feature_preset["feature_dtype"],
            context_selection=args.context_selection,
            context_seed=args.seed,
        )
    dataset_proprio_dim = dataset.infer_proprio_dim()
    if dataset_proprio_dim != model_config.proprio_dim:
        raise ValueError(
            f"Checkpoint expects proprio_dim={model_config.proprio_dim}, dataset produced "
            f"{dataset_proprio_dim}"
        )
    overlap_counts = Counter(
        entry["training_overlap"]
        for task_id in dataset.task_ids
        for entry in dataset.demo_by_task[task_id].values()
    )
    eval_summary["training_overlap"] = dict(sorted(overlap_counts.items()))
    eval_summary["processed_context_dir"] = str(processed_dir) if processed_dir else None
    eval_summary["context_native_feature_dir"] = context_native_feature_dir
    eval_summary["context_selection"] = args.context_selection
    return dataset, eval_summary, context_split, vfe_config


def _evaluate(args: argparse.Namespace) -> None:
    info = initialize_distributed_evaluation(args.device)
    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if info.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier(info)

        model = None if args.validate_only else load_model(args.checkpoint, info.device)
        model_config, data_split_config, data_split_preset = _runtime_checkpoint_config(
            args.checkpoint, model
        )
        dataset, eval_summary, context_split, vfe_config = _build_dataset(
            args=args,
            model_config=model_config,
            data_split_config=data_split_config,
            data_split_preset=data_split_preset,
        )
        eval_ground_truth_version = str(eval_summary["ground_truth_version"])
        if args.all_contexts:
            if not isinstance(dataset, IclContinuousEvalDataset):
                raise ValueError("--all-contexts requires --annotation-root")
            all_context_work_items, num_context_groups = build_all_context_eval_work_items(
                dataset
            )
            num_pair_frames = sum(
                int(dataset.demo_entry(task_id, query_demo_id)["demo_length"])
                for _, _, task_id, _, query_demo_id in all_context_work_items
            )
            eval_summary["context_mode"] = "all_leave_one_out"
            eval_summary["num_context_groups"] = num_context_groups
            eval_summary["num_query_context_evaluations"] = len(all_context_work_items)
            eval_summary["num_query_context_frames"] = num_pair_frames
        else:
            all_context_work_items = []
            num_context_groups = 0
            eval_summary["context_mode"] = (
                "selected_fixed_context"
                if model_config.backbone in CONTEXT_BACKBONES
                else "no_context_single_pass"
            )
            eval_summary["num_query_context_evaluations"] = sum(
                len(dataset.demo_by_task[task_id]) for task_id in dataset.task_ids
            )
            eval_summary["num_query_context_frames"] = sum(
                int(dataset.demo_entry(task_id, demo_id)["demo_length"])
                for task_id in dataset.task_ids
                for demo_id in dataset.demo_by_task[task_id]
            )
        print(
            f"ICL eval data ready: {eval_summary['num_tasks']} tasks, "
            f"{eval_summary['num_trajectories']} trajectories, {eval_summary['num_frames']} frames, "
            f"{eval_summary['num_query_context_evaluations']} query-context evaluations, "
            f"{eval_summary['num_query_context_frames']} evaluated frames, "
            f"overlap={eval_summary['training_overlap']}",
            flush=True,
        )
        if args.validate_only:
            if info.is_main_process:
                path = output_dir / "validation.json"
                path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"validation: {path}")
            return
        assert model is not None

        if args.all_contexts:
            # Assign whole context groups to ranks so all n-1 queries for one
            # context share one process and one context-token cache entry. Use
            # query-frame cost rather than group count so ranks finish together.
            local_work_items, estimated_frames_per_rank = shard_all_context_eval_work_items(
                all_context_work_items,
                dataset,
                rank=info.rank,
                world_size=info.world_size,
            )
            total_work_items = len(all_context_work_items)
        else:
            estimated_frames_per_rank = []
            work_items = [
                (int(task_id), demo_id)
                for task_id in dataset.task_ids
                for demo_id in dataset.demo_by_task[task_id]
            ]
            local_work_items = [
                (global_index, global_index, task_id, "", demo_id)
                for global_index, task_id, demo_id in shard_trajectory_work_items(
                    work_items, info.rank, info.world_size
                )
            ]
            total_work_items = len(work_items)
        context_cache = EvaluationContextTokenCache() if args.cache_context_tokens else None
        partial_path = (
            output_dir / "metrics.partial.json"
            if not info.enabled
            else output_dir / f"metrics.rank_{info.rank:03d}.partial.json"
        )
        local_results: list[dict[str, Any]] = []
        local_resumed_count = 0

        def write_partial_payload() -> None:
            partial_payload = {
                "checkpoint": str(args.checkpoint),
                "rank": info.rank,
                "world_size": info.world_size,
                "num_trajectories_completed": len(local_results),
                "num_trajectories_resumed": local_resumed_count,
                "aggregate": aggregate_metric_dicts(
                    [item["metrics"] for item in local_results]
                ),
                "context_token_cache": (
                    context_cache.summary()
                    if context_cache is not None
                    else {"enabled": False}
                ),
            }
            partial_path.write_text(
                json.dumps(partial_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        for local_index, (
            global_index,
            _context_group_index,
            task_id,
            forced_context_demo_id,
            demo_id,
        ) in enumerate(local_work_items, start=1):
            if args.all_contexts:
                assert isinstance(dataset, IclContinuousEvalDataset)
                dataset.set_fixed_context(task_id, forced_context_demo_id)
            entry = dataset.demo_entry(task_id, demo_id)
            context_entry = (
                dataset.all_demo_by_task[task_id][forced_context_demo_id]
                if forced_context_demo_id
                else None
            )
            if args.all_contexts:
                task_dir_name = task_output_dir_name(task_id, entry["task_name"])
                artifact_root = (
                    output_dir
                    / task_dir_name
                    / f"context_{forced_context_demo_id}"
                )
                stem = f"query_{demo_id}"
            else:
                artifact_root = output_dir
                stem = f"task_{task_id:03d}_{demo_id}"
            trajectory_path = artifact_root / "curves" / f"{stem}.npz"

            if args.resume_existing and trajectory_path.is_file():
                if context_entry is None:
                    raise RuntimeError(
                        "--resume-existing currently requires --all-contexts so the "
                        "saved context identity can be validated"
                    )
                record = load_existing_trajectory_record(
                    trajectory_path,
                    trajectory_index=global_index,
                    task_id=task_id,
                    demo_id=demo_id,
                    episode_uid=entry["source_episode_uid"],
                    context_demo_id=forced_context_demo_id,
                    context_episode_uid=context_entry["source_episode_uid"],
                    checkpoint=args.checkpoint,
                    ground_truth_version=eval_ground_truth_version,
                    vfe_preset=args.vfe_preset,
                )
                local_results.append(record)
                local_resumed_count += 1
                print(
                    f"[rank {info.rank}: {local_index}/{len(local_work_items)}; "
                    f"global {global_index + 1}/{total_work_items}] resume existing "
                    f"task {task_id} context={forced_context_demo_id} query={demo_id}",
                    flush=True,
                )
                write_partial_payload()
                continue

            print(
                f"[rank {info.rank}: {local_index}/{len(local_work_items)}; "
                f"global {global_index + 1}/{total_work_items}] task {task_id} "
                f"context={forced_context_demo_id or ('selected' if model_config.backbone in CONTEXT_BACKBONES else 'none')} "
                f"query={demo_id} "
                f"frames={entry['demo_length']}/{entry['demo_length']}",
                flush=True,
            )
            result = predict_trajectory(
                model,
                dataset,
                task_id,
                demo_id,
                info.device,
                batch_size=args.batch_size,
                context_token_cache=context_cache,
                prefetch_batches=args.prefetch_batches,
            )
            metrics = trajectory_metrics(
                result["target"], result["prediction"], args.monotonicity_threshold
            )
            metrics.update(
                {
                    "final_abs_error": float(abs(result["prediction"][-1] - result["target"][-1])),
                    "final_target": float(result["target"][-1]),
                    "final_prediction": float(result["prediction"][-1]),
                }
            )
            record_context_demo_id = resolve_record_context_demo_id(
                result["context_demo_ids"],
                forced_context_demo_id=forced_context_demo_id,
                model_context_enabled=model_config.backbone in CONTEXT_BACKBONES,
                query_demo_id=demo_id,
            )
            context_entry = (
                dataset.all_demo_by_task[task_id][record_context_demo_id]
                if record_context_demo_id
                else None
            )
            record = {
                "trajectory_index": global_index,
                "task_id": task_id,
                "task_name": entry["task_name"],
                "task_novelty": entry.get("task_novelty", "seen"),
                "demo_id": demo_id,
                "episode_index": entry["source_episode_index"],
                "episode_uid": entry["source_episode_uid"],
                "dataset_success": entry["dataset_success"],
                "effective_success": entry["effective_success"],
                "outcome_override": entry["outcome_override"],
                "training_overlap": entry["training_overlap"],
                "context_demo_id": record_context_demo_id,
                "context_episode_index": (
                    context_entry["source_episode_index"] if context_entry else None
                ),
                "context_episode_uid": (
                    context_entry["source_episode_uid"] if context_entry else None
                ),
                "num_source_frames": entry["demo_length"],
                "num_evaluated_frames": entry["demo_length"],
                "metrics": metrics,
            }
            local_results.append(record)
            trajectory_metadata = {
                **record,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "annotation_path": entry["annotation_path"],
                "ground_truth_version": eval_ground_truth_version,
                "vfe_preset": args.vfe_preset,
                "context_selection": (
                    args.context_selection
                    if model_config.backbone in CONTEXT_BACKBONES
                    else "none"
                ),
            }
            _save_trajectory_result(
                trajectory_path,
                result=result,
                entry=entry,
                metadata=trajectory_metadata,
                value_min=model.config.value_min,
                value_max=model.config.value_max,
            )
            if not args.no_plots:
                annotation_type = eval_summary.get("annotation_type", "manual_key_events")
                title = f"task {task_id} {demo_id} ({annotation_type} GT)"
                plot_value_curve(
                    result["timesteps"],
                    result["target"],
                    result["prediction"],
                    artifact_root / "curves" / f"{stem}.png",
                    title,
                )
                plot_error_curve(
                    result["timesteps"],
                    result["prediction"] - result["target"],
                    artifact_root / "errors" / f"{stem}.png",
                    f"{title} error",
                )
            write_partial_payload()

        local_payload = {
            "results": local_results,
            "num_trajectories_resumed": local_resumed_count,
            "context_token_cache": (
                context_cache.summary() if context_cache is not None else {"enabled": False}
            ),
        }
        if info.enabled:
            gathered: list[dict[str, Any] | None] = [None] * info.world_size
            dist.all_gather_object(gathered, local_payload)
            if any(payload is None for payload in gathered):
                raise RuntimeError("Distributed ICL eval did not receive every rank payload")
            rank_payloads = [payload for payload in gathered if payload is not None]
        else:
            rank_payloads = [local_payload]
        results = [record for payload in rank_payloads for record in payload["results"]]
        results.sort(key=lambda record: int(record["trajectory_index"]))
        observed = [int(record["trajectory_index"]) for record in results]
        if observed != list(range(total_work_items)):
            raise RuntimeError(
                "Distributed ICL eval did not produce exactly one result per query-context pair"
            )

        task_groups: dict[str, Any] = {}
        for task_id in dataset.task_ids:
            task_entries = [record for record in results if int(record["task_id"]) == task_id]
            task_groups[str(task_id)] = {
                "task_name": dataset.tasks[task_id]["task_name"],
                "task_novelty": task_entries[0]["task_novelty"],
                **_metric_group(task_entries),
            }
        metrics_payload = {
            "format_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "ground_truth_version": eval_ground_truth_version,
            "eval_data": eval_summary,
            "checkpoint_data_split": {
                "preset": data_split_preset,
                "config": data_split_config,
                "context_manifest": context_split.get("context_split_manifest"),
            },
            "vfe": {
                "preset": args.vfe_preset,
                "config": args.vfe_config,
                "resolved": vfe_config,
            },
            "evaluation": {
                "frame_selection": "every_timestep",
                "batch_size": args.batch_size,
                "context_selection": (
                    args.context_selection
                    if model_config.backbone in CONTEXT_BACKBONES
                    else "none"
                ),
                "context_mode": eval_summary["context_mode"],
                "batch_prefetch_workers_per_rank": int(args.prefetch_batches),
                "seed": args.seed,
                "monotonicity_threshold": args.monotonicity_threshold,
                "resume_existing": bool(args.resume_existing),
                "num_trajectories_resumed": sum(
                    int(payload.get("num_trajectories_resumed", 0))
                    for payload in rank_payloads
                ),
            },
            "num_tasks_evaluated": len(dataset.task_ids),
            "num_trajectories_evaluated": len(results),
            "num_unique_query_trajectories": eval_summary["num_trajectories"],
            "num_context_groups": num_context_groups,
            "aggregate": aggregate_metric_dicts([record["metrics"] for record in results]),
            "per_task": task_groups,
            "by_dataset_success": _group_metrics(results, "dataset_success"),
            "by_effective_success": _group_metrics(results, "effective_success"),
            "by_training_overlap": _group_metrics(results, "training_overlap"),
            "by_task_novelty": _group_metrics(results, "task_novelty"),
            "trajectories": results,
            "context_token_cache": aggregate_context_cache_summaries(
                [payload["context_token_cache"] for payload in rank_payloads]
            ),
            "distributed": {
                "enabled": info.enabled,
                "world_size": info.world_size,
                "sharding": (
                    "whole_context_group_frame_cost_balanced"
                    if args.all_contexts
                    else "whole_trajectory_round_robin"
                ),
                "trajectories_per_rank": [len(payload["results"]) for payload in rank_payloads],
                "estimated_query_context_frames_per_rank": estimated_frames_per_rank,
            },
        }
        if info.is_main_process:
            if args.all_contexts:
                for task_id in dataset.task_ids:
                    for context_demo_id in dataset.context_demo_ids_by_task[task_id]:
                        context_results = [
                            record
                            for record in results
                            if int(record["task_id"]) == task_id
                            and record["context_demo_id"] == context_demo_id
                        ]
                        if not context_results:
                            continue
                        context_payload = {
                            "task_id": task_id,
                            "task_name": dataset.tasks[task_id]["task_name"],
                            "task_novelty": context_results[0]["task_novelty"],
                            "context_demo_id": context_demo_id,
                            "context_episode_index": context_results[0][
                                "context_episode_index"
                            ],
                            "context_episode_uid": context_results[0]["context_episode_uid"],
                            **_metric_group(context_results),
                            "trajectories": context_results,
                        }
                        context_metrics_path = (
                            output_dir
                            / task_output_dir_name(
                                task_id, dataset.tasks[task_id]["task_name"]
                            )
                            / f"context_{context_demo_id}"
                            / "metrics.json"
                        )
                        context_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                        context_metrics_path.write_text(
                            json.dumps(context_payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
            path = output_dir / "metrics.json"
            path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"metrics: {path}")
            print(json.dumps(metrics_payload["aggregate"], indent=2, sort_keys=True))
        distributed_barrier(info)
    finally:
        if info.enabled and dist.is_initialized():
            dist.destroy_process_group()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ICL VFE on either sparse manual annotations or dense "
            "per-frame annotations."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--annotation-root",
        default=None,
        help=(
            "Dense per-frame annotation directory. When set, contexts and queries "
            "come from --dataset-root and --eval-root is unused."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vfe-config", default=str(DEFAULT_VFE_PRESET_PATH))
    parser.add_argument("--vfe-preset", default=DEFAULT_VFE_PRESET)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--context-selection",
        choices=["round_robin", "seeded"],
        default="round_robin",
        help="Select one fixed keep-true context per eval trajectory.",
    )
    parser.add_argument("--cache-context-tokens", action="store_true")
    parser.add_argument(
        "--prefetch-batches",
        action="store_true",
        help="Prepare the next trajectory batch in one background worker per rank.",
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--trajectories-per-task", type=int, default=0)
    parser.add_argument(
        "--task-split",
        choices=["all", "seen", "unseen"],
        default="all",
        help="Filter dense demo-dataset tasks by checkpoint-training task-name overlap.",
    )
    parser.add_argument(
        "--contexts-per-task",
        type=int,
        default=2,
        help=(
            "Number of earliest annotated same-task demo episodes in the fixed context pool. "
            "At least two are required so context episodes can be evaluated without self-context."
        ),
    )
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help=(
            "For dense annotation eval, use every episode as context once and evaluate "
            "all other same-task episodes (ordered leave-one-out pairs)."
        ),
    )
    parser.add_argument("--monotonicity-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse completed all-context trajectory NPZ files in --output-dir after "
            "strictly validating their checkpoint, query, context, and label metadata."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate trajectory/annotation/context alignment without loading the checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_tasks < 0 or args.trajectories_per_task < 0:
        raise ValueError("--max-tasks and --trajectories-per-task must be non-negative")
    if args.annotation_root and args.contexts_per_task < 2:
        raise ValueError("--contexts-per-task must be at least 2 for dense annotation eval")
    if args.all_contexts and not args.annotation_root:
        raise ValueError("--all-contexts requires --annotation-root")
    _evaluate(args)


if __name__ == "__main__":
    main()
