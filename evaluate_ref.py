from __future__ import annotations

import argparse
import gc
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import (
    FrameRef,
    VfeFrameDataset,
    collate_vfe_batch,
    derive_context_max_timesteps,
)
from .data_splits import (
    GROUND_TRUTH_VERSION_FIELD,
    load_preset_preprocess_config,
    resolve_data_split,
    resolve_processed_dir,
)
from .labels import validate_ground_truth_version
from .model import (
    CONTEXT_BACKBONES,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
    HF_IMAGE_BACKBONES,
    ValueFunctionEstimator,
    VfeModelConfig,
)
from .visualize import plot_bin_distribution, plot_error_curve, plot_value_curve
from .vfe_presets import (
    DEFAULT_VFE_PRESET_PATH,
    load_vfe_preset,
    resolve_vfe_backbone,
)


class DistributedEvaluationInfo:
    """Process placement for independent, trajectory-sharded evaluation."""

    def __init__(self, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed_evaluation(device_arg: str | None) -> DistributedEvaluationInfo:
    """Initialize a torchrun process group when evaluation has multiple ranks.

    Evaluation does not use DDP: every rank holds a read-only model replica and
    receives a disjoint subset of complete trajectories. The process group is
    used only to collect the small metric payloads after inference finishes.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
        return DistributedEvaluationInfo(rank=0, local_rank=0, world_size=1, device=device)

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank {rank} for WORLD_SIZE={world_size}")
    if torch.cuda.is_available():
        if device_arg and not str(device_arg).startswith("cuda"):
            raise ValueError("Multi-GPU evaluation requires --device cuda (or no device override)")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        if device_arg and str(device_arg).startswith("cuda"):
            raise ValueError("--device cuda was requested, but CUDA is not available")
        device = torch.device("cpu")
        backend = "gloo"
    timeout_seconds = int(os.environ.get("VFE_DISTRIBUTED_TIMEOUT_SECONDS", "7200"))
    if timeout_seconds <= 0:
        raise ValueError(
            "VFE_DISTRIBUTED_TIMEOUT_SECONDS must be positive, "
            f"got {timeout_seconds}"
        )
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return DistributedEvaluationInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def distributed_barrier(info: DistributedEvaluationInfo) -> None:
    if info.enabled:
        dist.barrier()


def shard_trajectory_work_items(
    work_items: list[tuple[int, str]],
    rank: int,
    world_size: int,
) -> list[tuple[int, int, str]]:
    """Assign whole trajectories round-robin, without duplication or padding."""

    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size={world_size}")
    return [
        (trajectory_index, task_id, demo_id)
        for trajectory_index, (task_id, demo_id) in enumerate(work_items)
        if trajectory_index % world_size == rank
    ]


def aggregate_context_cache_summaries(summaries: list[dict[str, int | float]]) -> dict[str, int | float]:
    """Combine rank-local GPU cache statistics for the final metrics file."""

    if not summaries or not any(bool(summary.get("enabled", False)) for summary in summaries):
        return {"enabled": False}
    requests = sum(int(summary.get("requests", 0)) for summary in summaries)
    hits = sum(int(summary.get("hits", 0)) for summary in summaries)
    return {
        "enabled": True,
        "entries": sum(int(summary.get("entries", 0)) for summary in summaries),
        "requests": requests,
        "hits": hits,
        "misses": sum(int(summary.get("misses", 0)) for summary in summaries),
        "hit_rate": float(hits / requests) if requests else 0.0,
        "resident_bytes": sum(int(summary.get("resident_bytes", 0)) for summary in summaries),
    }


def merge_sharded_trajectory_metrics(
    rank_payloads: list[list[tuple[int, int, dict[str, float]]]],
    task_ids: list[int],
    total_trajectories: int,
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    """Validate rank output and reconstruct serial-order aggregate metrics."""

    entries = [entry for payload in rank_payloads for entry in payload]
    entries.sort(key=lambda entry: entry[0])
    observed_indices = [entry[0] for entry in entries]
    expected_indices = list(range(total_trajectories))
    if observed_indices != expected_indices:
        raise RuntimeError(
            "Distributed evaluation did not produce exactly one result per trajectory: "
            f"expected indices {expected_indices[:5]}... ({total_trajectories} total), "
            f"got {observed_indices[:5]}... ({len(observed_indices)} total)"
        )

    all_metrics = [metrics for _, _, metrics in entries]
    metrics_by_task: dict[int, list[dict[str, float]]] = {int(task_id): [] for task_id in task_ids}
    for _, task_id, metrics in entries:
        metrics_by_task.setdefault(int(task_id), []).append(metrics)
    per_task_errors = {
        str(task_id): aggregate_metric_dicts(metrics_by_task[int(task_id)])
        for task_id in task_ids
    }
    return all_metrics, per_task_errors


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def trajectory_metrics(target: np.ndarray, prediction: np.ndarray, monotonicity_threshold: float = 1e-3) -> dict[str, float]:
    error = prediction - target
    diffs = np.diff(prediction)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _corr(target, prediction),
        "spearman": _corr(_rankdata(target), _rankdata(prediction)),
        "monotonicity_violation_rate": float(np.mean(diffs < -monotonicity_threshold)) if len(diffs) else 0.0,
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "target_std": float(np.std(target)),
        "prediction_std": float(np.std(prediction)),
        "target_range": float(np.max(target) - np.min(target)),
        "prediction_range": float(np.max(prediction) - np.min(prediction)),
    }


def load_model(checkpoint_path: str | Path, device: torch.device) -> ValueFunctionEstimator:
    print(f"loading checkpoint on CPU: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = VfeModelConfig.from_dict(checkpoint["model_config"])
    print("constructing model", flush=True)
    model = ValueFunctionEstimator(config)
    print("loading model state", flush=True)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_run_config = checkpoint.get("run_config", {})
    # Runtime consumers (for example RICL progress retrieval) need the exact
    # resolved observation/context contract recorded by training, not only the
    # data-split provenance used by this evaluator.
    model.checkpoint_run_config = checkpoint_run_config
    model.checkpoint_legacy_ground_truth_version = checkpoint_run_config.get(GROUND_TRUTH_VERSION_FIELD)
    model.checkpoint_data_split_config = checkpoint_run_config.get("data_split_config")
    model.checkpoint_data_split_preset = checkpoint_run_config.get("data_split_preset")
    del checkpoint
    gc.collect()
    print(f"moving model to {device}", flush=True)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("model ready", flush=True)
    return model


def validate_checkpoint_ground_truth_provenance(
    *,
    checkpoint_legacy_ground_truth_version: str | None,
    checkpoint_data_split_preset: str | None,
    requested_data_split_preset: str,
    config_ground_truth_version: str,
) -> str:
    """Resolve checkpoint labels only from explicit checkpoint/config metadata."""

    config_ground_truth_version = validate_ground_truth_version(config_ground_truth_version)
    if checkpoint_data_split_preset is None:
        raise ValueError(
            "Checkpoint does not record data_split_preset; its ground-truth version cannot be resolved "
            "through a data config."
        )
    elif checkpoint_data_split_preset != requested_data_split_preset:
        raise ValueError(
            f"Checkpoint was trained with data_split_preset={checkpoint_data_split_preset!r}, but evaluation "
            f"requested {requested_data_split_preset!r}. Use the checkpoint's recorded data config."
        )

    if checkpoint_legacy_ground_truth_version is not None:
        legacy_version = validate_ground_truth_version(checkpoint_legacy_ground_truth_version)
        if legacy_version != config_ground_truth_version:
            raise ValueError(
                f"Checkpoint contains deprecated ground-truth metadata {legacy_version!r}, but data config "
                f"{requested_data_split_preset!r} declares {config_ground_truth_version!r}."
            )
    return config_ground_truth_version


class EvaluationContextTokenCache:
    """GPU-resident raw context-token cache for deterministic evaluation context."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.requests = 0
        self.hits = 0
        self.misses = 0

    def contains(self, key: tuple[int, str]) -> bool:
        return key in self._entries

    def get_or_encode(
        self,
        model: ValueFunctionEstimator,
        batch: dict[str, Any],
        key: tuple[int, str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.requests += 1
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            return entry

        required_keys = {"context_proprio", "context_valid_mask"}
        if bool(getattr(model.config, "context_progress_enabled", False)):
            required_keys.add("context_progress")
        if not required_keys.issubset(batch):
            missing = ", ".join(sorted(required_keys.difference(batch)))
            raise RuntimeError(f"Context cache miss for {key}, but the batch lacks {missing}")
        has_images = "context_images" in batch
        has_native_features = {
            "context_native_features",
            "context_feature_positions",
        }.issubset(batch)
        if has_images == has_native_features:
            raise RuntimeError(
                f"Context cache miss for {key} requires exactly one of raw images or cached native features"
            )
        # All frames in one evaluation batch use the same deterministic context
        # demo. Encode only its first row, then expand that token sequence for
        # the query-frame batch below.
        native_features = None
        native_positions = None
        if has_native_features:
            first_row = batch["context_feature_positions"][:, 0] == 0
            native_features = batch["context_native_features"][first_row]
            native_positions = batch["context_feature_positions"][first_row].clone()
            native_positions[:, 0] = 0
        tokens, mask = model.encode_context_tokens_for_eval(
            batch.get("context_images", None)[:1] if has_images else None,
            batch["context_proprio"][:1],
            batch["context_valid_mask"][:1],
            batch.get("context_progress", None)[:1]
            if "context_progress" in batch
            else None,
            native_features,
            native_positions,
        )
        entry = (tokens.contiguous(), mask.contiguous())
        self._entries[key] = entry
        self.misses += 1
        return entry

    @staticmethod
    def expand(entry: tuple[torch.Tensor, torch.Tensor], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = entry
        return tokens.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def summary(self) -> dict[str, int | float]:
        bytes_used = sum(
            tokens.numel() * tokens.element_size() + mask.numel() * mask.element_size()
            for tokens, mask in self._entries.values()
        )
        return {
            "enabled": True,
            "entries": len(self._entries),
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": float(self.hits / self.requests) if self.requests else 0.0,
            "resident_bytes": bytes_used,
        }


@torch.no_grad()
def predict_trajectory(
    model: ValueFunctionEstimator,
    dataset: VfeFrameDataset,
    task_id: int,
    target_demo_id: str,
    device: torch.device,
    batch_size: int = 64,
    context_token_cache: EvaluationContextTokenCache | None = None,
    prefetch_batches: bool = False,
) -> dict[str, np.ndarray]:
    labels = dataset.load_label(task_id, target_demo_id)
    target = np.asarray(labels["value_continuous"], dtype=np.float32)
    length = len(target)
    model_uses_context = model.config.backbone in CONTEXT_BACKBONES
    if context_token_cache is not None and not model_uses_context:
        raise ValueError("A context-token cache requires a context-enabled model")

    preds: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    context_demo_ids_out: list[str] = []
    batch_ranges = [
        (start, min(start + batch_size, length))
        for start in range(0, length, batch_size)
    ]

    def load_query_batch(start: int, end: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        items = [
            dataset.item_from_ref(
                FrameRef(
                    task_id=int(task_id),
                    demo_id=target_demo_id,
                    frame_idx=frame_idx,
                ),
                sample_index=frame_idx,
                epoch=0,
                # A cache miss needs only one copy of the shared context. It is
                # loaded and collated separately below instead of decoding the
                # same context once for every query item in this batch.
                include_context=model_uses_context and context_token_cache is None,
            )
            for frame_idx in range(start, end)
        ]
        return items, collate_vfe_batch(items)

    executor = ThreadPoolExecutor(max_workers=1) if prefetch_batches else None
    pending: Future[tuple[list[dict[str, Any]], dict[str, Any]]] | None = None
    if executor is not None:
        pending = executor.submit(load_query_batch, *batch_ranges[0])
    try:
        for batch_index, (start, end) in enumerate(batch_ranges):
            if pending is None:
                items, batch = load_query_batch(start, end)
            else:
                items, batch = pending.result()
                pending = None
            cache_key: tuple[int, str] | None = None
            cache_hit = False
            if context_token_cache is not None:
                context_demo_ids = [
                    dataset.select_context_demo_id(
                        task_id,
                        target_demo_id,
                        frame_idx=frame_idx,
                        sample_index=frame_idx,
                        epoch=0,
                    )
                    for frame_idx in range(start, end)
                ]
                if len(set(context_demo_ids)) != 1:
                    raise RuntimeError(
                        "A cached evaluation batch must use one shared context demo"
                    )
                context_demo_ids_out.extend(context_demo_ids)
                cache_key = (int(task_id), context_demo_ids[0])
                cache_hit = context_token_cache.contains(cache_key)
            if context_token_cache is None:
                context_demo_ids_out.extend(
                    str(demo_id) for demo_id in batch.get("context_demo_id", [])
                )
            precomputed_context_tokens = None
            precomputed_context_mask = None
            if context_token_cache is not None:
                if cache_key is None:
                    raise AssertionError("Context cache did not assign a cache key")
                if not cache_hit:
                    context_item = dataset.item_from_ref(
                        FrameRef(
                            task_id=int(task_id),
                            demo_id=target_demo_id,
                            frame_idx=start,
                        ),
                        sample_index=start,
                        epoch=0,
                        include_context=True,
                    )
                    context_batch = collate_vfe_batch([context_item])
                    for key in (
                        "context_images",
                        "context_native_features",
                        "context_feature_positions",
                        "context_proprio",
                        "context_valid_mask",
                        "context_progress",
                    ):
                        if key in context_batch:
                            batch[key] = context_batch[key]
                if executor is not None and batch_index + 1 < len(batch_ranges):
                    pending = executor.submit(
                        load_query_batch, *batch_ranges[batch_index + 1]
                    )
                entry = context_token_cache.get_or_encode(model, batch, cache_key)
                precomputed_context_tokens, precomputed_context_mask = (
                    context_token_cache.expand(entry, len(items))
                )
                batch.pop("context_images", None)
                batch.pop("context_native_features", None)
                batch.pop("context_feature_positions", None)
                batch.pop("context_proprio", None)
                batch.pop("context_valid_mask", None)
                batch.pop("context_progress", None)
            elif executor is not None and batch_index + 1 < len(batch_ranges):
                pending = executor.submit(
                    load_query_batch, *batch_ranges[batch_index + 1]
                )
            keep_cpu_keys = (
                {"query_image"} if model.config.backbone in HF_IMAGE_BACKBONES else set()
            )
            if model.config.backbone in CONTEXT_BACKBONES and context_token_cache is None:
                keep_cpu_keys.add("context_images")
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor) and key not in keep_cpu_keys:
                    batch[key] = value.to(device)
            if context_token_cache is None:
                logits = model(batch=batch)
            else:
                logits = model(
                    query_image=batch["query_image"],
                    language=batch["language"],
                    query_proprio=batch.get("query_proprio"),
                    precomputed_context_tokens=precomputed_context_tokens,
                    precomputed_context_mask=precomputed_context_mask,
                )
            pred = model.expected_value(logits)
            preds.append(pred.cpu().numpy())
            logits_out.append(logits.cpu().numpy())
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    prediction = np.concatenate(preds, axis=0)
    logits_np = np.concatenate(logits_out, axis=0)
    return {
        "timesteps": np.arange(length, dtype=np.int64),
        "target": target,
        "prediction": prediction,
        "logits": logits_np,
        "context_demo_ids": np.asarray(context_demo_ids_out, dtype=str),
    }


def save_trajectory_data(
    path: str | Path,
    result: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Persist the arrays behind one curve next to its PNG visualization."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "timesteps": np.asarray(result["timesteps"], dtype=np.int64),
        "target": np.asarray(result["target"], dtype=np.float32),
        "prediction": np.asarray(result["prediction"], dtype=np.float32),
        "logits": np.asarray(result["logits"], dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    context_demo_ids = np.asarray(result.get("context_demo_ids", []), dtype=str)
    if context_demo_ids.size:
        arrays["context_demo_ids"] = context_demo_ids
    np.savez_compressed(path, **arrays)


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def _evaluate(args: argparse.Namespace, distributed_info: DistributedEvaluationInfo) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if distributed_info.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed_info)
    device = distributed_info.device
    print(
        f"evaluation device: {device} rank={distributed_info.rank}/{distributed_info.world_size}",
        flush=True,
    )
    model = load_model(args.checkpoint, device)
    checkpoint_data_split_config = getattr(model, "checkpoint_data_split_config", None)
    checkpoint_data_split_preset = getattr(model, "checkpoint_data_split_preset", None)
    if not checkpoint_data_split_config or not checkpoint_data_split_preset:
        raise ValueError(
            "Checkpoint must record both data_split_config and data_split_preset for evaluation."
        )
    args.data_split_config = str(checkpoint_data_split_config)
    args.data_split_preset = str(checkpoint_data_split_preset)
    data_split = resolve_data_split(
        preset_name=args.data_split_preset,
        phase=args.data_split_phase,
        preset_path=args.data_split_config,
    )
    preprocess_config = load_preset_preprocess_config(args.data_split_preset, args.data_split_config)
    ground_truth_version = preprocess_config["ground_truth_version"]
    processed_dir = resolve_processed_dir(
        preprocess_config["output_dir"],
        ground_truth_version,
    )
    print(f"data split preset: {args.data_split_preset}", flush=True)
    print(f"processed ground truth: {processed_dir}", flush=True)
    checkpoint_legacy_ground_truth_version = getattr(model, "checkpoint_legacy_ground_truth_version", None)
    validate_checkpoint_ground_truth_provenance(
        checkpoint_legacy_ground_truth_version=checkpoint_legacy_ground_truth_version,
        checkpoint_data_split_preset=checkpoint_data_split_preset,
        requested_data_split_preset=args.data_split_preset,
        config_ground_truth_version=ground_truth_version,
    )
    vfe_config = load_vfe_preset(args.vfe_preset, args.vfe_config)
    model_preset = vfe_config["model"]
    feature_preset = vfe_config["context_features"]
    expected_backbone = resolve_vfe_backbone(model_preset)
    if model.config.backbone != expected_backbone:
        raise ValueError(
            f"Checkpoint backbone {model.config.backbone!r} does not match VFE preset "
            f"{args.vfe_preset!r} effective backbone {expected_backbone!r}"
        )
    obs_key = model_preset["obs_key"]
    proprio_key = model_preset["proprio_key"]
    image_size = int(model_preset["image_size"])
    context_obs_key = model_preset["context_obs_key"]
    context_image_size = int(model_preset["context_image_size"])
    context_max_original_timesteps = int(model_preset["context_max_original_timesteps"])
    context_stride = int(model_preset["context_stride"])
    if model.config.proprio_dim == 0 and proprio_key:
        print(
            "checkpoint has proprio_dim=0; ignoring preset proprio_key for compatibility with older checkpoints",
            flush=True,
        )
        proprio_key = ""

    context_enabled = model.config.backbone in CONTEXT_BACKBONES
    if context_enabled != bool(model_preset["context_enabled"]):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_enabled does not match checkpoint "
            f"backbone {model.config.backbone!r}"
        )
    if model.config.context_progress_enabled != bool(model_preset["context_progress_enabled"]):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_progress_enabled does not match "
            f"checkpoint value {model.config.context_progress_enabled}"
        )
    if model.config.context_slot_embedding_enabled != bool(
        model_preset["context_slot_embedding_enabled"]
    ):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_slot_embedding_enabled does not match "
            f"checkpoint value {model.config.context_slot_embedding_enabled}"
        )
    context_native_feature_dir = (
        feature_preset["output_dir"]
        if bool(feature_preset["enabled"])
        and context_enabled
        and model.config.backbone == GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE
        and model.config.freeze_vision
        else None
    )
    if args.cache_context_tokens and not context_enabled:
        raise ValueError("--cache-context-tokens requires a context VFE backbone")
    context_token_cache = EvaluationContextTokenCache() if args.cache_context_tokens else None
    context_selection = (
        args.context_selection
        if args.context_selection is not None
        else data_split.get("context_selection", "eval_fixed")
    )
    if model.config.query_history_offsets != model_preset["query_history_offsets"]:
        raise ValueError("VFE preset query_history_offsets does not match checkpoint")
    context_max_timesteps = derive_context_max_timesteps(
        context_max_original_timesteps,
        context_stride,
    )
    if context_enabled and context_max_timesteps != model.config.context_max_timesteps:
        raise ValueError(
            f"Evaluation context sampling derives {context_max_timesteps} slots from "
            f"preset context_max_original_timesteps={context_max_original_timesteps} and "
            f"context_stride={context_stride}, but checkpoint expects "
            f"context_max_timesteps={model.config.context_max_timesteps}. Use the same context stride/max window "
            "as training."
        )
    dataset = VfeFrameDataset(
        processed_dir,
        split=data_split["demo_split"],
        obs_key=obs_key,
        proprio_key=proprio_key,
        image_size=image_size,
        query_history_offsets=model.config.query_history_offsets,
        seed=args.seed,
        task_id_filter=data_split["task_id_set"],
        query_demo_ids_by_task=data_split.get("query_demo_ids_by_task"),
        context_demo_ids_by_task=data_split.get("context_demo_ids_by_task"),
        context_enabled=context_enabled,
        context_progress_enabled=model.config.context_progress_enabled,
        context_selection=context_selection,
        context_seed=int(data_split.get("context_split_seed", args.seed)),
        context_obs_key=context_obs_key,
        context_image_size=context_image_size,
        context_max_timesteps=context_max_timesteps,
        context_stride=context_stride,
        context_max_original_timesteps=context_max_original_timesteps,
        context_native_feature_dir=context_native_feature_dir,
        native_vision_model_name=model.config.native_vision_model_name,
        native_vision_model_revision=model.config.native_vision_model_revision,
        context_feature_model_name=feature_preset.get("model_name"),
        context_feature_model_revision=feature_preset.get("model_revision"),
        context_feature_tokens_per_image=int(feature_preset.get("tokens_per_image", 256)),
        context_native_feature_stage=feature_preset["stage"],
        context_native_feature_dtype=feature_preset["feature_dtype"],
        data_split_preset=args.data_split_preset,
    )
    dataset_proprio_dim = dataset.infer_proprio_dim()
    if dataset_proprio_dim != model.config.proprio_dim:
        raise ValueError(
            f"Checkpoint expects proprio_dim={model.config.proprio_dim}, but preset proprio_key={proprio_key!r} "
            f"produced dim={dataset_proprio_dim}. Use the same VFE preset as training."
        )

    task_ids = dataset.task_ids[: args.max_tasks] if args.max_tasks > 0 else dataset.task_ids
    work_items: list[tuple[int, str]] = []
    for task_id in task_ids:
        all_demo_ids = list(dataset.demo_by_task[task_id].keys())
        demo_ids = all_demo_ids[: args.trajectories_per_task] if args.trajectories_per_task > 0 else all_demo_ids
        work_items.extend((int(task_id), demo_id) for demo_id in demo_ids)
    total_trajectories = len(work_items)
    local_work_items = shard_trajectory_work_items(
        work_items,
        rank=distributed_info.rank,
        world_size=distributed_info.world_size,
    )
    print(
        f"evaluating split={args.split}: {len(task_ids)} tasks, {total_trajectories} trajectories, "
        f"rank {distributed_info.rank} owns {len(local_work_items)}, batch_size={args.batch_size}, "
        f"preset={args.data_split_preset}, phase={args.data_split_phase}, "
        f"demo_split={data_split['demo_split']}, proprio_dim={dataset_proprio_dim}, context={context_enabled}, "
        f"context_selection={context_selection}, "
        f"context_stride={context_stride}, context_max_original_timesteps={context_max_original_timesteps}, "
        f"derived_context_max_timesteps={context_max_timesteps}",
        flush=True,
    )
    local_results: list[tuple[int, int, dict[str, float]]] = []
    partial_metrics_path = (
        output_dir / "metrics.partial.json"
        if not distributed_info.enabled
        else output_dir / f"metrics.rank_{distributed_info.rank:03d}.partial.json"
    )
    for local_trajectory_index, (trajectory_index, task_id, target_demo_id) in enumerate(local_work_items, start=1):
        print(
            f"[rank {distributed_info.rank}: {local_trajectory_index}/{len(local_work_items)}; "
            f"global {trajectory_index + 1}/{total_trajectories}] task {task_id} {target_demo_id}",
            flush=True,
        )
        result = predict_trajectory(
            model,
            dataset,
            task_id,
            target_demo_id,
            device,
            batch_size=args.batch_size,
            context_token_cache=context_token_cache,
            prefetch_batches=args.prefetch_batches,
        )
        metrics = trajectory_metrics(
            result["target"],
            result["prediction"],
            monotonicity_threshold=args.monotonicity_threshold,
        )
        local_results.append((trajectory_index, task_id, metrics))

        stem = f"task_{task_id:03d}_{target_demo_id}"
        context_demo_ids = [str(demo_id) for demo_id in result["context_demo_ids"].tolist()]
        unique_context_demo_ids = sorted(set(context_demo_ids))
        trajectory_metadata: dict[str, Any] = {
            "format_version": 1,
            "task_id": int(task_id),
            "demo_id": target_demo_id,
            "source_hdf5": str(dataset.demo_entry(task_id, target_demo_id).get("source_hdf5", "")),
            "checkpoint": str(args.checkpoint),
            "data_split": {
                "preset": args.data_split_preset,
                "phase": args.data_split_phase,
                "demo_split": data_split["demo_split"],
            },
            "vfe": {
                "preset": args.vfe_preset,
                "config": args.vfe_config,
                "resolved": vfe_config,
            },
            "model_config": model.config.to_dict(),
            "context": {
                "enabled": context_enabled,
                "selection": dataset.context_selection,
                "stride": context_stride,
                "native_feature_dir": context_native_feature_dir,
                "max_original_timesteps": context_max_original_timesteps,
                "max_timesteps": context_max_timesteps,
                "unique_demo_ids": unique_context_demo_ids,
            },
            "evaluation": {
                "batch_size": args.batch_size,
                "seed": args.seed,
                "monotonicity_threshold": args.monotonicity_threshold,
            },
            "distributed": {
                "rank": distributed_info.rank,
                "world_size": distributed_info.world_size,
            },
            "metrics": metrics,
        }
        if len(unique_context_demo_ids) == 1:
            trajectory_metadata["context"]["demo_id"] = unique_context_demo_ids[0]
        save_trajectory_data(
            output_dir / "curves" / f"{stem}.npz",
            result=result,
            metadata=trajectory_metadata,
        )
        plot_value_curve(
            result["timesteps"],
            result["target"],
            result["prediction"],
            output_dir / "curves" / f"{stem}.png",
            f"task {task_id} {target_demo_id}",
        )
        plot_error_curve(
            result["timesteps"],
            result["prediction"] - result["target"],
            output_dir / "errors" / f"{stem}.png",
            f"task {task_id} {target_demo_id} error",
        )

        if args.plot_distributions:
            for frame_idx in [0, len(result["timesteps"]) // 2, len(result["timesteps"]) - 1]:
                probs = F.softmax(torch.from_numpy(result["logits"][frame_idx]), dim=-1).numpy()
                bin_values = np.linspace(model.config.value_min, model.config.value_max, model.config.num_bins)
                plot_bin_distribution(
                    bin_values,
                    probs,
                    output_dir / "distributions" / f"{stem}_frame_{frame_idx:04d}.png",
                    f"task {task_id} frame {frame_idx}",
                )

        local_partial_payload = {
            "split": args.split,
            "data_split_preset": args.data_split_preset,
            "data_split_phase": args.data_split_phase,
            "checkpoint": str(args.checkpoint),
            "rank": distributed_info.rank,
            "world_size": distributed_info.world_size,
            "scope": "rank_local",
            "num_trajectories_completed": len(local_results),
            "aggregate": aggregate_metric_dicts([entry[2] for entry in local_results]),
            "context_token_cache": (
                context_token_cache.summary() if context_token_cache is not None else {"enabled": False}
            ),
        }
        partial_metrics_path.write_text(
            json.dumps(local_partial_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    local_payload = {
        "metrics": local_results,
        "context_token_cache": context_token_cache.summary() if context_token_cache is not None else {"enabled": False},
    }
    if distributed_info.enabled:
        gathered_payloads: list[dict[str, Any] | None] = [None] * distributed_info.world_size
        dist.all_gather_object(gathered_payloads, local_payload)
        if any(payload is None for payload in gathered_payloads):
            raise RuntimeError("Distributed evaluation did not receive a payload from every rank")
        rank_payloads = [payload for payload in gathered_payloads if payload is not None]
    else:
        rank_payloads = [local_payload]

    all_metrics, per_task_errors = merge_sharded_trajectory_metrics(
        [payload["metrics"] for payload in rank_payloads],
        task_ids=[int(task_id) for task_id in task_ids],
        total_trajectories=total_trajectories,
    )
    combined_cache_summary = aggregate_context_cache_summaries(
        [payload["context_token_cache"] for payload in rank_payloads]
    )
    metrics_payload = {
        "split": args.split,
        "data_split_preset": args.data_split_preset,
        "data_split_phase": args.data_split_phase,
        "resolved_data_split": {key: value for key, value in data_split.items() if key != "task_id_set"},
        "checkpoint": str(args.checkpoint),
        "num_tasks_evaluated": len(task_ids),
        "num_trajectories_evaluated": len(all_metrics),
        "trajectories_per_task": args.trajectories_per_task,
        "context_selection": context_selection,
        "aggregate": aggregate_metric_dicts(all_metrics),
        "per_task_average_error": per_task_errors,
        "context_token_cache": combined_cache_summary,
        "distributed": {
            "enabled": distributed_info.enabled,
            "world_size": distributed_info.world_size,
            "sharding": "whole_trajectory_round_robin",
            "trajectories_per_rank": [len(payload["metrics"]) for payload in rank_payloads],
        },
    }
    if distributed_info.is_main_process:
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"metrics: {metrics_path}")
        if context_token_cache is not None:
            print(f"context token cache: {json.dumps(combined_cache_summary, sort_keys=True)}")
        print(json.dumps(metrics_payload["aggregate"], indent=2, sort_keys=True))
    distributed_barrier(distributed_info)


def evaluate(args: argparse.Namespace) -> None:
    distributed_info = initialize_distributed_evaluation(args.device)
    try:
        _evaluate(args, distributed_info)
    finally:
        if distributed_info.enabled and dist.is_initialized():
            dist.destroy_process_group()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the LIBERO-100 context-demo VFE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/vfe_eval")
    parser.add_argument("--data-split-phase", default="eval_unseen")
    parser.add_argument("--vfe-config", default=str(DEFAULT_VFE_PRESET_PATH))
    parser.add_argument("--vfe-preset", required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cache-context-tokens",
        action="store_true",
        help="Cache raw context tokens on the model device by (task_id, context_demo_id) during evaluation.",
    )
    parser.add_argument(
        "--context-selection",
        choices=["train_seeded", "eval_fixed"],
        default=None,
        help="Override the data split's context selection policy.",
    )
    parser.add_argument(
        "--prefetch-batches",
        action="store_true",
        help="Prepare the next query-frame batch in one background worker.",
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--trajectories-per-task", type=int, default=0, help="Trajectories to plot per task; 0 means every demo in the selected split.")
    parser.add_argument("--plot-distributions", action="store_true")
    parser.add_argument("--monotonicity-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
