#!/usr/bin/env python
"""CPU-only reference rescoring of the saved paper-ablation predictions.

No model is loaded. Both reference columns use identical saved predictions,
including the original RoboDopamine context-progress inputs of R7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RUNS = {
    "R0": ("IC-VFE (main)", "1_vfe_icl_keep_true_annotation_gemma270m_dualres_stride30_3cam_bs160_4gpu_lr1e5_step30000", "8800", True),
    "R1": ("Same-architecture no-context", "vfe_icl_keep_true_annotation_no_context_gemma270m_dualres_3cam_bs160_2gpu_lr1e5_step30000", "20000", True),
    "R7": ("Native context + progress, no slot embedding", "vfe_icl_keep_true_annotation_gemma270m_dualres_stride30_3cam_context_progress_rope_only_bs160_4gpu_lr1e5_step20000", "icl_demo_keep_true_original640_sequential_all_contexts_step_20000", True),
    "R3": ("RECAP-style, SigLIP 224", "vfe_icl_keep_true_annotation_no_context_recap_style_patch14_224_3cam_bs160_2gpu_lr1e5_step20000", "20000", True),
    "R2": ("RECAP-style, SigLIP 384", "vfe_icl_keep_true_annotation_no_context_recap_style_3cam_bs160_2gpu_lr1e5_step20000", "20000", True),
    "R8": ("RECAP-style 512 dual-resolution context", "2_vfe_icl_keep_true_annotation_gemma270m_recap_style_qwen2_512_dualres_stride30_3cam_no_context_progress_frozen_detail_cache_bs160_4gpu_lr1e5_step10000", "icl_demo_keep_true_original640_sequential_all_contexts_step_10000", True),
}
METRICS = ("mae", "mse", "pearson")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    # Match vfe.evaluate.trajectory_metrics, including float32 MAE/MSE and
    # the zero-correlation convention for constant or single-frame sequences.
    error = prediction - target
    pearson = 0.0
    if len(target) >= 2 and np.std(target) != 0 and np.std(prediction) != 0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
    return {"mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error**2)), "pearson": pearson}


def mean_metrics(records: list[dict]) -> dict:
    return {key: float(np.mean([r[key] for r in records])) for key in METRICS}


def annotation_records(root: Path, source: Path, episodes: dict):
    """Read only the selected episode annotations, without extracting ZIP files."""
    source_label = str(source.relative_to(root)) if source.is_relative_to(root) else str(source)
    if source.is_dir():
        for uid in sorted(episodes):
            path = source / (uid.replace(":", "__") + ".json")
            yield uid, f"{source_label}/{path.name}", path.read_bytes()
        return
    require(source.is_file() and zipfile.is_zipfile(source), f"Not an annotation directory or ZIP: {source}")
    with zipfile.ZipFile(source) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or not info.filename.endswith(".json"):
                continue
            name = Path(info.filename).name
            require(name not in members, f"Duplicate annotation basename in ZIP: {name}")
            members[name] = info.filename
        for uid in sorted(episodes):
            name = uid.replace(":", "__") + ".json"
            require(name in members, f"Missing annotation in ZIP: {name}")
            member = members[name]
            yield uid, f"{source_label}!/{member}", archive.read(member)


def load_annotations(root: Path, folder: Path, episodes: dict, fps: float) -> tuple[dict, list]:
    targets, manifest = {}, []
    for uid, path, raw in annotation_records(root, folder, episodes):
        episode = episodes[uid]
        annotation = json.loads(raw)
        n = episode["length"]
        expected = {
            "episode_uid": uid, "episode_index": episode["source_episode_index"],
            "num_frames": n, "task": episode["tasks"][0],
            "fps": fps, "success": "success",
        }
        for key, value in expected.items():
            require(annotation.get(key) == value, f"Annotation {key} mismatch: {path}")
        points = annotation["points"]
        require(len(points) == n, f"Annotation length mismatch: {path}")
        require([p["frame_index"] for p in points] == list(range(n)), f"Frame index mismatch: {path}")
        times = np.asarray([p["timestamp"] for p in points], dtype=np.float64)
        require(np.isfinite(times).all(), f"Nonfinite timestamp: {path}")
        require(np.allclose(times, np.arange(n) / fps, atol=5.1e-5, rtol=0), f"Timestamp mismatch: {path}")
        progress = np.asarray([p["progress"] for p in points], dtype=np.float32)
        require(np.isfinite(progress).all(), f"Nonfinite annotation: {path}")
        targets[uid] = (-1.0 + np.clip(progress, 0, 100) / 100.0).astype(np.float32)
        manifest.append({"path": path, "sha256": sha256(raw),
                         "episode_uid": uid, "num_frames": n,
                         "interpolation": annotation.get("interpolation"),
                         "steepness": annotation.get("steepness"),
                         "window_seconds": annotation.get("window_seconds")})
    return targets, manifest


def score_run(root: Path, alias: str, spec: tuple, episodes: dict, targets: dict) -> dict:
    name, run, evaluation, visible = spec
    run_root = root / "outputs" / run
    source = run_root / "eval" / evaluation
    summary_paths = [source / "metrics.json"] if (source / "metrics.json").exists() else [
        source / split / "metrics.json" for split in ("seen", "unseen")]
    summaries = [read_json(path) for path in summary_paths]
    source_records = {}
    for summary in summaries:
        require(summary["evaluation"]["frame_selection"] == "every_timestep", f"Sparse source: {alias}")
        for record in summary["trajectories"]:
            key = (record["episode_uid"], record.get("context_episode_uid"))
            require(key not in source_records, f"Duplicate source record: {alias} {key}")
            source_records[key] = record
    files = sorted(source.rglob("*.npz"))
    require(len(files) == len(source_records), f"Curve/record count mismatch: {alias}")
    pairs, keys_seen, source_manifest = [], set(), []
    for i, path in enumerate(files):
        before = path.stat()
        with np.load(path, allow_pickle=False) as curve:
            metadata = json.loads(str(curve["metadata_json"]))
            ts = curve["timesteps"].copy()
            prediction = curve["prediction"].copy()
            original_target = curve["target"].copy()
            context_ids = curve["context_demo_ids"].copy()
        uid, context_uid = metadata["episode_uid"], metadata.get("context_episode_uid")
        key = (uid, context_uid)
        require(uid in episodes and key in source_records and key not in keys_seen, f"Unexpected curve: {path}")
        keys_seen.add(key)
        episode, source_record = episodes[uid], source_records[key]
        n = episode["length"]
        require(np.array_equal(ts, np.arange(n)), f"Timestep coverage mismatch: {path}")
        require(prediction.shape == original_target.shape == (n,), f"Curve shape mismatch: {path}")
        require(prediction.dtype == np.float32, f"Unexpected prediction dtype: {path}")
        require(np.isfinite(prediction).all(), f"Nonfinite prediction: {path}")
        require(np.array_equal(original_target, targets["robodopamine"][uid]), f"Original reference mismatch: {path}")
        for field in ("task_name", "task_novelty", "demo_id", "context_demo_id", "num_source_frames", "num_evaluated_frames"):
            require(metadata[field] == source_record[field], f"Metadata {field} mismatch: {path}")
        require(metadata["task_name"] == episode["tasks"][0], f"Task mismatch: {path}")
        require(metadata["dataset_success"] == "success", f"Non-success source: {path}")
        if context_uid is not None:
            require(context_uid in episodes and context_uid != uid, f"Invalid/self context: {path}")
            require(episodes[context_uid]["tasks"] == episode["tasks"], f"Cross-task context: {path}")
            require(len(context_ids) == n and np.all(context_ids == metadata["context_demo_id"]), f"Changing context: {path}")
        else:
            require(len(context_ids) == 0, f"Unexpected context IDs: {path}")
        scores = {reference: metrics(values[uid], prediction) for reference, values in targets.items()}
        for metric in METRICS:
            require(abs(scores["robodopamine"][metric] - source_record["metrics"][metric]) < 2e-6,
                    f"Original metric mismatch: {alias} {metric} {path}")
        pairs.append({"episode_uid": uid, "context_episode_uid": context_uid,
                      "task_name": metadata["task_name"], "task_novelty": metadata["task_novelty"],
                      "num_frames": n, "metrics": scores})
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"Source changed: {path}")
        source_manifest.append({"path": str(path.relative_to(root)), "size": after.st_size,
                                "mtime_ns": after.st_mtime_ns, "episode_uid": uid,
                                "context_episode_uid": context_uid,
                                "prediction_sha256": sha256(prediction.tobytes()),
                                "timesteps_sha256": sha256(ts.tobytes())})
        if (i + 1) % 500 == 0:
            print(f"{alias}: rescored {i + 1}/{len(files)} saved curves", flush=True)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["episode_uid"]].append(pair)
    require(set(grouped) == set(episodes), f"Query-set mismatch: {alias}")
    queries = []
    context_enabled = summaries[0]["vfe"]["resolved"]["model"]["context_enabled"]
    for uid, records in sorted(grouped.items()):
        require(len({r["task_novelty"] for r in records}) == 1, f"Split mismatch: {alias} {uid}")
        expected_contexts = {u for u, e in episodes.items() if e["tasks"] == episodes[uid]["tasks"] and u != uid} if context_enabled else {None}
        require({r["context_episode_uid"] for r in records} == expected_contexts, f"Context coverage mismatch: {alias} {uid}")
        queries.append({"episode_uid": uid, "task_name": records[0]["task_name"],
                        "task_novelty": records[0]["task_novelty"], "num_contexts": len(records),
                        "num_frames": records[0]["num_frames"],
                        "metrics": {ref: mean_metrics([r["metrics"][ref] for r in records]) for ref in targets}})
    aggregate = {}
    for split in ("seen", "unseen", "all"):
        selected = [q for q in queries if split == "all" or q["task_novelty"] == split]
        require(len(selected) == (240 if split == "all" else 120), f"Split count mismatch: {alias} {split}")
        aggregate[split] = {"num_queries": len(selected), "num_frames": sum(q["num_frames"] for q in selected),
                            "metrics": {ref: mean_metrics([q["metrics"][ref] for q in selected]) for ref in targets}}
    print(alias, json.dumps(aggregate), flush=True)
    return {"run_id": run, "method": name, "visible_in_table": visible,
            "source_eval": str(source.relative_to(root)),
            "source_metrics": [{"path": str(p.relative_to(root)), "sha256": sha256(p.read_bytes())} for p in summary_paths],
            "context_progress_enabled": summaries[0]["vfe"]["resolved"]["model"].get("context_progress_enabled", False),
            "num_prediction_curves": len(pairs), "query_equal": aggregate,
            "per_query": queries, "per_pair": pairs, "prediction_manifest": source_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--human-annotations", type=Path,
                        default=Path("data/annotations/manual_icl_demo_dataset_continuous"),
                        help="Human annotation directory or ZIP; relative paths are resolved against --repo-root.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repo_root.resolve(), args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    human_source = (root / args.human_annotations).resolve()
    human_source_sha256 = sha256(human_source.read_bytes()) if human_source.is_file() else None
    dataset = root / "data/lerobot/adityx23/icl-demo-dataset-keep-true"
    rows = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    episodes = {row["episode_uid"]: row for row in rows if row["keep"]}
    require(len(episodes) == len(rows) == 240, "Expected the current 240-query evaluation dataset")
    require(all(row["success"] for row in episodes.values()), "Expected successful trajectories")
    targets, annotations = {}, {}
    for reference, folder in (("robodopamine", root / "data/annotations/zs_robodopamine_icl_demo_dataset_continuous"),
                              ("human", human_source)):
        targets[reference], annotations[reference] = load_annotations(
            root, folder, episodes, float(read_json(dataset / "meta/info.json")["fps"]))
    results = {alias: score_run(root, alias, spec, episodes, targets) for alias, spec in RUNS.items()}
    split_maps = [{q["episode_uid"]: q["task_novelty"] for q in r["per_query"]} for r in results.values()]
    require(all(m == split_maps[0] for m in split_maps), "Cross-run seen/unseen split mismatch")
    if human_source_sha256 is not None:
        require(sha256(human_source.read_bytes()) == human_source_sha256, "Human annotation archive changed during rescoring")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "gpu_used": False,
               "model_inference_rerun": False, "prediction_postprocessing": "none (raw saved predictions)",
               "reference_mapping": "clip(progress_percent,0,100)/100-1",
               "human_reference": "existing frame-aligned continuous human annotations; interpolation is not regenerated",
               "human_annotation_source": str(human_source.relative_to(root)) if human_source.is_relative_to(root) else str(human_source),
               "human_annotation_source_sha256": human_source_sha256,
               "aggregation": "compute per-query/context metrics, average contexts within each query, then average queries equally within each split",
               "context_inputs": "unchanged from original inference; R7 retains RoboDopamine context progress, not human progress",
               "num_queries": 240, "annotation_manifest": annotations, "runs": {}}
    for alias, result in results.items():
        detail = {key: result.pop(key) for key in ("per_query", "per_pair", "prediction_manifest")}
        (output / f"{alias}_details.json").write_text(json.dumps(detail, indent=2, allow_nan=False) + "\n")
        result["details_file"] = f"{alias}_details.json"
        summary["runs"][alias] = result
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved verified rescoring to {output}", flush=True)


if __name__ == "__main__":
    main()
