#!/usr/bin/env python
"""CPU-only causal EMA rescoring of saved VFE trajectory predictions."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.stats import rankdata


def causal_ema(prediction: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("prediction must be a nonempty finite 1D array")
    return lfilter([alpha], [1, -(1-alpha)], p, zi=[(1-alpha)*p[0]])[0]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    delta = np.diff(p)
    return {
        "pearson": corr(y, p),
        "spearman": corr(rankdata(y), rankdata(p)),
        "mae": float(np.mean(np.abs(y-p))),
        "mse": float(np.mean((y-p)**2)),
        "monotonicity_violation_rate": float(np.mean(delta < -threshold)) if len(delta) else 0.0,
        "mean_abs_frame_delta": float(np.mean(np.abs(delta))) if len(delta) else 0.0,
        "prediction_mean": float(np.mean(p)),
        "prediction_std": float(np.std(p)),
        "prediction_range": float(np.ptp(p)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_range": float(np.ptp(y)),
        "final_abs_error": float(abs(y[-1]-p[-1])),
        "final_prediction": float(p[-1]),
        "final_target": float(y[-1]),
    }


def group(records: list[dict]) -> dict:
    return {
        "num_trajectories": len(records),
        "aggregate": {k: float(np.mean([r["metrics"][k] for r in records]))
                      for k in records[0]["metrics"]},
    }


def payload(records: list[dict], provenance: dict) -> dict:
    result = {**provenance, **group(records),
              "num_trajectories_evaluated": len(records), "trajectories": records}
    result["num_unique_query_trajectories"] = len({(r["task_id"], r["demo_id"]) for r in records})
    for field, key in [("task_id", "per_task"), ("task_novelty", "by_task_novelty"),
                       ("training_overlap", "by_training_overlap"),
                       ("dataset_success", "by_dataset_success"),
                       ("effective_success", "by_effective_success")]:
        values = sorted({str(r.get(field, "unknown")) for r in records})
        result[key] = {v: group([r for r in records if str(r.get(field, "unknown")) == v])
                       for v in values}
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def plot_episode(destination: Path, ts: np.ndarray, y: np.ndarray, raw: np.ndarray,
                 smooth: np.ndarray, metadata: dict, alpha: float,
                 raw_metrics: dict, smooth_metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = (f'{metadata["task_name"]}\n'
             f'Query: {metadata["demo_id"]} | Context: {metadata["context_demo_id"]}')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, raw, color="gray", alpha=.55, linewidth=.8, label="Raw VFE")
    ax.plot(ts, y, color="black", linewidth=1.8, label="Target")
    ax.plot(ts, smooth, color="tab:blue", linewidth=1.5, label=f"Causal EMA alpha={alpha:g}")
    ax.set(xlabel="Timestep", ylabel="Value", title=title)
    ax.text(.02, .03,
            f'Pearson: {raw_metrics["pearson"]:.4f} -> {smooth_metrics["pearson"]:.4f}\n'
            f'MAE: {raw_metrics["mae"]:.4f} -> {smooth_metrics["mae"]:.4f}',
            transform=ax.transAxes, bbox={"facecolor":"white", "alpha":.85, "edgecolor":"none"})
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=130); plt.close(fig)
    error_path = destination.parent.parent / "errors" / destination.with_suffix(".png").name
    error_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, raw-y, color="gray", alpha=.55, linewidth=.8, label="Raw - target")
    ax.plot(ts, smooth-y, color="tab:blue", linewidth=1.4, label=f"EMA alpha={alpha:g} - target")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Timestep", ylabel="Signed prediction error", title=title)
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(error_path, dpi=130); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-trajectories", type=int)
    parser.add_argument("--plots", type=int, default=6)
    parser.add_argument("--all-plots", action="store_true", help="Save curve and error PNG for every query-context pair")
    parser.add_argument("--plot-workers", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="Add new alphas to a previously completed export")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    alphas = sorted(set(args.alphas))
    if not alphas or any(not 0 < a <= 1 for a in alphas):
        parser.error("alphas must be in (0, 1]")
    if output == source or output in source.parents or source in output.parents:
        parser.error("output must be separate from the source eval tree")
    if args.plot_workers < 1:
        parser.error("plot-workers must be positive")
    previous = None
    if args.append:
        previous = json.loads((output / "comparison.json").read_text())
        if previous["source_eval"] != str(source):
            parser.error("append source differs from existing export")
        for a in alphas:
            if (output / f"ema_{a:g}").exists():
                parser.error(f"alpha={a:g} already exists; will not overwrite it")
    elif output.exists():
        parser.error("output already exists; choose a new directory to preserve previous results")
    source_summary_path = source / "metrics.json"
    summary = json.loads(source_summary_path.read_text()) if source_summary_path.exists() else None
    if summary is None and not args.allow_partial:
        parser.error("source eval is incomplete; --allow-partial explicitly permits a snapshot")
    files = sorted(source.glob("*/context_*/curves/*.npz"))
    if not files:
        parser.error("no saved context trajectory NPZ files found")
    expected = summary["num_trajectories_evaluated"] if summary else args.expected_trajectories
    if summary and len(files) != expected:
        raise ValueError(f"Expected {expected} trajectories, found {len(files)}")
    if previous:
        prior_manifest = json.loads((output / "source_manifest.json").read_text())
        current_files = [{"path": str(p.relative_to(source)), "size": p.stat().st_size,
                          "mtime_ns": p.stat().st_mtime_ns} for p in files]
        if prior_manifest["files"] != current_files or prior_manifest.get("skipped"):
            raise ValueError("Append requires exactly the same unchanged source snapshot")
    threshold = summary.get("evaluation", {}).get("monotonicity_threshold", .001) if summary else .001
    provenance = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_eval": str(source),
        "source_eval_complete_at_start": summary is not None,
        "is_partial": summary is None,
        "expected_trajectories": expected,
        "aggregation": "unweighted mean of metrics computed separately for each query-context trajectory",
        "postprocessing": "causal scalar EMA; reset to first prediction for each trajectory; no future or target inputs",
        "parameter_selection": "exploratory fixed alphas, not selected on held-out validation",
        "monotonicity_threshold": threshold,
        "gpu_required": False,
    }
    variants = {"raw": []} | {f"ema_{a:g}": [] for a in alphas}
    snapshots, skipped = [], []
    plot_indices = set(np.linspace(0, len(files)-1, min(max(args.plots, 0), len(files)), dtype=int))
    plot_pool = ProcessPoolExecutor(max_workers=args.plot_workers) if args.all_plots else None
    pending_plots = []
    for i, path in enumerate(files):
        try:
            stat = path.stat()
            with np.load(path, allow_pickle=False) as z:
                y, p = z["target"].astype(float), z["prediction"].astype(float)
                ts = z["timesteps"].copy()
                metadata = json.loads(str(z["metadata_json"]))
                contexts = z["context_demo_ids"].copy()
        except (OSError, ValueError, EOFError) as exc:
            if summary:
                raise
            skipped.append({"path": str(path), "error": str(exc)})
            continue
        if y.shape != p.shape or ts.shape != p.shape or not len(p):
            raise ValueError(f"Invalid trajectory shapes: {path}")
        if not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(np.diff(ts) <= 0):
            raise ValueError(f"Invalid values or timestep ordering: {path}")
        # Fixed-context artifacts are required so EMA never mixes context identities.
        if len(np.unique(contexts)) != 1:
            raise ValueError(f"Context changes within trajectory: {path}")
        base = {k: v for k, v in metadata.items() if k != "metrics"}
        relative = path.relative_to(source)
        base["source_curve"] = str(relative)
        raw_metrics = metrics(y, p, threshold)
        # Confirm our CPU metrics agree with the inference-time saved metrics.
        for k, value in metadata.get("metrics", {}).items():
            if k in raw_metrics and abs(raw_metrics[k]-value) > 2e-6:
                raise ValueError(f"Raw metric mismatch {path}: {k}")
        variants["raw"].append({**base, "metrics": raw_metrics})
        curves = {}
        for a in alphas:
            name = f"ema_{a:g}"
            smooth = causal_ema(p, a)
            m = metrics(y, smooth, threshold)
            record = {**base, "metrics": m}
            variants[name].append(record)
            curves[name] = smooth
            destination = output / name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, timesteps=ts, target=y, prediction=smooth,
                                raw_prediction=p, context_demo_ids=contexts,
                                metadata_json=np.asarray(json.dumps({**record, "ema_alpha": a,
                                                                    "source_curve": str(path)})))
            if plot_pool is not None:
                pending_plots.append(plot_pool.submit(plot_episode, destination, ts, y, p,
                                                      smooth, metadata, a, raw_metrics, m))
                if len(pending_plots) >= 4*args.plot_workers:
                    for job in pending_plots:
                        job.result()
                    pending_plots.clear()
        snapshots.append({"path": str(relative), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if i in plot_indices and not args.all_plots and not args.append:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(ts, p, color="gray", alpha=.45, linewidth=.8, label="Raw")
            ax.plot(ts, y, color="black", linewidth=2, label="Target (reference only)")
            for name, smooth in curves.items():
                ax.plot(ts, smooth, linewidth=1.2, label=name)
            ax.set(xlabel="Timestep", ylabel="Value", title=f'{metadata["task_name"]} / {metadata["demo_id"]}')
            ax.legend(); fig.tight_layout()
            plot_path = output / "plots" / f"trajectory_{i:04d}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=130); plt.close(fig)
        if (i+1) % 250 == 0:
            print(f"Processed {i+1}/{len(files)}", flush=True)
    if plot_pool is not None:
        for job in pending_plots:
            job.result()
        plot_pool.shutdown(wait=True)
    if not variants["raw"]:
        raise ValueError("No readable trajectories")
    comparison = {**provenance, "num_trajectories_evaluated": len(variants["raw"]),
                  "variants": dict(previous["variants"]) if previous else {}}
    for name, records in variants.items():
        alpha = None if name == "raw" else float(name[4:])
        result = payload(records, {**provenance, "ema_alpha": alpha})
        if not (previous and name == "raw"):
            result["all_episode_plots"] = bool(args.all_plots and name != "raw")
            write_json(output / name / "metrics.json", result)
        if name != "raw":
            context_groups = {}
            for record in records:
                context_path = Path(record["source_curve"]).parent.parent
                context_groups.setdefault(context_path, []).append(record)
            for context_path, context_records in context_groups.items():
                first = context_records[0]
                write_json(output/name/context_path/"metrics.json", {
                    "task_id": first["task_id"], "task_name": first["task_name"],
                    "context_demo_id": first["context_demo_id"], "ema_alpha": alpha,
                    **group(context_records), "trajectories": context_records})
        comparison["variants"][name] = result["aggregate"]
    if summary:
        for k, value in summary["aggregate"].items():
            if k in comparison["variants"]["raw"] and abs(comparison["variants"]["raw"][k]-value) > 2e-6:
                raise ValueError(f"Source aggregate mismatch: {k}")
    if not previous:
        write_json(output / "source_manifest.json", {**provenance, "files": snapshots, "skipped": skipped})
    write_json(output / "comparison.json", comparison)
    rows = ["# Causal EMA rescoring", "", f"Source: `{source}`", "",
            f"Trajectories: {len(variants['raw'])}/{expected or 'unknown'}; partial: {summary is None}.", "",
            "Pearson is averaged per trajectory, matching the original eval. Parameters are exploratory.", "",
            "| Variant | Pearson | Spearman | MAE | MSE | Mean absolute frame change |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, m in comparison["variants"].items():
        rows.append(f"| {name} | " + " | ".join(f"{m[k]:.6f}" for k in
                    ("pearson", "spearman", "mae", "mse", "mean_abs_frame_delta")) + " |")
    (output / "comparison.md").write_text("\n".join(rows)+"\n")
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
