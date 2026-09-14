"""
Multi-source progress-curve visualizer for ICVFE Evals.

A read-only clone of the video-annotation tool's "value vs time" chart
(../../Manual Annotation/static/{index.html,app.js,style.css}), adapted to
show all four evaluator curves for an episode at once -- Human, RoboDopamine,
RoboMeter, ICVFE -- as toggleable overlaid lines, instead of one editable
curve. No video: this project's data export doesn't carry video, only the
progress-value JSON/npz curves already used by analysis/pipeline.py.

Uses only the standard library (no FastAPI/uvicorn) so it needs no extra
install beyond numpy, which the rest of this repo already depends on.

Usage:
    python server.py            # http://localhost:8000
    python server.py --port 8001
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).parent
ANALYSIS_DIR = HERE.parent / "analysis"
STATIC_DIR = HERE / "static"

sys.path.insert(0, str(ANALYSIS_DIR))
import pipeline as pl  # noqa: E402  (analysis/pipeline.py -- extraction + cache paths)

SOURCE_LABELS = {
    "human": "Human",
    "robodopamine": "RoboDopamine",
    "robometer": "RoboMeter",
    "icvfe": "ICVFE",
}


# --------------------------------------------------------------------------
# Data loading (reuses analysis/pipeline.py's extraction + cache)
# --------------------------------------------------------------------------

def load_source(cache_subdir: str) -> dict:
    """Returns {episode_uid: {task, episode_index, success, num_frames, fps, progress}}."""
    root = pl.CACHE_DIR / cache_subdir if isinstance(pl.CACHE_DIR, Path) else Path(pl.CACHE_DIR) / cache_subdir
    json_dir = Path(pl._find_json_dir(str(root)))
    out = {}
    for path in json_dir.glob("*.json"):
        d = json.loads(path.read_text())
        progress = np.array([p["progress"] for p in d["points"]], dtype=np.float64)
        out[d["episode_uid"]] = {
            "task": d["task"],
            "episode_index": d["episode_index"],
            "success": d.get("success"),
            "num_frames": d["num_frames"],
            "fps": d["fps"],
            "progress": progress,
        }
    return out


def load_icvfe_averages(common_uids: set) -> dict:
    """episode_uid -> {"progress": np.ndarray (0-100), "n_reps": int}.

    Averages ICVFE's prediction curve across its in-context replicates (each
    query episode is scored once per context example drawn from the same
    task) -- the same aggregation analysis/pipeline.py uses for its
    episode-level icvfe_* numbers.
    """
    ckpt_root = Path(pl._icvfe_root())
    meta = json.loads((ckpt_root / "metrics.json").read_text())

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for t in meta["trajectories"]:
        uid = t["episode_uid"]
        if uid not in common_uids:
            continue
        task_dir = f"task_{t['task_id']:03d}+{t['task_name'].replace(' ', '_')}"
        npz_path = ckpt_root / task_dir / f"context_{t['context_demo_id']}" / "curves" / f"query_{t['demo_id']}.npz"
        with np.load(npz_path) as d:
            pred = np.asarray(d["prediction_progress"], dtype=np.float64) * 100.0
        if uid in sums and len(sums[uid]) != len(pred):
            n = min(len(sums[uid]), len(pred))
            sums[uid] = sums[uid][:n]
            pred = pred[:n]
        sums[uid] = sums.get(uid, np.zeros_like(pred)) + pred
        counts[uid] = counts.get(uid, 0) + 1

    return {uid: {"progress": sums[uid] / counts[uid], "n_reps": counts[uid]} for uid in sums}


print("Loading curve data (extracting archives on first run)...", file=sys.stderr)
pl.extract_archives(force=False)
MANUAL = load_source("manual")
ROBOMETER = load_source("finetuned_robometer")
ROBODOP = load_source("zs_robodopamine")

COMMON_UIDS = set(MANUAL) & set(ROBOMETER) & set(ROBODOP)
ICVFE_AVG = load_icvfe_averages(COMMON_UIDS)
print(f"Loaded {len(COMMON_UIDS)} episodes ({len(ICVFE_AVG)} with an ICVFE curve)", file=sys.stderr)

TASK_NAMES = sorted({MANUAL[u]["task"] for u in COMMON_UIDS})
TASK_INDEX_BY_NAME = {name: i for i, name in enumerate(TASK_NAMES)}

EPISODES_BY_TASK: dict[int, list[str]] = {i: [] for i in range(len(TASK_NAMES))}
for uid in COMMON_UIDS:
    t_idx = TASK_INDEX_BY_NAME[MANUAL[uid]["task"]]
    EPISODES_BY_TASK[t_idx].append(uid)
for t_idx in EPISODES_BY_TASK:
    EPISODES_BY_TASK[t_idx].sort(key=lambda u: MANUAL[u]["episode_index"])


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------

def build_task_list() -> list:
    return [
        {"task_index": i, "task": name, "n_episodes": len(EPISODES_BY_TASK[i])}
        for i, name in enumerate(TASK_NAMES)
    ]


def build_episode_list(task_index: int) -> list:
    uids = EPISODES_BY_TASK[task_index]
    return [
        {
            "position": i + 1,
            "episode_uid": uid,
            "num_frames": MANUAL[uid]["num_frames"],
            "success": MANUAL[uid]["success"],
            "has_icvfe": uid in ICVFE_AVG,
        }
        for i, uid in enumerate(uids)
    ]


def build_episode_payload(task_index: int, position: int) -> dict:
    uids = EPISODES_BY_TASK[task_index]
    uid = uids[position - 1]
    m = MANUAL[uid]

    def series(entry_progress, n_reps=None):
        payload = {"available": True, "progress": [round(v, 3) for v in entry_progress.tolist()]}
        if n_reps is not None:
            payload["n_reps"] = n_reps
        return payload

    icvfe = ICVFE_AVG.get(uid)

    return {
        "task_index": task_index,
        "task": TASK_NAMES[task_index],
        "episode_uid": uid,
        "position": position,
        "total_in_task": len(uids),
        "num_frames": m["num_frames"],
        "fps": m["fps"],
        "dataset_success": m["success"],
        "series": {
            "human": series(m["progress"]),
            "robodopamine": series(ROBODOP[uid]["progress"]),
            "robometer": series(ROBOMETER[uid]["progress"]),
            "icvfe": series(icvfe["progress"], icvfe["n_reps"]) if icvfe else {"available": False},
        },
    }


# --------------------------------------------------------------------------
# HTTP server (stdlib only)
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json({"detail": message}, status=status)

    def _send_static(self, rel_path: str):
        if rel_path == "" or rel_path == "/":
            rel_path = "index.html"
        file_path = (STATIC_DIR / rel_path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
            self._send_error_json(403, "forbidden")
            return
        if not file_path.is_file():
            self._send_error_json(404, "not found")
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/tasks":
                self._send_json(build_task_list())
                return

            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "tasks" and parts[3] == "episodes":
                t_idx = int(parts[2])
                if t_idx not in EPISODES_BY_TASK:
                    self._send_error_json(404, "task not found")
                    return
                self._send_json(build_episode_list(t_idx))
                return

            if len(parts) == 4 and parts[0] == "api" and parts[1] == "episode":
                t_idx, pos = int(parts[2]), int(parts[3])
                uids = EPISODES_BY_TASK.get(t_idx)
                if uids is None or not (1 <= pos <= len(uids)):
                    self._send_error_json(404, "episode not found")
                    return
                self._send_json(build_episode_payload(t_idx, pos))
                return

            if path.startswith("/api/"):
                self._send_error_json(404, "not found")
                return

            self._send_static(path)
        except Exception as e:  # keep the server alive on a bad request
            self._send_error_json(500, str(e))


def main():
    parser = argparse.ArgumentParser(description="ICVFE Evals multi-source visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
