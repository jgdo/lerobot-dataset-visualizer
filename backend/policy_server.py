"""LeRobot dataset visualizer — policy evaluation service.

Runs a trained LeRobot policy (the ``last`` checkpoint of a training run) over
every frame of a locally stored episode and returns the predicted actions next
to the recorded ones, so the visualizer can chart the deviation.

The policy is queried every ``actions_per_chunk`` frames and the first
``actions_per_chunk`` actions of each predicted chunk are used, i.e. with 1 the
policy runs on every frame and only its first action is compared.

Everything is read from ``$LOCAL_LEROBOT_DATA_ROOT``:

- datasets from ``datasets/<org>/<dataset>``
- policies from ``train/<run>/checkpoints/last/pretrained_model``

Nothing is downloaded (``HF_HUB_OFFLINE=1``) or written.

Only the Python standard library is used for HTTP, so this runs in any Python
environment that has ``lerobot`` and ``torch`` installed:

    LOCAL_LEROBOT_DATA_ROOT=/path/to/root python backend/policy_server.py --port 7862

The Next.js app forwards ``/api/policy/*`` here when ``POLICY_BACKEND_URL`` is
set (see ``next.config.ts`` and the ``dev:local`` script).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")

BATCH_SIZE = int(os.environ.get("POLICY_BATCH_SIZE", "16"))


def data_root() -> Path:
    root = os.environ.get("LOCAL_LEROBOT_DATA_ROOT")
    if not root:
        raise RuntimeError("LOCAL_LEROBOT_DATA_ROOT is not set")
    return Path(root).resolve()


def dataset_root() -> Path:
    return data_root() / "datasets"


def train_root() -> Path:
    return data_root() / "train"


def is_policy_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        path / "model.safetensors"
    ).is_file()


def checkpoint_dir(run: str) -> Path:
    """The `last` checkpoint of a training run under train/."""
    return train_root() / run / "checkpoints" / "last" / "pretrained_model"


def list_policies() -> list[dict]:
    """Training runs under train/ whose `last` checkpoint holds a policy."""
    out = []
    root = train_root()
    for run in sorted(root.iterdir() if root.is_dir() else [], key=lambda p: p.name):
        ckpt = checkpoint_dir(run.name)
        if run.name.startswith(".") or not is_policy_dir(ckpt):
            continue
        out.append({"name": run.name, "checkpoint": ckpt.parent.resolve().name})
    return out


def policy_dir(run: str) -> Path:
    """Checkpoint folder of a listed training run; rejects anything else."""
    if run not in {p["name"] for p in list_policies()}:
        raise FileNotFoundError(f"No trained policy {run!r} under {train_root()}")
    return checkpoint_dir(run)


# --------------------------------------------------------------------------
# Policy loading (one policy kept on the GPU at a time)
# --------------------------------------------------------------------------

_policy_lock = threading.Lock()
_loaded: dict | None = None


def load_policy(policy_path: str) -> dict:
    """Returns {path, policy, cfg, pre, post}, reusing the loaded one if same path."""
    global _loaded
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = str(Path(policy_path).resolve())
    if _loaded and _loaded["path"] == path:
        return _loaded
    if _loaded:
        _loaded = None
        torch.cuda.empty_cache()

    cfg = PreTrainedConfig.from_pretrained(path)
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(cfg.device)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    _loaded = {"path": path, "policy": policy, "cfg": cfg, "pre": pre, "post": post}
    return _loaded


def saved_rename_map(policy_path: Path) -> dict[str, str]:
    """The observation renames the policy was trained with, if recorded."""
    try:
        train_cfg = json.loads((policy_path / "train_config.json").read_text())
        if train_cfg.get("rename_map"):
            return dict(train_cfg["rename_map"])
    except (OSError, ValueError):
        pass
    try:
        pre_cfg = json.loads((policy_path / "policy_preprocessor.json").read_text())
        for step in pre_cfg.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                return dict(step.get("config", {}).get("rename_map") or {})
    except (OSError, ValueError):
        pass
    return {}


def dataset_dir(repo_id: str) -> Path:
    root = dataset_root()
    ds_dir = (root / repo_id).resolve()
    if not str(ds_dir).startswith(str(root) + os.sep) or not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset {repo_id} not found under {root}")
    return ds_dir


def dataset_observation_keys(repo_id: str) -> list[str]:
    info = json.loads((dataset_dir(repo_id) / "meta" / "info.json").read_text())
    return [k for k in info.get("features", {}) if k.startswith("observation.")]


def policy_summary(policy_path: str, repo_id: str | None = None) -> dict:
    cfg = json.loads((Path(policy_path) / "config.json").read_text())
    out = {
        "type": cfg.get("type"),
        "chunk_size": cfg.get("chunk_size") or cfg.get("n_action_steps") or 1,
        "input_features": [
            {"key": key, "type": feat.get("type")}
            for key, feat in (cfg.get("input_features") or {}).items()
        ],
        "default_remap": saved_rename_map(Path(policy_path)),
    }
    if repo_id:
        out["dataset_observation_keys"] = dataset_observation_keys(repo_id)
    return out


def resolve_inputs(
    dataset_keys: list[str], input_features: dict, remap: dict[str, str]
) -> dict[str, str]:
    """Maps dataset observation keys to the policy inputs they feed.

    `remap` maps a dataset key to a policy input key, or to "" to leave it
    out; unmapped keys feed the policy input of the same name, if any.
    """
    feed: dict[str, str] = {}
    for key in dataset_keys:
        target = remap.get(key, key)
        if not target:
            continue
        if target not in input_features:
            if key in remap:
                raise ValueError(
                    f"{key} is remapped to {target}, which is not a policy input"
                )
            continue
        if target in feed.values():
            raise ValueError(f"More than one dataset stream is mapped to {target}")
        feed[key] = target

    # Missing cameras are fine (policies like SmolVLA pad them), as long as
    # at least one is fed; every other input is required.
    fed = set(feed.values())
    visual = [k for k, f in input_features.items() if f.type.value == "VISUAL"]
    missing_required = [k for k in input_features if k not in fed and k not in visual]
    if missing_required or (visual and not any(k in fed for k in visual)):
        need = missing_required or visual
        raise ValueError(
            "The dataset provides no stream for the policy input(s) "
            + ", ".join(need)
            + ". Add a remapping, e.g. observation.images.<dataset camera> -> "
            + need[0]
        )
    return feed


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.status = "queued"  # queued | loading | running | done | error | cancelled
        self.message = ""
        self.progress = 0
        self.total = 0
        self.result: dict | None = None
        self.cancel_event = threading.Event()

    def to_json(self, with_result: bool = True) -> dict:
        out = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "total": self.total,
        }
        if with_result and self.result is not None:
            out["result"] = self.result
        return out


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def run_job(job: Job) -> None:
    try:
        with _policy_lock:  # one GPU job at a time; queued jobs wait here
            if job.cancel_event.is_set():
                raise Cancelled()
            _run_job_locked(job)
        job.status = "done"
    except Cancelled:
        job.status = "cancelled"
        job.message = "Cancelled"
    except Exception as e:  # noqa: BLE001 — reported to the UI
        traceback.print_exc()
        job.status = "error"
        job.message = f"{type(e).__name__}: {e}"


def _run_job_locked(job: Job) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    p = job.params
    repo_id: str = p["repo_id"]
    episode = int(p["episode"])
    k = int(p["actions_per_chunk"])

    job.status = "loading"
    job.message = "Loading policy"
    loaded = load_policy(str(policy_dir(p["policy"])))
    policy, cfg, pre, post = (
        loaded["policy"],
        loaded["cfg"],
        loaded["pre"],
        loaded["post"],
    )
    chunk_size = int(getattr(cfg, "chunk_size", None) or cfg.n_action_steps or 1)
    if k < 1 or k > chunk_size:
        raise ValueError(f"actions_per_chunk must be between 1 and {chunk_size}")

    job.message = "Loading episode"
    ds = LeRobotDataset(repo_id, root=dataset_dir(repo_id), episodes=[episode])
    n = len(ds)
    names = ds.meta.features["action"].get("names") or []
    # Dataset stream -> policy input. The batch is built with the policy's
    # key names, so a rename step saved in the preprocessor has nothing left
    # to rename and the remapping chosen here is the one that applies.
    feed = resolve_inputs(
        [k for k in ds.meta.features if k.startswith("observation.")],
        cfg.input_features,
        p.get("remap") or {},
    )

    if job.cancel_event.is_set():
        raise Cancelled()

    query_frames = list(range(0, n, k))
    job.total = n
    job.status = "running"
    job.message = "Running policy"

    timestamps = [0.0] * n
    recorded = [None] * n
    predicted = [None] * n

    # Recorded actions and timestamps for every frame, read from the parquet
    # rows directly (ds[i] would also decode the video frames).
    cols = ds.hf_dataset.with_format("numpy")[:]
    if len(cols["episode_index"]) != n or (cols["episode_index"] != episode).any():
        raise RuntimeError("unexpected rows in the episode's parquet selection")
    actions_col = cols["action"]
    ts_col = cols["timestamp"]
    for i in range(n):
        recorded[i] = [float(x) for x in actions_col[i]]
        timestamps[i] = float(ts_col[i])

    done = 0
    for b in range(0, len(query_frames), BATCH_SIZE):
        if job.cancel_event.is_set():
            raise Cancelled()
        frames = query_frames[b : b + BATCH_SIZE]
        items = [ds[i] for i in frames]
        batch = {
            target: torch.stack([it[src] for it in items])
            for src, target in feed.items()
        }
        batch["task"] = [it.get("task", "") for it in items]
        with torch.inference_mode():
            obs = pre(batch)
            chunk = policy.predict_action_chunk(obs)  # (B, chunk_size, A)
            chunk = post(chunk)
        chunk = chunk.detach().float().cpu()
        for row, start in enumerate(frames):
            for j in range(min(k, n - start)):
                predicted[start + j] = [float(x) for x in chunk[row, j]]
        done = min(n, frames[-1] + k)
        job.progress = done

    action_dim = len(recorded[0]) if n else 0
    if not names or len(names) != action_dim:
        names = [str(i) for i in range(action_dim)]

    per_joint = []
    for d in range(action_dim):
        errs = [predicted[i][d] - recorded[i][d] for i in range(n)]
        mae = sum(abs(e) for e in errs) / n
        rmse = math.sqrt(sum(e * e for e in errs) / n)
        max_abs = max(abs(e) for e in errs)
        rec = [recorded[i][d] for i in range(n)]
        per_joint.append(
            {
                "name": names[d],
                "mae": mae,
                "rmse": rmse,
                "max_abs": max_abs,
                "recorded_range": max(rec) - min(rec),
            }
        )

    job.result = {
        "repo_id": repo_id,
        "episode": episode,
        "policy": p["policy"],
        "checkpoint": Path(loaded["path"]).parent.name,
        "policy_type": cfg.type,
        "chunk_size": chunk_size,
        "actions_per_chunk": k,
        "num_policy_calls": len(query_frames),
        "inputs": feed,
        "names": names,
        "timestamps": timestamps,
        "recorded": recorded,
        "predicted": predicted,
        "per_joint": per_joint,
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "LeRobotPolicyServer/0.1"

    def log_message(self, fmt, *args):  # quieter: skip job polling
        if "/jobs/" in self.path and self.command == "GET":
            return
        super().log_message(fmt, *args)

    def _json(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        parts = url.path.rstrip("/").split("/")
        try:
            if url.path == "/api/policy/health":
                return self._json(200, {"ok": True})
            if url.path == "/api/policy/policies":
                return self._json(200, list_policies())
            if url.path == "/api/policy/info":
                try:
                    path = policy_dir(q.get("policy", ""))
                except FileNotFoundError as e:
                    return self._json(400, {"error": str(e)})
                return self._json(200, policy_summary(str(path), q.get("repo_id")))
            if len(parts) == 5 and parts[3] == "jobs":
                job = _jobs.get(parts[4])
                if not job:
                    return self._json(404, {"error": "unknown job"})
                return self._json(200, job.to_json())
            return self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": str(e)})

    def do_POST(self):
        if urlparse(self.path).path != "/api/policy/jobs":
            return self._json(404, {"error": "not found"})
        try:
            body = self._body()
            policy = str(body.get("policy", ""))
            try:
                policy_dir(policy)
            except FileNotFoundError as e:
                return self._json(400, {"error": str(e)})
            params = {
                "policy": policy,
                "repo_id": str(body["repo_id"]),
                "episode": int(body["episode"]),
                "actions_per_chunk": int(body.get("actions_per_chunk", 1)),
                "remap": {
                    str(k): str(v) for k, v in (body.get("remap") or {}).items()
                },
            }
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"bad request: {e}"})
        job = Job(params)
        with _jobs_lock:
            # Drop finished jobs so results don't pile up in memory.
            for jid in [
                j.id for j in _jobs.values() if j.status in ("done", "error", "cancelled")
            ]:
                del _jobs[jid]
            _jobs[job.id] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        return self._json(200, job.to_json(with_result=False))

    def do_DELETE(self):
        parts = urlparse(self.path).path.rstrip("/").split("/")
        if len(parts) == 5 and parts[3] == "jobs" and parts[4] in _jobs:
            _jobs[parts[4]].cancel_event.set()
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "unknown job"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()
    data_root()  # fail early if unset
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"policy server on http://{args.host}:{args.port} (data: {data_root()})")
    server.serve_forever()


if __name__ == "__main__":
    main()
