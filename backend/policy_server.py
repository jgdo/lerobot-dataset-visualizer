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

Evaluation downloads nothing (``HF_HUB_OFFLINE=1``) and writes nothing. The
training queue (``/api/policy/train/*``, see ``training.py``) writes new runs
to ``train/`` and may download base models. Trained policies are uploaded to
the Hub only on request (``/api/policy/train/uploads``, see ``hf_upload.py``);
datasets are never uploaded. Target annotations
(``/api/policy/annotations/*``, see ``target_annotations.py``) are written to
``annotations/<org>/<dataset>/targets.json``, never into the dataset.

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

from hf_upload import UploadManager
from target_annotations import TargetAnnotations
from training import POLICY_TYPES, TrainingManager

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


def list_runs() -> list[dict]:
    """All training runs under train/, with the dataset each was trained on."""
    out = []
    root = train_root()
    for run in sorted(root.iterdir() if root.is_dir() else [], key=lambda p: p.name):
        if run.name.startswith(".") or not run.is_dir():
            continue
        ckpts = run / "checkpoints"
        last = ckpts / "last"
        names = sorted(p.name for p in ckpts.iterdir() if p.name != "last") if ckpts.is_dir() else []
        cfg_dir = last / "pretrained_model" if last.exists() else None
        if cfg_dir is None and names:
            cfg_dir = ckpts / names[-1] / "pretrained_model"
        train_cfg = {}
        if cfg_dir is not None:
            try:
                train_cfg = json.loads((cfg_dir / "train_config.json").read_text())
            except (OSError, ValueError):
                pass
        out.append(
            {
                "name": run.name,
                "type": (train_cfg.get("policy") or {}).get("type"),
                "dataset": (train_cfg.get("dataset") or {}).get("repo_id"),
                "steps": train_cfg.get("steps"),
                "checkpoint": last.resolve().name if last.exists() else None,
                "checkpoints": names,
                "has_policy": is_policy_dir(checkpoint_dir(run.name)),
                "modified": run.stat().st_mtime,
            }
        )
    return out


def list_datasets() -> list[dict]:
    """Local datasets (datasets/<org>/<name>) with their cameras."""
    out = []
    root = dataset_root()
    for info_file in sorted(root.glob("*/*/meta/info.json")):
        try:
            info = json.loads(info_file.read_text())
        except (OSError, ValueError):
            continue
        ds = info_file.parent.parent
        out.append(
            {
                "repo_id": f"{ds.parent.name}/{ds.name}",
                "cameras": [
                    k
                    for k, f in info.get("features", {}).items()
                    if f.get("dtype") in ("video", "image")
                ],
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "fps": info.get("fps"),
            }
        )
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
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()  # plugin policies such as target_reach
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


def release_policy() -> None:
    """Frees the GPU memory of the evaluation policy unless it is in use."""
    global _loaded
    if not _policy_lock.acquire(blocking=False):
        return
    try:
        if _loaded:
            import torch

            _loaded = None
            torch.cuda.empty_cache()
    finally:
        _policy_lock.release()


targets = TargetAnnotations(data_root, lambda repo_id: dataset_dir(repo_id))
training = TrainingManager(
    train_root,
    lambda repo_id: dataset_dir(repo_id),
    before_start=release_policy,
    annotations=targets,
    after_done=lambda job: uploads.start(job["name"], **job["upload"], job_id=job["id"]),
)
uploads = UploadManager(training.state_dir, policy_dir)


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
    out["training_active"] = training.active()
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

    # What the prediction is compared with: the recorded action of the same
    # frame, or for target_reach the pose at the episode's tagged target frame.
    reference, target = recorded, None
    if cfg.type == "target_reach":
        target = target_reference(repo_id, episode, cfg, cols, n)
        reference = target["pose"]
    compared = [i for i in range(n) if reference[i] is not None]

    per_joint = []
    for d in range(action_dim if compared else 0):
        errs = [predicted[i][d] - reference[i][d] for i in compared]
        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        max_abs = max(abs(e) for e in errs)
        rec = [recorded[i][d] for i in compared]
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
        "target": target,
    }


def target_reference(repo_id: str, episode: int, cfg, cols, n: int) -> dict:
    """Per-frame target pose of a target_reach policy for one episode.

    Uses the dataset's current annotations for the tag the policy was trained
    on: frames 0..t get the pose (``cfg.target_key``) at the tagged frame t,
    later frames None, like in training. No tag in the episode: all None.
    """
    tag, key = cfg.target_tag, cfg.target_key
    data = targets.load(repo_id)
    frame = next(
        (
            t["frame"]
            for t in data["episodes"].get(str(episode), {}).get("targets", [])
            if t["tag"] == tag
        ),
        None,
    )
    pose: list = [None] * n
    if frame is not None:
        if frame >= n:
            raise ValueError(f"target frame {frame} is outside the episode ({n} frames)")
        value = [float(x) for x in cols[key][frame]]
        pose[: frame + 1] = [value] * (frame + 1)
    return {"tag": tag, "key": key, "frame": frame, "pose": pose}


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
            if url.path == "/api/policy/train/options":
                types = [{"type": k, **v} for k, v in POLICY_TYPES.items()]
                return self._json(200, {"types": types, "datasets": list_datasets()})
            if url.path == "/api/policy/train/runs":
                return self._json(200, list_runs())
            if url.path == "/api/policy/train/jobs":
                return self._json(200, training.snapshot())
            if url.path == "/api/policy/train/hf":
                return self._json(200, uploads.account())
            if url.path == "/api/policy/train/uploads":
                return self._json(200, uploads.snapshot())
            if url.path == "/api/policy/train/name":
                return self._json(200, {"problem": training.name_problem(q.get("name", ""))})
            if url.path == "/api/policy/annotations/targets":
                return self._json(200, targets.snapshot(q.get("repo_id", "")))
            if url.path == "/api/policy/annotations/tags/usage":
                return self._json(200, targets.tag_usage(q.get("repo_id", ""), q.get("tag", "")))
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

    def do_PUT(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path != "/api/policy/annotations/targets":
            return self._json(404, {"error": "not found"})
        try:
            body = self._body()
            episode = int(q.get("episode", ""))
            return self._json(
                200, targets.set_episode(q.get("repo_id", ""), episode, list(body["targets"]))
            )
        except FileNotFoundError as e:
            return self._json(404, {"error": str(e)})
        except (ValueError, KeyError, TypeError) as e:
            return self._json(400, {"error": str(e)})

    def do_POST(self):
        url = urlparse(self.path)
        parts = url.path.rstrip("/").split("/")
        if url.path == "/api/policy/annotations/tags":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                body = self._body()
                repo_id = q.get("repo_id", "")
                if "add" in body:
                    return self._json(200, targets.add_tag(repo_id, str(body["add"]).strip()))
                if "remove" in body:
                    return self._json(200, targets.remove_tag(repo_id, str(body["remove"])))
                return self._json(400, {"error": "expected add or remove"})
            except FileNotFoundError as e:
                return self._json(404, {"error": str(e)})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if url.path == "/api/policy/train/uploads":
            try:
                body = self._body()
                up = uploads.start(
                    str(body.get("run", "")), str(body.get("repo_id") or ""), bool(body.get("private", True))
                )
                return self._json(200, up)
            except FileNotFoundError as e:
                return self._json(404, {"error": str(e)})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if parts[3:5] == ["train", "jobs"]:
            try:
                body = self._body()
                if len(parts) == 5:
                    return self._json(200, training.enqueue(body))
                if len(parts) == 7 and parts[6] == "stop":
                    training.stop(parts[5], str(body.get("mode", "")))
                    return self._json(200, {"ok": True})
            except KeyError:
                return self._json(404, {"error": "unknown training job"})
            except (ValueError, FileNotFoundError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(404, {"error": "not found"})
        if url.path != "/api/policy/jobs":
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
        if len(parts) == 6 and parts[3:5] == ["train", "jobs"]:
            try:
                training.cancel(parts[5])
            except KeyError:
                return self._json(404, {"error": "unknown training job"})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"ok": True})
        if len(parts) == 6 and parts[3:5] == ["train", "uploads"]:
            try:
                uploads.dismiss(parts[5])
            except KeyError:
                return self._json(404, {"error": "unknown upload"})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"ok": True})
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
    training.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"policy server on http://{args.host}:{args.port} (data: {data_root()})")
    server.serve_forever()


if __name__ == "__main__":
    main()
