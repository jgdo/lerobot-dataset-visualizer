"""Training queue for the policy service (see ``policy_server.py``).

Runs ``lerobot-train`` jobs one after another, writing each run to
``$LOCAL_LEROBOT_DATA_ROOT/train/<name>`` like the existing training runs.
The queue lives in ``train/.visualizer/queue.json`` and every training runs as
its own process group with its output in ``train/.visualizer/<id>.log``, so
both survive a restart of the visualizer: on startup a still running training
is picked up again and the queue continues.

Progress, loss and the remaining time are parsed from the log (tqdm progress
bar and lerobot's ``step:… loss:…`` lines). Unlike evaluation, training may
download base models from the Hub (SmolVLA, pi0-fast). lerobot-train itself
uploads nothing (``policy.push_to_hub=false``, wandb off); a job queued with
``upload`` hands its finished run to ``after_done`` (``hf_upload.py``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from hf_upload import check_repo_id

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
# Only lerobot's own bar; downloads and dataset loading print tqdm bars too.
PROGRESS_RE = re.compile(r"Training:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<")
LOSS_RE = re.compile(r"\bstep:\S+ .*?\bloss:([-+\d.eE]+|nan|inf)")
# Logged by the target_reach plugin's sampler (lerobot_policy_target_reach).
SAMPLER_RE = re.compile(r"target_reach: (tag .* usable frames .*)$")
ACTIVE = ("queued", "running", "stopping")

# Policy types offered for training. Types with a ``base`` fine-tune that Hub
# model; ``cameras`` are the image inputs it expects, which dataset cameras are
# renamed to (missing ones become ``empty_cameras``).
POLICY_TYPES = {
    "act": {"label": "ACT", "base": None, "cameras": None},
    "smolvla": {
        "label": "SmolVLA",
        "base": "lerobot/smolvla_base",
        "cameras": [f"observation.images.camera{i}" for i in (1, 2, 3)],
    },
    "pi0_fast": {
        "label": "pi0-fast",
        "base": "lerobot/pi0fast-base",
        "cameras": [
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ],
    },
    # Plugin policy (lerobot_policy_target_reach): predicts the pose at the
    # frame annotated with ``target_tag`` (see target_annotations.py).
    "target_reach": {"label": "Target reach", "base": None, "cameras": None},
}

# Parameters the UI may set, as lerobot-train flags. Values are validated and
# formatted here; anything else in a request is rejected.
INT_PARAMS = {
    "steps": "--steps",
    "batch_size": "--batch_size",
    "save_freq": "--save_freq",
    "log_freq": "--log_freq",
    "num_workers": "--num_workers",
    "seed": "--seed",
    "chunk_size": "--policy.chunk_size",
    "n_action_steps": "--policy.n_action_steps",
}
FLOAT_PARAMS = {"optimizer_lr": "--policy.optimizer_lr"}
BOOL_PARAMS = {
    "freeze_vision_encoder": "--policy.freeze_vision_encoder",
    "train_expert_only": "--policy.train_expert_only",
    "gradient_checkpointing": "--policy.gradient_checkpointing",
    "use_vae": "--policy.use_vae",
}
CHOICE_PARAMS = {
    "dtype": ("--policy.dtype", ("bfloat16", "float32")),
    "target_key": ("--policy.target_key", ("observation.state", "action")),
}
# Image augmentation, split into two groups of LeRobot's default transforms
# (same types and ranges as ImageTransformsConfig). Each training frame gets up
# to max_num_transforms (3) of the enabled ones.
COLOR_TRANSFORMS = {
    "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
    "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
    "saturation": {"type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
    "hue": {"type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
    "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
}
POSITION_TRANSFORMS = {
    "affine": {"type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
}
AUGMENT_PARAMS = {"color_augmentation": COLOR_TRANSFORMS, "position_augmentation": POSITION_TRANSFORMS}


def augmentation_args(params: dict, policy_type: str) -> list[str]:
    # Jobs queued before the split carry the single "image_transforms" flag (= both groups).
    legacy = params.get("image_transforms", False)
    if policy_type == "target_reach":
        # Augmented batched on the GPU after resizing to 240x320 (the plugin's augment.py), so
        # the data loader workers only decode. Same transforms and ranges as below.
        return [
            "--dataset.image_transforms.enable=false",
            *(f"--policy.{key}={'true' if params.get(key, legacy) else 'false'}" for key in AUGMENT_PARAMS),
        ]
    tfs: dict = {}
    for key, group in AUGMENT_PARAMS.items():
        if params.get(key, legacy):
            tfs.update({name: {"weight": 1.0, **tf} for name, tf in group.items()})
    if not tfs:
        return ["--dataset.image_transforms.enable=false"]
    return ["--dataset.image_transforms.enable=true", f"--dataset.image_transforms.tfs={json.dumps(tfs)}"]


# target_reach predicts a single pose, so it has no action chunk.
CHUNK_PARAMS = {"chunk_size", "n_action_steps"}


def build_args(job: dict, dataset_dir: Path, output_dir: Path) -> list[str]:
    spec = POLICY_TYPES[job["policy_type"]]
    args = [
        f"--dataset.repo_id={job['repo_id']}",
        f"--dataset.root={dataset_dir}",
        f"--output_dir={output_dir}",
        f"--job_name={job['name']}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]
    if spec["base"]:
        args.append(f"--policy.path={spec['base']}")
    else:
        args.append(f"--policy.type={job['policy_type']}")
    params = job["params"]
    for key, flag in INT_PARAMS.items():
        if key in params:
            args.append(f"{flag}={params[key]}")
    for key, flag in FLOAT_PARAMS.items():
        if key in params:
            args.append(f"{flag}={params[key]}")
    for key, flag in BOOL_PARAMS.items():
        if key in params:
            args.append(f"{flag}={'true' if params[key] else 'false'}")
    for key, (flag, _) in CHOICE_PARAMS.items():
        if key in params:
            args.append(f"{flag}={params[key]}")
    args += augmentation_args(params, job["policy_type"])
    if job.get("target_tag"):
        args.append(f"--policy.annotation_path={job['annotation_path']}")
        args.append(f"--policy.target_tag={job['target_tag']}")
    if job.get("rename_map"):
        args.append(f"--rename_map={json.dumps(job['rename_map'])}")
    if spec["cameras"] is not None:
        args.append(f"--policy.empty_cameras={job.get('empty_cameras', 0)}")
    return args


def clean_params(policy_type: str, raw: dict) -> dict:
    allowed = {"act": {"use_vae"}, "smolvla": {"freeze_vision_encoder", "train_expert_only"}}
    allowed["pi0_fast"] = {"gradient_checkpointing", "dtype"}
    allowed["target_reach"] = {"target_key"}
    common = set(INT_PARAMS) | set(FLOAT_PARAMS) | set(AUGMENT_PARAMS)
    if policy_type == "target_reach":
        common -= CHUNK_PARAMS
    out: dict = {}
    for key, value in raw.items():
        if value is None or value == "":
            continue
        if key not in common and key not in allowed[policy_type]:
            raise ValueError(f"unknown parameter {key} for {policy_type}")
        if key in INT_PARAMS:
            v = int(value)
            if v < (0 if key in ("num_workers", "seed") else 1):
                raise ValueError(f"{key} must be positive")
            out[key] = v
        elif key in FLOAT_PARAMS:
            v = float(value)
            if not v > 0:
                raise ValueError(f"{key} must be positive")
            out[key] = v
        elif key in BOOL_PARAMS or key in AUGMENT_PARAMS:
            out[key] = bool(value)
        else:
            if value not in CHOICE_PARAMS[key][1]:
                raise ValueError(f"{key} must be one of {CHOICE_PARAMS[key][1]}")
            out[key] = value
    if "steps" not in out:
        raise ValueError("steps is required")
    if out.get("n_action_steps", 0) > out.get("chunk_size", 10**9):
        raise ValueError("actions per step must not exceed the chunk size")
    return out


def default_name(policy_type: str, repo_id: str, now: datetime, tag: str | None = None) -> str:
    tag_part = f"_{tag}" if tag else ""
    return f"{policy_type}_{repo_id.replace('/', '_')}{tag_part}_{now:%d%m%Y_%H%M%S}"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (exited, not yet reaped by us) counts as finished.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class LogFollower:
    """Incrementally parses a training log: tqdm counter, loss curve, last lines.

    Logs of long runs grow large (tqdm rewrites its bar many times a second),
    so each call only reads what was appended since the previous one.
    """

    def __init__(self, path: Path, log_freq: int):
        self.path = path
        self.log_freq = log_freq
        self.offset = 0
        self.partial = ""
        self.step = 0
        self.total: int | None = None
        self.elapsed_s: int | None = None
        self.losses: list[list[float]] = []
        self.tail: list[str] = []
        self.note: str | None = None

    def update(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            chunk = b""
        self.offset += len(chunk)
        lines = re.split(r"[\r\n]", self.partial + chunk.decode(errors="replace"))
        self.partial = lines.pop()
        for ln in lines:
            if not ln.strip():
                continue
            for m in PROGRESS_RE.finditer(ln):
                self.step, self.total = int(m.group(1)), int(m.group(2))
                parts = [int(p) for p in m.group(3).split(":")]
                self.elapsed_s = sum(p * 60**i for i, p in enumerate(reversed(parts)))
            m = SAMPLER_RE.search(ln)
            if m:
                self.note = m.group(1)
            m = LOSS_RE.search(ln)
            if m:
                # lerobot logs exactly every log_freq steps; its own step
                # counter is abbreviated ("1K"), so count the lines instead.
                self.losses.append([(len(self.losses) + 1) * self.log_freq, float(m.group(1))])
            if not ln.startswith("Training:"):
                self.tail = (self.tail + [ln])[-15:]
        return {
            "step": self.step,
            "total": self.total,
            "elapsed_s": self.elapsed_s,
            "losses": self.losses,
            "tail": self.tail,
            "note": self.note,
        }


class TrainingManager:
    def __init__(
        self,
        train_root: Callable[[], Path],
        dataset_dir: Callable[[str], Path],
        before_start: Callable[[], None] = lambda: None,
        annotations=None,
        after_done: Callable[[dict], None] = lambda job: None,
    ):
        self._train_root = train_root
        self._dataset_dir = dataset_dir
        self._before_start = before_start
        self._annotations = annotations  # TargetAnnotations, for target_reach
        self._after_done = after_done  # e.g. starts the Hub upload of job["upload"]
        self._lock = threading.RLock()
        self._jobs: list[dict] = []
        self._procs: dict[str, subprocess.Popen] = {}
        self._progress: dict[str, dict] = {}
        self._followers: dict[str, LogFollower] = {}
        self._loaded = False

    # -- persistence -------------------------------------------------------

    def state_dir(self) -> Path:
        d = self._train_root() / ".visualizer"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        f = self.state_dir() / "queue.json"
        if f.exists():
            self._jobs = json.loads(f.read_text()).get("jobs", [])

    def _save(self) -> None:
        f = self.state_dir() / "queue.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": self._jobs}, indent=1))
        tmp.replace(f)

    def _job(self, job_id: str) -> dict:
        for job in self._jobs:
            if job["id"] == job_id:
                return job
        raise KeyError(job_id)

    def _run_dir(self, name: str) -> Path:
        root = self._train_root().resolve()
        d = (root / name).resolve()
        if d.parent != root:
            raise ValueError(f"invalid run name {name!r}")
        return d

    # -- public API --------------------------------------------------------

    def name_problem(self, name: str) -> str | None:
        if not NAME_RE.match(name):
            return "Use letters, digits, '_', '-' and '.' only (max 128 characters)"
        with self._lock:
            self._load()
            if (self._train_root() / name).exists():
                return f"A model named {name} already exists"
            if any(j["name"] == name and j["status"] in ACTIVE for j in self._jobs):
                return f"A queued training is already named {name}"
        return None

    def enqueue(self, body: dict) -> dict:
        policy_type = str(body.get("policy_type", ""))
        if policy_type not in POLICY_TYPES:
            raise ValueError(f"unsupported policy type {policy_type!r}")
        repo_id = str(body.get("repo_id", ""))
        self._dataset_dir(repo_id)  # raises if unknown
        params = clean_params(policy_type, dict(body.get("params") or {}))
        rename_map = {str(k): str(v) for k, v in (body.get("rename_map") or {}).items() if v}
        empty_cameras = int(body.get("empty_cameras") or 0)
        target = self._target(repo_id, body) if policy_type == "target_reach" else {}
        upload = body.get("upload")
        if upload:
            upload = {
                "repo_id": check_repo_id(str(upload.get("repo_id") or "")),
                "private": bool(upload.get("private", True)),
            }
        with self._lock:
            self._load()
            name = str(body.get("name") or "").strip()
            if not name:
                name = default_name(policy_type, repo_id, datetime.now(), target.get("target_tag"))
            problem = self.name_problem(name)
            if problem:
                raise ValueError(problem)
            job = {
                "id": uuid.uuid4().hex[:12],
                "name": name,
                "policy_type": policy_type,
                "repo_id": repo_id,
                "params": params,
                "rename_map": rename_map,
                "empty_cameras": empty_cameras,
                **target,
                "upload": upload or None,
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "pid": None,
                "error": None,
                "stop_mode": None,
            }
            self._jobs.append(job)
            self._save()
        self.tick()
        return job

    def _target(self, repo_id: str, body: dict) -> dict:
        """Checks the selected tag of a target_reach training against the annotations."""
        tag = str(body.get("target_tag") or "")
        if self._annotations is None:
            raise ValueError("target annotations are not available")
        data = self._annotations.snapshot(repo_id)
        if tag not in data["tags"]:
            raise ValueError(f"select a target tag (unknown tag {tag!r})")
        if not self._annotations.tag_usage(repo_id, tag)["episodes"]:
            raise ValueError(f"no episode has a target tagged {tag!r}")
        if data["mismatched_episodes"]:
            eps = ", ".join(map(str, data["mismatched_episodes"][:5]))
            raise ValueError(f"annotations do not match the dataset (episode length changed: {eps})")
        return {"target_tag": tag, "annotation_path": str(self._annotations.path(repo_id))}

    def cancel(self, job_id: str) -> None:
        """Removes a queued job, or a finished one from the history."""
        with self._lock:
            self._load()
            job = self._job(job_id)
            if job["status"] in ("running", "stopping"):
                raise ValueError("training is running; stop it instead")
            self._jobs.remove(job)
            self._progress.pop(job_id, None)
            self._followers.pop(job_id, None)
            self._save()

    def stop(self, job_id: str, mode: str) -> None:
        if mode not in ("keep", "delete"):
            raise ValueError("mode must be keep or delete")
        with self._lock:
            self._load()
            job = self._job(job_id)
            if job["status"] not in ("running", "stopping"):
                raise ValueError("training is not running")
            job["status"] = "stopping"
            job["stop_mode"] = mode
            job["stop_requested_at"] = time.time()
            self._save()
            self._signal(job, signal.SIGINT)

    def snapshot(self) -> dict:
        with self._lock:
            self._load()
            jobs = []
            for job in self._jobs:
                out = dict(job)
                if job["status"] != "queued":
                    out["progress"] = self._progress.get(job["id"]) or self._read_progress(job)
                jobs.append(out)
            return {"jobs": jobs, "active": self.active()}

    def active(self) -> bool:
        return any(j["status"] in ("running", "stopping") for j in self._jobs)

    # -- scheduler ---------------------------------------------------------

    def start(self) -> None:
        def loop() -> None:
            while True:
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001
                    print(f"training scheduler: {e}", file=sys.stderr)
                time.sleep(1.0)

        threading.Thread(target=loop, daemon=True).start()

    def tick(self) -> None:
        with self._lock:
            self._load()
            changed = False
            for job in self._jobs:
                if job["status"] in ("running", "stopping"):
                    changed |= self._check_running(job)
            if not self.active():
                nxt = next((j for j in self._jobs if j["status"] == "queued"), None)
                if nxt:
                    self._launch(nxt)
                    changed = True
            if changed:
                self._save()

    def _signal(self, job: dict, sig: int) -> None:
        try:
            os.killpg(job["pid"], sig)
        except (ProcessLookupError, PermissionError, TypeError):
            pass

    def _log_path(self, job: dict) -> Path:
        return self.state_dir() / f"{job['id']}.log"

    def _read_progress(self, job: dict) -> dict:
        follower = self._followers.get(job["id"])
        if follower is None:
            follower = LogFollower(self._log_path(job), job["params"].get("log_freq", 200))
            self._followers[job["id"]] = follower
        prog = follower.update()
        total = prog["total"] or job["params"]["steps"]
        prog["total"] = total
        remaining = None
        if prog["step"] and prog["elapsed_s"]:
            remaining = (total - prog["step"]) * prog["elapsed_s"] / prog["step"]
        prog["remaining_s"] = remaining
        if job["status"] in ("running", "stopping"):
            prog["phase"] = "training" if prog["step"] else "preparing"
        self._progress[job["id"]] = prog
        return prog

    def _launch(self, job: dict) -> None:
        try:
            self._before_start()
        except Exception:  # noqa: BLE001
            pass
        output_dir = self._run_dir(job["name"])
        if output_dir.exists():
            job.update(status="failed", error=f"{output_dir} already exists", finished_at=time.time())
            return
        args = build_args(job, self._dataset_dir(job["repo_id"]), output_dir)
        exit_file = self.state_dir() / f"{job['id']}.exit"
        # The shell records the exit code so it can be read after a restart.
        cmd = [
            "bash",
            "-c",
            'exec 0</dev/null; "$@"; echo $? > "$EXIT_FILE"',
            "lerobot-train",
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            *args,
        ]
        env = os.environ.copy()
        env.pop("HF_HUB_OFFLINE", None)  # base models may need downloading
        env["EXIT_FILE"] = str(exit_file)
        env["VISUALIZER_TRAINING_JOB"] = job["id"]
        log = open(self._log_path(job), "wb")
        log.write(("$ lerobot-train " + " ".join(args) + "\n").encode())
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.state_dir()),
            start_new_session=True,
        )
        log.close()
        self._procs[job["id"]] = proc
        job.update(status="running", pid=proc.pid, started_at=time.time())

    def _check_running(self, job: dict) -> bool:
        """Updates progress; finalizes the job once its process is gone."""
        self._read_progress(job)
        proc = self._procs.get(job["id"])
        alive = proc.poll() is None if proc else pid_alive(job["pid"]) and self._is_ours(job)
        if alive:
            if job["status"] == "stopping":
                waited = time.time() - job.get("stop_requested_at", time.time())
                if waited > 60:
                    self._signal(job, signal.SIGKILL)
                elif waited > 20:
                    self._signal(job, signal.SIGTERM)
            return False
        self._procs.pop(job["id"], None)
        job["finished_at"] = time.time()
        exit_file = self.state_dir() / f"{job['id']}.exit"
        code = None
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                pass
        if job["status"] == "stopping":
            job["status"] = "stopped"
            if job["stop_mode"] == "delete":
                run_dir = self._run_dir(job["name"])
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                job["deleted"] = True
        elif code == 0:
            job["status"] = "done"
            if job.get("upload"):
                try:
                    self._after_done(job)
                except Exception as e:  # noqa: BLE001
                    job["upload_error"] = str(e)[:500]
        else:
            job["status"] = "failed"
            tail = self._progress[job["id"]].get("tail") or []
            errors = [ln.strip() for ln in tail if "Error" in ln or "error" in ln]
            if code is None:
                msg = "training process ended unexpectedly"
            elif code > 128:
                msg = f"training process was killed (signal {code - 128})"
            else:
                msg = f"lerobot-train exited with code {code}"
            job["error"] = (msg + (f": {errors[-1]}" if errors else ""))[:500]
        return True

    def _is_ours(self, job: dict) -> bool:
        try:
            environ = Path(f"/proc/{job['pid']}/environ").read_bytes()
        except OSError:
            return False
        return f"VISUALIZER_TRAINING_JOB={job['id']}".encode() in environ
