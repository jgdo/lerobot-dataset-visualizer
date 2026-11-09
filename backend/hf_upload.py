"""Uploads trained policies to the Hugging Face Hub (see ``policy_server.py``).

Only on request: the training page's "Upload" button for an existing run, or a
training queued with "Upload to Hugging Face when done". The upload holds what
``lerobot-train --policy.push_to_hub`` would push: the files of
``train/<run>/checkpoints/last/pretrained_model`` (weights, config, train
config, processors) plus a LeRobot model card. Nothing local is changed.

Each upload runs this file as a subprocess (the server itself is offline,
``HF_HUB_OFFLINE=1``). The login comes from ``hf auth login`` or ``HF_TOKEN``.
Uploads are listed in ``train/.visualizer/uploads.json``, with their output in
``train/.visualizer/upload-<id>.log``. The subprocess prints
``PROGRESS {phase, done, total, rate}`` lines (bytes, from the Xet upload
callback), kept in memory as the upload's ``progress``.

``VISUALIZER_HF_DRY_RUN=1`` (or ``--dry-run``) does everything except talking to
the Hub, for testing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$")
ALLOW = (".safetensors", ".json", ".yaml", ".md")
DRY_RUN = os.environ.get("VISUALIZER_HF_DRY_RUN") == "1"
PROGRESS = "PROGRESS "


def check_repo_id(repo_id: str) -> str:
    """'' (= <user>/<run name>), 'name' (= <user>/name) or 'org/name'."""
    repo_id = repo_id.strip()
    if repo_id and (not REPO_ID_RE.match(repo_id) or ".." in repo_id or repo_id.endswith(".git")):
        raise ValueError(f"invalid Hugging Face repo id {repo_id!r} (expected user/name)")
    return repo_id


def _hub_env() -> dict:
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    return env


class UploadManager:
    def __init__(self, state_dir: Callable[[], Path], model_dir: Callable[[str], Path]):
        self._state_dir = state_dir
        self._model_dir = model_dir  # raises for unknown runs
        self._lock = threading.RLock()
        self._uploads: list[dict] = []
        self._loaded = False
        self._account: tuple[float, dict] | None = None

    def _file(self) -> Path:
        return self._state_dir() / "uploads.json"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._file().exists():
            self._uploads = json.loads(self._file().read_text()).get("uploads", [])
        for up in self._uploads:
            if up["status"] == "running":  # the server stopped during the upload
                up.update(status="failed", error="interrupted (server restarted)")
        self._save()

    def _save(self) -> None:
        tmp = self._file().with_suffix(".tmp")
        saved = [{k: v for k, v in u.items() if k != "progress"} for u in self._uploads]
        tmp.write_text(json.dumps({"uploads": saved}, indent=1))
        tmp.replace(self._file())

    def snapshot(self) -> list[dict]:
        with self._lock:
            self._load()
            return [dict(u) for u in self._uploads]  # includes the in-memory progress

    def account(self) -> dict:
        """``{user}`` of the Hub login, or ``{user: None, error}``; cached briefly."""
        now = time.time()
        if self._account and now - self._account[0] < (300 if self._account[1].get("user") else 20):
            return self._account[1]
        if DRY_RUN:
            info = {"user": "dry-run-user", "dry_run": True}
        else:
            code = "import json,huggingface_hub as h;print(json.dumps(h.whoami()['name']))"
            try:
                res = subprocess.run(
                    [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=_hub_env()
                )
                if res.returncode == 0:
                    info = {"user": json.loads(res.stdout.strip().splitlines()[-1])}
                else:
                    last = (res.stderr.strip().splitlines() or ["unknown error"])[-1]
                    info = {"user": None, "error": f"not logged in to Hugging Face ({last[:200]})"}
            except (OSError, subprocess.TimeoutExpired) as e:
                info = {"user": None, "error": f"could not reach Hugging Face: {e}"}
        self._account = (now, info)
        return info

    def start(self, run: str, repo_id: str = "", private: bool = True, job_id: str | None = None) -> dict:
        repo_id = check_repo_id(repo_id)
        model_dir = self._model_dir(run)
        with self._lock:
            self._load()
            if any(u["run"] == run and u["status"] == "running" for u in self._uploads):
                raise ValueError(f"{run} is already being uploaded")
            up = {
                "id": uuid.uuid4().hex[:12],
                "run": run,
                "repo_id": repo_id or None,
                "private": bool(private),
                "job_id": job_id,
                "status": "running",
                "progress": {"phase": "preparing"},
                "url": None,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
            self._uploads.append(up)
            self._save()
        threading.Thread(target=self._run, args=(up, model_dir), daemon=True).start()
        return dict(up)

    def dismiss(self, upload_id: str) -> None:
        with self._lock:
            self._load()
            up = next((u for u in self._uploads if u["id"] == upload_id), None)
            if up is None:
                raise KeyError(upload_id)
            if up["status"] == "running":
                raise ValueError("upload is running")
            self._uploads.remove(up)
            self._save()

    def _run(self, up: dict, model_dir: Path) -> None:
        log_path = self._state_dir() / f"upload-{up['id']}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), str(model_dir), up["run"], up["repo_id"] or ""]
        cmd += ["--private"] if up["private"] else []
        cmd += ["--dry-run"] if DRY_RUN else []
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=_hub_env()
            )
            for line in proc.stdout:
                if line.startswith(PROGRESS):
                    try:
                        progress = json.loads(line[len(PROGRESS) :])
                    except ValueError:
                        continue
                    with self._lock:
                        up["progress"] = progress
                else:
                    log.write(line)
                    log.flush()
            proc.wait()

        lines = log_path.read_text(errors="replace").strip().splitlines()
        result = {}
        if lines and lines[-1].startswith("{"):
            try:
                result = json.loads(lines[-1])
            except ValueError:
                pass
        with self._lock:
            up.pop("progress", None)
            up["finished_at"] = time.time()
            if proc.returncode == 0 and result.get("url"):
                up.update(status="done", url=result["url"], repo_id=result["repo_id"])
            else:
                errors = [ln for ln in lines if "Error" in ln or "error" in ln] or lines[-1:]
                up.update(status="failed", error=(errors[-1] if errors else f"exit code {proc.returncode}")[:500])
            self._save()


# -- subprocess: the actual upload ------------------------------------------


def _progress(phase: str, done: int | None = None, total: int | None = None, rate: float | None = None) -> None:
    print(PROGRESS + json.dumps({"phase": phase, "done": done, "total": total, "rate": rate}), flush=True)


def _report_xet_progress() -> None:
    """Mirror the Xet upload callback (bytes processed of all files) as PROGRESS lines."""
    from huggingface_hub.utils import _xet_progress_reporting as xpr

    original = xpr.XetUploadProgressReporter.update_progress
    last = [0.0]

    def update_progress(self, group_report, item_reports):
        original(self, group_report, item_reports)
        now = time.monotonic()
        if now - last[0] >= 0.5:
            last[0] = now
            _progress(
                "uploading",
                getattr(group_report, "total_bytes_completed", None),
                getattr(group_report, "total_bytes", None),
                getattr(group_report, "total_bytes_completion_rate", None),
            )

    xpr.XetUploadProgressReporter.update_progress = update_progress


def _model_card(model_dir: Path, repo_id: str):
    from types import SimpleNamespace

    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.pretrained import PreTrainedPolicy

    config = PreTrainedConfig.from_pretrained(model_dir)
    config.repo_id = repo_id
    try:
        cfg = TrainPipelineConfig.from_pretrained(model_dir)
        dataset = cfg.dataset.repo_id
    except Exception as e:  # noqa: BLE001
        print(f"train config not readable for the model card: {e}")
        cfg, dataset = None, json.loads((model_dir / "train_config.json").read_text())["dataset"]["repo_id"]
    return PreTrainedPolicy.generate_model_card(
        SimpleNamespace(config=config), dataset, config.type, config.license, config.tags, cfg=cfg
    )


def upload(model_dir: Path, run: str, repo_id: str, private: bool, dry_run: bool) -> dict:
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()  # plugin configs such as target_reach
    if dry_run:
        from huggingface_hub import ModelCard

        ModelCard.validate = lambda self, *a, **k: None  # validation calls the Hub
    if "/" not in repo_id:
        if dry_run:
            user = "dry-run-user"
        else:
            from huggingface_hub import whoami

            user = whoami()["name"]
        repo_id = f"{user}/{repo_id or run}"
    check_repo_id(repo_id)

    files = sorted(p for p in model_dir.iterdir() if p.is_file() and p.name.endswith(ALLOW) and p.name != "README.md")
    card = str(_model_card(model_dir, repo_id))
    print(f"uploading {len(files)} files + README.md from {model_dir} to {repo_id} (private={private})")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    url = f"https://huggingface.co/{repo_id}"
    total = sum(f.stat().st_size for f in files)
    if dry_run:  # pretend to upload for ~5 s so the progress display can be tested
        for i in range(11):
            _progress("uploading", total * i // 10, total, total / 5)
            time.sleep(0.5)
        _progress("committing")
        print("dry run: nothing uploaded")
        return {"repo_id": repo_id, "url": url, "files": [f.name for f in files], "card_chars": len(card)}

    from huggingface_hub import CommitOperationAdd, HfApi

    try:
        _report_xet_progress()
    except (ImportError, AttributeError) as e:  # without Xet: phases only, no byte progress
        print(f"no upload progress: {e}")
    api = HfApi()
    repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
    ops = [CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    _progress("uploading", 0, total)
    info = api.create_commit(repo_id=repo_id, operations=ops, commit_message=f"Upload policy {run}")
    print(f"uploaded: {info.commit_url}")
    return {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a trained policy to the Hugging Face Hub")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("run")
    ap.add_argument("repo_id", nargs="?", default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = upload(args.model_dir, args.run, args.repo_id, args.private, args.dry_run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
