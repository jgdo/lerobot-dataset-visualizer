"""Per-episode target-frame annotations, stored next to (not inside) a dataset.

``<root>/annotations/<org>/<dataset>/targets.json``::

    {
      "version": 1,
      "types": ["lin", "circ", "coll"],
      "tags": ["above_handle", "handle_grasped"],
      "episodes": {
        "0": {"length": 512, "targets": [{"frame": 180, "type": "lin", "tag": "above_handle"}]}
      }
    }

``types`` is fixed, ``tags`` is edited by the user. A frame and a tag appear at
most once per episode. The dataset itself is never written.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable

TYPES = ["lin", "circ", "coll"]
TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def empty() -> dict:
    return {"version": 1, "types": list(TYPES), "tags": [], "episodes": {}}


class TargetAnnotations:
    def __init__(self, data_root: Callable[[], Path], dataset_dir: Callable[[str], Path]):
        self._data_root = data_root
        self._dataset_dir = dataset_dir
        self._lock = threading.Lock()

    # -- files ---------------------------------------------------------------

    def path(self, repo_id: str) -> Path:
        self._dataset_dir(repo_id)  # only known datasets
        base = (self._data_root() / "annotations").resolve()
        path = (base / repo_id / "targets.json").resolve()
        if not str(path).startswith(str(base) + os.sep):
            raise FileNotFoundError(f"Dataset {repo_id} not found")
        return path

    def load(self, repo_id: str) -> dict:
        path = self.path(repo_id)
        if not path.is_file():
            return empty()
        data = json.loads(path.read_text())
        data["types"] = list(TYPES)
        data.setdefault("tags", [])
        data.setdefault("episodes", {})
        return data

    def _save(self, repo_id: str, data: dict) -> None:
        path = self.path(repo_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop episodes without targets so the file only lists annotated ones.
        data["episodes"] = {
            k: v for k, v in sorted(data["episodes"].items(), key=lambda kv: int(kv[0])) if v["targets"]
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".targets.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)

    def episode_lengths(self, repo_id: str) -> dict[int, int]:
        import pyarrow.parquet as pq

        lengths: dict[int, int] = {}
        for file in sorted((self._dataset_dir(repo_id) / "meta" / "episodes").glob("*/*.parquet")):
            table = pq.read_table(file, columns=["episode_index", "length"]).to_pydict()
            lengths.update(zip(table["episode_index"], table["length"]))
        return lengths

    # -- API -----------------------------------------------------------------

    def snapshot(self, repo_id: str) -> dict:
        data = self.load(repo_id)
        lengths = self.episode_lengths(repo_id)
        data["mismatched_episodes"] = [
            int(k) for k, v in data["episodes"].items() if lengths.get(int(k)) != v.get("length")
        ]
        data["num_episodes"] = len(lengths)
        return data

    def set_episode(self, repo_id: str, episode: int, targets: list) -> dict:
        length = self.episode_lengths(repo_id).get(episode)
        if length is None:
            raise ValueError(f"episode {episode} not in dataset")
        with self._lock:
            data = self.load(repo_id)
            clean = []
            for t in targets:
                frame, kind, tag = int(t["frame"]), str(t["type"]), str(t["tag"])
                if not 0 <= frame < length:
                    raise ValueError(f"frame {frame} outside episode (0..{length - 1})")
                if kind not in TYPES:
                    raise ValueError(f"unknown type {kind!r}, must be one of {TYPES}")
                if tag not in data["tags"]:
                    raise ValueError(f"unknown tag {tag!r}")
                clean.append({"frame": frame, "type": kind, "tag": tag})
            frames = [t["frame"] for t in clean]
            tags = [t["tag"] for t in clean]
            if len(set(frames)) != len(frames):
                raise ValueError("two targets on the same frame")
            if len(set(tags)) != len(tags):
                raise ValueError("a tag may be used only once per episode")
            clean.sort(key=lambda t: t["frame"])
            data["episodes"][str(episode)] = {"length": length, "targets": clean}
            self._save(repo_id, data)
        return self.snapshot(repo_id)

    def tag_usage(self, repo_id: str, tag: str) -> dict:
        data = self.load(repo_id)
        episodes = [
            int(k) for k, v in data["episodes"].items() if any(t["tag"] == tag for t in v["targets"])
        ]
        return {"tag": tag, "targets": len(episodes), "episodes": len(episodes)}

    def add_tag(self, repo_id: str, tag: str) -> dict:
        if not TAG_RE.match(tag):
            raise ValueError("tags may only contain letters, digits, '_' and '-'")
        with self._lock:
            data = self.load(repo_id)
            if tag in data["tags"]:
                raise ValueError(f"tag {tag!r} already exists")
            data["tags"].append(tag)
            self._save(repo_id, data)
        return self.snapshot(repo_id)

    def remove_tag(self, repo_id: str, tag: str) -> dict:
        """Remove ``tag`` and every target that uses it, in one write."""
        with self._lock:
            data = self.load(repo_id)
            if tag not in data["tags"]:
                raise ValueError(f"unknown tag {tag!r}")
            data["tags"].remove(tag)
            removed = 0
            for ep in data["episodes"].values():
                keep = [t for t in ep["targets"] if t["tag"] != tag]
                removed += len(ep["targets"]) - len(keep)
                ep["targets"] = keep
            self._save(repo_id, data)
        return {**self.snapshot(repo_id), "removed_targets": removed}
