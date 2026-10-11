"""Reuse of finished 3D builds for an identical 3D input: memoisation of our own pipeline output, labelled as such.

The DA3 point cloud takes seconds, the high-quality Gaussian splats ~15-36 min of GPU. When a new inspection has
exactly the same 3D input as a finished build, the stored artifacts are attached to the new job instead of
rebuilding them, and the job state says so (``reused``, ``reused_from_job``, ``reused_from_created_utc``,
``original_seconds``, ``original_device``). Nothing is invented: the numbers shown are the measured numbers of the
original build.

Cache key (a readable string, one per kind):
- the inspection's ``result.frames_sha256`` = SHA-256 over the blurred frames the pipeline saved, i.e. exactly the
  frames both 3D stages read (the raw video is never hashed or kept);
- the frame-sampling config that produced them (``result.extra.config``: fps, max_frames, work_side);
- the model identity: DA3 model + pinned revision + requested process resolution for ``recon3d``; Brush training
  steps for ``recon3d_hq``.
A different input or config gives a different key: no reuse. A job without frames_sha256 or config gets no key.

Index: ``DATA_DIR/cache/recon3d_index.json`` = {"format": 1, "entries": {key: entry}}; entry = {job_id, kind,
created_utc (the source inspection), finished_utc, seconds (measured build time), device, summary, registered_utc}.
Writes take a process lock + an fcntl file lock and replace the file atomically (the server and the admin script
may write at the same time). When several real builds share a key, the most recently finished one wins (the
pipeline code improved over time; reused copies are never indexed).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("avtopassport.api.reuse3d")

INDEX_NAME = "recon3d_index.json"
FORMAT = 1
KINDS = ("recon3d", "recon3d_hq")
# recon3d.hq.available() only lets the HQ stage run after it found a Vulkan GPU driver + device node, so a finished
# HQ build was made on one (the HQ report does not name the card).
HQ_DEVICE = "Vulkan GPU"
# files copied for a reuse (the served file is always required; see required_files)
DA3_FILES = ("points_web.ply", "points.ply", "cameras.json", "depth.npz", "result.json")
HQ_FILES = ("splat_web.ply", "splat_car.ply", "splat.ply", "hq_report.json")
REUSE_DIRS = {"recon3d": "recon3d_reused", "recon3d_hq": "recon3d_hq_reused"}

_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------- key

def da3_identity(device: str) -> dict[str, Any]:
    """Model identity of a DA3 build the server would run now on ``device`` (defaults of recon3d.build)."""
    from ..recon3d.build import DEFAULT_MODEL  # noqa: PLC0415 - stdlib-only module
    from ..recon3d.worker import DEFAULT_PROCESS_RES, MODEL_REVISIONS  # noqa: PLC0415 - numpy + stdlib only

    return {"model": DEFAULT_MODEL, "revision": MODEL_REVISIONS.get(DEFAULT_MODEL),
            "process_res": DEFAULT_PROCESS_RES["cuda" if str(device).startswith("cuda") else "cpu"]}


def input_fields(state: dict[str, Any]) -> dict[str, Any] | None:
    """{frames_sha256, fps, max_frames, work_side} of a finished inspection, or None when anything is missing."""
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    sha = result.get("frames_sha256")
    cfg = (result.get("extra") or {}).get("config") if isinstance(result.get("extra"), dict) else None
    if not (isinstance(sha, str) and len(sha) == 64 and isinstance(cfg, dict)):
        return None
    try:
        fps, max_frames, work_side = float(cfg["fps"]), int(cfg["max_frames"]), int(cfg["work_side"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"frames_sha256": sha, "fps": fps, "max_frames": max_frames, "work_side": work_side}


def make_key(kind: str, inp: dict[str, Any], model: dict[str, Any]) -> str:
    base = (f"frames={inp['frames_sha256']}|fps={inp['fps']:g}|max_frames={inp['max_frames']}"
            f"|work_side={inp['work_side']}")
    if kind == "recon3d":
        return f"recon3d|{base}|da3={model['model']}@{model['revision']}|res={model['process_res']}"
    if kind == "recon3d_hq":
        return f"recon3d_hq|{base}|steps={int(model['steps'])}"
    raise ValueError(kind)


def key_for_new(kind: str, state: dict[str, Any], *, device: str, hq_steps: int) -> str | None:
    """Key of the build this server WOULD run now for the job (None = cannot be keyed: always build)."""
    inp = input_fields(state)
    if inp is None:
        return None
    try:
        model = da3_identity(device) if kind == "recon3d" else {"steps": hq_steps}
    except Exception:  # noqa: BLE001 - identity unknown: never reuse
        return None
    if kind == "recon3d" and not model.get("revision"):
        return None
    return make_key(kind, inp, model)


def key_for_finished(kind: str, state: dict[str, Any]) -> str | None:
    """Key of a finished, real (not reused) build recorded in a job state, from what that build recorded."""
    rec = state.get(kind) if isinstance(state.get(kind), dict) else {}
    if state.get("status") != "done" or rec.get("status") != "done" or rec.get("reused"):
        return None
    inp = input_fields(state)
    if inp is None:
        return None
    if kind == "recon3d":
        info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
        model = {"model": info.get("model"), "revision": info.get("revision"),
                 "process_res": info.get("process_res_requested") or info.get("process_res")}
        if not all(model.values()):
            return None
    else:
        report = rec.get("report") if isinstance(rec.get("report"), dict) else {}
        steps = rec.get("steps") or report.get("steps")
        if not isinstance(steps, int):
            return None
        model = {"steps": steps}
    return make_key(kind, inp, model)


def entry_for(kind: str, state: dict[str, Any]) -> dict[str, Any]:
    """Index entry of a finished build: measured seconds + a short summary, all copied from the job state."""
    rec = state.get(kind) or {}
    if kind == "recon3d":
        info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
        device = info.get("device")
        summary = {k: info.get(k) for k in ("model", "revision", "n_frames", "process_res", "n_points",
                                            "n_points_web")}
    else:
        rep = rec.get("report") if isinstance(rec.get("report"), dict) else {}
        device = rec.get("device") or HQ_DEVICE
        summary = {"steps": rep.get("steps") or rec.get("steps"), "n_images": rep.get("n_images"),
                   "n_registered": rep.get("n_registered"), "psnr": rep.get("psnr"), "psnr_car": rep.get("psnr_car"),
                   "seconds": rep.get("seconds"), "cropped": rec.get("cropped")}
    return {"job_id": state.get("id"), "kind": kind, "created_utc": state.get("created_utc"),
            "finished_utc": rec.get("finished_utc"), "seconds": rec.get("seconds"), "device": device,
            "summary": summary, "registered_utc": utc_now()}


# --------------------------------------------------------------------------------------------- index

class Index:
    """``DATA_DIR/cache/recon3d_index.json`` with locked, atomic read-modify-write."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "cache" / INDEX_NAME

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(self.path.with_suffix(".lock"), "a+") as fh:
            try:
                import fcntl  # noqa: PLC0415 - POSIX only

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            yield

    def read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        entries = data.get("entries") if isinstance(data, dict) else None
        return {k: v for k, v in entries.items() if isinstance(v, dict)} if isinstance(entries, dict) else {}

    def _write(self, entries: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"format": FORMAT, "entries": entries}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.read().get(key)

    def register(self, items: list[tuple[str, dict[str, Any]]]) -> int:
        """Add entries; an existing entry is replaced unless it finished later. Returns # entries changed."""
        if not items:
            return 0
        with self._locked():
            entries = self.read()
            changed = 0
            for key, entry in items:
                old = entries.get(key)
                if old is not None and old.get("job_id") == entry.get("job_id"):
                    continue
                if old is not None and str(old.get("finished_utc") or "") > str(entry.get("finished_utc") or ""):
                    continue  # keep a build that finished later (equal times: the newest registration wins)
                entries[key] = entry
                changed += 1
            if changed:
                self._write(entries)
            return changed

    def drop(self, key: str, job_id: str) -> None:
        with self._locked():
            entries = self.read()
            if (entries.get(key) or {}).get("job_id") == job_id:
                del entries[key]
                self._write(entries)


# --------------------------------------------------------------------------------------------- files

def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink (same filesystem, instant, no extra disk) else copy. Returns "link" or "copy"."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def attach_files(src_dir: Path, dst_dir: Path, names: tuple[str, ...], required: set[str]) -> dict[str, str]:
    """Copy ``names`` present in src_dir into dst_dir (built in a temp dir, then renamed into place).

    Raises FileNotFoundError when a required file is missing (nothing is left behind)."""
    missing = sorted(n for n in required if not (src_dir / n).is_file())
    if missing:
        raise FileNotFoundError(f"source files missing in {src_dir}: {', '.join(missing)}")
    tmp = dst_dir.with_name(f".{dst_dir.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    how: dict[str, str] = {}
    try:
        for n in names:
            if (src_dir / n).is_file():
                how[n] = link_or_copy(src_dir / n, tmp / n)
        shutil.rmtree(dst_dir, ignore_errors=True)  # only ever our own reuse dir (links / copies)
        tmp.rename(dst_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return how
