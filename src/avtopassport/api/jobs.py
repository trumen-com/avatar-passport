"""Background jobs for the API: one worker thread for inspections, a second one for 3D reconstruction.

Flow of one job (``DATA_DIR/jobs/<id>/``):
1. ``upload/`` receives the raw video (+ optional VIN photo). It is deleted as soon as the inspection ends, whatever
   the outcome (privacy: only the pipeline's blurred frames are kept).
2. The inspection worker runs ``pipeline.run.Inspector`` (created lazily, once) with ``out_dir`` = the job folder.
3. On success: ``passport.store`` (``DATA_DIR/passports.sqlite``) -> ``passport.diff`` against the previous inspection of
   the same VIN -> ``passport.report`` JSON + PDF (first spin frames as images).
4. The 3D job goes to the second worker: ``recon3d.build`` on ``<job>/frames`` then ``recon3d.pins.project_pins``.
   If the 3D environment is missing the status is ``unavailable`` (never a fake model). The viewer orientation the
   worker measured (cameras.json: center, up, forward0, walkaround_radius) is copied into ``recon3d.view``.
5. Optional, GPU only (``AVTO_HQ3D=1`` and ``recon3d.hq.available()``): after the DA3 job, a third worker runs
   ``recon3d.hq.build_hq`` (COLMAP SfM + Brush Gaussian splats) on ALL the blurred frames the inspection saved; state
   in ``recon3d_hq``. More frames = better splats: raise ``AVTO_FPS`` / ``AVTO_MAX_FRAMES`` for HQ demos.
6. Reuse (``AVTO_REUSE_3D``, default on; see ``api/reuse3d.py``): when a finished DA3 / HQ build exists for exactly
   the same 3D input (same ``frames_sha256`` of the blurred frames + frame-sampling config + model identity), its
   files are linked into the new job and the state says ``reused`` with the original's measured numbers, instead of
   queueing a rebuild. Every successful real build is registered in ``DATA_DIR/cache/recon3d_index.json``; the
   index is backfilled from the job states at startup.

Job state is written to ``<job>/state.json`` after every change, so a restarted server still serves finished jobs.
Every error is captured into the job state; nothing here may crash the server. The sandbox registry
(``avtopassport.synthetic``) is never called from a job: a scanned VIN never gets synthetic history.

The heavy modules are imported lazily through :class:`Backends`, so this module imports even when they are missing and
tests replace them with stubs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..paths import REPO_ROOT
from . import reuse3d

log = logging.getLogger("avtopassport.api.jobs")

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._ -]*$")
IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp"})
API_PREFIX = "/api/inspections"
MAX_ARRAY_ITEMS = 4096           # numpy arrays larger than this are not copied into JSON state (e.g. masks)

ProgressFn = Callable[[str, float], None]


# --------------------------------------------------------------------------------------------- helpers

def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_dir_from_env() -> Path:
    """``AVTO_DATA_DIR`` or ``REPO_ROOT/var`` (gitignored)."""
    env = os.environ.get("AVTO_DATA_DIR")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "var"


def auto_device() -> str:
    """``cuda`` when torch sees a GPU, else ``cpu`` (torch missing -> ``cpu``)."""
    try:
        import torch  # noqa: PLC0415 - heavy, only when no device is configured

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch absent or broken CUDA install
        return "cpu"


def jsonable(obj: Any) -> Any:
    """Deep-convert to plain JSON types (numpy scalars/arrays, Paths, dataclasses, tuples, non-finite floats)."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    if hasattr(obj, "dtype") and hasattr(obj, "tolist"):  # numpy array or scalar
        size = int(getattr(obj, "size", 1))
        if getattr(obj, "ndim", 0) > 0 and size > MAX_ARRAY_ITEMS:
            return {"omitted": "array", "shape": list(getattr(obj, "shape", ())), "dtype": str(obj.dtype)}
        return jsonable(obj.tolist())
    return str(obj)


def safe_relpath(name: str) -> str | None:
    """A relative path made of safe parts (no '..', no absolute paths, no hidden files), or None."""
    if not name or len(name) > 255 or "\\" in name:
        return None
    parts = name.split("/")
    if any(not _SAFE_PART_RE.match(p) or p in {".", ".."} for p in parts):
        return None
    return "/".join(parts)


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _item_name(item: Any, frames: Sequence[str]) -> str | None:
    """Frame name from an ``extra['spin']`` / ``extra['frames']`` item (name, path, dict or frame index)."""
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return frames[item] if 0 <= item < len(frames) else None
    if isinstance(item, (str, Path)):
        return Path(item).name or None
    if isinstance(item, dict):
        for key in ("name", "file", "filename", "path", "frame", "image", "source"):
            value = item.get(key)
            if isinstance(value, (str, Path)) and str(value):
                return Path(value).name
            if isinstance(value, int) and not isinstance(value, bool):
                return _item_name(value, frames)
    return None


# --------------------------------------------------------------------------------------------- backends

class InspectorLike(Protocol):
    def run(self, video_path: Path, *, out_dir: Path, vin_image_path: Path | None = None,
            vin_text: str | None = None, progress: ProgressFn | None = None) -> Any: ...


class StoreLike(Protocol):
    def add_inspection(self, result: Any, media: dict[str, Any]) -> int: ...

    def get_inspection(self, inspection_id: int) -> dict[str, Any] | None: ...

    def history(self, vin: str) -> list[dict[str, Any]]: ...

    def previous(self, vin: str, inspection_id: int) -> dict[str, Any] | None: ...


def inspection_overrides() -> dict[str, Any]:
    """InspectionConfig overrides from the environment: AVTO_FPS (frames per second kept), AVTO_MAX_FRAMES.

    The defaults (2.5 fps, 72 frames) suit the passport; the optional HQ 3D stage gets better with more frames.
    Invalid values are ignored (logged)."""
    out: dict[str, Any] = {}
    for env, key, cast in (("AVTO_FPS", "fps", float), ("AVTO_MAX_FRAMES", "max_frames", int)):
        raw = os.environ.get(env, "").strip()
        if not raw:
            continue
        try:
            value = cast(raw)
        except ValueError:
            log.warning("ignoring %s=%r (not a number)", env, raw)
            continue
        if value > 0:
            out[key] = value
    return out


def _make_inspector(device: str) -> InspectorLike:
    from ..pipeline.run import InspectionConfig, Inspector  # noqa: PLC0415 - heavy (torch / ultralytics)

    return Inspector(InspectionConfig(device=device, **inspection_overrides()))


def _open_store(path: Path) -> StoreLike:
    from ..passport.store import PassportStore  # noqa: PLC0415

    return PassportStore(path)


def _diff(previous: Any, current: Any) -> dict[str, Any]:
    from ..passport.diff import diff  # noqa: PLC0415

    return diff(previous, current)


def _build_json(result: Any, *, diff: dict[str, Any] | None, decode: Any, extra: dict[str, Any]) -> dict[str, Any]:
    from ..passport.report import build_json  # noqa: PLC0415

    return build_json(result, diff=diff, decode=decode, extra=extra)


def _build_pdf(report: dict[str, Any], out_path: Path, images: list[Path]) -> Path:
    from ..passport.report import build_pdf  # noqa: PLC0415

    return build_pdf(report, out_path, images=images)


def _recon3d_available() -> tuple[bool, str]:
    from ..recon3d.build import available  # noqa: PLC0415

    ok, msg = available()
    return bool(ok), str(msg)


def _recon3d_build(frames_dir: Path, out_dir: Path, device: str, masks_dir: Path | None) -> dict[str, Any]:
    from ..recon3d.build import build  # noqa: PLC0415

    return build(frames_dir, out_dir, device=device, masks_dir=masks_dir)


def _project_pins(out_dir: Path, pins2d: Any, frame_size: Any) -> list[dict[str, Any]]:
    from ..recon3d.pins import project_pins  # noqa: PLC0415

    return project_pins(out_dir, pins2d, frame_size)


def _hq_available() -> tuple[bool, str]:
    from ..recon3d.hq import available  # noqa: PLC0415

    ok, msg = available()
    return bool(ok), str(msg)


def _hq_build(frames_dir: Path, out_dir: Path, steps: int, pins2d: Any, masks_dir: Path | None) -> dict[str, Any]:
    from ..recon3d.hq import build_hq  # noqa: PLC0415

    return build_hq(frames_dir, out_dir, steps=steps, pins2d=pins2d, masks_dir=masks_dir)


def read_view(out_dir: Path) -> dict[str, Any] | None:
    """Viewer orientation measured by the 3D worker (cameras.json) -> {center, up, forward0, radius}, or None."""
    try:
        meta = json.loads((Path(out_dir) / "cameras.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    view: dict[str, Any] = {}
    for key, src in (("center", "center"), ("up", "up"), ("forward0", "forward0")):
        v = meta.get(src)
        if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x)
                                                             for x in v)):
            return None
        view[key] = [float(x) for x in v]
    r = meta.get("walkaround_radius")
    if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
        return None
    view["radius"] = float(r)
    return view


def _load_palette() -> Any:
    from ..tryon.recolor import load_palette  # noqa: PLC0415

    return load_palette()


def _recolor(img: Any, mask: Any, hex_colour: str) -> Any:
    from ..tryon.recolor import recolor  # noqa: PLC0415

    return recolor(img, mask, hex_colour)


@dataclass
class Backends:
    """Entry points of the modules the API depends on. Defaults import lazily; tests pass stubs."""

    make_inspector: Callable[[str], InspectorLike] = _make_inspector
    open_store: Callable[[Path], StoreLike] = _open_store
    diff: Callable[[Any, Any], dict[str, Any]] = _diff
    build_json: Callable[..., dict[str, Any]] = _build_json
    build_pdf: Callable[[dict[str, Any], Path, list[Path]], Path] = _build_pdf
    recon3d_available: Callable[[], tuple[bool, str]] = _recon3d_available
    recon3d_build: Callable[[Path, Path, str, Path | None], dict[str, Any]] = _recon3d_build
    project_pins: Callable[[Path, Any, Any], list[dict[str, Any]]] = _project_pins
    hq_available: Callable[[], tuple[bool, str]] = _hq_available
    hq_build: Callable[[Path, Path, int, Any, Path | None], dict[str, Any]] = _hq_build
    load_palette: Callable[[], Any] = _load_palette
    recolor: Callable[[Any, Any, str], Any] = _recolor


# --------------------------------------------------------------------------------------------- settings

def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class Settings:
    data_dir: Path
    device: str | None = None              # None -> auto (cuda if available, else cpu), resolved lazily
    recon3d_device: str | None = None      # None -> same as device
    max_video_bytes: int = 200 * 1024 * 1024
    max_photo_bytes: int = 25 * 1024 * 1024
    enable_recon3d: bool = True
    warmup: bool = True                    # load the models in the worker right after startup
    pdf_images: int = 4
    enable_hq3d: bool = False              # AVTO_HQ3D=1: optional GPU-only SfM + Gaussian-splat stage
    hq3d_steps: int = 30_000               # AVTO_HQ3D_STEPS: Brush training steps
    reuse_3d: bool = True                  # AVTO_REUSE_3D=0: always rebuild, never attach a stored build

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> Settings:
        return cls(
            data_dir=Path(data_dir) if data_dir is not None else data_dir_from_env(),
            device=os.environ.get("AVTO_DEVICE") or None,
            recon3d_device=os.environ.get("AVTO_RECON3D_DEVICE") or None,
            max_video_bytes=int(float(os.environ.get("AVTO_MAX_VIDEO_MB", "200")) * 1024 * 1024),
            enable_recon3d=_env_flag("AVTO_RECON3D", True),
            warmup=_env_flag("AVTO_WARMUP", True),
            enable_hq3d=_env_flag("AVTO_HQ3D", False),
            hq3d_steps=int(os.environ.get("AVTO_HQ3D_STEPS") or 30_000),
            reuse_3d=_env_flag("AVTO_REUSE_3D", True),
        )


# --------------------------------------------------------------------------------------------- manager

def _new_hq_state() -> dict[str, Any]:
    return {"status": "not_started", "splat": None, "seconds": None, "report": None, "error": None}


def _new_state(job_id: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "id": job_id,
        "status": "queued",
        "created_utc": now,
        "updated_utc": now,
        "progress": [],
        "error": None,
        "warnings": [],
        "inspection_id": None,
        "result": None,
        "report": None,
        "diff": None,
        "media": {"frames": [], "spin": [], "pdf": None, "masks": []},
        "recon3d": {"status": "not_started", "ply": None, "pins": [], "seconds": None, "error": None},
        "recon3d_hq": _new_hq_state(),
        "timings_s": {},
    }


@dataclass
class _InspectionTask:
    job_id: str
    video: Path
    vin_photo: Path | None
    vin_text: str | None


@dataclass
class JobManager:
    settings: Settings
    backends: Backends = field(default_factory=Backends)
    on_stored: Callable[[str], None] | None = None  # called with the job id after a passport is stored (snapshots)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._inspector_lock = threading.Lock()
        self._states: dict[str, dict[str, Any]] = {}
        self._last_save: dict[str, float] = {}
        self._inspect_q: queue.Queue[_InspectionTask | str | None] = queue.Queue()
        self._recon_q: queue.Queue[str | None] = queue.Queue()
        self._hq_q: queue.Queue[str | None] = queue.Queue()
        self._hq_avail: tuple[bool, str] | None = None
        self._threads: list[threading.Thread] = []
        self._started = False
        self._inspector: InspectorLike | None = None
        self._inspector_error: str | None = None
        self._device: str | None = self.settings.device
        self._recon_avail: tuple[bool, str] | None = None
        self.reuse_index = reuse3d.Index(self.settings.data_dir)

    # ----- paths
    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir

    @property
    def jobs_dir(self) -> Path:
        return self.settings.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.settings.data_dir / "passports.sqlite"

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise ValueError("bad job id")
        return self.jobs_dir / job_id

    def upload_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "upload"

    def frames_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "frames"

    def masks_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "masks"

    # ----- device / readiness
    @property
    def device(self) -> str:
        if self._device is None:
            self._device = auto_device()
        return self._device

    @property
    def recon3d_device(self) -> str:
        return self.settings.recon3d_device or self.device

    @property
    def models_ready(self) -> bool:
        return self._inspector is not None

    @property
    def models_error(self) -> str | None:
        return self._inspector_error

    def recon3d_available(self) -> tuple[bool, str]:
        """Cached ``recon3d.build.available()``; a missing module counts as unavailable."""
        if not self.settings.enable_recon3d:
            return False, "3D reconstruction disabled on this server (AVTO_RECON3D=0)"
        if self._recon_avail is None:
            try:
                self._recon_avail = self.backends.recon3d_available()
            except Exception as exc:  # module missing / broken env
                self._recon_avail = (False, f"3D module not available: {type(exc).__name__}: {exc}")
        return self._recon_avail

    def recon3d_available_nowait(self) -> tuple[bool, str]:
        """Like recon3d_available() but never blocks (the check imports torch in a subprocess at startup)."""
        if not self.settings.enable_recon3d or self._recon_avail is not None:
            return self.recon3d_available()
        return False, "checking the 3D environment"

    def hq_available(self) -> tuple[bool, str]:
        """Cached ``recon3d.hq.available()`` when AVTO_HQ3D=1; otherwise disabled."""
        if not self.settings.enable_hq3d:
            return False, "high-quality 3D disabled (set AVTO_HQ3D=1 on a GPU worker)"
        if self._hq_avail is None:
            try:
                self._hq_avail = self.backends.hq_available()
            except Exception as exc:
                self._hq_avail = (False, f"HQ 3D module not available: {type(exc).__name__}: {exc}")
        return self._hq_avail

    def queue_sizes(self) -> dict[str, int]:
        return {"inspections": self._inspect_q.unfinished_tasks, "recon3d": self._recon_q.unfinished_tasks,
                "recon3d_hq": self._hq_q.unfinished_tasks}

    # ----- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            self._recover()
            for name, q, handler in (("avto-inspect", self._inspect_q, self._handle_inspection),
                                     ("avto-recon3d", self._recon_q, self._handle_recon3d),
                                     ("avto-recon3d-hq", self._hq_q, self._handle_hq)):
                t = threading.Thread(target=self._loop, args=(q, handler), name=name, daemon=True)
                t.start()
                self._threads.append(t)
            if self.settings.warmup:
                self._inspect_q.put("__warmup__")
            if self.settings.enable_recon3d:
                threading.Thread(target=self.recon3d_available, name="avto-recon3d-check", daemon=True).start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self._inspect_q.put(None)
        self._recon_q.put(None)
        self._hq_q.put(None)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until both queues are empty and idle (tests, scripts). Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (self._inspect_q.unfinished_tasks == 0 and self._recon_q.unfinished_tasks == 0
                    and self._hq_q.unfinished_tasks == 0):
                return True
            time.sleep(0.02)
        return False

    def _loop(self, q: queue.Queue[Any], handler: Callable[[Any], None]) -> None:
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                handler(item)
            except Exception:  # a handler bug must never kill the worker
                log.exception("job worker error")
            finally:
                q.task_done()

    def _recover(self) -> None:
        """After a restart: unfinished jobs become errors, raw uploads are deleted, interrupted 3D is re-queued."""
        for state_file in sorted(self.jobs_dir.glob("*/state.json")):
            job_id = state_file.parent.name
            if not JOB_ID_RE.match(job_id):
                continue
            shutil.rmtree(state_file.parent / "upload", ignore_errors=True)
            try:
                st = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            changed = False
            if st.get("status") in {"queued", "running"}:
                st["status"] = "error"
                st["error"] = "interrupted by a server restart before the inspection finished; please upload again"
                changed = True
            rec = st.get("recon3d") or {}
            if st.get("status") == "done" and rec.get("status") in {"queued", "running"}:
                rec.update(status="queued", error=None)
                st["recon3d"] = rec
                self._recon_q.put(job_id)
                changed = True
            hq = st.get("recon3d_hq") or {}
            waiting_lost = hq.get("status") == "waiting" and rec.get("status") not in {"queued", "running"}
            if hq.get("status") in {"queued", "running"} or waiting_lost:  # ~45 min of GPU: never restarted silently
                hq.update(status="error", error="interrupted by a server restart; the high-quality 3D was not finished")
                st["recon3d_hq"] = hq
                changed = True
            with self._lock:
                self._states[job_id] = st
            if changed:
                self._save(job_id)
        self.backfill_index()

    # ----- state
    def _save(self, job_id: str) -> None:
        with self._lock:
            st = self._states.get(job_id)
            if st is None:
                return
            st["updated_utc"] = utc_now()
            text = json.dumps(st, ensure_ascii=False)
            self._last_save[job_id] = time.monotonic()
        path = self.job_dir(job_id) / "state.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"state.{threading.get_ident()}.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        except OSError:
            log.exception("could not write %s", path)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._states[job_id].update(fields)
        self._save(job_id)

    def _update_recon(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._states[job_id]["recon3d"].update(fields)
        self._save(job_id)

    def _update_hq(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._states[job_id].setdefault("recon3d_hq", _new_hq_state()).update(fields)
        self._save(job_id)

    def _warn(self, job_id: str, msg: str) -> None:
        log.warning("job %s: %s", job_id, msg)
        with self._lock:
            self._states[job_id].setdefault("warnings", []).append(msg)

    def _state(self, job_id: str) -> dict[str, Any] | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            st = self._states.get(job_id)
        if st is not None:
            return st
        path = self.jobs_dir / job_id / "state.json"
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        with self._lock:
            return self._states.setdefault(job_id, st)

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Public JSON view of a job (internal keys starting with '_' are dropped), or None."""
        st = self._state(job_id)
        if st is None:
            return None
        rec = st.get("recon3d") or {}
        if rec.get("status") == "done" and "view" not in rec:  # finished before the view existed: read it once
            raw = rec.get("_ply_path")
            view = read_view(Path(raw).parent) if raw else None
            with self._lock:
                rec["view"] = view
        with self._lock:
            view = json.loads(json.dumps(st))
        view["recon3d"] = {k: v for k, v in (view.get("recon3d") or {}).items() if not k.startswith("_")}
        view["recon3d_hq"] = {k: v for k, v in (view.get("recon3d_hq") or _new_hq_state()).items()
                              if not k.startswith("_")}
        view["status_url"] = f"{API_PREFIX}/{job_id}"
        return {k: v for k, v in view.items() if not k.startswith("_")}

    def internal(self, job_id: str) -> dict[str, Any] | None:
        """Raw state including internal keys (file paths). Read-only use by the API."""
        st = self._state(job_id)
        if st is None:
            return None
        with self._lock:
            return json.loads(json.dumps(st))

    # ----- submission
    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def submit(self, job_id: str, video: Path, *, vin_photo: Path | None = None, vin_text: str | None = None) -> None:
        self.start()
        with self._lock:
            self._states[job_id] = _new_state(job_id)
        self._save(job_id)
        self._inspect_q.put(_InspectionTask(job_id, Path(video), Path(vin_photo) if vin_photo else None,
                                            vin_text or None))

    # ----- store
    @contextmanager
    def store(self) -> Iterator[StoreLike]:
        """A fresh store per use (SQLite connections are not shared between threads)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        st = self.backends.open_store(self.db_path)
        try:
            yield st
        finally:
            close = getattr(st, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.exception("store close failed")

    def compute_diff(self, previous_findings: list[Any] | None, current_findings: list[Any]) -> dict[str, Any]:
        """``passport.diff.diff``; with no previous inspection it is called with ``None`` (first inspection)."""
        try:
            out = self.backends.diff(previous_findings, current_findings)
        except TypeError:
            if previous_findings is not None:
                raise
            out = dict(self.backends.diff([], current_findings))
            out.setdefault("first_inspection", True)
        return jsonable(out)

    # ----- inspection worker
    def _get_inspector(self) -> InspectorLike:
        with self._inspector_lock:
            if self._inspector is None:
                try:
                    self._inspector = self.backends.make_inspector(self.device)
                    self._inspector_error = None
                except Exception as exc:
                    self._inspector_error = f"{type(exc).__name__}: {exc}"
                    raise
            return self._inspector

    def _handle_inspection(self, task: _InspectionTask | str) -> None:
        if isinstance(task, str):  # "__warmup__"
            try:
                t0 = time.perf_counter()
                self._get_inspector()
                log.info("models loaded in %.1f s on %s", time.perf_counter() - t0, self.device)
            except Exception:
                log.exception("model warm-up failed")
            return
        job_id = task.job_id
        if self._state(job_id) is None:
            return
        last_step: list[str | None] = [None]

        def progress(step: str, fraction: float) -> None:
            try:
                f = max(0.0, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                f = 0.0
            with self._lock:
                prog = self._states[job_id]["progress"]
                if prog and prog[-1]["step"] == str(step):
                    prog[-1]["fraction"] = f
                else:
                    prog.append({"step": str(step), "fraction": f})
            if step != last_step[0] or time.monotonic() - self._last_save.get(job_id, 0.0) > 1.0:
                last_step[0] = step
                self._save(job_id)

        self._update(job_id, status="running")
        t0 = time.perf_counter()
        try:
            progress("load_models", 0.0)
            inspector = self._get_inspector()
            progress("load_models", 1.0)
            result = inspector.run(task.video, out_dir=self.job_dir(job_id), vin_image_path=task.vin_photo,
                                   vin_text=task.vin_text, progress=progress)
        except Exception as exc:
            log.exception("inspection %s failed", job_id)
            self._update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            # privacy: the raw upload (unblurred video / VIN photo) never outlives the inspection
            shutil.rmtree(self.upload_dir(job_id), ignore_errors=True)
        with self._lock:
            self._states[job_id]["timings_s"]["inspection"] = round(time.perf_counter() - t0, 3)
        try:
            self._finish(job_id, result)
        except Exception as exc:  # defensive: post-processing bug
            log.exception("post-processing %s failed", job_id)
            self._update(job_id, status="error", error=f"post-processing failed: {type(exc).__name__}: {exc}")

    def collect_media(self, job_id: str, extra: dict[str, Any]) -> dict[str, Any]:
        """URLs of the blurred frames, spin frames and masks the pipeline wrote into the job folder."""
        base = f"{API_PREFIX}/{job_id}"
        fdir, sdir, mdir = self.frames_dir(job_id), self.job_dir(job_id) / "spin", self.masks_dir(job_id)
        on_disk = sorted(p.name for p in fdir.iterdir() if p.suffix.lower() in IMAGE_EXT) if fdir.is_dir() else []
        listed = [n for n in (_item_name(x, on_disk) for x in extra.get("frames") or []) if n in set(on_disk)]
        frames = list(dict.fromkeys(listed)) if listed else on_disk
        spin_files = sorted(p.name for p in sdir.iterdir() if p.suffix.lower() in IMAGE_EXT) if sdir.is_dir() else []
        known = set(frames) | set(spin_files)
        spin: list[str] = []
        for item in extra.get("spin") or []:
            name = _item_name(item, frames)
            if name and name in known and name not in spin:
                spin.append(name)
        if not spin:
            spin = spin_files or frames
        masks = (sorted(p.relative_to(mdir).as_posix() for p in mdir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
                 if mdir.is_dir() else [])
        return {
            "frames": [f"{base}/frames/{n}" for n in frames],
            "spin": [f"{base}/frames/{n}" for n in spin],
            "pdf": None,
            "masks": [f"{base}/masks/{n}" for n in masks],
        }

    def resolve_frame(self, job_id: str, name: str) -> Path | None:
        rel = safe_relpath(name)
        if rel is None or "/" in rel:
            return None
        for folder in (self.frames_dir(job_id), self.job_dir(job_id) / "spin"):
            p = folder / rel
            if p.is_file() and _contained(p, folder):
                return p
        return None

    def resolve_mask(self, job_id: str, name: str) -> Path | None:
        rel = safe_relpath(name)
        if rel is None:
            return None
        p = self.masks_dir(job_id) / rel
        return p if p.is_file() and _contained(p, self.masks_dir(job_id)) else None

    def _finish(self, job_id: str, result: Any) -> None:
        job_dir = self.job_dir(job_id)
        result_dict: dict[str, Any] = jsonable(result)
        if not isinstance(result_dict, dict):
            raise TypeError(f"Inspector returned {type(result).__name__}, expected InspectionResult")
        extra = result_dict.get("extra") or {}
        vin = result_dict.get("vin") or None
        findings = result_dict.get("findings") or []
        media = self.collect_media(job_id, extra)

        inspection_id: int | None = None
        previous: dict[str, Any] | None = None
        t0 = time.perf_counter()
        try:
            with self.store() as store:
                inspection_id = int(store.add_inspection(result, media={**media, "job_id": job_id}))
                if vin:
                    previous = store.previous(vin, inspection_id)
        except Exception as exc:
            self._warn(job_id, f"passport store failed: {type(exc).__name__}: {exc}")

        diff_obj: dict[str, Any] | None = None
        if vin and inspection_id is not None:
            try:
                prev_findings = ((previous.get("result") or {}).get("findings") or []) if previous else None
                diff_obj = self.compute_diff(prev_findings, findings)
                if previous:
                    diff_obj.setdefault("previous_inspection_id", previous.get("id"))
                    diff_obj.setdefault("previous_created_utc", previous.get("created_utc"))
            except Exception as exc:
                self._warn(job_id, f"diff failed: {type(exc).__name__}: {exc}")

        report: dict[str, Any] | None = None
        try:
            report = jsonable(self.backends.build_json(
                result, diff=diff_obj, decode=extra.get("vin_decode"),
                extra={"inspection_id": inspection_id, "job_id": job_id, "media": media}))
        except Exception as exc:
            self._warn(job_id, f"report JSON failed: {type(exc).__name__}: {exc}")

        if report is not None:
            images = [p for u in media["spin"][: self.settings.pdf_images]
                      if (p := self.resolve_frame(job_id, u.rsplit("/", 1)[-1])) is not None]
            try:
                pdf = Path(self.backends.build_pdf(report, job_dir / "report.pdf", images) or job_dir / "report.pdf")
                if pdf.is_file() and _contained(pdf, job_dir):
                    media["pdf"] = f"{API_PREFIX}/{job_id}/report.pdf"
                    with self._lock:
                        self._states[job_id]["_pdf"] = pdf.resolve().relative_to(job_dir.resolve()).as_posix()
                    self._store_media(job_id, inspection_id, {"pdf": media["pdf"]})
                else:
                    self._warn(job_id, "PDF builder returned no file inside the job folder")
            except Exception as exc:
                self._warn(job_id, f"PDF failed: {type(exc).__name__}: {exc}")

        with self._lock:
            st = self._states[job_id]
            st["timings_s"]["passport"] = round(time.perf_counter() - t0, 3)
            st.update(status="done", result=result_dict, report=report, diff=diff_obj, inspection_id=inspection_id,
                      media=media, error=None)
        self._save(job_id)
        if self.on_stored is not None and inspection_id is not None:
            try:
                self.on_stored(job_id)
            except Exception:  # a persistence hook must never fail the inspection
                log.exception("on_stored hook failed")
        self._prepare_hq(job_id)
        self._schedule_recon3d(job_id)

    def _store_media(self, job_id: str, inspection_id: int | None, media: dict[str, Any]) -> None:
        """Merge media URLs into the stored inspection (PassportStore.update_media, when it exists)."""
        if inspection_id is None:
            return
        try:
            with self.store() as store:
                update = getattr(store, "update_media", None)
                if callable(update):
                    update(inspection_id, media)
        except Exception as exc:
            self._warn(job_id, f"store media update failed: {type(exc).__name__}: {exc}")

    # ----- 3D worker
    def _schedule_recon3d(self, job_id: str) -> None:
        if self.settings.enable_recon3d and self.settings.reuse_3d and self.reuse_build(job_id, "recon3d")[0]:
            self._schedule_hq(job_id)  # no-op unless the HQ stage is waiting
            return
        ok, msg = self.recon3d_available()
        if not ok:
            self._update_recon(job_id, status="unavailable", error=msg)
            self._schedule_hq(job_id)
            return
        if not self.frames_dir(job_id).is_dir():
            self._update_recon(job_id, status="error", error="no blurred frames to reconstruct from")
            self._schedule_hq(job_id)
            return
        self._update_recon(job_id, status="queued", error=None)
        self._recon_q.put(job_id)

    def _prepare_hq(self, job_id: str) -> None:
        """Third, optional stage (GPU only): decided when the passport is done, queued after the DA3 job.

        disabled / unavailable are statuses the web page does not show; "waiting" = waiting for the DA3 job."""
        if not self.settings.enable_hq3d:
            self._update_hq(job_id, status="disabled", error=None)
            return
        if self.settings.reuse_3d and self.reuse_build(job_id, "recon3d_hq")[0]:
            return
        ok, msg = self.hq_available()
        if not ok:
            self._update_hq(job_id, status="unavailable", error=msg)
            return
        if not self.frames_dir(job_id).is_dir():
            self._update_hq(job_id, status="error", error="no blurred frames to reconstruct from")
            return
        self._update_hq(job_id, status="waiting", error=None, steps=self.settings.hq3d_steps,
                        previous_run=self.last_hq_run())

    def _schedule_hq(self, job_id: str) -> None:
        """Queue the HQ stage if it is waiting for the DA3 job (no-op otherwise)."""
        st = self._state(job_id)
        if st is None or (st.get("recon3d_hq") or {}).get("status") != "waiting":
            return
        self._update_hq(job_id, status="queued", error=None)
        self._hq_q.put(job_id)

    def last_hq_run(self) -> dict[str, Any] | None:
        """Measured numbers of the most recent successful HQ build on this server (from its hq_report.json)."""
        best: tuple[float, dict[str, Any]] | None = None
        for rp in self.jobs_dir.glob("*/recon3d_hq/hq_report.json"):
            try:
                rep = json.loads(rp.read_text(encoding="utf-8"))
                mtime = rp.stat().st_mtime
            except (OSError, ValueError):
                continue
            if rep.get("ok") and (best is None or mtime > best[0]):
                best = (mtime, rep)
        if best is None:
            return None
        rep = best[1]
        secs = rep.get("seconds") or {}
        return {"seconds_total": secs.get("total"), "seconds_sfm": secs.get("sfm"), "seconds_brush": secs.get("brush"),
                "steps": rep.get("steps"), "n_images": rep.get("n_images"),
                "finished_utc": datetime.fromtimestamp(best[0], UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}

    def _handle_hq(self, job_id: str) -> None:
        st = self._state(job_id)
        if st is None:
            return
        out_dir = self.job_dir(job_id) / "recon3d_hq"
        extra = (st.get("result") or {}).get("extra") or {}
        frames = self.frames_dir(job_id)
        n_frames = sum(1 for p in frames.iterdir() if p.suffix.lower() in IMAGE_EXT) if frames.is_dir() else 0
        self._update_hq(job_id, status="running", error=None, started_utc=utc_now(), n_frames=n_frames)
        t0 = time.perf_counter()
        mdir = self.masks_dir(job_id)
        try:
            res = jsonable(self.backends.hq_build(frames, out_dir, self.settings.hq3d_steps,
                                                  extra.get("pins2d") or [], mdir if mdir.is_dir() else None)) or {}
        except Exception as exc:
            log.exception("hq3d %s failed", job_id)
            self._update_hq(job_id, status="error", error=f"{type(exc).__name__}: {exc}",
                            seconds=round(time.perf_counter() - t0, 1))
            return
        seconds = res.get("seconds_wall")
        seconds = float(seconds) if isinstance(seconds, (int, float)) else round(time.perf_counter() - t0, 1)
        report = res.get("report") if isinstance(res.get("report"), dict) else None
        splat = Path(res["splat"]) if isinstance(res.get("splat"), str) else out_dir / "splat.ply"
        if not res.get("ok") or not splat.is_file() or not _contained(splat, self.data_dir):
            self._update_hq(job_id, status="error", seconds=seconds, report=report,
                            error=str(res.get("error") or "high-quality 3D failed (no splat file)"))
            return
        url = f"{API_PREFIX}/{job_id}/recon3d/splat.ply"
        full = Path(res["splat_full"]) if isinstance(res.get("splat_full"), str) else None
        full_ok = full is not None and full.is_file() and _contained(full, self.data_dir) and full != splat
        crop = (report or {}).get("crop") or {}
        self._update_hq(job_id, status="done", splat=url, seconds=seconds, report=report, error=None,
                        view=(report or {}).get("view"), pins=(report or {}).get("pins") or [],
                        cropped=bool(crop.get("ok")) and full_ok,
                        splat_full=f"{API_PREFIX}/{job_id}/recon3d/splat_full.ply" if full_ok else None,
                        _splat_path=str(splat.resolve()), _splat_full_path=str(full.resolve()) if full_ok else None,
                        finished_utc=utc_now())
        self._store_media(job_id, st.get("inspection_id"), {"recon3d_splat": url})
        self._register_build(job_id, "recon3d_hq")

    def resolve_splat(self, job_id: str, full: bool = False) -> Path | None:
        """The served HQ splat (car crop when available) or, with full=True, the uncropped one."""
        st = self._state(job_id)
        if st is None:
            return None
        hq = st.get("recon3d_hq") or {}
        raw = hq.get("_splat_full_path" if full else "_splat_path")
        if hq.get("status") != "done" or not raw:
            return None
        p = Path(raw)
        return p if p.is_file() and _contained(p, self.data_dir) else None

    def _handle_recon3d(self, job_id: str) -> None:
        st = self._state(job_id)
        if st is None:
            return
        try:
            self._run_recon3d(job_id, st)
        finally:
            self._schedule_hq(job_id)

    def _run_recon3d(self, job_id: str, st: dict[str, Any]) -> None:
        job_dir = self.job_dir(job_id)
        out_dir = job_dir / "recon3d"
        extra = (st.get("result") or {}).get("extra") or {}
        self._update_recon(job_id, status="running", error=None, started_utc=utc_now())
        t0 = time.perf_counter()
        try:
            mdir = self.masks_dir(job_id)
            res = jsonable(self.backends.recon3d_build(self.frames_dir(job_id), out_dir, self.recon3d_device,
                                                       mdir if mdir.is_dir() else None)) or {}
        except Exception as exc:
            log.exception("recon3d %s failed", job_id)
            self._update_recon(job_id, status="error", error=f"{type(exc).__name__}: {exc}",
                               seconds=round(time.perf_counter() - t0, 3))
            return
        seconds = res.get("seconds_wall")
        seconds = float(seconds) if isinstance(seconds, (int, float)) else round(time.perf_counter() - t0, 3)
        if not res.get("ok"):
            self._update_recon(job_id, status="error", seconds=seconds,
                               error=str(res.get("error") or "3D reconstruction failed"))
            return
        res_out = Path(res.get("out_dir") or out_dir)
        info = res.get("result") if isinstance(res.get("result"), dict) else {}
        # points_web.ply (decimated for the browser) first, then the full cloud
        candidates = [Path(v) for v in (res.get("ply_web"), res.get("ply"), info.get("ply")) if isinstance(v, str)]
        candidates += [res_out / "points_web.ply", res_out / "points.ply", *sorted(res_out.glob("*.ply"))]
        ply = next((p for p in candidates if p.is_file()), None)
        if ply is None or not _contained(ply, self.data_dir):
            self._update_recon(job_id, status="error", seconds=seconds, error="3D build reported ok but no .ply found")
            return
        pins: list[Any] = []
        pins_error: str | None = None
        try:
            pins = jsonable(self.backends.project_pins(res_out, extra.get("pins2d") or [], extra.get("frame_size")))
        except Exception as exc:
            pins_error = f"{type(exc).__name__}: {exc}"
            log.exception("project_pins %s failed", job_id)
        ply_url = f"{API_PREFIX}/{job_id}/recon3d/points.ply"
        self._update_recon(job_id, status="done", ply=ply_url, pins=pins, seconds=seconds, error=None,
                           pins_error=pins_error, info=info, view=read_view(res_out), _ply_path=str(ply.resolve()),
                           finished_utc=utc_now())
        self._store_media(job_id, st.get("inspection_id"), {"recon3d_ply": ply_url, "recon3d_pins": pins})
        self._register_build(job_id, "recon3d")

    def resolve_ply(self, job_id: str) -> Path | None:
        st = self._state(job_id)
        if st is None:
            return None
        rec = st.get("recon3d") or {}
        raw = rec.get("_ply_path")
        if rec.get("status") != "done" or not raw:
            return None
        p = Path(raw)
        return p if p.is_file() and _contained(p, self.data_dir) else None

    def resolve_pdf(self, job_id: str) -> Path | None:
        st = self._state(job_id)
        if st is None or not st.get("_pdf"):
            return None
        p = self.job_dir(job_id) / st["_pdf"]
        return p if p.is_file() and _contained(p, self.job_dir(job_id)) else None

    # ----- reuse of finished 3D builds (api/reuse3d.py)
    def _register_build(self, job_id: str, kind: str) -> None:
        """Index a successful real build so the same 3D input is not rebuilt (never fails the job)."""
        try:
            st = self.internal(job_id)
            key = reuse3d.key_for_finished(kind, st) if st else None
            if key:
                self.reuse_index.register([(key, reuse3d.entry_for(kind, st))])
        except Exception:
            log.exception("could not register the %s build of %s", kind, job_id)

    def backfill_index(self) -> int:
        """Index every finished real build found in the loaded job states (startup). Returns # entries changed."""
        with self._lock:
            states = [json.loads(json.dumps(st)) for st in self._states.values()]
        items = [(key, reuse3d.entry_for(kind, st)) for st in states for kind in reuse3d.KINDS
                 if (key := reuse3d.key_for_finished(kind, st))]
        try:
            return self.reuse_index.register(items)
        except Exception:
            log.exception("3D reuse index backfill failed")
            return 0

    def reuse_build(self, job_id: str, kind: str) -> tuple[bool, str]:
        """Attach a finished build of the same 3D input to this job (no queueing). (reused?, reason).

        False (= build normally) when the job has no key, no finished build matches, or the source files are gone."""
        try:
            return self._reuse_build(job_id, kind)
        except Exception as exc:  # a reuse bug must never block the normal build
            log.exception("3D reuse %s for %s failed", kind, job_id)
            shutil.rmtree(self.job_dir(job_id) / reuse3d.REUSE_DIRS[kind], ignore_errors=True)
            return False, f"reuse failed: {type(exc).__name__}: {exc}"

    def _reuse_build(self, job_id: str, kind: str) -> tuple[bool, str]:
        st = self.internal(job_id)
        if st is None:
            return False, "job not found"
        device = self.recon3d_device if kind == "recon3d" else "n/a"
        key = reuse3d.key_for_new(kind, st, device=device, hq_steps=self.settings.hq3d_steps)
        if key is None:
            return False, "no cache key (the inspection has no frames_sha256 or frame config)"
        entry = self.reuse_index.get(key)
        if entry is None:
            return False, f"no finished build for key {key}"
        src_id = str(entry.get("job_id") or "")
        if src_id == job_id or not JOB_ID_RE.match(src_id):
            return False, "the index entry is this job"
        src = self.internal(src_id)
        if src is None or reuse3d.key_for_finished(kind, src) != key:
            self.reuse_index.drop(key, src_id)
            return False, f"stale index entry (job {src_id} has no finished {kind} build with this key)"
        rec = src[kind]
        served_raw = rec.get("_ply_path" if kind == "recon3d" else "_splat_path")
        served = Path(served_raw) if isinstance(served_raw, str) else None
        if served is None or not _contained(served, self.data_dir):
            return False, f"job {src_id} has no served {kind} file"
        src_dir, dst_dir = served.parent, self.job_dir(job_id) / reuse3d.REUSE_DIRS[kind]
        names = reuse3d.DA3_FILES if kind == "recon3d" else reuse3d.HQ_FILES
        required = {served.name, "cameras.json", "depth.npz"} if kind == "recon3d" else {served.name}
        full_raw = rec.get("_splat_full_path") if kind == "recon3d_hq" else None
        full_name = Path(full_raw).name if isinstance(full_raw, str) and Path(full_raw).parent == src_dir else None
        names = tuple(dict.fromkeys((*names, served.name, *([full_name] if full_name else []))))
        t0 = time.perf_counter()
        try:
            how = reuse3d.attach_files(src_dir, dst_dir, names, required)
        except OSError as exc:  # source files deleted / disk error: build normally
            log.warning("job %s: cannot reuse %s of %s: %s", job_id, kind, src_id, exc)
            return False, f"cannot reuse: {exc}"
        extra = (st.get("result") or {}).get("extra") or {}
        common = {"status": "done", "error": None, "seconds": rec.get("seconds"), "finished_utc": utc_now(),
                  "reused": True, "reused_from_job": src_id, "reused_from_created_utc": src.get("created_utc"),
                  "original_seconds": rec.get("seconds"), "original_finished_utc": rec.get("finished_utc"),
                  "original_device": entry.get("device"), "reuse_key": key,
                  "reuse_files": {"linked": sum(v == "link" for v in how.values()),
                                  "copied": sum(v == "copy" for v in how.values())}}
        if kind == "recon3d":
            pins: list[Any] = []
            pins_error: str | None = None
            try:  # the NEW inspection's 2D boxes, projected with the reused cameras + depth (numpy, ~1 s)
                pins = jsonable(self.backends.project_pins(dst_dir, extra.get("pins2d") or [], extra.get("frame_size")))
            except Exception as exc:
                pins_error = f"{type(exc).__name__}: {exc}"
            url = f"{API_PREFIX}/{job_id}/recon3d/points.ply"
            new = {**common, "ply": url, "pins": pins, "pins_error": pins_error, "info": rec.get("info"),
                   "view": read_view(dst_dir), "_ply_path": str((dst_dir / served.name).resolve()),
                   "reuse_seconds": round(time.perf_counter() - t0, 3)}
            with self._lock:
                self._states[job_id]["recon3d"] = new
            self._save(job_id)
            self._store_media(job_id, st.get("inspection_id"), {"recon3d_ply": url, "recon3d_pins": pins})
        else:
            same_boxes = (extra.get("pins2d") or []) == (((src.get("result") or {}).get("extra") or {}).get("pins2d")
                                                          or [])
            # HQ pins need the SfM model (pycolmap, not in the main venv): reuse them only for identical 2D boxes
            pins = (rec.get("pins") or []) if same_boxes else []
            pins_note = None if same_boxes else ("damage boxes differ from the original inspection and the "
                                                 "high-quality pins cannot be re-placed without re-running SfM: "
                                                 "no pins on the reused high-quality model")
            full_ok = full_name is not None and (dst_dir / full_name).is_file() and full_name != served.name
            url = f"{API_PREFIX}/{job_id}/recon3d/splat.ply"
            old = st.get("recon3d_hq") or {}
            new = {**_new_hq_state(), **common, "splat": url, "report": rec.get("report"), "view": rec.get("view"),
                   "pins": pins, "pins_note": pins_note, "steps": rec.get("steps"), "n_frames": rec.get("n_frames"),
                   "cropped": bool(rec.get("cropped")) and full_ok,
                   "splat_full": f"{API_PREFIX}/{job_id}/recon3d/splat_full.ply" if full_ok else None,
                   "_splat_path": str((dst_dir / served.name).resolve()),
                   "_splat_full_path": str((dst_dir / full_name).resolve()) if full_ok and full_name else None,
                   "reuse_seconds": round(time.perf_counter() - t0, 3)}
            if old.get("status") == "error":
                new["replaced_error"] = old.get("error")
            with self._lock:
                self._states[job_id]["recon3d_hq"] = new
            self._save(job_id)
            self._store_media(job_id, st.get("inspection_id"), {"recon3d_splat": url})
        log.info("job %s: %s reused from job %s (%s)", job_id, kind, src_id, key)
        return True, f"{kind} reused from job {src_id} (key {key})"
