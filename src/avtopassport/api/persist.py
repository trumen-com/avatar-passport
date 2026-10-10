"""Opt-in persistence for a Hugging Face Docker Space (ADR 003, "Persistence"): snapshots to a PRIVATE dataset repo.

The Space disk is not persistent, so the passport history (``DATA_DIR/passports.sqlite``) and the small files of each
finished job would be lost on restart. With ``AVTO_SNAPSHOT_REPO`` and ``HF_TOKEN`` set:

- **restore()** at startup downloads ``snapshots/latest.tar.gz`` from the dataset repo and unpacks it into the data dir,
  but only when there is no local DB (a local DB is always newer than the snapshot and is never overwritten). Errors
  only log a warning: the app then starts with an empty history.
- **snapshot_now()** makes a consistent DB copy with SQLite's backup API (never a byte copy of the live file), adds
  ``jobs/<id>/state.json``, ``report.json`` and ``report.pdf`` of finished jobs, writes a manifest (created_utc,
  counts, sha256 + size per file) and uploads the archive with ``HfApi.upload_file``. Nothing is uploaded when the
  content hash is unchanged, and never to a repo that is not private.
- A daemon thread runs it every ``AVTO_SNAPSHOT_EVERY_MIN`` minutes (default 5) and shortly after each finished
  inspection (``notify()``; notifications within ``debounce_s`` are coalesced into one snapshot).

NOT stored: blurred frames, masks, try-on images, 3D models and uploads. Reasons: size (tens of MB per inspection vs a
few hundred kB here) and privacy minimisation - frames are blurred, but the passport history does not need them, so
they never leave the container. Raw uploads are deleted after each inspection anyway. After a restore, the restored
jobs say so in their warnings and show no frames or 3D model (their media lists are emptied, never faked).

With ``AVTO_SNAPSHOT_REPO`` unset (the default) every method is a no-op and nothing is imported from huggingface_hub.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("avtopassport.api.persist")

ARCHIVE_IN_REPO = "snapshots/latest.tar.gz"   # overwritten each time; the repo's git history keeps older ones
MANIFEST = "manifest.json"
DB_NAME = "passports.sqlite"
JOB_FILES = ("state.json", "report.json", "report.pdf")
MAX_FILE_BYTES = 10 * 1024 * 1024             # per job file; anything bigger is skipped (listed in the manifest)
FORMAT = 1
_JOB_MEMBER_RE = re.compile(r"^jobs/([0-9a-f]{32})/(state\.json|report\.json|report\.pdf)$")
_NOT_FOUND = {"EntryNotFoundError", "RemoteEntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError"}
RESTORED_NOTE = ("restored from a snapshot after a server restart: only the passport, report and PDF are kept; "
                 "blurred frames, masks and the 3D model were not stored")


def utc_now() -> str:
    """UTC timestamp, second precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


# --------------------------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class PersistSettings:
    """``AVTO_SNAPSHOT_REPO`` (dataset repo id), ``HF_TOKEN``, ``AVTO_SNAPSHOT_EVERY_MIN`` (default 5)."""

    repo_id: str | None = None
    token: str | None = None
    every_min: float = 5.0
    debounce_s: float = 30.0
    final_timeout_s: float = 20.0

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id and self.token)

    @classmethod
    def from_env(cls) -> PersistSettings:
        """Read the environment; invalid intervals fall back to the default (logged)."""
        every = 5.0
        raw = os.environ.get("AVTO_SNAPSHOT_EVERY_MIN", "").strip()
        if raw:
            try:
                every = float(raw)
                if every <= 0:
                    raise ValueError(raw)
            except ValueError:
                log.warning("ignoring AVTO_SNAPSHOT_EVERY_MIN=%r (need a positive number of minutes)", raw)
                every = 5.0
        return cls(repo_id=os.environ.get("AVTO_SNAPSHOT_REPO", "").strip() or None,
                   token=os.environ.get("HF_TOKEN", "").strip() or None, every_min=every)


# --------------------------------------------------------------------------------------------- SQLite copy

def backup_db(src: Path, dest: Path, timeout_s: float = 30.0) -> None:
    """Consistent copy of a live SQLite DB with the backup API (committed data only; gives up after ``timeout_s``)."""
    deadline = time.monotonic() + timeout_s

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"SQLite backup of {src.name} did not finish in {timeout_s:.0f} s")

    dest.unlink(missing_ok=True)
    source = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        target = sqlite3.connect(str(dest))
        try:
            # one step (pages=-1): the read lock is held for the whole copy, so a concurrent commit cannot force
            # endless restarts (a multi-step backup restarts whenever another connection writes in between)
            source.backup(target, pages=-1, progress=progress, sleep=0.05)
        finally:
            target.close()
    finally:
        source.close()


def _db_ok(path: Path) -> tuple[bool, int]:
    """(integrity_check passed, number of inspections) for a DB file."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        try:
            n = int(conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0])
        except sqlite3.Error:
            n = 0
        return ok, n
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------- archive

def collect_files(data_dir: Path, work: Path) -> tuple[dict[str, bytes], list[str]]:
    """Archive members {name: bytes} (DB copy + small job files of finished jobs) and the skipped names."""
    files: dict[str, bytes] = {}
    skipped: list[str] = []
    db = data_dir / DB_NAME
    if not db.is_file():
        return files, skipped
    copy = work / DB_NAME
    backup_db(db, copy)
    files[DB_NAME] = copy.read_bytes()
    jobs_dir = data_dir / "jobs"
    for job in sorted(jobs_dir.iterdir()) if jobs_dir.is_dir() else []:
        state_path = job / "state.json"
        if not state_path.is_file() or not re.fullmatch(r"[0-9a-f]{32}", job.name):
            continue
        try:
            state_bytes = state_path.read_bytes()  # written atomically by JobManager._save
            if json.loads(state_bytes).get("status") != "done":
                continue  # unfinished or failed jobs are not in the passport history
        except (OSError, ValueError):
            continue
        for name in JOB_FILES:
            path = job / name
            if not path.is_file():
                continue
            member = f"jobs/{job.name}/{name}"
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped.append(member)
                continue
            try:
                files[member] = state_bytes if name == "state.json" else path.read_bytes()
            except OSError:
                skipped.append(member)
    return files, skipped


def digest(files: dict[str, bytes]) -> str:
    """Content hash of a snapshot (DB copy + job files); equal digest -> nothing changed."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + _sha256(files[name]).encode() + b"\n")
    return h.hexdigest()


def build_snapshot(data_dir: Path, out_path: Path) -> dict[str, Any] | None:
    """Write ``out_path`` (tar.gz: manifest.json + DB copy + job files); returns the manifest, None if no DB yet."""
    with tempfile.TemporaryDirectory(prefix="avto-snap-") as tmp:
        files, skipped = collect_files(data_dir, Path(tmp))
        if DB_NAME not in files:
            return None
        _, n_inspections = _db_ok(Path(tmp) / DB_NAME)
    names = sorted(files)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "created_utc": utc_now(),
        "digest": digest(files),
        "counts": {"inspections": n_inspections,
                   "jobs": len({n.split("/")[1] for n in names if n.startswith("jobs/")}),
                   "states": sum(n.endswith("/state.json") for n in names),
                   "reports_json": sum(n.endswith("/report.json") for n in names),
                   "pdfs": sum(n.endswith("/report.pdf") for n in names)},
        "files": {n: {"sha256": _sha256(files[n]), "bytes": len(files[n])} for n in names},
        "skipped_too_large": skipped,
        "not_stored": "blurred frames, masks, try-on images, 3D models, uploads (size and privacy minimisation)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for name, data in [(MANIFEST, json.dumps(manifest, indent=1).encode())] + [(n, files[n]) for n in names]:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(data))
    return manifest


def _annotate_restored(data: bytes, snapshot_utc: str) -> bytes:
    """Mark a restored job: frames/masks/3D are gone, so their URLs are dropped and the reason is a warning."""
    st = json.loads(data)
    media = st.get("media") if isinstance(st.get("media"), dict) else {}
    st["media"] = {**media, "frames": [], "spin": [], "masks": []}
    for key in ("recon3d", "recon3d_hq"):
        rec = st.get(key)
        if isinstance(rec, dict) and rec.get("status") not in {None, "not_started", "disabled", "unavailable"}:
            st[key] = {**{k: v for k, v in rec.items() if not k.startswith("_")},
                       "status": "unavailable", "error": "the 3D model was not kept across a server restart"}
    warnings = st.get("warnings") if isinstance(st.get("warnings"), list) else []
    if RESTORED_NOTE not in warnings:
        st["warnings"] = [*warnings, RESTORED_NOTE]
    st["restored_from_snapshot_utc"] = snapshot_utc
    return json.dumps(st, ensure_ascii=False).encode()


def extract_snapshot(archive: Path, data_dir: Path) -> dict[str, Any]:
    """Unpack a snapshot into ``data_dir`` without overwriting anything; checks names, sha256 and DB integrity."""
    restored: dict[str, Any] = {"db": False, "inspections": 0, "job_files": 0, "snapshot_utc": None}
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        if MANIFEST not in members:
            raise ValueError("snapshot has no manifest.json")
        manifest = json.loads(_read_member(tar, members[MANIFEST]))
        restored["snapshot_utc"] = manifest.get("created_utc")
        expected: dict[str, Any] = manifest.get("files") or {}

        def checked(name: str) -> bytes:
            data = _read_member(tar, members[name])
            if _sha256(data) != (expected.get(name) or {}).get("sha256"):
                raise ValueError(f"sha256 mismatch for {name}")
            return data

        db = data_dir / DB_NAME
        if DB_NAME in members and not db.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            tmp = data_dir / f".{DB_NAME}.restore"
            tmp.write_bytes(checked(DB_NAME))
            ok, n = _db_ok(tmp)
            if not ok:
                tmp.unlink(missing_ok=True)
                raise ValueError("restored DB failed PRAGMA integrity_check")
            os.replace(tmp, db)
            restored.update(db=True, inspections=n)
        for name in sorted(members):
            if not _JOB_MEMBER_RE.match(name) or name not in expected:
                continue  # unknown or unlisted member: ignored (no path traversal, no surprises)
            dest = data_dir / name
            if dest.exists():
                continue  # never overwrite local files
            data = checked(name)
            if name.endswith("/state.json"):
                data = _annotate_restored(data, str(restored["snapshot_utc"]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            restored["job_files"] += 1
    return restored


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    f = tar.extractfile(member)
    if f is None:
        raise ValueError(f"cannot read {member.name}")
    return f.read()


# --------------------------------------------------------------------------------------------- Hugging Face

class HfApiLike(Protocol):
    def create_repo(self, repo_id: str, **kwargs: Any) -> Any: ...

    def repo_info(self, repo_id: str, **kwargs: Any) -> Any: ...

    def upload_file(self, **kwargs: Any) -> Any: ...


def _hf_api(token: str | None) -> HfApiLike:
    from huggingface_hub import HfApi  # noqa: PLC0415 - only when persistence is enabled

    return HfApi(token=token)


def _hf_download(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return str(hf_hub_download(**kwargs))


class Persister:
    """Restore at boot + periodic / on-demand snapshots to a private HF dataset repo. Never raises into the app."""

    def __init__(self, data_dir: Path, settings: PersistSettings | None = None, *,
                 api_factory: Callable[[str | None], HfApiLike] | None = None,
                 download: Callable[..., str] | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.settings = settings or PersistSettings()
        self._api_factory = api_factory
        self._download = download
        self._api: HfApiLike | None = None
        self._lock = threading.Lock()          # status fields
        self._snap_lock = threading.Lock()     # one snapshot at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_at: float | None = None
        self._last_digest: str | None = None
        self.last_snapshot_utc: str | None = None
        self.last_error: str | None = None
        self.last_restore: str | None = None
        self.n_uploads = 0

    @classmethod
    def from_env(cls, data_dir: Path) -> Persister:
        """Settings from the environment; disabled unless AVTO_SNAPSHOT_REPO and HF_TOKEN are both set."""
        s = PersistSettings.from_env()
        p = cls(data_dir, s)
        if s.repo_id and not s.token:
            p.last_error = "AVTO_SNAPSHOT_REPO is set but HF_TOKEN is not: persistence disabled"
            log.warning(p.last_error)
        return p

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def status(self) -> dict[str, Any]:
        """For /api/health: {"enabled", "last_snapshot_utc", "last_error", "last_restore", "every_min"}."""
        with self._lock:
            return {"enabled": self.enabled, "last_snapshot_utc": self.last_snapshot_utc,
                    "last_error": self.last_error, "last_restore": self.last_restore,
                    "every_min": self.settings.every_min if self.enabled else None}

    def _set(self, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self, k, v)

    # ----- restore
    def restore(self) -> str:
        """Download + unpack the latest snapshot when there is no local DB. Returns a one-line summary."""
        if not self.enabled:
            return "disabled"
        if (self.data_dir / DB_NAME).exists():
            msg = "local passports.sqlite exists; snapshot not restored (the local DB is newer)"
            log.info("persistence: %s", msg)
            self._set(last_restore=msg)
            return msg
        try:
            with tempfile.TemporaryDirectory(prefix="avto-restore-") as tmp:
                download = self._download or _hf_download
                path = download(repo_id=self.settings.repo_id, filename=ARCHIVE_IN_REPO, repo_type="dataset",
                                token=self.settings.token, cache_dir=tmp)
                res = extract_snapshot(Path(path), self.data_dir)
            msg = (f"restored snapshot {res['snapshot_utc']}: {res['inspections']} inspections, "
                   f"{res['job_files']} job files")
            log.info("persistence: %s", msg)
        except Exception as exc:
            if type(exc).__name__ in _NOT_FOUND:
                msg = f"no snapshot to restore yet ({type(exc).__name__}); starting empty"
                log.info("persistence: %s", msg)
            else:
                msg = f"restore failed, starting empty: {_short(exc)}"
                log.warning("persistence: %s", msg)
                self._set(last_error=msg)
        self._set(last_restore=msg)
        return msg

    # ----- snapshot
    def _ensure_repo(self) -> HfApiLike:
        if self._api is None:
            api = (self._api_factory or _hf_api)(self.settings.token)
            api.create_repo(self.settings.repo_id, repo_type="dataset", private=True, exist_ok=True)
            info = api.repo_info(self.settings.repo_id, repo_type="dataset")
            if getattr(info, "private", None) is not True:
                raise PermissionError(f"refusing to upload: dataset repo {self.settings.repo_id} is not private")
            self._api = api
        return self._api

    def snapshot_now(self) -> str:
        """Build and upload one snapshot. Returns "uploaded", "unchanged", "empty", "disabled" or "error"."""
        if not self.enabled:
            return "disabled"
        with self._snap_lock:
            try:
                with tempfile.TemporaryDirectory(prefix="avto-upload-") as tmp:
                    archive = Path(tmp) / "latest.tar.gz"
                    manifest = build_snapshot(self.data_dir, archive)
                    if manifest is None:
                        return "empty"
                    if manifest["digest"] == self._last_digest:
                        return "unchanged"
                    api = self._ensure_repo()
                    c = manifest["counts"]
                    api.upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE_IN_REPO,
                                    repo_id=self.settings.repo_id, repo_type="dataset",
                                    commit_message=(f"snapshot {manifest['created_utc']}: {c['inspections']} "
                                                    f"inspections, {c['jobs']} jobs"))
                self._last_digest = manifest["digest"]
                self._set(last_snapshot_utc=manifest["created_utc"], last_error=None, n_uploads=self.n_uploads + 1)
                log.info("persistence: snapshot uploaded (%s)", manifest["counts"])
                return "uploaded"
            except Exception as exc:
                self._set(last_error=f"snapshot failed: {_short(exc)}")
                log.warning("persistence: snapshot failed: %s", _short(exc))
                return "error"

    # ----- background thread
    def notify(self, _job_id: str | None = None) -> None:
        """An inspection was stored: snapshot after ``debounce_s`` (repeated calls join the same snapshot)."""
        if not self.enabled:
            return
        with self._lock:
            if self._pending_at is None:
                self._pending_at = time.monotonic() + self.settings.debounce_s
        self._wake.set()

    def start(self) -> None:
        """Start the daemon thread (no-op when disabled or already running)."""
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="avto-snapshot", daemon=True)
        self._thread.start()

    def stop(self, final_snapshot: bool = True) -> None:
        """Stop the thread; then one last snapshot, bounded by ``final_timeout_s`` (shutdown must not hang)."""
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(self.settings.final_timeout_s)
        self._thread = None
        if final_snapshot:
            t = threading.Thread(target=self.snapshot_now, name="avto-snapshot-final", daemon=True)
            t.start()
            t.join(self.settings.final_timeout_s)

    def _loop(self) -> None:
        every_s = self.settings.every_min * 60.0
        next_periodic = time.monotonic() + every_s
        while not self._stop.is_set():
            with self._lock:
                due = next_periodic if self._pending_at is None else min(next_periodic, self._pending_at)
            self._wake.wait(max(0.0, due - time.monotonic()))
            self._wake.clear()
            if self._stop.is_set():
                return
            now = time.monotonic()
            with self._lock:
                pending_due = self._pending_at is not None and now >= self._pending_at
                if pending_due:
                    self._pending_at = None
            if pending_due or now >= next_periodic:
                self.snapshot_now()
                next_periodic = time.monotonic() + every_s
