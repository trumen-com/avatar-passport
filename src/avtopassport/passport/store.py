"""SQLite passport store: one row per inspection, looked up by VIN.

One connection (check_same_thread=False) guarded by a lock, so a worker thread can write while API
handlers read. VIN None is stored as NULL and has no history. Rows come back as plain dicts:
{id, vin, created_utc, frames_sha256, result (dict), media (dict)}.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schema import InspectionResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vin TEXT,
    created_utc TEXT,
    frames_sha256 TEXT,
    result_json TEXT,
    media_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_inspections_vin ON inspections (vin, created_utc, id);
"""
_COLS = "id, vin, created_utc, frames_sha256, result_json, media_json"


def json_default(obj: Any) -> Any:
    """json.dumps fallback for numpy scalars/arrays, Paths, sets and dataclasses."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if hasattr(obj, "tolist"):          # numpy arrays and scalars
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=json_default, ensure_ascii=False)


def _norm_vin(vin: str | None) -> str | None:
    vin = (vin or "").strip().upper()
    return vin or None


class PassportStore:
    """Inspections by VIN in one SQLite file (``":memory:"`` works for tests)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> PassportStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _row(row: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row[0],
            "vin": row[1],
            "created_utc": row[2],
            "frames_sha256": row[3],
            "result": json.loads(row[4]) if row[4] else {},
            "media": json.loads(row[5]) if row[5] else {},
        }

    def add_inspection(self, result: InspectionResult | dict[str, Any], *, media: dict[str, Any] | None = None) -> int:
        """Insert one inspection and return its id."""
        data = result.to_dict() if isinstance(result, InspectionResult) else dict(result)
        created = data.get("created_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO inspections (vin, created_utc, frames_sha256, result_json, media_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_norm_vin(data.get("vin")), created, data.get("frames_sha256") or "", _dumps(data),
                 _dumps(media or {})),
            )
            return int(cur.lastrowid or 0)

    def update_media(self, inspection_id: int, media: dict[str, Any], *, merge: bool = True) -> bool:
        """Set (or merge into) an inspection's media dict, e.g. when the async 3D model is ready."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT media_json FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
            if row is None:
                return False
            new = {**(json.loads(row[0]) if row[0] and merge else {}), **media}
            self._conn.execute("UPDATE inspections SET media_json = ? WHERE id = ?", (_dumps(new), inspection_id))
            return True

    def get_inspection(self, inspection_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(f"SELECT {_COLS} FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
        return self._row(row)

    def history(self, vin: str | None) -> list[dict[str, Any]]:
        """All inspections of a VIN, newest first (VIN None -> [])."""
        vin = _norm_vin(vin)
        if vin is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM inspections WHERE vin = ? ORDER BY created_utc DESC, id DESC", (vin,)
            ).fetchall()
        return [r for r in map(self._row, rows) if r is not None]

    def previous(self, vin: str | None, inspection_id: int | None) -> dict[str, Any] | None:
        """The latest inspection of ``vin`` earlier than ``inspection_id`` (None id -> latest of the VIN)."""
        vin = _norm_vin(vin)
        if vin is None:
            return None
        with self._lock:
            if inspection_id is None:
                row = self._conn.execute(
                    f"SELECT {_COLS} FROM inspections WHERE vin = ? ORDER BY created_utc DESC, id DESC LIMIT 1",
                    (vin,),
                ).fetchone()
                return self._row(row)
            cur = self._conn.execute("SELECT created_utc FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
            if cur is None:
                return None
            row = self._conn.execute(
                f"SELECT {_COLS} FROM inspections WHERE vin = ? AND id != ? "
                "AND (created_utc < ? OR (created_utc = ? AND id < ?)) "
                "ORDER BY created_utc DESC, id DESC LIMIT 1",
                (vin, inspection_id, cur[0], cur[0], inspection_id),
            ).fetchone()
        return self._row(row)
