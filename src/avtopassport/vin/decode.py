"""VIN decoding: our cited WMI table (primary for Uzbek/Chinese cars) + NHTSA vPIC (optional, cached).

STUB written by main; the vin writer implements the bodies without changing signatures.
Never guess: a field nobody provides stays None; every filled field records its source.
vPIC: GET https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json[&modelyear=YYYY]

Implementation notes:
- The WMI table is read with the csv module (header names are case/space-insensitive, extra columns are
  ignored, rows whose WMI is not 3 VIN-alphabet characters are skipped, a duplicate WMI keeps the
  higher-confidence row). It is cached per (path, mtime, size), so edits to the CSV are picked up.
- A table row rated below "medium" confidence (or unrated) is still used, with a note naming its caveat.
- vPIC is sent ``modelyear`` only when position 10 gives a single candidate year. Successful responses
  are cached as ``<cache_dir>/<vin>.json``; network/HTTP/JSON failures become notes, never exceptions.
- A vPIC ModelYear outside our position-10 candidates is not used (note added). Nor is one that picks
  between several candidates (or comes with no candidate) unless vPIC decoded cleanly (ErrorCode "0"):
  without our ``modelyear`` hint vPIC chooses the 30-year cycle itself (its docs only say "it is
  recommended to always send in the model year"); on the team's real VIN photos it answered ErrorCode
  "5,14" (errors in several positions) yet returned a ModelYear. Any other non-clean ErrorCode adds a note.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from ..paths import OUTPUTS, REFERENCE
from .checkdigit import VIN_ALPHABET, VIN_RE
from .validate import model_year_candidates

log = logging.getLogger("avtopassport.vin.decode")

WMI_TABLE = REFERENCE / "wmi_table.csv"
VPIC_CACHE = OUTPUTS / "cache" / "vpic"
VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}"

# vPIC DecodeVinValues field -> VinDecode attribute (ModelYear, ErrorCode, ErrorText handled separately)
_VPIC_FIELDS = {
    "Make": "make",
    "Model": "model",
    "BodyClass": "body_class",
    "FuelTypePrimary": "fuel_type",
    "PlantCountry": "plant_country",
    "PlantCity": "plant_city",
}
_VPIC_EMPTY = {"", "null", "none", "not applicable"}
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
_NO_DETAIL_CODES = {"7", "8"}


@dataclass(frozen=True)
class WmiRow:
    wmi: str
    manufacturer: str
    country: str
    plant_or_notes: str
    source_url: str
    confidence: str


@dataclass
class VinDecode:
    vin: str
    wmi: str
    manufacturer: str | None = None
    country: str | None = None
    make: str | None = None
    model: str | None = None
    model_year: int | None = None
    model_year_candidates: tuple[int, ...] = ()
    body_class: str | None = None
    fuel_type: str | None = None
    plant_country: str | None = None
    plant_city: str | None = None
    sources: dict[str, str] = field(default_factory=dict)   # field -> "wmi_table" | "vpic" | "vin_rules"
    vpic_error_code: str | None = None
    vpic_error_text: str | None = None
    notes: list[str] = field(default_factory=list)


def _cell(record: dict[str | None, Any], key: str) -> str:
    value = record.get(key)
    return value.strip() if isinstance(value, str) else ""


@lru_cache(maxsize=8)
def _read_table(path: str, mtime_ns: int, size: int) -> dict[str, WmiRow]:
    """Parse the CSV once per (path, mtime, size). Callers get a copy of the dict."""
    rows: dict[str, WmiRow] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        reader.fieldnames = [(name or "").strip().lower() for name in reader.fieldnames]
        if "wmi" not in reader.fieldnames:
            raise ValueError(f"{path}: no 'wmi' column")
        for line_no, record in enumerate(reader, start=2):
            wmi = _cell(record, "wmi").upper()
            if len(wmi) != 3 or any(c not in VIN_ALPHABET for c in wmi):
                if any(_cell(record, k) for k in reader.fieldnames):
                    log.warning("%s:%d: skipped row with WMI %r (not 3 VIN-alphabet characters)", path, line_no, wmi)
                continue
            row = WmiRow(
                wmi=wmi,
                manufacturer=_cell(record, "manufacturer"),
                country=_cell(record, "country"),
                plant_or_notes=_cell(record, "plant_or_notes"),
                source_url=_cell(record, "source_url"),
                confidence=_cell(record, "confidence").lower(),
            )
            old = rows.get(wmi)
            if old is not None:
                log.warning("%s:%d: duplicate WMI %s; keeping the higher-confidence row", path, line_no, wmi)
                if _CONFIDENCE_RANK.get(row.confidence, 0) <= _CONFIDENCE_RANK.get(old.confidence, 0):
                    continue
            rows[wmi] = row
    return rows


def load_wmi_table(path: Path = WMI_TABLE) -> dict[str, WmiRow]:
    """WMI -> row from our cited table. Raises FileNotFoundError / ValueError for a missing or headerless file."""
    p = Path(path).resolve()
    st = p.stat()
    return dict(_read_table(str(p), st.st_mtime_ns, st.st_size))


def known_wmis(path: Path = WMI_TABLE) -> frozenset[str]:
    """Every WMI in our table (for validate(..., known_wmis=...))."""
    return frozenset(load_wmi_table(path))


def _vpic_value(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _VPIC_EMPTY else text


def _first_result(payload: Any) -> dict[str, Any] | None:
    results = payload.get("Results") if isinstance(payload, dict) else None
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0]
    return None


def _write_cache(cache_file: Path, payload: Any) -> None:
    """Atomic write so a crash never leaves half a JSON file behind."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_file.parent, prefix=f".{cache_file.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, cache_file)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _fetch_vpic(vin: str, years: tuple[int, ...], timeout: float, cache_dir: Path | None,
                notes: list[str]) -> dict[str, Any] | None:
    """The DecodeVinValues record for ``vin`` from cache or network; problems become notes."""
    cache_file = Path(cache_dir) / f"{vin}.json" if cache_dir is not None else None
    if cache_file is not None and cache_file.is_file():
        try:
            record = _first_result(json.loads(cache_file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            record = None
        if record is not None:
            return record
        notes.append(f"ignored unreadable vPIC cache file {cache_file.name}")

    params = {"format": "json"}
    if len(years) == 1:
        params["modelyear"] = str(years[0])
    try:
        resp = requests.get(VPIC_URL.format(vin=vin), params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.Timeout:
        notes.append(f"vPIC timed out after {timeout:g} s; not decoded by vPIC")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        notes.append(f"vPIC HTTP error {status}; not decoded by vPIC")
        return None
    except Exception as e:  # connection errors, bad JSON, anything else: never raise from decode()
        notes.append(f"vPIC unavailable ({type(e).__name__}); not decoded by vPIC")
        return None

    record = _first_result(payload)
    if record is None:
        notes.append("vPIC returned no results")
        return None
    if cache_file is not None:
        try:
            _write_cache(cache_file, payload)
        except OSError as e:
            notes.append(f"could not write vPIC cache ({type(e).__name__})")
    return record


def _apply_vpic(out: VinDecode, record: dict[str, Any]) -> None:
    for key, attr in _VPIC_FIELDS.items():
        value = _vpic_value(record, key)
        if value is not None:
            setattr(out, attr, value)
            out.sources[attr] = "vpic"

    out.vpic_error_code = _vpic_value(record, "ErrorCode")
    out.vpic_error_text = _vpic_value(record, "ErrorText")
    codes = {c.strip() for c in (out.vpic_error_code or "").split(",") if c.strip()}
    if codes & _NO_DETAIL_CODES:
        out.notes.append(f"vPIC ErrorCode {out.vpic_error_code}: "
                         "manufacturer not registered with NHTSA / no detailed data")
    elif codes - {"0"}:
        out.notes.append(f"vPIC ErrorCode {out.vpic_error_code}: vPIC decoded this VIN with errors "
                         "(see vpic_error_text)")

    year_text = _vpic_value(record, "ModelYear")
    if year_text is not None:
        try:
            year = int(year_text)
        except ValueError:
            out.notes.append(f"vPIC ModelYear {year_text!r} is not a year; ignored")
            return
        candidates = out.model_year_candidates
        if candidates and year not in candidates:
            out.notes.append(f"vPIC ModelYear {year} is not among the position-10 candidates "
                             f"{candidates}; model year left empty")
        elif len(candidates) != 1 and codes != {"0"}:
            # No model-year hint was sent, so vPIC picked the 30-year cycle itself (undocumented) while
            # reporting errors: a guess, not a decode.
            allowed = f"position 10 allows {candidates}" if candidates else "position 10 is not a year code"
            out.notes.append(f"vPIC ModelYear {year} not used: {allowed}, no model year was sent and vPIC "
                             f"reported ErrorCode {out.vpic_error_code or 'none'}; model year left empty")
        else:
            out.model_year = year
            out.sources["model_year"] = "vpic"


def decode(vin: str, *, use_vpic: bool = True, timeout: float = 4.0, cache_dir: Path | None = VPIC_CACHE,
           table_path: Path = WMI_TABLE) -> VinDecode:
    """Decode a (validated) 17-char VIN. vPIC failures/timeouts are recorded in notes, never raised."""
    v = (vin or "").strip().upper()
    out = VinDecode(vin=v, wmi=v[:3])
    if not VIN_RE.match(v):
        out.notes.append("not a 17-character VIN (VIN alphabet, no I/O/Q); not decoded")
        return out

    try:
        row = load_wmi_table(table_path).get(out.wmi)
    except (OSError, ValueError, csv.Error) as e:
        row = None
        out.notes.append(f"WMI table unavailable ({type(e).__name__})")
    else:
        if row is None:
            out.notes.append(f"WMI {out.wmi} is not in our WMI table")
    if row is not None:
        if row.manufacturer:
            out.manufacturer = row.manufacturer
            out.sources["manufacturer"] = "wmi_table"
        if row.country:
            out.country = row.country
            out.sources["country"] = "wmi_table"
        if _CONFIDENCE_RANK.get(row.confidence, 0) < _CONFIDENCE_RANK["medium"]:
            caveat = f": {row.plant_or_notes}" if row.plant_or_notes else ""
            out.notes.append(f"WMI {out.wmi}: {row.confidence or 'unrated'}-confidence row in our WMI table{caveat}")

    years = model_year_candidates(v)
    if years:
        out.model_year_candidates = years
        out.sources["model_year_candidates"] = "vin_rules"

    if use_vpic:
        record = _fetch_vpic(v, years, timeout, cache_dir, out.notes)
        if record is not None:
            _apply_vpic(out, record)
    return out
