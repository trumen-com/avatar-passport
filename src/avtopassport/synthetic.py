"""Synthetic "registry connector - sandbox" data access. Every record is synthetic and flagged.

- Synthetic VINs start with "0": row "0" of the ISO 3780 WMI chart (rev. 2021-04-13) is "spare"
  (currently unassigned) - so they cannot be a real assigned VIN today. They also carry a valid
  check digit and `"synthetic": true` in every record.
- lookup(..., origin="scan") ALWAYS refuses: synthetic history is never attached to a VIN that was
  read from a real car/photo, even if the strings happen to match.
- The data files are written by scripts/generate_synthetic.py (method and assumptions:
  data/synthetic/README.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .paths import SYNTHETIC
from .vin.checkdigit import check_digit_ok

SEED = 20261010
SYNTHETIC_WMI = "0AP"
REGISTRY_JSON = SYNTHETIC / "registry.json"
CUSTOMERS_JSON = SYNTHETIC / "customers.json"
SOURCE_LABEL = "registry connector - sandbox"

_SCAN_REFUSAL = "synthetic registry is never attached to a scanned VIN"


@dataclass(frozen=True)
class RegistryRefusal:
    """Returned instead of registry history when a VIN must not get synthetic data."""

    vin: str
    reason: str


def _canonical(vin: str) -> str:
    return vin.strip().upper()


def is_synthetic_vin(vin: str) -> bool:
    """True for a sandbox VIN: 17 VIN-alphabet characters, first character "0", valid check digit.

    Case and surrounding whitespace are ignored.
    """
    if not isinstance(vin, str):
        return False
    v = _canonical(vin)
    return len(v) == 17 and v[0] == "0" and check_digit_ok(v)


def load_registry(path: Path = REGISTRY_JSON) -> dict[str, dict[str, Any]]:
    """Load the sandbox registry as {VIN: record}.

    Accepts the generator's wrapper ``{"synthetic": true, "records": [...]}`` or a bare list of records.
    Raises ValueError unless the wrapper and every record carry ``"synthetic": true`` (the boolean) and a
    synthetic VIN, or if a VIN repeats; FileNotFoundError if the file is missing.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict):
        if doc.get("synthetic") is not True:
            raise ValueError(f'{path}: file is not flagged "synthetic": true')
        doc = doc.get("records")
    if not isinstance(doc, list):
        raise ValueError(f"{path}: expected a list of registry records")
    registry: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(doc):
        if not isinstance(record, dict) or record.get("synthetic") is not True:
            raise ValueError(f'{path}: record {i} is not flagged "synthetic": true')
        vin = record.get("vin")
        if not is_synthetic_vin(vin):
            raise ValueError(f'{path}: record {i} has no synthetic VIN (17 chars, leading "0", valid check digit)')
        key = _canonical(vin)
        if key in registry:
            raise ValueError(f"{path}: record {i} repeats VIN {key}")
        registry[key] = record
    return registry


def lookup(vin: str, *, origin: Literal["sandbox", "scan"], path: Path = REGISTRY_JSON) -> dict[str, Any] | RegistryRefusal | None:
    """origin='scan' -> RegistryRefusal always. origin='sandbox' -> the record (synthetic=True) or None.

    The scan refusal does not even read the registry, so it holds when the scanned string matches a
    sandbox VIN. Any other origin raises ValueError.
    """
    if origin == "scan":
        return RegistryRefusal(vin, _SCAN_REFUSAL)
    if origin != "sandbox":
        raise ValueError(f"origin must be 'sandbox' or 'scan', got {origin!r}")
    if not is_synthetic_vin(vin):
        return None
    return load_registry(path).get(_canonical(vin))
