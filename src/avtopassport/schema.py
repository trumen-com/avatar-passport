"""Shared data contract for the Phase 3 pipeline (plain dataclasses, JSON-serialisable via ``to_dict``).

Image coordinates are pixels of the image the masks belong to. Masks are boolean ``np.ndarray`` (H, W).
Every number produced from these objects must come from the code that fills them - no defaults are presented
as measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Severity = Literal["minor", "moderate", "severe"]
Action = Literal["polish", "paint", "repair", "replace"]


@dataclass
class Segment:
    """One instance from a segmentation model on one image."""

    cls: str                      # class name from model.names (e.g. "scratch", "left_front-door")
    score: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    mask: np.ndarray | None       # bool (H, W) in the image's pixel grid, or None for detection-only models
    area_px: int                  # mask pixel count (box area when mask is None)


@dataclass
class DamageObservation:
    """A damage instance in one frame/photo, assigned to a car part (mapping.assign)."""

    source: str                   # input id, e.g. "360VIEW_f00288" or "CAR3"
    damage: str                   # damage class
    score: float
    part: str                     # part class, or "unknown" when no part overlaps enough
    part_overlap: float           # |damage ∩ part| / |damage|
    damage_area_px: int
    part_area_px: int             # 0 when part == "unknown"
    area_ratio: float             # damage_area_px / part_area_px (0 when unknown)
    box: tuple[float, float, float, float]


@dataclass
class DamageFinding:
    """Same part + damage type merged across frames (dedupe.merge)."""

    part: str
    damage: str
    n_observations: int
    sources: list[str]
    best_source: str              # observation with the highest score
    max_score: float
    max_area_ratio: float         # largest damage/part ratio seen (drives severity)
    severity: Severity | None = None
    action: Action | None = None
    severity_rule: str | None = None   # human-readable rule that fired (from severity.rate)


@dataclass
class CostLine:
    part: str
    damage: str
    action: Action
    price_uzs: int | None         # None when no matching price row exists
    price_kind: str | None        # "from" | "flat" | "range" ...
    basis: str                    # e.g. "repair_prices.csv: paint (from), auto.ustabor.uz, 2026-10-10"
    is_estimate: bool             # True when the price is not an observed row for this exact service


@dataclass
class InspectionResult:
    vin: str | None
    vin_status: str               # read | ambiguous | could_not_read | manual
    created_utc: str
    frames_sha256: str            # SHA-256 over the per-frame SHA-256 list (report.hash_frames)
    findings: list[DamageFinding] = field(default_factory=list)
    costs: list[CostLine] = field(default_factory=list)
    total_from_uzs: int | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
