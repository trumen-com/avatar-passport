"""Damage -> car part assignment: each damage instance goes to the part that covers most of it.

Overlap = |damage ∩ part| / |damage| on boolean masks of the same image. When either mask is None
(detection-only model) the box intersection over the damage box area is used instead.
``MIN_OVERLAP`` is an UNCALIBRATED default (assumption), to be calibrated on the local test set in Phase 2.
"""

from __future__ import annotations

import numpy as np

from ..schema import DamageObservation, Segment

UNKNOWN_PART = "unknown"
MIN_OVERLAP = 0.2   # uncalibrated default (assumption): below this best overlap the part is "unknown"

Box = tuple[float, float, float, float]


def _box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_overlap(damage: Box, part: Box) -> float:
    """|damage box ∩ part box| / |damage box| (0 for a degenerate damage box)."""
    area = _box_area(damage)
    if area <= 0.0:
        return 0.0
    inter = (max(damage[0], part[0]), max(damage[1], part[1]), min(damage[2], part[2]), min(damage[3], part[3]))
    return _box_area(inter) / area


def _mask_crop(mask: np.ndarray) -> tuple[slice, slice] | None:
    """Tight row/column slices around the True pixels (None for an empty mask); speeds up the intersections."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def overlap(damage: Segment, part: Segment) -> float:
    """|damage ∩ part| / |damage| in 0..1. Masks when both exist, otherwise boxes.
    Raises ValueError when both masks exist but their shapes differ (caller bug: different images)."""
    if damage.mask is None or part.mask is None:
        return _box_overlap(damage.box, part.box)
    if damage.mask.shape != part.mask.shape:
        raise ValueError(f"mask shapes differ: damage {damage.mask.shape} vs part {part.mask.shape}")
    dm = damage.mask.astype(bool, copy=False)
    crop = _mask_crop(dm)
    if crop is None:
        return 0.0
    d = dm[crop]
    inter = np.count_nonzero(d & part.mask[crop].astype(bool, copy=False))
    return inter / np.count_nonzero(d)


def assign(
    source: str, damages: list[Segment], parts: list[Segment], *, min_overlap: float = MIN_OVERLAP
) -> list[DamageObservation]:
    """One DamageObservation per damage, in input order. The part with the largest overlap wins
    (ties: first part in input order). Below ``min_overlap`` the part is "unknown", with part_area_px=0
    and area_ratio=0.0; part_overlap then still records the best overlap found (0.0 when no parts).
    area_ratio = damage.area_px / part.area_px. "license-plate" is treated as a normal part."""
    out: list[DamageObservation] = []
    for d in damages:
        best_part: Segment | None = None
        best = 0.0
        for p in parts:
            ov = overlap(d, p)
            if best_part is None or ov > best:
                best_part, best = p, ov
        if best_part is not None and best >= min_overlap:
            part, part_area = best_part.cls, int(best_part.area_px)
            ratio = d.area_px / part_area if part_area > 0 else 0.0
        else:
            part, part_area, ratio = UNKNOWN_PART, 0, 0.0
        out.append(
            DamageObservation(
                source=source,
                damage=d.cls,
                score=float(d.score),
                part=part,
                part_overlap=float(best),
                damage_area_px=int(d.area_px),
                part_area_px=part_area,
                area_ratio=float(ratio),
                box=tuple(float(v) for v in d.box),  # type: ignore[arg-type]
            )
        )
    return out
