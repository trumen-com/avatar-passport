"""Compare two inspections of the same VIN, keyed by (part, damage).

new        in current only            resolved   in previous only
worse      severity rank went up      improved   severity rank went down (extra key)
unchanged  same severity rank
Severity rank: None < minor < moderate < severe. Items: {part, damage, severity_before, severity_after}.
previous None -> every current finding is "new" and "first_inspection" is True.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from ..schema import DamageFinding

SEVERITY_RANK: dict[str | None, int] = {None: 0, "minor": 1, "moderate": 2, "severe": 3}

FindingLike = DamageFinding | dict[str, Any]


def _as_dict(item: FindingLike) -> dict[str, Any]:
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    return dict(item)  # type: ignore[arg-type]


def _rank(severity: str | None) -> int:
    return SEVERITY_RANK.get(severity, 0)


def _index(findings: Sequence[FindingLike]) -> dict[tuple[str, str], str | None]:
    """(part, damage) -> most severe severity seen for that key."""
    out: dict[tuple[str, str], str | None] = {}
    for item in findings:
        d = _as_dict(item)
        key = (str(d.get("part") or "unknown"), str(d.get("damage") or "unknown"))
        sev = d.get("severity")
        if key not in out or _rank(sev) > _rank(out[key]):
            out[key] = sev
    return out


def _item(key: tuple[str, str], before: str | None, after: str | None) -> dict[str, Any]:
    return {"part": key[0], "damage": key[1], "severity_before": before, "severity_after": after}


def diff(previous: Sequence[FindingLike] | None, current: Sequence[FindingLike]) -> dict[str, Any]:
    """Diff findings (DamageFinding objects or their dicts); lists are sorted by (part, damage)."""
    cur = _index(current)
    out: dict[str, Any] = {"new": [], "resolved": [], "unchanged": [], "worse": [], "improved": [],
                           "first_inspection": previous is None}
    prev = _index(previous) if previous is not None else {}
    for key in sorted(set(cur) | set(prev)):
        if key not in prev:
            out["new"].append(_item(key, None, cur[key]))
        elif key not in cur:
            out["resolved"].append(_item(key, prev[key], None))
        else:
            before, after = prev[key], cur[key]
            bucket = "worse" if _rank(after) > _rank(before) else "improved" if _rank(after) < _rank(before) \
                else "unchanged"
            out[bucket].append(_item(key, before, after))
    return out
