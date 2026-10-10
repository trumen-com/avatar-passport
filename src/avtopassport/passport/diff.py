"""Compare two inspections of the same VIN, keyed by (part, damage).

new        in current only            resolved   in previous only
worse      severity rank went up      improved   severity rank went down (extra key)
unchanged  same severity rank
Severity rank: None < minor < moderate < severe. Items: {part, damage, severity_before, severity_after}.
previous None -> every current finding is "new" and "first_inspection" is True.

Only CONFIRMED findings drive new / resolved / worse / improved: max_score >= CONFIRM_MIN_SCORE and seen in at least
CONFIRM_MIN_VIEWS frames (uncalibrated defaults). Reason: re-inspecting the same undamaged car with a different frame
rate made the uncalibrated damage models report a different set of low-confidence false positives, which showed up as
phantom "new damage". Differences that involve only unconfirmed findings go to "possible_changes"; a confirmed finding
that is now only seen unconfirmed goes to "uncertain" instead of "resolved". Findings without max_score /
n_observations (older stored results) count as confirmed.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from ..schema import DamageFinding

SEVERITY_RANK: dict[str | None, int] = {None: 0, "minor": 1, "moderate": 2, "severe": 3}
CONFIRM_MIN_SCORE = 0.5   # uncalibrated default; same threshold as the UI's low-confidence group
CONFIRM_MIN_VIEWS = 2     # uncalibrated default: seen in at least this many frames

FindingLike = DamageFinding | dict[str, Any]


def _as_dict(item: FindingLike) -> dict[str, Any]:
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    return dict(item)  # type: ignore[arg-type]


def _rank(severity: str | None) -> int:
    return SEVERITY_RANK.get(severity, 0)


def _confirmed(d: dict[str, Any], min_score: float, min_views: int) -> bool:
    score, views = d.get("max_score"), d.get("n_observations")
    return (score is None or float(score) >= min_score) and (views is None or int(views) >= min_views)


def _index(findings: Sequence[FindingLike], *, confirmed: bool | None = None, min_score: float = CONFIRM_MIN_SCORE,
           min_views: int = CONFIRM_MIN_VIEWS) -> dict[tuple[str, str], str | None]:
    """(part, damage) -> most severe severity seen for that key; confirmed=True/False keeps only that kind."""
    out: dict[tuple[str, str], str | None] = {}
    for item in findings:
        d = _as_dict(item)
        if confirmed is not None and _confirmed(d, min_score, min_views) != confirmed:
            continue
        key = (str(d.get("part") or "unknown"), str(d.get("damage") or "unknown"))
        sev = d.get("severity")
        if key not in out or _rank(sev) > _rank(out[key]):
            out[key] = sev
    return out


def _item(key: tuple[str, str], before: str | None, after: str | None) -> dict[str, Any]:
    return {"part": key[0], "damage": key[1], "severity_before": before, "severity_after": after}


def diff(previous: Sequence[FindingLike] | None, current: Sequence[FindingLike], *,
         min_score: float = CONFIRM_MIN_SCORE, min_views: int = CONFIRM_MIN_VIEWS) -> dict[str, Any]:
    """Diff findings (DamageFinding objects or their dicts); lists are sorted by (part, damage)."""
    kw = {"min_score": min_score, "min_views": min_views}
    cur_all = _index(current)
    cur_conf = _index(current, confirmed=True, **kw)
    out: dict[str, Any] = {"new": [], "resolved": [], "unchanged": [], "worse": [], "improved": [], "uncertain": [],
                           "possible_changes": [], "first_inspection": previous is None,
                           "rule": f"changes are judged on confirmed findings only: score >= {min_score} and seen in "
                                   f">= {min_views} frames (uncalibrated defaults)"}
    if previous is None:
        out["new"] = [_item(k, None, v) for k, v in sorted(cur_all.items())]
        return out
    prev_all = _index(previous)
    prev_conf = _index(previous, confirmed=True, **kw)
    for key in sorted(set(cur_all) | set(prev_all)):
        in_cur, in_prev = key in cur_conf, key in prev_conf
        if in_cur and key in prev_all:
            before, after = prev_all[key], cur_conf[key]
            bucket = "worse" if _rank(after) > _rank(before) else "improved" if _rank(after) < _rank(before) \
                else "unchanged"
            out[bucket].append(_item(key, before, after))
        elif in_cur:
            out["new"].append(_item(key, None, cur_conf[key]))
        elif in_prev:
            if key in cur_all:
                out["uncertain"].append(_item(key, prev_conf[key], cur_all[key]))
            else:
                out["resolved"].append(_item(key, prev_conf[key], None))
        elif key in cur_all and key in prev_all:
            out["unchanged"].append(_item(key, prev_all[key], cur_all[key]))
        else:  # unconfirmed on both sides of a change
            item = _item(key, prev_all.get(key), cur_all.get(key))
            item["direction"] = "appeared" if key in cur_all else "disappeared"
            out["possible_changes"].append(item)
    return out
