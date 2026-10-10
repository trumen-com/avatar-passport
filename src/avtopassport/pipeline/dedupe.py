"""Merge per-frame damage observations into one finding per (part, damage type).

The same scratch seen in several walkaround frames becomes one finding. Grouping is on the exact
(part, damage) strings as produced by mapping.assign; severity/action are left None for severity.rate.
"""

from __future__ import annotations

from ..schema import DamageFinding, DamageObservation


def merge(observations: list[DamageObservation]) -> list[DamageFinding]:
    """Group by (part, damage). sources are unique in input order; best_source is the source of the
    highest-score observation (ties: first seen). Findings are sorted by (part, damage)."""
    groups: dict[tuple[str, str], list[DamageObservation]] = {}
    for o in observations:
        groups.setdefault((o.part, o.damage), []).append(o)
    findings: list[DamageFinding] = []
    for (part, damage), obs in sorted(groups.items()):
        best = max(obs, key=lambda o: o.score)  # max() keeps the first of equal scores
        findings.append(
            DamageFinding(
                part=part,
                damage=damage,
                n_observations=len(obs),
                sources=list(dict.fromkeys(o.source for o in obs)),
                best_source=best.source,
                max_score=float(best.score),
                max_area_ratio=float(max(o.area_ratio for o in obs)),
            )
        )
    return findings
