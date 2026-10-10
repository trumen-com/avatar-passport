"""Severity and recommended action for a merged finding.

Severity comes from ``max_area_ratio`` (fraction of the part covered by the damage):
``< MINOR_MAX`` minor, ``< MODERATE_MAX`` moderate, otherwise severe. Both cut-offs are UNCALIBRATED
defaults (assumption), to be calibrated on the local test set in Phase 2. When the part is "unknown"
(no area ratio) the severity is the damage-type default in ``UNKNOWN_PART_SEVERITY`` (assumption: "moderate").

Action table ``ACTIONS[(damage, severity)]`` (assumption, not from a body-shop source):

    damage          minor    moderate  severe
    scratch         polish   paint     paint
    dent            repair   repair    replace
    crack           repair   replace   replace
    glass shatter   replace  replace   replace
    lamp broken     replace  replace   replace
    tire flat       repair   repair    replace

Damage classes not in the table get ``FALLBACK_ACTION`` ("repair") and the rule text says so.
Class names are matched case-insensitively with "_", "-" and " " treated the same.
"""

from __future__ import annotations

import dataclasses
import math
import re

from ..schema import Action, DamageFinding, Severity
from .mapping import UNKNOWN_PART

MINOR_MAX = 0.02      # uncalibrated default (assumption): area ratio below this -> minor
MODERATE_MAX = 0.10   # uncalibrated default (assumption): area ratio below this -> moderate, else severe

UNKNOWN_PART_DEFAULT: Severity = "moderate"   # assumption: severity when the part (and ratio) is unknown
UNKNOWN_PART_SEVERITY: dict[str, Severity] = {   # assumption: per damage type, all "moderate" for now
    "scratch": "moderate",
    "dent": "moderate",
    "crack": "moderate",
    "glass shatter": "moderate",
    "lamp broken": "moderate",
    "tire flat": "moderate",
}

FALLBACK_ACTION: Action = "repair"   # assumption: action for damage classes missing from ACTIONS

ACTIONS: dict[tuple[str, Severity], Action] = {   # assumption (see module docstring)
    ("scratch", "minor"): "polish",
    ("scratch", "moderate"): "paint",
    ("scratch", "severe"): "paint",
    ("dent", "minor"): "repair",
    ("dent", "moderate"): "repair",
    ("dent", "severe"): "replace",
    ("crack", "minor"): "repair",
    ("crack", "moderate"): "replace",
    ("crack", "severe"): "replace",
    ("glass shatter", "minor"): "replace",
    ("glass shatter", "moderate"): "replace",
    ("glass shatter", "severe"): "replace",
    ("lamp broken", "minor"): "replace",
    ("lamp broken", "moderate"): "replace",
    ("lamp broken", "severe"): "replace",
    ("tire flat", "minor"): "repair",
    ("tire flat", "moderate"): "repair",
    ("tire flat", "severe"): "replace",
}

_SEP = re.compile(r"[\s_\-]+")


def normalise_class(name: str) -> str:
    """Lower-case, "_" / "-" / whitespace runs -> one space: "Glass_Shatter" -> "glass shatter"."""
    return _SEP.sub(" ", name).strip().lower()


def _pct(x: float) -> str:
    """Threshold as a percentage without trailing zeros: 0.02 -> "2%", 0.025 -> "2.5%"."""
    return f"{x * 100:g}%"


def _ratio_pct(ratio: float, severity: Severity) -> str:
    """Ratio as "4.1%"; more decimals when rounding would cross a band edge (0.0199 -> "1.99%", not "2.0%")."""
    for digits in (1, 2, 3, 4):
        s = f"{ratio * 100:.{digits}f}"
        if severity_from_ratio(float(s) / 100)[0] == severity:
            break
    return f"{s}%"


def severity_from_ratio(ratio: float) -> tuple[Severity, str]:
    """(severity, band text) for a damage/part area ratio, e.g. ("moderate", ">= 2%, < 10%")."""
    if ratio < MINOR_MAX:
        return "minor", f"< {_pct(MINOR_MAX)}"
    if ratio < MODERATE_MAX:
        return "moderate", f">= {_pct(MINOR_MAX)}, < {_pct(MODERATE_MAX)}"
    return "severe", f">= {_pct(MODERATE_MAX)}"


def rate(finding: DamageFinding) -> DamageFinding:
    """Return a copy of ``finding`` with severity, action and severity_rule set (input is not mutated)."""
    dmg = normalise_class(finding.damage)
    ratio = finding.max_area_ratio
    if finding.part == UNKNOWN_PART or not math.isfinite(ratio) or ratio < 0:
        severity = UNKNOWN_PART_SEVERITY.get(dmg, UNKNOWN_PART_DEFAULT)
        where = "unknown part (no part overlapped enough)" if finding.part == UNKNOWN_PART else (
            f"{finding.part} (area ratio {ratio!r} not usable)"
        )
        cause = f"{finding.damage} on {where} -> {severity} (damage-type default, assumption)"
    else:
        severity, band = severity_from_ratio(ratio)
        cause = f"{finding.damage} covering {_ratio_pct(ratio, severity)} of {finding.part} ({band}) -> {severity}"
    action = ACTIONS.get((dmg, severity))
    if action is None:
        action = FALLBACK_ACTION
        rule = f"{cause} -> {action} (fallback: no action rule for damage class {finding.damage!r})"
    else:
        rule = f"{cause} -> {action}"
    return dataclasses.replace(finding, severity=severity, action=action, severity_rule=rule)


def rate_all(findings: list[DamageFinding]) -> list[DamageFinding]:
    """``rate`` applied to each finding, same order."""
    return [rate(f) for f in findings]
