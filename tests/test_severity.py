"""severity.rate on hand-built findings (CPU, offline, no data/raw)."""

from __future__ import annotations

import copy

import pytest

from avtopassport.pipeline.severity import (
    ACTIONS,
    FALLBACK_ACTION,
    MINOR_MAX,
    MODERATE_MAX,
    normalise_class,
    rate,
    rate_all,
)
from avtopassport.schema import DamageFinding


def finding(damage: str, ratio: float, part: str = "left_front-door") -> DamageFinding:
    return DamageFinding(
        part=part, damage=damage, n_observations=1, sources=["f0"], best_source="f0",
        max_score=0.9, max_area_ratio=ratio,
    )


def test_constants() -> None:
    assert MINOR_MAX == 0.02 and MODERATE_MAX == 0.10


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "minor"),
        (0.0199, "minor"),
        (MINOR_MAX, "moderate"),        # boundary belongs to the upper band
        (0.0999, "moderate"),
        (MODERATE_MAX, "severe"),
        (0.5, "severe"),
        (1.7, "severe"),                # ratio > 1 is possible when a part mask is cut off
    ],
)
def test_severity_boundaries(ratio: float, expected: str) -> None:
    assert rate(finding("dent", ratio)).severity == expected


def test_rule_text_example() -> None:
    r = rate(finding("dent", 0.041))
    assert r.severity_rule == "dent covering 4.1% of left_front-door (>= 2%, < 10%) -> moderate -> repair"
    assert rate(finding("scratch", 0.01)).severity_rule == "scratch covering 1.0% of left_front-door (< 2%) -> minor -> polish"
    assert rate(finding("dent", 0.1)).severity_rule.endswith("(>= 10%) -> severe -> replace")


def test_rule_text_does_not_round_across_a_band_edge() -> None:
    assert "1.99% of" in rate(finding("dent", 0.0199)).severity_rule


@pytest.mark.parametrize(("key", "action"), sorted(ACTIONS.items()))
def test_every_action_entry(key: tuple[str, str], action: str) -> None:
    damage, severity = key
    ratio = {"minor": 0.0, "moderate": MINOR_MAX, "severe": MODERATE_MAX}[severity]
    r = rate(finding(damage, ratio))
    assert (r.severity, r.action) == (severity, action)
    assert "fallback" not in r.severity_rule


def test_action_table_contents() -> None:
    expected = {
        "scratch": ("polish", "paint", "paint"),
        "dent": ("repair", "repair", "replace"),
        "crack": ("repair", "replace", "replace"),
        "glass shatter": ("replace", "replace", "replace"),
        "lamp broken": ("replace", "replace", "replace"),
        "tire flat": ("repair", "repair", "replace"),
    }
    table = {(d, s): a for d, acts in expected.items() for s, a in zip(("minor", "moderate", "severe"), acts)}
    assert ACTIONS == table


@pytest.mark.parametrize(
    "name", ["glass shatter", "Glass_Shatter", "glass-shatter", "GLASS  SHATTER", " glass__shatter "]
)
def test_name_normalisation(name: str) -> None:
    assert normalise_class(name) == "glass shatter"
    r = rate(finding(name, 0.0))
    assert r.action == "replace" and "fallback" not in r.severity_rule
    assert r.damage == name   # the finding keeps the model's class name


def test_tire_flat_and_lamp_broken_variants() -> None:
    assert rate(finding("Tire_Flat", 0.2)).action == "replace"
    assert rate(finding("lamp-broken", 0.0)).action == "replace"


def test_unknown_part_uses_damage_type_default() -> None:
    r = rate(finding("dent", 0.0, part="unknown"))
    assert r.severity == "moderate" and r.action == "repair"
    assert "unknown" in r.severity_rule and "damage-type default" in r.severity_rule
    assert rate(finding("scratch", 0.0, part="unknown")).action == "paint"
    assert rate(finding("crack", 0.0, part="unknown")).action == "replace"


def test_non_finite_ratio_uses_default() -> None:
    r = rate(finding("dent", float("nan")))
    assert r.severity == "moderate" and "not usable" in r.severity_rule


def test_unknown_damage_class_falls_back_to_repair() -> None:
    r = rate(finding("rust", 0.5))
    assert r.severity == "severe" and r.action == FALLBACK_ACTION == "repair"
    assert "fallback" in r.severity_rule and "'rust'" in r.severity_rule
    u = rate(finding("rust", 0.0, part="unknown"))
    assert u.severity == "moderate" and u.action == "repair" and "fallback" in u.severity_rule


def test_rate_does_not_mutate() -> None:
    f = finding("dent", 0.05)
    before = copy.deepcopy(f)
    r = rate(f)
    assert f == before and f.severity is None and f.action is None and f.severity_rule is None
    assert r is not f and r.severity == "moderate"
    assert r.sources == f.sources and r.n_observations == f.n_observations


def test_rate_all_keeps_order() -> None:
    fs = [finding("scratch", 0.0, part="hood"), finding("dent", 0.5, part="door"), finding("crack", 0.05)]
    out = rate_all(fs)
    assert [(r.part, r.action) for r in out] == [("hood", "polish"), ("door", "replace"), ("left_front-door", "replace")]
    assert rate_all([]) == []
