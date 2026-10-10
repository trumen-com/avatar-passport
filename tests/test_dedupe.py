"""dedupe.merge on hand-built observations (CPU, offline, no data/raw)."""

from __future__ import annotations

import pytest

from avtopassport.pipeline.dedupe import merge
from avtopassport.schema import DamageObservation


def obs(source: str, part: str, damage: str, score: float, ratio: float) -> DamageObservation:
    return DamageObservation(
        source=source, damage=damage, score=score, part=part, part_overlap=1.0,
        damage_area_px=100, part_area_px=0 if part == "unknown" else 1000, area_ratio=ratio,
        box=(0.0, 0.0, 10.0, 10.0),
    )


def test_groups_by_part_and_damage_sorted() -> None:
    observations = [
        obs("f3", "rear_bumper", "scratch", 0.5, 0.01),
        obs("f1", "hood", "dent", 0.6, 0.03),
        obs("f2", "hood", "scratch", 0.7, 0.02),
        obs("f4", "hood", "dent", 0.8, 0.01),
    ]
    findings = merge(observations)
    assert [(f.part, f.damage) for f in findings] == [
        ("hood", "dent"), ("hood", "scratch"), ("rear_bumper", "scratch"),
    ]
    hood_dent = findings[0]
    assert hood_dent.n_observations == 2
    assert hood_dent.sources == ["f1", "f4"]
    assert hood_dent.best_source == "f4" and hood_dent.max_score == pytest.approx(0.8)
    assert hood_dent.max_area_ratio == pytest.approx(0.03)   # max ratio need not come from the best score
    assert all(f.severity is None and f.action is None and f.severity_rule is None for f in findings)


def test_sources_unique_in_input_order() -> None:
    observations = [
        obs("f9", "hood", "scratch", 0.4, 0.01),
        obs("f2", "hood", "scratch", 0.9, 0.02),
        obs("f9", "hood", "scratch", 0.5, 0.05),   # same frame seen twice (two instances)
        obs("f1", "hood", "scratch", 0.3, 0.01),
    ]
    (f,) = merge(observations)
    assert f.n_observations == 4
    assert f.sources == ["f9", "f2", "f1"]
    assert f.best_source == "f2"
    assert f.max_area_ratio == pytest.approx(0.05)


def test_best_source_tie_is_first_seen() -> None:
    (f,) = merge([obs("a", "hood", "dent", 0.7, 0.0), obs("b", "hood", "dent", 0.7, 0.0)])
    assert f.best_source == "a"


def test_unknown_part_kept_and_empty_input() -> None:
    (f,) = merge([obs("x", "unknown", "crack", 0.6, 0.0)])
    assert f.part == "unknown" and f.max_area_ratio == 0.0
    assert merge([]) == []


def test_same_damage_on_different_parts_not_merged() -> None:
    findings = merge([obs("a", "left_front-door", "dent", 0.6, 0.1), obs("a", "left_rear-door", "dent", 0.6, 0.1)])
    assert [f.part for f in findings] == ["left_front-door", "left_rear-door"]
