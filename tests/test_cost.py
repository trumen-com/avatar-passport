"""cost.estimator on synthetic price rows (plus a schema check of the real reference CSV)."""

from __future__ import annotations

from typing import Any

import pytest

from avtopassport.cost.estimator import (
    EXCLUDED_CATEGORIES,
    USABLE_CATEGORIES,
    estimate,
    load_prices,
    service_for,
)
from avtopassport.schema import DamageFinding


def _row(category: str, price: int, *, part: str = "any", kind: str = "from", url: str = "https://www.shop-a.uz/x",
         date: str = "2026-10-01", estimate_flag: bool = False) -> dict[str, Any]:
    return {"service": category, "service_category": category, "part": part, "price_uzs": price,
            "price_kind": kind, "source_url": url, "listing_date": date, "retrieved_on": "2026-10-10",
            "is_estimate": estimate_flag}


PRICES = [
    _row("paint", 300_000),
    _row("paint", 700_000, part="panel", url="https://olx.uz/a", date="2026-09-25"),
    _row("paint", 500_000, part="panel", url="https://olx.uz/b", date="2026-10-03"),
    _row("dent", 400_000),
    _row("scratch", 250_000),
    _row("polish", 800_000),
    _row("polish", 100_000, part="whole_body", kind="flat"),       # whole-body price: not per part
    _row("panel_replacement", 1_500_000, part="panel"),
    _row("inspection", 9_000_000),
    _row("promo", 9_000_000, kind="promo"),
    _row("unspecified", 9_000_000, kind="flat"),
]


def _f(part: str, damage: str, action: str | None, severity: str = "moderate") -> DamageFinding:
    return DamageFinding(part=part, damage=damage, n_observations=1, sources=["s"], best_source="s",
                         max_score=0.9, max_area_ratio=0.1, severity=severity, action=action)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("part", "damage", "action", "category", "price"),
    [
        ("hood", "scratch", "polish", "polish", 800_000),
        ("hood", "scratch", "paint", "paint", 500_000),
        ("front_door", "dent", "repair", "dent", 400_000),
        ("front_door", "scratch", "repair", "scratch", 250_000),
        ("front_bumper", "crack", "repair", "paint", 500_000),
        ("front_bumper", "rust", "repair", "dent", 400_000),
        ("trunk", "dent", "replace", "panel_replacement", 1_500_000),
    ],
)
def test_action_mapping(part: str, damage: str, action: str, category: str, price: int) -> None:
    lines, total = estimate([_f(part, damage, action)], PRICES)
    assert len(lines) == 1
    line = lines[0]
    assert service_for(_f(part, damage, action))[0] == category
    assert line.price_uzs == price
    assert line.price_kind == "from"
    assert line.basis.startswith(f"repair_prices.csv: {category} ")
    assert line.is_estimate is False
    assert total == price


@pytest.mark.parametrize(
    ("part", "damage", "action", "service"),
    [
        ("front_glass", "glass shatter", "replace", "glass replacement"),
        ("front_light", "lamp broken", "replace", "lamp replacement"),
        ("wheel", "tire flat", "repair", "tyre repair or replacement"),
        ("front_glass", "crack", "replace", "replace of glass"),
        ("left_mirror", "dent", "replace", "replacement of left_mirror"),
    ],
)
def test_no_price_rows(part: str, damage: str, action: str, service: str) -> None:
    lines, total = estimate([_f(part, damage, action)], PRICES)
    assert lines[0].price_uzs is None and lines[0].price_kind is None
    assert lines[0].basis.startswith(f"no observed price for {service}")
    assert total is None


def test_median_and_sources() -> None:
    (line,), _ = estimate([_f("hood", "scratch", "paint")], PRICES)
    assert line.price_uzs == 500_000                       # median of 300k, 500k, 700k
    assert "median of 3 rows" in line.basis
    for src in ("shop-a.uz 2026-10-01", "olx.uz 2026-09-25", "olx.uz 2026-10-03"):
        assert src in line.basis


def test_even_median_rounds_half_up() -> None:
    rows = [_row("dent", 100_001), _row("dent", 100_002)]
    (line,), total = estimate([_f("hood", "dent", "repair")], rows)
    assert line.price_uzs == 100_002                      # (100001 + 100002) / 2 = 100001.5 -> 100002
    assert total == 100_002


def test_excluded_categories_and_whole_body_ignored() -> None:
    only_excluded = [r for r in PRICES if r["service_category"] in EXCLUDED_CATEGORIES]
    lines, total = estimate([_f("hood", "scratch", "polish"), _f("hood", "dent", "repair")], only_excluded)
    assert all(ln.price_uzs is None for ln in lines) and total is None
    (polish,), _ = estimate([_f("hood", "scratch", "polish")], PRICES)
    assert polish.price_uzs == 800_000 and "1 row" in polish.basis


def test_is_estimate_propagates() -> None:
    rows = [_row("scratch", 200_000), _row("scratch", 260_000, estimate_flag=True)]
    (line,), _ = estimate([_f("hood", "scratch", "repair")], rows)
    assert line.is_estimate is True and line.price_uzs == 230_000


def test_total_skips_unpriced_and_unrated() -> None:
    findings = [_f("hood", "scratch", "polish"), _f("front_glass", "glass shatter", "replace"),
                _f("trunk", "dent", "replace"), _f("roof", "dent", None)]
    lines, total = estimate(findings, PRICES)
    assert len(lines) == 3                                # the unrated finding gets no line
    assert total == 800_000 + 1_500_000


def test_empty_findings() -> None:
    assert estimate([], PRICES) == ([], None)


def test_reference_csv_loads() -> None:
    rows = load_prices()
    assert rows
    for r in rows:
        assert r["service_category"] in USABLE_CATEGORIES | EXCLUDED_CATEGORIES
        assert isinstance(r["price_uzs"], int) and r["price_uzs"] > 0
        assert isinstance(r["is_estimate"], bool)
    lines, _ = estimate([_f("hood", "scratch", "paint")])
    assert lines[0].price_uzs is not None
    assert lines[0].price_uzs in {r["price_uzs"] for r in rows} or "median" in lines[0].basis
