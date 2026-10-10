"""Repair cost lines from observed Tashkent prices (data/reference/repair_prices.csv).

Rules (no invented numbers: every price is an observed CSV row or the median of observed rows)
- Only rows with service_category in USABLE_CATEGORIES are used; "inspection", "promo" and "unspecified"
  rows are context, not repair costs (see scripts/build_repair_prices.py).
- Only rows priced per part (``part`` "any" or "panel") are used. "whole_body" rows price the whole car body
  and "wheels" rows are not body work, so neither is a price for repairing one part.
- action -> service category:
    polish  -> polish
    paint   -> paint
    repair  -> dent for dents, scratch for scratches, paint for cracks (a body crack is filled and
               repainted), dent for any other body damage
    replace -> panel_replacement, only for body panels (not glass, lights, wheels, mirrors)
- Glass shatter, broken lamps, flat tyres and any damage on glass / lights / wheels have no price row:
  price_uzs is None and basis says "no observed price for <service>".
- Several matching rows -> the MEDIAN of their price_uzs (statistics.median; with an even count it is the
  mean of the two middle observed prices, rounded half up to whole UZS). Basis lists every row used.
- price_kind: the shared kind of the rows used; "from" when any used row is a "from" (lower-bound) price.
- is_estimate is True only when a used row has is_estimate true.
- total = sum of the known price_uzs (a "from" total); None when no line has a price.
"""

from __future__ import annotations

import csv
import logging
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..paths import REFERENCE
from ..schema import Action, CostLine, DamageFinding

log = logging.getLogger("avtopassport.cost.estimator")

PRICES_CSV = REFERENCE / "repair_prices.csv"
USABLE_CATEGORIES = frozenset({"paint", "dent", "polish", "scratch", "panel_replacement"})
EXCLUDED_CATEGORIES = frozenset({"inspection", "promo", "unspecified"})
PER_PART_SCOPES = frozenset({"any", "panel"})

# Damage types that are never body work -> no price row exists (service name used in the basis text).
NO_PRICE_DAMAGE: dict[str, str] = {
    "glass shatter": "glass replacement",
    "lamp broken": "lamp replacement",
    "tire flat": "tyre repair or replacement",
}
# Part-name fragments that are not paintable body panels (carparts-seg names such as front_glass, back_light).
_NON_BODY_PARTS: dict[str, str] = {
    "glass": "glass",
    "windshield": "glass",
    "light": "lamp",
    "lamp": "lamp",
    "wheel": "wheel",
    "tire": "tyre",
    "tyre": "tyre",
}
_NOT_A_PANEL = ("mirror", "object")

_ACTION_CATEGORY: dict[str, str] = {"polish": "polish", "paint": "paint", "replace": "panel_replacement"}
_REPAIR_CATEGORY: dict[str, str] = {"dent": "dent", "scratch": "scratch", "crack": "paint"}


def _norm(text: str | None) -> str:
    return (text or "").strip().lower().replace("_", " ").replace("-", " ")


def load_prices(path: Path = PRICES_CSV) -> list[dict[str, Any]]:
    """Read the price CSV; ``price_uzs`` becomes int (None if empty) and ``is_estimate`` bool."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            try:
                row["price_uzs"] = int(row.get("price_uzs") or "")
            except ValueError:
                row["price_uzs"] = None
            row["is_estimate"] = str(row.get("is_estimate", "")).lower() in {"true", "1", "yes"}
            rows.append(row)
    return rows


def _usable(row: dict[str, Any]) -> bool:
    return (
        row.get("service_category") in USABLE_CATEGORIES
        and _norm(row.get("part")) in PER_PART_SCOPES
        and isinstance(row.get("price_uzs"), int)
        and row["price_uzs"] > 0
    )


def _non_body(part: str) -> str | None:
    p = _norm(part)
    for frag, label in _NON_BODY_PARTS.items():
        if frag in p:
            return label
    return None


def service_for(finding: DamageFinding) -> tuple[str | None, str]:
    """(price category or None, service description) for a rated finding."""
    damage = _norm(finding.damage)
    action = finding.action
    part = finding.part or "unknown"
    if damage in NO_PRICE_DAMAGE:
        return None, NO_PRICE_DAMAGE[damage]
    non_body = _non_body(part)
    if non_body:
        return None, f"{action} of {non_body} ({part})"
    if action == "repair":
        category = _REPAIR_CATEGORY.get(damage, "dent")
        return category, f"repair of {damage} -> {category}"
    if action == "replace":
        if any(word in _norm(part) for word in _NOT_A_PANEL):
            return None, f"replacement of {part}"
        return "panel_replacement", "panel replacement"
    if action in _ACTION_CATEGORY:
        return _ACTION_CATEGORY[action], _ACTION_CATEGORY[action]
    return None, f"action {action!r}"


def _source_label(row: dict[str, Any]) -> str:
    url = str(row.get("source_url", ""))
    if url.startswith("phone:"):
        where = url.split(",")[0]
    else:
        where = urlparse(url).netloc.removeprefix("www.") or url or "unknown source"
    date = row.get("listing_date") or (f"retrieved {row['retrieved_on']}" if row.get("retrieved_on") else "no date")
    return f"{where} {date}"


def _median_uzs(values: list[int]) -> int:
    med = Decimal(str(statistics.median(values)))
    return int(med.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _line(finding: DamageFinding, action: Action, rows: list[dict[str, Any]]) -> CostLine:
    category, service = service_for(finding)
    no_price = CostLine(finding.part, finding.damage, action, None, None, "", False)
    if category is None:
        no_price.basis = f"no observed price for {service}"
        return no_price
    used = [r for r in rows if r.get("service_category") == category]
    if not used:
        no_price.basis = f"no observed price for {service} (repair_prices.csv has no usable {category} row)"
        return no_price
    kinds = sorted({str(r.get("price_kind") or "unknown") for r in used})
    kind = "from" if "from" in kinds else kinds[0]
    price = _median_uzs([int(r["price_uzs"]) for r in used])
    sources = "; ".join(sorted(_source_label(r) for r in used))
    how = "1 row" if len(used) == 1 else f"median of {len(used)} rows"
    basis = f"repair_prices.csv: {category} ({'/'.join(kinds)}), {how}: {sources}"
    return CostLine(
        part=finding.part,
        damage=finding.damage,
        action=action,
        price_uzs=price,
        price_kind=kind,
        basis=basis,
        is_estimate=any(bool(r.get("is_estimate")) for r in used),
    )


def estimate(findings: list[DamageFinding], prices: list[dict[str, Any]] | None = None
             ) -> tuple[list[CostLine], int | None]:
    """One CostLine per rated finding (findings without an action are skipped) and the "from" total."""
    rows = [r for r in (load_prices() if prices is None else prices) if _usable(r)]
    lines: list[CostLine] = []
    for f in findings:
        if f.action is None:
            log.warning("finding %s/%s has no action; no cost line", f.part, f.damage)
            continue
        lines.append(_line(f, f.action, rows))
    known = [ln.price_uzs for ln in lines if ln.price_uzs is not None]
    total = sum(known) if known else None
    return lines, total
