"""Generate the synthetic sandbox (SYNTHETIC DEMO DATA): data/synthetic/registry.json and customers.json.

Owners, accidents, insurance claims, mileage and dealer-CRM customers live in police, insurer and dealer
systems we cannot access, so the demo uses this seeded, flagged sandbox instead (CLAUDE.md sections 2 and 5).

- Stdlib only, plus avtopassport's VIN check digit. Fixed seed: avtopassport.synthetic.SEED.
- Every draw is built on random.Random.random(), the one method Python promises to reproduce across
  versions for the same seed, so the output is byte-identical on every run.
- Every tunable parameter lives in ASSUMPTIONS with its basis ("assumption" or a source URL). The README
  blocks between <!-- BEGIN:name --> and <!-- END:name --> are rendered from ASSUMPTIONS and the generated data.

Usage:
    python scripts/generate_synthetic.py                 # data/synthetic/{registry,customers}.json + README tables
    python scripts/generate_synthetic.py --out-dir DIR   # same into DIR; DIR/README.md is rewritten only if present
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from avtopassport.synthetic import CUSTOMERS_JSON, REGISTRY_JSON, SEED, SOURCE_LABEL, SYNTHETIC_WMI  # noqa: E402
from avtopassport.vin.checkdigit import VIN_ALPHABET, compute_check_digit  # noqa: E402

T = TypeVar("T")

CRM_SOURCE_LABEL = "dealer CRM - sandbox"
BADGE = "SYNTHETIC DEMO DATA"
REGION_LABEL = "Tashkent region (synthetic)"
NOTICE = (
    "Generated sandbox data: not real registry, police, insurer or dealer records. Show the SYNTHETIC DEMO DATA "
    "badge wherever it is used and never attach it to a scanned VIN. Method: data/synthetic/README.md."
)
YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"  # VIN position 10 for model years 2010-2039 (49 CFR 565.15)
YEAR_CODE_BASE = 2010
DAYS_PER_YEAR = 365.25


def _assumed(value: Any, note: str) -> dict[str, Any]:
    return {"value": value, "basis": "assumption", "note": note}


def _sourced(value: Any, url: str, note: str) -> dict[str, Any]:
    return {"value": value, "basis": url, "note": note}


ASSUMPTIONS: dict[str, dict[str, Any]] = {
    # --- size and time window ---
    "n_vehicles": _assumed(40, "registry records, one synthetic VIN each"),
    "n_customers": _assumed(25, "dealer-CRM customer records"),
    "date_window": _assumed(("2016-01-01", "2026-09-30"), "every generated date falls inside this window"),
    "model_years": _assumed((2016, 2025), "model-year range (VIN position 10 codes G to S)"),
    # --- vehicles ---
    "make_share": _assumed({"Toyota": 0.5, "BYD": 0.5},
                           "share of vehicles per make; the model is then drawn uniformly from that make's line-up"),
    "lineup_toyota": _sourced(["Camry", "RAV4", "Land Cruiser Prado", "Land Cruiser 300"],
                              "https://uzbekistan.toyota-centralasia.com/ru/",
                              "models on sale (docs/research/phase0_findings_deploy_market.md section 6)"),
    "lineup_byd": _sourced(["Chazor", "Song Plus DM-i", "Seal 6", "Sealion 7", "Yuan Up", "e2", "Seagull"],
                           "https://bydauto.uz",
                           "models listed with UZS prices (docs/research/phase0_findings_deploy_market.md section 6)"),
    "first_model_year": _assumed(
        {"Song Plus DM-i": 2021, "e2": 2020, "Seagull": 2023, "Land Cruiser 300": 2022, "Chazor": 2024,
         "Yuan Up": 2024, "Seal 6": 2025, "Sealion 7": 2025},
        "earliest model year allowed per model, so the sandbox avoids obvious anachronisms; approximate, not "
        "verified against manufacturer data; other models start at the range start"),
    "colour_share": _assumed({"white": 0.30, "black": 0.20, "silver": 0.15, "grey": 0.15, "blue": 0.08, "red": 0.05,
                              "green": 0.04, "beige": 0.03},
                             "generic colour names; not market data"),
    # --- owners ---
    "ownership_years": _assumed((1.5, 7.0),
                                "holding period per owner, uniform; the last owner is the current one (to = null)"),
    # --- accidents and insurance claims ---
    "event_window_days": _assumed(30, "accidents and service visits: one Bernoulli trial per window, "
                                      "p = annual rate x window / 365.25"),
    "accident_rate_per_year": _assumed(0.12, "accidents per vehicle-year since first registration"),
    "severity_share": _assumed({"minor": 0.6, "moderate": 0.3, "severe": 0.1}, "accident severity"),
    "impact_zone_share": _assumed({"front": 0.4, "rear": 0.3, "left": 0.15, "right": 0.15},
                                  "side of the car that takes the impact"),
    "impact_zone_parts": _assumed(
        {
            "front": ["front_bumper", "grille", "hood", "windshield", "left_headlight", "right_headlight",
                      "left_fender", "right_fender"],
            "rear": ["back_bumper", "trunk", "back_windshield", "left_tail_light", "right_tail_light",
                     "left_quarter_panel", "right_quarter_panel"],
            "left": ["left_front_door", "left_back_door", "left_fender", "left_quarter_panel", "left_mirror",
                     "left_rocker_panel"],
            "right": ["right_front_door", "right_back_door", "right_fender", "right_quarter_panel", "right_mirror",
                      "right_rocker_panel"],
        },
        "parts an impact in each zone can damage; names follow the parts-model catalog (README, Sources)"),
    "parts_per_accident": _assumed({"minor": (1, 1), "moderate": (1, 2), "severe": (2, 4)},
                                   "damaged parts per accident, uniform, drawn without repetition from the zone"),
    "claim_probability": _assumed({"minor": 0.35, "moderate": 0.75, "severe": 0.95},
                                  "P(insurance claim) by accident severity"),
    "claim_delay_days": _assumed((1, 30), "days from accident to claim, uniform"),
    "claim_amount_per_part_uzs": _assumed(
        {"minor": (300_000, 1_500_000), "moderate": (1_000_000, 5_000_000), "severe": (4_000_000, 20_000_000)},
        "claimed amount = sum of one uniform draw per damaged part; synthetic, not price data"),
    "insurers": _assumed(["Sandbox Insurer A", "Sandbox Insurer B", "Sandbox Insurer C"],
                         "fictitious insurer labels, drawn uniformly"),
    "claim_open_window_days": _assumed(
        120, "only claims filed this many days or fewer before the window end can be open"),
    "claim_open_probability": _assumed(0.6, "P(open) for a claim inside the open window"),
    "claim_rejected_probability": _assumed(0.1, "P(rejected) for a closed claim; otherwise paid"),
    # --- mileage ---
    "odometer_at_registration_km": _assumed((5, 50), "first reading, at first registration"),
    "annual_km": _assumed((8_000, 25_000), "annual distance per vehicle, uniform"),
    "annual_km_variation": _assumed((0.8, 1.2), "multiplier on the annual distance per interval"),
    "mileage_interval_days": _assumed((300, 430), "days between odometer readings, uniform"),
    # --- dealer CRM ---
    "two_vehicle_customer_probability": _assumed(
        0.2, "P(customer has two registry vehicles); each VIN goes to at most one customer"),
    "tashkent_region_probability": _assumed(
        0.2, f"P(district = '{REGION_LABEL}'); otherwise a city district, uniform"),
    "tashkent_districts": _sourced(
        ["Bektemir", "Chilanzar", "Yashnobod", "Mirobod", "Mirzo Ulugbek", "Sergeli", "Shayxontoxur", "Olmazor",
         "Uchtepa", "Yakkasaray", "Yunusabad", "Yangihayot"],
        "https://en.wikipedia.org/w/index.php?title=Tashkent&oldid=1376716206",
        "the 12 Tashkent city districts as spelled in the district table; a coarse area, never an address"),
    "service_visits_per_year": _assumed(1.5, "dealer visits per vehicle-year since the current owner's 'from' date"),
    "visit_type_share": _assumed(
        {"maintenance": 0.6, "body_repair": 0.12, "warranty": 0.13, "accessories": 0.15},
        "visit type; warranty is dropped (and the rest renormalised) after the warranty period"),
    "warranty_years": _assumed(3, "warranty visits only within this many years of first registration"),
    "visit_amount_uzs": _assumed(
        {"maintenance": (600_000, 3_000_000), "body_repair": (1_000_000, 15_000_000), "warranty": (0, 0),
         "accessories": (300_000, 8_000_000)},
        "amount billed to the customer, uniform; warranty work is billed at 0; synthetic, not price data"),
    "amount_rounding_uzs": _assumed(10_000, "every UZS amount is rounded to this step"),
}

N_VEHICLES: int = ASSUMPTIONS["n_vehicles"]["value"]
N_CUSTOMERS: int = ASSUMPTIONS["n_customers"]["value"]

_V: dict[str, Any] = {name: spec["value"] for name, spec in ASSUMPTIONS.items()}
_LINEUPS: dict[str, list[str]] = {"Toyota": _V["lineup_toyota"], "BYD": _V["lineup_byd"]}
DATE_MIN, DATE_MAX = (date.fromisoformat(d) for d in _V["date_window"])


class _Rng:
    """Seeded draws built only on ``random.Random.random()``: the one method whose sequence Python keeps
    for a given seed across versions (https://docs.python.org/3/library/random.html#notes-on-reproducibility)."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed).random

    def chance(self, p: float) -> bool:
        return self._random() < p

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self._random()

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi]."""
        return min(hi, lo + int(self._random() * (hi - lo + 1)))

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.randint(0, len(seq) - 1)]

    def weighted(self, weights: Mapping[str, float]) -> str:
        x = self._random() * sum(weights.values())
        for key, weight in weights.items():
            x -= weight
            if x < 0:
                return key
        return list(weights)[-1]  # float round-off

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        """k distinct items in draw order (k >= len(seq) gives a shuffle)."""
        pool = list(seq)
        return [pool.pop(self.randint(0, len(pool) - 1)) for _ in range(min(k, len(pool)))]

    def date_between(self, lo: date, hi: date) -> date:
        return date.fromordinal(self.randint(lo.toordinal(), hi.toordinal()))


class _Ids:
    """Sequential IDs per kind: SYN-OWNER-0001, SYN-ACC-0001, ..."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def next(self, kind: str) -> str:
        self._counts[kind] += 1
        return f"SYN-{kind}-{self._counts[kind]:04d}"


def _check_assumptions() -> None:
    """Fail fast if a tuned parameter breaks an invariant the generator relies on."""
    first, last = _V["model_years"]
    if not (DATE_MIN.year <= first <= last <= DATE_MAX.year and YEAR_CODE_BASE <= first and last < YEAR_CODE_BASE + 30):
        raise ValueError("model_years must lie inside date_window and the 2010-2039 VIN year-code cycle")
    if set(_V["make_share"]) != set(_LINEUPS):
        raise ValueError("make_share keys must match the line-ups")
    models = {model for lineup in _LINEUPS.values() for model in lineup}
    if not set(_V["first_model_year"]) <= models or any(y > last for y in _V["first_model_year"].values()):
        raise ValueError("first_model_year must name line-up models and lie inside the model-year range")
    if N_VEHICLES < N_CUSTOMERS:
        raise ValueError("need at least one registry vehicle per customer")


def _round_uzs(amount: float) -> int:
    step = _V["amount_rounding_uzs"]
    return int(round(amount / step)) * step


def _event_dates(rng: _Rng, start: date, end: date, rate_per_year: float) -> list[date]:
    """Event days between start and end: one Bernoulli trial per window, p = rate x window length / 365.25."""
    days: list[date] = []
    lo, last = start.toordinal(), end.toordinal()
    while lo <= last:
        hi = min(lo + _V["event_window_days"] - 1, last)
        if rng.chance(rate_per_year * (hi - lo + 1) / DAYS_PER_YEAR):
            days.append(date.fromordinal(rng.randint(lo, hi)))
        lo = hi + 1
    return days


def _vin(rng: _Rng, model_year: int, seen: set[str]) -> str:
    """0AP + 5 VDS chars + check digit + model-year code + plant char + 6-digit serial; unique, never contains
    "998" (Uzbekistan's calling code) so phone-number scanners never flag the sandbox."""
    year_code = YEAR_CODES[model_year - YEAR_CODE_BASE]
    while True:
        vds = "".join(rng.choice(VIN_ALPHABET) for _ in range(5))
        body = f"{SYNTHETIC_WMI}{vds}0{year_code}{rng.choice(VIN_ALPHABET)}{rng.randint(0, 999_999):06d}"
        vin = body[:8] + compute_check_digit(body) + body[9:]
        if vin not in seen and "998" not in vin:
            seen.add(vin)
            return vin


def _owners(rng: _Rng, ids: _Ids, registered: date) -> list[dict[str, Any]]:
    owners: list[dict[str, Any]] = []
    start = registered
    while True:
        end = start + timedelta(days=round(rng.uniform(*_V["ownership_years"]) * DAYS_PER_YEAR))
        current = end >= DATE_MAX
        owners.append({
            "owner_id": ids.next("OWNER"),
            "from": start.isoformat(),
            "to": None if current else end.isoformat(),
        })
        if current:
            return owners
        start = end


def _accidents(rng: _Rng, ids: _Ids, registered: date) -> list[dict[str, Any]]:
    accidents: list[dict[str, Any]] = []
    for day in _event_dates(rng, registered, DATE_MAX, _V["accident_rate_per_year"]):
        severity = rng.weighted(_V["severity_share"])
        zone = rng.weighted(_V["impact_zone_share"])
        parts = sorted(rng.sample(_V["impact_zone_parts"][zone], rng.randint(*_V["parts_per_accident"][severity])))
        accidents.append({
            "accident_id": ids.next("ACC"),
            "date": day.isoformat(),
            "severity": severity,
            "parts": parts,
            "description": f"{severity.capitalize()} {zone} impact affecting {', '.join(parts)} (synthetic record)",
        })
    return accidents


def _claims(rng: _Rng, ids: _Ids, accidents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for accident in accidents:
        severity = accident["severity"]
        if not rng.chance(_V["claim_probability"][severity]):
            continue
        delay = timedelta(days=rng.randint(*_V["claim_delay_days"]))
        filed = min(DATE_MAX, date.fromisoformat(accident["date"]) + delay)
        amount = _round_uzs(sum(rng.uniform(*_V["claim_amount_per_part_uzs"][severity]) for _ in accident["parts"]))
        insurer = rng.choice(_V["insurers"])
        if (DATE_MAX - filed).days <= _V["claim_open_window_days"] and rng.chance(_V["claim_open_probability"]):
            status = "open"
        elif rng.chance(_V["claim_rejected_probability"]):
            status = "rejected"
        else:
            status = "paid"
        claims.append({
            "claim_id": ids.next("CLAIM"),
            "accident_id": accident["accident_id"],
            "date": filed.isoformat(),
            "insurer": insurer,
            "amount_uzs": amount,
            "status": status,
        })
    return claims


def _mileage(rng: _Rng, registered: date) -> list[dict[str, Any]]:
    annual = rng.uniform(*_V["annual_km"])
    km = float(rng.randint(*_V["odometer_at_registration_km"]))
    day = registered
    readings = [{"date": day.isoformat(), "km": int(km)}]
    while True:
        interval = rng.randint(*_V["mileage_interval_days"])
        day += timedelta(days=interval)
        if day > DATE_MAX:
            return readings
        km += annual * interval / DAYS_PER_YEAR * rng.uniform(*_V["annual_km_variation"])
        readings.append({"date": day.isoformat(), "km": int(round(km))})


def _vehicle(rng: _Rng, ids: _Ids, seen: set[str]) -> dict[str, Any]:
    make = rng.weighted(_V["make_share"])
    model = rng.choice(_LINEUPS[make])
    first, last = _V["model_years"]
    model_year = rng.randint(max(first, _V["first_model_year"].get(model, first)), last)
    vin = _vin(rng, model_year, seen)
    colour = rng.weighted(_V["colour_share"])
    registered = rng.date_between(max(DATE_MIN, date(model_year, 1, 1)), min(DATE_MAX, date(model_year, 12, 31)))
    owners = _owners(rng, ids, registered)
    accidents = _accidents(rng, ids, registered)
    claims = _claims(rng, ids, accidents)
    mileage = _mileage(rng, registered)
    return {
        "synthetic": True,
        "source": SOURCE_LABEL,
        "vin": vin,
        "make": make,
        "model": model,
        "model_year": model_year,
        "colour": colour,
        "owners": owners,
        "accidents": accidents,
        "insurance_claims": claims,
        "mileage_km": mileage,
    }


def _visits(rng: _Rng, vehicle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dealer visits of one vehicle during its current owner's period (the CRM customer)."""
    registered = date.fromisoformat(vehicle["owners"][0]["from"])
    since = date.fromisoformat(vehicle["owners"][-1]["from"])
    warranty_end = registered + timedelta(days=round(_V["warranty_years"] * DAYS_PER_YEAR))
    shares = _V["visit_type_share"]
    after_warranty = {kind: w for kind, w in shares.items() if kind != "warranty"}
    visits: list[dict[str, Any]] = []
    for day in _event_dates(rng, since, DATE_MAX, _V["service_visits_per_year"]):
        kind = rng.weighted(shares if day <= warranty_end else after_warranty)
        visits.append({
            "vin": vehicle["vin"],
            "date": day.isoformat(),
            "type": kind,
            "amount_uzs": _round_uzs(rng.uniform(*_V["visit_amount_uzs"][kind])),
        })
    return visits


def _customers(rng: _Rng, ids: _Ids, vehicles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_vin = {v["vin"]: v for v in vehicles}
    pool = rng.sample(list(by_vin), len(by_vin))  # shuffled; dealt out 1 or 2 per customer
    customers: list[dict[str, Any]] = []
    for i in range(1, N_CUSTOMERS + 1):
        wants_two = rng.chance(_V["two_vehicle_customer_probability"])
        spare = len(pool) - (N_CUSTOMERS - i + 1)  # VINs left over if every remaining customer gets one
        take = 2 if wants_two and spare >= 1 else 1
        vins = sorted(pool[:take])
        del pool[:take]
        in_region = rng.chance(_V["tashkent_region_probability"])
        district = REGION_LABEL if in_region else rng.choice(_V["tashkent_districts"])
        visits: list[dict[str, Any]] = []
        for vin in vins:
            visits += _visits(rng, by_vin[vin])
        visits.sort(key=lambda v: (v["date"], v["vin"]))
        for visit in visits:
            visit["visit_id"] = ids.next("VISIT")
        customers.append({
            "synthetic": True,
            "source": CRM_SOURCE_LABEL,
            "customer_id": f"SYN-CUST-{i:04d}",
            "display_name": f"Synthetic Customer {i:03d}",
            "email": f"cust{i:03d}@example.invalid",
            "district": district,
            "vins": vins,
            "service_visits": visits,
        })
    return customers


def _wrap(records: list[dict[str, Any]], source: str) -> dict[str, Any]:
    return {
        "synthetic": True,
        "badge": BADGE,
        "source": source,
        "notice": NOTICE,
        "generator": "scripts/generate_synthetic.py",
        "seed": SEED,
        "records": records,
    }


def generate() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (registry document, customers document) from SEED. Same output on every call."""
    _check_assumptions()
    rng, ids, seen = _Rng(SEED), _Ids(), set()
    vehicles = [_vehicle(rng, ids, seen) for _ in range(N_VEHICLES)]
    customers = _customers(rng, ids, vehicles)
    return _wrap(vehicles, SOURCE_LABEL), _wrap(customers, CRM_SOURCE_LABEL)


# --- README rendering ---

_YEAR_VALUED = {"model_years", "first_model_year"}  # ints shown without thousands separators


def _fmt(value: Any, *, group: bool = True) -> str:
    """Markdown cell text: tuples are ranges, lists are comma lists, dicts are 'key: value; ...'."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_fmt(v, group=group)}" for k, v in value.items())
    if isinstance(value, tuple):
        if len(set(value)) == 1:
            return _fmt(value[0], group=group)
        return " to ".join(_fmt(v, group=group) for v in value)
    if isinstance(value, list):
        return ", ".join(_fmt(v, group=group) for v in value)
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}" if group else str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _tally(items: Iterable[Any]) -> str:
    return ", ".join(f"{key}: {n}" for key, n in sorted(Counter(items).items(), key=lambda kv: str(kv[0])))


def render_assumptions() -> str:
    """Markdown table of ASSUMPTIONS (parameter, value, basis, meaning)."""
    lines = ["| Parameter | Value | Basis | Meaning |", "|---|---|---|---|"]
    for name, spec in ASSUMPTIONS.items():
        value = _fmt(spec["value"], group=name not in _YEAR_VALUED)
        lines.append(f"| `{name}` | {_cell(value)} | {_cell(spec['basis'])} | {_cell(spec['note'])} |")
    return "\n".join(lines)


def render_summary(registry: Mapping[str, Any], customers: Mapping[str, Any]) -> str:
    """Markdown table of counts computed from the generated documents."""
    vehicles, clients = registry["records"], customers["records"]
    years = [v["model_year"] for v in vehicles]
    accidents = [a for v in vehicles for a in v["accidents"]]
    claims = [c for v in vehicles for c in v["insurance_claims"]]
    visits = [x for c in clients for x in c["service_visits"]]
    areas = ["Tashkent region" if c["district"] == REGION_LABEL else "Tashkent city" for c in clients]
    rows = [
        ("Seed", str(registry["seed"])),
        ("Registry vehicles", f"{len(vehicles)} ({_tally(v['make'] for v in vehicles)})"),
        ("Models", _tally(f"{v['make']} {v['model']}" for v in vehicles)),
        ("Model years", f"{min(years)} to {max(years)}"),
        ("Owners", f"{sum(len(v['owners']) for v in vehicles)}; vehicles by owner count: "
                   + _tally(f"{len(v['owners'])} owner{'s' if len(v['owners']) > 1 else ''}" for v in vehicles)),
        ("Accidents", f"{len(accidents)} ({_tally(a['severity'] for a in accidents)}) on "
                      f"{sum(1 for v in vehicles if v['accidents'])} vehicles"),
        ("Insurance claims", f"{len(claims)} ({_tally(c['status'] for c in claims)})"),
        ("Mileage readings", str(sum(len(v["mileage_km"]) for v in vehicles))),
        ("CRM customers", f"{len(clients)} ({_tally(areas)})"),
        ("Registry VINs linked to a customer", f"{sum(len(c['vins']) for c in clients)} of {len(vehicles)}"),
        ("Service visits", f"{len(visits)} ({_tally(x['type'] for x in visits)})"),
    ]
    return "\n".join(["| Item | Value |", "|---|---|", *(f"| {k} | {_cell(v)} |" for k, v in rows)])


def _replace_block(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < i:
        raise ValueError(f"README has no {begin} ... {end} block")
    return f"{text[: i + len(begin)]}\n{body.strip()}\n{text[j:]}"


def _dumps(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_outputs(out_dir: Path, readme: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Write registry.json, customers.json and (if given) the README blocks. Nothing is written if the README
    lacks its markers. Returns (registry, customers, written paths)."""
    registry, customers = generate()
    readme_text = None
    if readme is not None:
        readme_text = readme.read_text(encoding="utf-8")
        readme_text = _replace_block(readme_text, "assumptions", render_assumptions())
        readme_text = _replace_block(readme_text, "summary", render_summary(registry, customers))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [(out_dir / REGISTRY_JSON.name, _dumps(registry)), (out_dir / CUSTOMERS_JSON.name, _dumps(customers))]
    if readme is not None and readme_text is not None:
        outputs.append((readme, readme_text))
    for path, text in outputs:
        path.write_text(text, encoding="utf-8", newline="\n")
    return registry, customers, [path for path, _ in outputs]


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic sandbox (SYNTHETIC DEMO DATA).")
    parser.add_argument("--out-dir", type=Path, default=REGISTRY_JSON.parent,
                        help="output directory (default: data/synthetic); its README.md is rewritten if present")
    args = parser.parse_args(argv)
    readme = args.out_dir / "README.md"
    try:
        registry, customers, written = write_outputs(args.out_dir, readme if readme.is_file() else None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    vehicles, clients = registry["records"], customers["records"]
    print(f"seed {SEED}: {len(vehicles)} vehicles, {sum(len(v['owners']) for v in vehicles)} owners, "
          f"{sum(len(v['accidents']) for v in vehicles)} accidents, "
          f"{sum(len(v['insurance_claims']) for v in vehicles)} insurance claims, "
          f"{sum(len(v['mileage_km']) for v in vehicles)} mileage readings; "
          f"{len(clients)} customers, {sum(len(c['vins']) for c in clients)} linked VINs, "
          f"{sum(len(c['service_visits']) for c in clients)} service visits")
    for path in written:
        print(f"wrote {_shown(path)}")
    if not readme.is_file():
        print(f"no README.md in {_shown(args.out_dir)}: tables not rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
