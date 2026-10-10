"""Synthetic sandbox: deterministic generator, privacy of the generated data, and the scan-refusal rule.

Synthetic inputs only. The generator always writes into tmp_path; the committed data/synthetic files are only read.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import pytest

from avtopassport.paths import REPO_ROOT, SYNTHETIC
from avtopassport.synthetic import (
    CUSTOMERS_JSON,
    REGISTRY_JSON,
    SEED,
    SOURCE_LABEL,
    RegistryRefusal,
    is_synthetic_vin,
    load_registry,
    lookup,
)
from avtopassport.vin.checkdigit import check_digit_ok, compute_check_digit

SCRIPT = REPO_ROOT / "scripts" / "generate_synthetic.py"
REFUSAL = "synthetic registry is never attached to a scanned VIN"
CRM_SOURCE = "dealer CRM - sandbox"
YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"  # position 10, model years 2010-2039
DATE_MIN, DATE_MAX = date(2016, 1, 1), date(2026, 9, 30)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
PLATE_PATTERNS = (  # Uzbek plate layouts: "01 A 123 BC" (private) and "01 123 ABC" (legal entities)
    re.compile(r"\b\d{2}\s?[A-Z]\s?\d{3}\s?[A-Z]{2}\b", re.IGNORECASE),
    re.compile(r"\b\d{5}\s?[A-Z]{3}\b", re.IGNORECASE),
    re.compile(r"\b\d{2}\s?\d{3}\s?[A-Z]{3}\b", re.IGNORECASE),
)
PHONE_PATTERNS = (  # +998 country code, long digit runs, "+<digit>", local "90 123 45 67" / "(90) 123-45-67"
    re.compile(r"\+?998"),
    re.compile(r"\d{9,}"),
    re.compile(r"\+\s*\d"),
    re.compile(r"\(?\b\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b"),
)
ADDRESS_WORDS = re.compile(
    r"\b(street|avenue|ave|road|house|apt|apartment|flat|building|ko'?chasi|kuchasi|ulitsa|prospekt|mahalla|kvartal)\b",
    re.IGNORECASE,
)


def _make_vin(prefix: str) -> str:
    """17-char VIN with a valid check digit, built from an 8-char prefix (no real VIN involved)."""
    body = f"{prefix}0SA123456"
    return body[:8] + compute_check_digit(body) + body[9:]


def _strings(obj: Any) -> Iterator[str]:
    """Every key and string value in a JSON document."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    """scripts/generate_synthetic.py imported as a module."""
    spec = importlib.util.spec_from_file_location("generate_synthetic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def docs(gen: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    return gen.generate()


@pytest.fixture()
def registry_file(tmp_path: Path, docs: tuple[dict[str, Any], dict[str, Any]]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(docs[0]), encoding="utf-8")
    return path


# --- generator: determinism and committed files ---

def test_cli_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        out.mkdir()
        shutil.copy(SYNTHETIC / "README.md", out / "README.md")
        subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(out)], check=True, capture_output=True, timeout=60)
        runs.append({p.name: p.read_bytes() for p in sorted(out.iterdir())})
    assert set(runs[0]) == {"README.md", REGISTRY_JSON.name, CUSTOMERS_JSON.name}
    assert runs[0] == runs[1]


def test_committed_files_match_a_fresh_run(gen: ModuleType, tmp_path: Path) -> None:
    shutil.copy(SYNTHETIC / "README.md", tmp_path / "README.md")
    assert gen.main(["--out-dir", str(tmp_path)]) == 0
    for name in (REGISTRY_JSON.name, CUSTOMERS_JSON.name, "README.md"):
        assert (tmp_path / name).read_bytes() == (SYNTHETIC / name).read_bytes(), (
            f"data/synthetic/{name} is stale: run scripts/generate_synthetic.py")
    registry = load_registry()  # the committed file, default path
    assert len(registry) == gen.N_VEHICLES and all(is_synthetic_vin(v) for v in registry)


def test_readme_without_markers_fails_before_writing(gen: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# no markers here\n", encoding="utf-8")
    assert gen.main(["--out-dir", str(tmp_path)]) == 2
    assert not (tmp_path / REGISTRY_JSON.name).exists()


def test_assumptions_have_value_basis_and_note(gen: ModuleType) -> None:
    for name, spec in gen.ASSUMPTIONS.items():
        assert "value" in spec, name
        assert spec["basis"] == "assumption" or spec["basis"].startswith("https://"), name
        assert spec["note"], name
    table = gen.render_assumptions().splitlines()
    assert len(table) == 2 + len(gen.ASSUMPTIONS)
    assert gen.N_VEHICLES == 40 and gen.N_CUSTOMERS == 25


# --- generated content ---

def test_counts_and_synthetic_flags(gen: ModuleType, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    registry, customers = docs
    assert len(registry["records"]) == gen.N_VEHICLES
    assert len(customers["records"]) == gen.N_CUSTOMERS
    for doc, source in ((registry, SOURCE_LABEL), (customers, CRM_SOURCE)):
        assert doc["synthetic"] is True and doc["badge"] == "SYNTHETIC DEMO DATA"
        assert doc["source"] == source and doc["seed"] == SEED
        for record in doc["records"]:
            assert record["synthetic"] is True and record["source"] == source


def test_vins_are_synthetic_valid_and_unique(docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    records = docs[0]["records"]
    vins = [r["vin"] for r in records]
    assert len(set(vins)) == len(vins)
    for record in records:
        vin = record["vin"]
        assert len(vin) == 17 and vin.startswith("0AP")
        assert vin[8] == compute_check_digit(vin) and check_digit_ok(vin) and is_synthetic_vin(vin)
        assert 2016 <= record["model_year"] <= 2025
        assert vin[9] == YEAR_CODES[record["model_year"] - 2010]
        assert re.fullmatch(r"\d{6}", vin[11:])


def test_vehicles_use_research_lineups(gen: ModuleType, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    lineups = {"Toyota": gen.ASSUMPTIONS["lineup_toyota"]["value"], "BYD": gen.ASSUMPTIONS["lineup_byd"]["value"]}
    first_year = gen.ASSUMPTIONS["first_model_year"]["value"]
    for record in docs[0]["records"]:
        assert record["model"] in lineups[record["make"]]
        assert record["model_year"] >= first_year.get(record["model"], 2016)
        assert record["colour"] in gen.ASSUMPTIONS["colour_share"]["value"]


@pytest.mark.parametrize(
    "text", ["00 A 000 AA", "00A000AA", "00 000 AAA", "00000AAA", "+998 00 000 00 00", "00 000 00 00", "(00) 000-00-00",
             "000000000", "Ko'chasi 1", "1 Example street"],
)
def test_privacy_scanners_catch_known_formats(text: str) -> None:
    """The scanners below must flag plate, phone and address formats (all-zero examples, not real data)."""
    assert any(p.search(text) for p in PLATE_PATTERNS + PHONE_PATTERNS) or ADDRESS_WORDS.search(text)


def test_no_plates_phones_or_addresses(docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    for doc in docs:
        for text in _strings(doc):
            for pattern in PLATE_PATTERNS + PHONE_PATTERNS:
                assert not pattern.search(text), (pattern.pattern, text)
            assert not ADDRESS_WORDS.search(text), text


def test_emails_names_insurers_and_districts_are_obviously_synthetic(
    gen: ModuleType, docs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    registry, customers = docs
    for doc in docs:
        for text in _strings(doc):
            if "@" in text:
                assert re.fullmatch(r"cust\d{3}@example\.invalid", text), text
    districts = set(gen.ASSUMPTIONS["tashkent_districts"]["value"]) | {"Tashkent region (synthetic)"}
    for i, customer in enumerate(customers["records"], start=1):
        assert customer["customer_id"] == f"SYN-CUST-{i:04d}"
        assert customer["display_name"] == f"Synthetic Customer {i:03d}"
        assert customer["email"] == f"cust{i:03d}@example.invalid"
        assert customer["district"] in districts
    insurers = {c["insurer"] for r in registry["records"] for c in r["insurance_claims"]}
    assert insurers <= {"Sandbox Insurer A", "Sandbox Insurer B", "Sandbox Insurer C"}


def test_ids_are_unique_and_prefixed(docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    ids: dict[str, list[str]] = {"owner_id": [], "accident_id": [], "claim_id": [], "visit_id": [], "customer_id": []}
    registry, customers = docs
    for record in registry["records"]:
        ids["owner_id"] += [o["owner_id"] for o in record["owners"]]
        ids["accident_id"] += [a["accident_id"] for a in record["accidents"]]
        ids["claim_id"] += [c["claim_id"] for c in record["insurance_claims"]]
    for customer in customers["records"]:
        ids["customer_id"].append(customer["customer_id"])
        ids["visit_id"] += [v["visit_id"] for v in customer["service_visits"]]
    kinds = {"owner_id": "OWNER", "accident_id": "ACC", "claim_id": "CLAIM", "visit_id": "VISIT", "customer_id": "CUST"}
    for field, values in ids.items():
        assert len(set(values)) == len(values), field
        assert all(re.fullmatch(rf"SYN-{kinds[field]}-\d{{4}}", v) for v in values), field


def test_dates_in_window_and_histories_consistent(docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    registry, _ = docs
    for doc in docs:
        for text in _strings(doc):
            if ISO_DATE.fullmatch(text):
                assert DATE_MIN <= date.fromisoformat(text) <= DATE_MAX, text
    for record in registry["records"]:
        owners = record["owners"]
        assert owners[-1]["to"] is None and all(o["to"] is not None for o in owners[:-1])
        for prev, nxt in zip(owners, owners[1:]):
            assert prev["from"] < prev["to"] == nxt["from"]
        registered = owners[0]["from"]
        assert date.fromisoformat(registered).year == record["model_year"]
        accidents = {a["accident_id"]: a for a in record["accidents"]}
        for accident in accidents.values():
            assert accident["date"] >= registered
            assert accident["severity"] in {"minor", "moderate", "severe"}
            assert accident["parts"] and len(set(accident["parts"])) == len(accident["parts"])
            assert all(re.fullmatch(r"[a-z_]+", p) for p in accident["parts"])
        for claim in record["insurance_claims"]:
            assert claim["accident_id"] in accidents
            assert claim["date"] >= accidents[claim["accident_id"]]["date"]
            assert claim["status"] in {"paid", "rejected", "open"}
            assert isinstance(claim["amount_uzs"], int) and claim["amount_uzs"] > 0
        readings = record["mileage_km"]
        assert readings[0]["date"] == registered
        assert all(a["date"] < b["date"] and a["km"] <= b["km"] for a, b in zip(readings, readings[1:]))


def test_customers_reference_registry_vehicles(gen: ModuleType, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    registry, customers = docs
    by_vin = {r["vin"]: r for r in registry["records"]}
    warranty_days = round(gen.ASSUMPTIONS["warranty_years"]["value"] * 365.25)
    seen: set[str] = set()
    for customer in customers["records"]:
        vins = set(customer["vins"])
        assert vins and vins <= set(by_vin) and not vins & seen
        seen |= vins
        for visit in customer["service_visits"]:
            vehicle = by_vin[visit["vin"]]
            assert visit["vin"] in vins
            assert visit["date"] >= vehicle["owners"][-1]["from"]
            assert visit["type"] in {"maintenance", "body_repair", "warranty", "accessories"}
            assert isinstance(visit["amount_uzs"], int) and visit["amount_uzs"] >= 0
            if visit["type"] == "warranty":
                registered = date.fromisoformat(vehicle["owners"][0]["from"])
                assert visit["amount_uzs"] == 0
                assert date.fromisoformat(visit["date"]) <= registered + timedelta(days=warranty_days)


# --- avtopassport.synthetic: VIN rule, loading, lookup ---

def test_is_synthetic_vin() -> None:
    vin = _make_vin("0APBCDEF")
    assert is_synthetic_vin(vin) and is_synthetic_vin(f" {vin.lower()} ")
    wrong_check = vin[:8] + ("1" if vin[8] != "1" else "2") + vin[9:]
    assert not is_synthetic_vin(wrong_check)
    assert not is_synthetic_vin(_make_vin("1APBCDEF"))  # valid check digit, but an assignable WMI row
    assert not is_synthetic_vin(vin[:16])
    assert not is_synthetic_vin("0APBCDEFI" + vin[9:])  # I is not a VIN character
    assert not is_synthetic_vin(None)  # type: ignore[arg-type]


def test_scan_lookup_always_refuses_even_for_a_registry_vin(
    registry_file: Path, docs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    vin = docs[0]["records"][0]["vin"]
    assert lookup(vin, origin="sandbox", path=registry_file) is not None  # the VIN really is in the registry
    assert lookup(vin, origin="scan", path=registry_file) == RegistryRefusal(vin, REFUSAL)
    assert lookup(vin.lower(), origin="scan", path=registry_file) == RegistryRefusal(vin.lower(), REFUSAL)
    assert lookup(vin, origin="scan", path=registry_file.with_name("missing.json")) == RegistryRefusal(vin, REFUSAL)
    real_like = _make_vin("1APBCDEF")
    assert lookup(real_like, origin="scan", path=registry_file) == RegistryRefusal(real_like, REFUSAL)


def test_sandbox_lookup_returns_the_record(registry_file: Path, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    records = docs[0]["records"]
    record = records[3]
    found = lookup(record["vin"], origin="sandbox", path=registry_file)
    assert found == record and found["synthetic"] is True and found["source"] == SOURCE_LABEL
    assert lookup(record["vin"].lower(), origin="sandbox", path=registry_file) == record
    known = {r["vin"] for r in records}
    unknown = next(v for v in (_make_vin(f"0APZZZ{a}{b}") for a in "0123456789" for b in "0123456789") if v not in known)
    assert lookup(unknown, origin="sandbox", path=registry_file) is None
    assert lookup(_make_vin("1APBCDEF"), origin="sandbox", path=registry_file) is None


@pytest.mark.parametrize("origin", ["api", "", "SCAN", "Sandbox"])
def test_lookup_rejects_unknown_origin(registry_file: Path, docs: tuple[dict[str, Any], dict[str, Any]], origin: str) -> None:
    with pytest.raises(ValueError):
        lookup(docs[0]["records"][0]["vin"], origin=origin, path=registry_file)  # type: ignore[arg-type]


def _drop_flag(record: dict[str, Any]) -> None:
    del record["synthetic"]


@pytest.mark.parametrize(
    "corrupt",
    [
        _drop_flag,
        lambda r: r.update(synthetic=False),
        lambda r: r.update(synthetic="true"),
        lambda r: r.update(synthetic=1),
        lambda r: r.update(vin=_make_vin("1APBCDEF")),
        lambda r: r.update(vin=r["vin"][:8] + ("1" if r["vin"][8] != "1" else "2") + r["vin"][9:]),
    ],
    ids=["missing", "false", "string-true", "int-1", "non-synthetic-vin", "bad-check-digit"],
)
def test_load_registry_rejects_unflagged_or_unsafe_records(
    tmp_path: Path, docs: tuple[dict[str, Any], dict[str, Any]], corrupt: Callable[[dict[str, Any]], None]
) -> None:
    doc = copy.deepcopy(docs[0])
    corrupt(doc["records"][5])
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path)


def test_load_registry_file_level_checks(tmp_path: Path, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    path = tmp_path / "registry.json"
    unflagged = {**docs[0], "synthetic": False}
    path.write_text(json.dumps(unflagged), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path)
    duplicated = copy.deepcopy(docs[0])
    duplicated["records"].append(copy.deepcopy(duplicated["records"][0]))
    path.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path)
    path.write_text(json.dumps(docs[0]["records"]), encoding="utf-8")  # a bare list of records is accepted
    assert set(load_registry(path)) == {r["vin"] for r in docs[0]["records"]}
