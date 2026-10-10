"""VIN decoding: real cited WMI table + vPIC with requests monkeypatched (never touches the network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from avtopassport.vin.decode import WMI_TABLE, VinDecode, decode, known_wmis, load_wmi_table

# Made-up VINs; only the WMI (first three characters) is real.
XWB_VIN = "XWB3L32EDKA123456"
LGX_VIN = "LGXCE4CB5N0123456"
NA_VIN = "1HGCM82633A004352"   # public sample VIN (check digit verified in test_vin_validate)


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)  # type: ignore[arg-type]

    def json(self) -> Any:
        return self.payload


def vpic_payload(**fields: str) -> dict[str, Any]:
    record = {key: "" for key in ("Make", "Model", "ModelYear", "BodyClass", "FuelTypePrimary",
                                  "PlantCountry", "PlantCity", "ErrorCode", "ErrorText", "Trim")}
    record.update(fields)
    return {"Count": 1, "Message": "Results returned successfully", "Results": [record]}


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Fail loudly if anything reaches requests.get; tests that want vPIC replace it again."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def forbidden(url: str, **kwargs: Any) -> None:
        calls.append((url, kwargs))
        raise AssertionError("network access in a unit test")

    monkeypatch.setattr(requests, "get", forbidden)
    return calls


@pytest.mark.parametrize(("wmi", "maker", "country"), [
    ("XWB", "UzAuto", "Uzbekistan"),
    ("LGX", "BYD", "China"),
    ("JTD", "Toyota", "Japan"),
])
def test_wmi_lookup_in_real_table(wmi: str, maker: str, country: str) -> None:
    row = load_wmi_table()[wmi]
    assert maker in row.manufacturer
    assert row.country == country
    assert row.source_url.startswith("https://")
    assert row.confidence in {"high", "medium", "low"}
    assert wmi in known_wmis()


def test_table_is_cached_and_copied() -> None:
    a, b = load_wmi_table(WMI_TABLE), load_wmi_table(WMI_TABLE)
    assert a == b and a is not b
    a.clear()
    assert load_wmi_table(WMI_TABLE)


def test_table_tolerates_extra_columns_bad_rows_and_duplicates(tmp_path: Path) -> None:
    csv_path = tmp_path / "wmi.csv"
    csv_path.write_text(
        "﻿WMI , Manufacturer,country,plant_or_notes,source_url,extra_col,confidence\n"
        "abc,Maker A,Nowhere,,https://example.org/a,x,LOW\n"
        "ABC,Maker A2,Nowhere,,https://example.org/a2,y,high\n"
        "AB,Too short,Nowhere,,https://example.org/b,z,high\n"
        "AOC,Has O,Nowhere,,https://example.org/c,z,high\n"
        "DEF,Maker D,Somewhere\n",
        encoding="utf-8",
    )
    table = load_wmi_table(csv_path)
    assert set(table) == {"ABC", "DEF"}
    assert table["ABC"].manufacturer == "Maker A2"      # higher confidence wins
    assert table["DEF"].confidence == "" and table["DEF"].source_url == ""
    assert known_wmis(csv_path) == frozenset({"ABC", "DEF"})


def test_decode_offline_uses_wmi_table_and_vin_rules(no_network: list) -> None:
    d = decode(XWB_VIN, use_vpic=False)
    assert isinstance(d, VinDecode)
    assert d.wmi == "XWB"
    assert "UzAuto" in (d.manufacturer or "")
    assert d.country == "Uzbekistan"
    assert d.model_year_candidates == (1989, 2019)       # position 10 'K', not North American
    assert d.sources == {"manufacturer": "wmi_table", "country": "wmi_table", "model_year_candidates": "vin_rules"}
    assert d.make is d.model is d.model_year is None    # never guessed
    assert no_network == []


def test_decode_unknown_wmi_and_low_confidence_rows(no_network: list) -> None:
    d = decode(NA_VIN, use_vpic=False)
    assert d.manufacturer is None and d.country is None
    assert any("not in our WMI table" in n for n in d.notes)

    low = {r.wmi for r in load_wmi_table().values() if r.confidence == "low"}
    if low:
        d = decode(sorted(low)[0] + "3L32EDKA123456", use_vpic=False)
        assert d.manufacturer is not None
        assert any("low-confidence" in n for n in d.notes)


def test_decode_malformed_vin_is_not_decoded(no_network: list) -> None:
    d = decode("XWB123")
    assert d.manufacturer is None
    assert d.notes and no_network == []


def test_decode_vpic_maps_fields_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    payload = vpic_payload(Make="HONDA", Model="Accord", ModelYear="2003", BodyClass="Coupe",
                           FuelTypePrimary="Gasoline", PlantCountry="UNITED STATES (USA)", PlantCity="MARYSVILLE",
                           ErrorCode="0", ErrorText="0 - VIN decoded clean.", Trim="Not Applicable")

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    d = decode(NA_VIN, cache_dir=tmp_path, timeout=2.5)
    assert (d.make, d.model, d.model_year) == ("HONDA", "Accord", 2003)
    assert (d.body_class, d.fuel_type) == ("Coupe", "Gasoline")
    assert (d.plant_country, d.plant_city) == ("UNITED STATES (USA)", "MARYSVILLE")
    assert d.vpic_error_code == "0"
    for name in ("make", "model", "model_year", "body_class", "fuel_type", "plant_country", "plant_city"):
        assert d.sources[name] == "vpic"
    assert d.sources["model_year_candidates"] == "vin_rules"
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url.endswith(f"/DecodeVinValues/{NA_VIN}")
    assert kwargs["params"]["format"] == "json" and kwargs["timeout"] == 2.5
    assert json.loads((tmp_path / f"{NA_VIN}.json").read_text(encoding="utf-8")) == payload

    # Second decode is served from the cache: the network is not touched.
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    again = decode(NA_VIN, cache_dir=tmp_path)
    assert (again.make, again.model_year) == ("HONDA", 2003)


def test_decode_vpic_sends_model_year_only_when_unique(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(requests, "get", lambda url, **k: seen.append(k["params"]) or FakeResponse(vpic_payload()))
    decode("1HGCM8L63AA004352", cache_dir=None)    # letter at position 7 -> 2010 only
    decode(NA_VIN, cache_dir=None)                  # digit at position 7 -> 2003 or 2033
    assert seen[0].get("modelyear") == "2010"
    assert "modelyear" not in seen[1]


def test_decode_vpic_timeout_becomes_a_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def timeout(url: str, **kwargs: Any) -> None:
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(requests, "get", timeout)
    d = decode(XWB_VIN, cache_dir=tmp_path, timeout=0.5)
    assert d.country == "Uzbekistan"
    assert d.make is None
    assert any("timed out" in n for n in d.notes)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("error", [requests.ConnectionError("offline"), ValueError("bad json")])
def test_decode_vpic_other_failures_become_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                 error: Exception) -> None:
    def broken(url: str, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(requests, "get", broken)
    d = decode(XWB_VIN, cache_dir=tmp_path)
    assert any("vPIC unavailable" in n for n in d.notes)


def test_decode_vpic_http_error_becomes_a_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse({}, status=503))
    d = decode(XWB_VIN, cache_dir=tmp_path)
    assert any("HTTP error 503" in n for n in d.notes)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("payload", [{"Count": 0, "Message": "No results", "Results": []}, [], None,
                                     {"Results": ["not a record"]}, "<html>maintenance</html>"])
def test_decode_vpic_odd_payloads_become_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                               payload: Any) -> None:
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(payload))
    d = decode(XWB_VIN, cache_dir=tmp_path)
    assert d.make is None and d.model_year is None
    assert "UzAuto" in (d.manufacturer or "")            # our table still answers
    assert any("no results" in n for n in d.notes)
    assert not any(tmp_path.iterdir())                   # nothing cached


def test_decode_corrupt_cache_file_is_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_file = tmp_path / f"{NA_VIN}.json"
    cache_file.write_text('{"Results": [', encoding="utf-8")   # truncated JSON
    clean = vpic_payload(Make="HONDA", ModelYear="2003", ErrorCode="0")
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(clean))
    d = decode(NA_VIN, cache_dir=tmp_path)
    assert (d.make, d.model_year) == ("HONDA", 2003)
    assert any("unreadable vPIC cache file" in n for n in d.notes)
    assert json.loads(cache_file.read_text(encoding="utf-8"))["Results"][0]["Make"] == "HONDA"


def test_decode_vpic_unregistered_manufacturer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = vpic_payload(ErrorCode="7", ErrorText="7 - Manufacturer is not registered with NHTSA")
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(payload))
    d = decode(XWB_VIN, cache_dir=tmp_path)
    assert d.make is None and d.model is None           # empty vPIC fields are not set
    assert "UzAuto" in (d.manufacturer or "")            # our table still answers
    assert d.vpic_error_code == "7"
    assert any("manufacturer not registered with NHTSA / no detailed data" in n for n in d.notes)


def test_decode_vpic_cycle_pick_with_errors_is_not_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Regression (seen on the rehearsal run): vPIC answered "5,14" (errors in several positions) for a VIN whose
    # position 10 allows two 30-year cycles; no model year had been sent, so its ModelYear was its own cycle pick.
    payload = vpic_payload(Make="BYD", ModelYear="2022", ErrorCode="5,14",
                           ErrorText="5 - VIN has errors in few positions; 14 - Unable to provide information")
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(payload))
    d = decode(LGX_VIN, cache_dir=tmp_path)                 # position 10 'N' -> 1992 or 2022
    assert d.make == "BYD" and d.sources["make"] == "vpic"
    assert d.model_year is None and "model_year" not in d.sources
    assert d.model_year_candidates == (1992, 2022)
    assert any("ModelYear 2022 not used" in n for n in d.notes)
    assert any("decoded this VIN with errors" in n for n in d.notes)


def test_decode_vpic_year_with_a_sent_hint_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = vpic_payload(Make="HONDA", ModelYear="2010", ErrorCode="1")
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(payload))
    d = decode("1HGCM8L63AA004352", cache_dir=None)         # letter at position 7 -> 2010 only, sent to vPIC
    assert (d.model_year, d.sources["model_year"]) == (2010, "vpic")


def test_decode_vpic_model_year_conflict_is_not_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = vpic_payload(Make="BYD", ModelYear="2015", ErrorCode="8")
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(payload))
    d = decode(LGX_VIN, cache_dir=tmp_path)    # position 10 'N' -> 1992 or 2022
    assert d.make == "BYD"
    assert d.model_year is None
    assert any("not among the position-10 candidates" in n for n in d.notes)
    assert any("no detailed data" in n for n in d.notes)
