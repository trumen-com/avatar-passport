"""passport.store / diff / report on synthetic inspections (CPU, offline, temp files only)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from avtopassport.passport.diff import diff
from avtopassport.passport.report import build_json, build_pdf, hash_frames
from avtopassport.passport.store import PassportStore
from avtopassport.schema import CostLine, DamageFinding, InspectionResult

VIN = "SYNTHVIN000000001"


def _f(part: str, damage: str, severity: str | None, action: str | None = "paint") -> DamageFinding:
    return DamageFinding(part=part, damage=damage, n_observations=2, sources=["a", "b"], best_source="a",
                         max_score=0.8, max_area_ratio=0.05, severity=severity, action=action)  # type: ignore[arg-type]


def _result(vin: str | None, created: str, findings: list[DamageFinding] | None = None) -> InspectionResult:
    return InspectionResult(vin=vin, vin_status="read" if vin else "could_not_read", created_utc=created,
                            frames_sha256="ab" * 32, findings=findings or [],
                            extra={"np_value": np.float32(1.5)})


# ------------------------------------------------------------------------------------------- store


def test_store_add_get_history_previous(tmp_path: Path) -> None:
    store = PassportStore(tmp_path / "db" / "passports.sqlite")
    a = store.add_inspection(_result(VIN, "2026-10-01T10:00:00Z", [_f("hood", "dent", "minor")]),
                             media={"frames": ["f1.jpg"]})
    b = store.add_inspection(_result(VIN, "2026-10-05T10:00:00Z"))
    other = store.add_inspection(_result("OTHERVIN000000002", "2026-10-06T10:00:00Z"))
    c = store.add_inspection(_result(VIN.lower(), "2026-10-09T10:00:00Z"))
    none_vin = store.add_inspection(_result(None, "2026-10-09T11:00:00Z"))

    got = store.get_inspection(a)
    assert got is not None
    assert set(got) == {"id", "vin", "created_utc", "frames_sha256", "result", "media"}
    assert got["vin"] == VIN and got["media"] == {"frames": ["f1.jpg"]}
    assert got["result"]["findings"][0]["part"] == "hood"
    assert got["result"]["extra"]["np_value"] == 1.5
    assert store.get_inspection(9999) is None

    assert [r["id"] for r in store.history(VIN)] == [c, b, a]
    assert store.history(None) == [] and store.history("") == []
    assert store.previous(VIN, c)["id"] == b          # type: ignore[index]
    assert store.previous(VIN, b)["id"] == a          # type: ignore[index]
    assert store.previous(VIN, a) is None
    assert store.previous(VIN, None)["id"] == c       # type: ignore[index]
    assert store.previous(None, none_vin) is None
    assert store.get_inspection(none_vin)["vin"] is None  # type: ignore[index]
    assert [r["id"] for r in store.history("OTHERVIN000000002")] == [other]

    assert store.update_media(b, {"model_3d": "m.glb"})
    assert store.update_media(b, {"spin": "s.json"})
    assert store.get_inspection(b)["media"] == {"model_3d": "m.glb", "spin": "s.json"}  # type: ignore[index]
    assert not store.update_media(9999, {})
    store.close()

    reopened = PassportStore(tmp_path / "db" / "passports.sqlite")
    assert len(reopened.history(VIN)) == 3
    reopened.close()


def test_store_same_timestamp_orders_by_id() -> None:
    with PassportStore(":memory:") as store:   # type: ignore[arg-type]
        ids = [store.add_inspection(_result(VIN, "2026-10-01T10:00:00Z")) for _ in range(3)]
        assert [r["id"] for r in store.history(VIN)] == ids[::-1]
        assert store.previous(VIN, ids[2])["id"] == ids[1]  # type: ignore[index]


def test_store_threaded_writes(tmp_path: Path) -> None:
    store = PassportStore(tmp_path / "p.sqlite")

    def work(i: int) -> None:
        store.add_inspection(_result(VIN, f"2026-10-01T10:00:{i:02d}Z"))
        store.history(VIN)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.history(VIN)) == 8
    store.close()


# -------------------------------------------------------------------------------------------- diff


def test_diff_first_inspection() -> None:
    d = diff(None, [_f("hood", "dent", "minor")])
    assert d["first_inspection"] is True
    assert d["new"] == [{"part": "hood", "damage": "dent", "severity_before": None, "severity_after": "minor"}]
    assert d["resolved"] == d["unchanged"] == d["worse"] == []


def test_diff_buckets_mixed_inputs() -> None:
    prev = [_f("hood", "dent", "minor"), _f("trunk", "scratch", "moderate"), _f("roof", "dent", "moderate"),
            _f("front_door", "scratch", "severe")]
    cur = [{"part": "hood", "damage": "dent", "severity": "severe"},          # worse
           {"part": "roof", "damage": "dent", "severity": "moderate"},        # unchanged
           {"part": "front_door", "damage": "scratch", "severity": "minor"},  # improved (extra key)
           {"part": "back_bumper", "damage": "crack", "severity": "minor"}]   # new
    d = diff(prev, cur)
    assert d["first_inspection"] is False
    assert [(i["part"], i["damage"]) for i in d["new"]] == [("back_bumper", "crack")]
    assert d["resolved"] == [{"part": "trunk", "damage": "scratch", "severity_before": "moderate",
                              "severity_after": None}]
    assert d["worse"] == [{"part": "hood", "damage": "dent", "severity_before": "minor", "severity_after": "severe"}]
    assert [(i["part"], i["severity_after"]) for i in d["unchanged"]] == [("roof", "moderate")]
    assert [(i["part"], i["severity_after"]) for i in d["improved"]] == [("front_door", "minor")]


def test_diff_empty_previous_is_not_first() -> None:
    d = diff([], [_f("hood", "dent", "minor")])
    assert d["first_inspection"] is False and len(d["new"]) == 1


# ------------------------------------------------------------------------------------------ report


def _frames(tmp_path: Path, n: int = 3) -> list[Path]:
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(p, quality=90)
        paths.append(p)
    return paths


def test_hash_frames_deterministic(tmp_path: Path) -> None:
    paths = _frames(tmp_path)
    h1, files1 = hash_frames(paths)
    h2, files2 = hash_frames(list(reversed(paths)))
    assert h1 == h2 and files1 == files2
    assert [f["name"] for f in files1] == sorted(p.name for p in paths)
    assert len(h1) == 64 and all(len(f["sha256"]) == 64 for f in files1)
    paths[0].write_bytes(paths[0].read_bytes() + b"\0")
    assert hash_frames(paths)[0] != h1


def test_build_json_and_pdf_with_cyrillic(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 5)
    sha, _ = hash_frames(frames)
    findings = [_f("hood", "dent", "moderate", "repair"), _f("front_glass", "glass shatter", "severe", "replace")]
    result = InspectionResult(
        vin=VIN, vin_status="read", created_utc="2026-10-11T00:50:00Z", frames_sha256=sha, findings=findings,
        costs=[CostLine("hood", "dent", "repair", 500_000, "from", "repair_prices.csv: dent (from), 1 row: x 2026", False),
               CostLine("front_glass", "glass shatter", "replace", None, None, "no observed price for glass replacement",
                        False)],
        total_from_uzs=500_000, notes=["Синтетическая заметка: кузов <проверен> & OK"],
    )
    decode = {"make": "Chevrolet", "model": "Кобальт", "model_year": 2024, "plant_country": "Узбекистан"}
    report = build_json(result, diff=diff(None, findings), decode=decode, extra={"synthetic": True})
    assert report["schema_version"] and report["generated_utc"].endswith("Z")
    assert report["result"]["frames_sha256"] == sha and report["hash"]["frames_sha256"] == sha
    assert report["diff"]["first_inspection"] is True and report["decode"]["model"] == "Кобальт"

    out = build_pdf(report, tmp_path / "out" / "passport.pdf", images=[tmp_path / "missing.jpg", *frames])
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1024
    assert b"DejaVu" in data or not Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").exists()


def test_build_pdf_font_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVTO_FONT", str(tmp_path / "no_such_font.ttf"))
    report = build_json(_result(None, "2026-10-11T00:00:00Z"))
    out = build_pdf(report, tmp_path / "fallback.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1024
