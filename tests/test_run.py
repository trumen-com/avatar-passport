"""run.Inspector end to end on a tiny synthetic mp4 with every heavy piece stubbed: no weights, no
ultralytics, no OCR engine, no network. Downstream modules (mapping, dedupe, severity, cost, report) are
replaced in sys.modules so the test checks the orchestrator, not their rules."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from avtopassport.pipeline.privacy import PrivacyReport
from avtopassport.pipeline.run import InspectionConfig, Inspector, mask_vin, spin_indices
from avtopassport.schema import DamageFinding, DamageObservation, Segment

MARK = (0, 255, 0)        # the fake anonymiser paints this block: proof a saved frame went through it
VALID_VIN = "1HGCM82633A004352"   # textbook VIN with a valid check digit (not a real car of ours)


def make_video(path: Path, n: int = 30, fps: float = 10.0, dark: tuple[int, ...] = (8,)) -> Path:
    w, h = 360, 640
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8) if i in dark else rng.integers(40, 255, (h, w, 3), dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path


class FakeAnonymizer:
    """Every 3rd call fails closed (None), the 5th call raises; others paint MARK on the top-left block."""

    def __init__(self) -> None:
        self.calls = 0

    def anonymize(self, img: np.ndarray) -> tuple[np.ndarray | None, PrivacyReport]:
        self.calls += 1
        if self.calls == 5:
            raise RuntimeError("detector exploded")
        if self.calls % 3 == 0:
            return None, PrivacyReport(ok=False, errors=["fake failure"])
        out = img.copy()
        out[:96, :96] = MARK
        return out, PrivacyReport(ok=True, n_plates=1, n_faces=0)


def rect_mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


class FakeSegmenter:
    def __init__(self, kind: str, fail_on: int | None = None) -> None:
        self.kind, self.fail_on, self.calls = kind, fail_on, 0
        self.shapes: list[tuple[int, ...]] = []
        self.model_path = Path("unused.pt")

    def segment(self, img: np.ndarray) -> list[Segment]:
        self.calls += 1
        self.shapes.append(img.shape)
        if self.calls == self.fail_on:
            raise RuntimeError("segmentation failed")
        hw = img.shape[:2]
        if self.kind == "damage":
            return [Segment("scratch", 0.8, (20, 200, 60, 240), rect_mask(hw, 20, 200, 60, 240), 1600)]
        return [Segment("front-door", 0.9, (0, 100, 180, 250), rect_mask(hw, 0, 100, 180, 250), 180 * 150),
                Segment("front-wheel", 0.9, (0, 250, 180, 320), rect_mask(hw, 0, 250, 180, 320), 180 * 70)]


def stub_downstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def assign(source: str, damages: list[Segment], parts: list[Segment], *, min_overlap: float = 0.2
               ) -> list[DamageObservation]:
        return [DamageObservation(source, d.cls, d.score, parts[0].cls, 1.0, d.area_px, parts[0].area_px,
                                  d.area_px / parts[0].area_px, d.box) for d in damages]

    def merge(obs: list[DamageObservation]) -> list[DamageFinding]:
        seen["observations"] = obs
        if not obs:
            return []
        return [DamageFinding(obs[0].part, obs[0].damage, len(obs), [o.source for o in obs], obs[0].source,
                              max(o.score for o in obs), max(o.area_ratio for o in obs))]

    def rate_all(findings: list[DamageFinding]) -> list[DamageFinding]:
        for f in findings:
            f.severity, f.action = "minor", "polish"
        return findings

    def estimate(findings: list[DamageFinding]) -> tuple[list[Any], int | None]:
        return [], None

    def hash_frames(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
        seen["hashed"] = list(paths)
        return "f" * 64, [{"file": p.name} for p in paths]

    for name, attrs in {
        "avtopassport.pipeline.mapping": {"assign": assign},
        "avtopassport.pipeline.dedupe": {"merge": merge},
        "avtopassport.pipeline.severity": {"rate_all": rate_all},
        "avtopassport.cost.estimator": {"estimate": estimate},
        "avtopassport.passport.report": {"hash_frames": hash_frames},
    }.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def make_inspector(**kw: Any) -> Inspector:
    cfg = InspectionConfig(use_vpic=False, spin_frames=3, work_side=320)
    parts = {"anonymizer": FakeAnonymizer(), "damage": FakeSegmenter("damage"), "parts": FakeSegmenter("parts"),
             "ocr_engine": object(), "known_wmis": frozenset()}
    parts.update(kw)
    return Inspector(cfg, **parts)


def test_run_structure_privacy_and_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_downstream(monkeypatch)
    video = make_video(tmp_path / "walk.mp4")
    insp = make_inspector()
    calls: list[tuple[str, float]] = []
    out = tmp_path / "out"
    res = insp.run(video, out_dir=out, vin_text=f" {VALID_VIN.lower()} ", progress=lambda s, f: calls.append((s, f)))

    q, p, x = res.extra["quality"], res.extra["privacy"], res.extra
    assert q["sampled"] == 8 and q["rejected"] == 1 and q["reasons"] == {"too_dark": 1}
    assert insp.anonymizer.calls == q["kept"] == 7 == p["frames_in"]
    # calls 3 and 6 return None, call 5 raises: 3 frames dropped, 4 saved
    assert p["frames_dropped"] == 3 and p["frames_ok"] == 4 and p["plates"] == 4

    saved = sorted((out / "frames").iterdir())
    assert [f.name for f in saved] == [Path(r).name for r in x["frames"]] and len(saved) == 4
    assert all(r.startswith("frames/f_") and r.endswith(".jpg") for r in x["frames"])
    for f in saved:   # every saved frame carries the anonymiser's mark: no raw frame was written
        img = cv2.imread(str(f))
        assert img.shape[:2] == (320, 180)
        assert np.abs(img[5:40, 5:40].reshape(-1, 3).mean(0) - MARK).max() < 12
    assert x["frame_size"] == [180, 320]
    assert all(s == (320, 180, 3) for s in insp.damage.shapes)   # models see the saved working copy

    masks = sorted((out / "masks").iterdir())
    assert len(masks) == 4 and x["masks"] == [f"masks/{f.name}" for f in masks]
    m = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
    assert m.dtype == np.uint8 and m.shape == (320, 180) and set(np.unique(m)) <= {0, 255}
    assert m[150, 50] == 255 and m[240, 10] == 255 and m[300, 170] == 0 and m[50, 50] == 0

    assert x["spin"] == [0, 2, 3]
    assert len(x["pins2d"]) == 4
    pin = x["pins2d"][0]
    assert pin["frame"] == x["frames"][0] and pin["damage"] == "scratch" and pin["part"] == "front-door"
    assert pin["box"] == [20.0, 200.0, 60.0, 240.0] and pin["score"] == 0.8

    assert len(res.findings) == 1 and res.findings[0].n_observations == 4 and res.findings[0].severity == "minor"
    assert [o.source for o in seen["observations"]] == [Path(r).stem for r in x["frames"]]
    assert [h.name for h in seen["hashed"]] == [f.name for f in saved]
    assert res.frames_sha256 == "f" * 64 and len(x["frame_hashes"]) == 4

    assert res.vin == VALID_VIN and res.vin_status == "manual"
    assert x["vin_masked"] == VALID_VIN[:11] + "******" and x["vin_decode"]["vin"] == VALID_VIN
    assert set(x["models"]) >= {"damage_mitbersh", "parts_mitbersh"}

    for k in ("extract_frames", "filter_frames", "anonymize", "segment.damage", "segment.parts", "hash_frames", "total"):
        assert k in res.timings_ms and res.timings_ms[k] >= 0
    assert x["timings"]["anonymize"]["count"] == 7

    names = [s for s, _ in calls]
    assert names[0] == "extract_frames" and calls[-1] == ("done", 1.0)
    assert {"filter_frames", "anonymize", "segment", "dedupe", "vin", "hash_frames"} <= set(names)
    fracs = [f for _, f in calls]
    assert fracs == sorted(fracs) and all(0.0 <= f <= 1.0 for f in fracs)
    assert any("detector exploded" in n for n in res.notes)
    res.to_dict()   # JSON-shaped dataclass


def test_failed_segmentation_frame_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_downstream(monkeypatch)
    video = make_video(tmp_path / "walk.mp4", dark=())

    def broken_progress(step: str, frac: float) -> None:
        raise ValueError("ui went away")

    insp = make_inspector(damage=FakeSegmenter("damage", fail_on=2))
    res = insp.run(video, out_dir=tmp_path / "o", progress=broken_progress)
    a = res.extra["analysis"]
    assert a["segment_failed"] == 1 and a["frames_segmented"] == len(res.extra["frames"]) - 1
    assert len(res.extra["pins2d"]) == len(res.extra["frames"]) - 1
    assert len(res.extra["masks"]) == len(res.extra["frames"])  # mask list mirrors frames
    assert any("segmentation failed" in n for n in res.notes)
    assert res.vin is None and res.vin_status == "could_not_read" and res.extra["vin_masked"] is None


def test_vin_ocr_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_downstream(monkeypatch)
    video = make_video(tmp_path / "walk.mp4", n=10)
    photo = tmp_path / "vin.png"
    cv2.imwrite(str(photo), np.full((50, 200, 3), 200, np.uint8))
    got: dict[str, Any] = {}

    def read_vin(img: np.ndarray, *, engine: Any = None, known_wmis: Any = None) -> Any:
        got["shape"] = img.shape
        val = types.SimpleNamespace(vin=VALID_VIN)
        best = types.SimpleNamespace(text=VALID_VIN, validation=val)
        return types.SimpleNamespace(status=got.get("status", "read"), best=best, candidates=[best],
                                     timings_ms={"total": 1.0})

    fake_ocr = types.ModuleType("avtopassport.vin.ocr")
    fake_ocr.read_vin = read_vin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "avtopassport.vin.ocr", fake_ocr)
    insp = make_inspector()

    res = insp.run(video, out_dir=tmp_path / "a", vin_image_path=photo, vin_text="NOT A VIN")
    assert got["shape"] == (50, 200, 3)   # full resolution by default
    assert res.vin == VALID_VIN and res.vin_status == "read" and res.extra["vin_source"] == "ocr"
    assert any("typed VIN rejected" in n for n in res.notes)
    assert not list((tmp_path / "a").rglob("vin*"))   # the VIN photo is never written

    got["status"] = "ambiguous"
    res = insp.run(video, out_dir=tmp_path / "b", vin_image_path=photo)
    assert res.vin is None and res.vin_status == "ambiguous" and res.extra["vin_decode"] is None
    assert res.extra["vin_ocr"]["candidates_masked"] == [mask_vin(VALID_VIN)]

    res = insp.run(video, out_dir=tmp_path / "c", vin_image_path=tmp_path / "missing.jpg")
    assert res.vin_status == "could_not_read" and any("VIN photo" in n for n in res.notes)


def test_helpers() -> None:
    assert spin_indices(0, 5) == [] and spin_indices(3, 5) == [0, 1, 2]
    s = spin_indices(100, 36)
    assert len(s) == 36 and s[0] == 0 and s[-1] == 99
    assert mask_vin(None) is None and mask_vin(VALID_VIN) == "1HGCM82633A******"


def test_run_with_real_downstream_modules(tmp_path: Path) -> None:
    """Same fakes for the models, but the real mapping / dedupe / severity / cost / report modules."""
    for name in ("pipeline.mapping", "pipeline.dedupe", "pipeline.severity", "cost.estimator", "passport.report"):
        pytest.importorskip(f"avtopassport.{name}")
    video = make_video(tmp_path / "walk.mp4", n=10, dark=())
    res = make_inspector().run(video, out_dir=tmp_path / "o", vin_text=VALID_VIN)
    assert res.extra["frames"] and len(res.frames_sha256) == 64
    assert len(res.extra["frame_hashes"]) == len(res.extra["frames"])
    assert [(f.part, f.damage) for f in res.findings] == [("front-door", "scratch")]
    assert res.findings[0].n_observations == len(res.extra["frames"]) and res.findings[0].action is not None
    res.to_dict()
