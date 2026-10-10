"""privacy: fail-closed loading and detection, pixelation, box maths. Fake detectors only: no model weights,
no ultralytics/torch import, synthetic images only."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from avtopassport.pipeline import privacy
from avtopassport.pipeline.privacy import (
    Anonymizer,
    Box,
    PrivacyUnavailable,
    _detect_faces_multiscale,
    _face_views,
    _merge_boxes,
    _overlap,
    _tile_starts,
    make_parts_plate_detector,
    pixelate,
)


def noise(h: int = 200, w: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- loading fails closed


def test_anonymizer_cannot_be_built_without_model_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("DEFAULT_PLATE_PT", "DEFAULT_PLATE_ONNX", "YUNET_V5", "YUNET_V4"):
        monkeypatch.setattr(privacy, name, tmp_path / f"missing_{name}.onnx")
    for device in ("cpu", "cuda"):
        with pytest.raises(PrivacyUnavailable, match="missing"):
            Anonymizer(device=device)


def test_anonymizer_rejects_unloadable_face_model(tmp_path: Path) -> None:
    plate, face = tmp_path / "plate.onnx", tmp_path / "face.onnx"
    plate.write_bytes(b"not a model")
    face.write_bytes(b"not a model")
    with pytest.raises(PrivacyUnavailable, match="face"):
        Anonymizer(plate_model=plate, face_model=face)


def test_anonymizer_fails_closed_without_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plate, face = tmp_path / "plate.onnx", tmp_path / "face.onnx"
    plate.write_bytes(b"not a model")
    face.write_bytes(b"not a model")
    monkeypatch.setattr(Anonymizer, "_load_face", lambda self: object())
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes `import ultralytics` raise ImportError
    with pytest.raises(PrivacyUnavailable, match="plate"):
        Anonymizer(plate_model=plate, face_model=face)


def test_anonymizer_rejects_unknown_device() -> None:
    with pytest.raises(ValueError):
        Anonymizer(device="mps")


def test_modules_import_without_heavy_packages() -> None:
    code = (
        "import sys, avtopassport.io, avtopassport.pipeline.frames, avtopassport.pipeline.quality, "
        "avtopassport.pipeline.privacy; "
        "heavy = [m for m in ('torch', 'ultralytics', 'easyocr') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, env=env)


# --------------------------------------------------------------------------- detection fails closed


def test_anonymize_fails_closed_when_a_detector_raises() -> None:
    def plates(img: np.ndarray) -> list[Box]:
        return [Box(50, 60, 110, 100, 0.9, "plate", "fake")]

    def broken(img: np.ndarray) -> list[Box]:
        raise RuntimeError("detector crashed")

    out, rep = Anonymizer.from_detectors([plates, broken]).anonymize(noise())
    assert out is None and rep.ok is False
    assert rep.errors and "detector crashed" in rep.errors[0]
    assert rep.n_plates == rep.n_faces == 0 and rep.boxes == []


@pytest.mark.parametrize(
    "bad",
    [
        [(50, 60, 110, 100)],                                         # not a Box
        [Box(float("nan"), 60, 110, 100, 0.9, "plate", "fake")],      # non-finite coordinates
    ],
)
def test_anonymize_fails_closed_on_malformed_detections(bad: list) -> None:
    out, rep = Anonymizer.from_detectors([lambda img: bad]).anonymize(noise())
    assert out is None and not rep.ok and rep.errors


@pytest.mark.parametrize(
    "img", [np.zeros((20, 20), np.uint8), np.zeros((20, 20, 4), np.uint8), np.zeros((20, 20, 3), np.float32), None]
)
def test_anonymize_fails_closed_on_invalid_images(img: object) -> None:
    out, rep = Anonymizer.from_detectors([lambda im: []]).anonymize(img)  # type: ignore[arg-type]
    assert out is None and not rep.ok


def test_from_detectors_needs_a_detector() -> None:
    with pytest.raises(ValueError):
        Anonymizer.from_detectors([])


@pytest.fixture()
def no_model_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the default model paths at missing files and forbid loading, so a settings test can never
    touch real weights, even if the validation it checks regresses."""
    for name in ("DEFAULT_PLATE_PT", "DEFAULT_PLATE_ONNX", "YUNET_V5", "YUNET_V4"):
        monkeypatch.setattr(privacy, name, tmp_path / f"missing_{name}.onnx")

    def forbidden(self: Anonymizer) -> None:
        raise AssertionError("unit tests must not load model weights")

    monkeypatch.setattr(Anonymizer, "_load_face", forbidden)
    monkeypatch.setattr(Anonymizer, "_load_plate", forbidden)


@pytest.mark.parametrize("pad", [-0.5, -0.01, float("nan"), float("inf")])
def test_negative_or_non_finite_pad_is_rejected(pad: float, no_model_files: None) -> None:
    # Regression: pad=-0.5 shrank every box to nothing, so a detected plate was left as it was with ok=True.
    box = Box(50, 60, 110, 100, 0.9, "plate", "t")
    with pytest.raises(ValueError):
        pixelate(noise(), [box], pad=pad)
    with pytest.raises(ValueError):
        Anonymizer.from_detectors([lambda im: [box]], pad=pad)
    with pytest.raises(ValueError):
        Anonymizer(pad=pad)  # rejected before any model file is looked at


@pytest.mark.parametrize(
    "settings",
    [{"plate_conf": 1.0}, {"plate_conf_hi": 1.5}, {"face_score": float("nan")}, {"face_score": -0.1},
     {"plate_imgsz_hi": 0}],
)
def test_settings_that_switch_a_detector_off_are_rejected(settings: dict, no_model_files: None) -> None:
    with pytest.raises(ValueError):
        Anonymizer(**settings)


def test_unknown_box_kind_fails_closed() -> None:
    # it would be pixelated, but counted as neither plate nor face in the report and the manifest
    car = Box(10, 10, 60, 60, 0.9, "car", "x")  # type: ignore[arg-type]
    out, rep = Anonymizer.from_detectors([lambda im: [car]]).anonymize(noise())
    assert out is None and not rep.ok and "kind" in rep.errors[0]


def test_anonymize_pixelates_and_counts_merged_boxes() -> None:
    img = noise()
    original = img.copy()
    a = Box(50, 60, 110, 100, 0.8, "plate", "plate@640")
    b = Box(52, 61, 112, 101, 0.6, "plate", "plate@1536")  # the same plate found by the second pass
    f = Box(200, 20, 240, 70, 0.7, "face", "face@full")

    def plates(im: np.ndarray) -> list[Box]:
        return [a, b]

    def faces(im: np.ndarray) -> list[Box]:
        return [f]

    out, rep = Anonymizer.from_detectors([plates, faces], pad=0.0).anonymize(img)
    assert rep.ok and out is not None and rep.errors == []
    assert (rep.n_plates, rep.n_faces) == (1, 1)
    assert rep.passes == {"plates": 2, "faces": 1}
    merged = next(x for x in rep.boxes if x.kind == "plate")
    assert (merged.x1, merged.y1, merged.x2, merged.y2) == (50, 60, 112, 101)
    assert merged.score == 0.8 and merged.source == "plate@1536+plate@640"
    mask = np.zeros(img.shape[:2], bool)
    mask[60:101, 50:112] = mask[20:70, 200:240] = True
    assert np.array_equal(out[~mask], img[~mask])
    assert not np.array_equal(out[60:101, 50:112], img[60:101, 50:112])
    assert not np.array_equal(out[20:70, 200:240], img[20:70, 200:240])
    assert np.array_equal(img, original)  # the input is never modified


def test_no_detections_is_ok_and_unchanged() -> None:
    img = noise()
    out, rep = Anonymizer.from_detectors([lambda im: []]).anonymize(img)
    assert rep.ok and out is not None and out is not img and np.array_equal(out, img)
    assert (rep.n_plates, rep.n_faces, rep.boxes) == (0, 0, [])


# --------------------------------------------------------------------------- pixelate


def test_pixelate_changes_only_the_padded_box() -> None:
    img = noise()
    box = Box(50, 60, 110, 100, 0.9, "plate", "t")      # 60 x 40 px
    out = pixelate(img, [box], pad=0.1)                  # +6 px per side -> x 44..116, y 54..106
    mask = np.zeros(img.shape[:2], bool)
    mask[54:106, 44:116] = True
    assert np.array_equal(out[~mask], img[~mask])
    roi, before = out[54:106, 44:116], img[54:106, 44:116]
    assert not np.array_equal(roi, before)
    # 72 x 52 px with 12-px cells -> a 6 x 4 mosaic: at most 24 distinct colours
    assert len(np.unique(roi.reshape(-1, 3), axis=0)) <= 24
    assert np.array_equal(img, noise())  # input untouched


def test_pixelate_caps_the_number_of_cells_on_large_boxes() -> None:
    img = noise(1200, 1600)
    box = Box(100, 100, 1100, 300, 0.9, "plate", "t")   # a 1000-px plate in a full-resolution photo

    def distinct_columns(out: np.ndarray) -> int:
        roi = out[100:300, 100:1100]
        return len({roi[:, x].tobytes() for x in range(roi.shape[1])})

    assert 3 <= distinct_columns(pixelate(img, [box], pad=0.0)) <= privacy.MOSAIC_MAX_CELLS
    assert distinct_columns(pixelate(img, [box], pad=0.0, max_cells=None)) > 50  # plain 12-px cells


def test_pixelate_clips_and_skips_outside_boxes() -> None:
    img = noise(50, 50)
    boxes = [Box(-20, -20, 5, 5, 0.9, "face", "t"), Box(100, 100, 120, 120, 0.9, "face", "t")]
    out = pixelate(img, boxes, pad=0.0)
    assert out.shape == img.shape
    assert np.array_equal(out[5:, :], img[5:, :]) and np.array_equal(out[:, 5:], img[:, 5:])
    assert not np.array_equal(out[:5, :5], img[:5, :5])


# --------------------------------------------------------------------------- box maths


def test_merge_boxes_unions_overlaps_of_the_same_kind_only() -> None:
    a = Box(0, 0, 10, 10, 0.5, "plate", "p@640")
    b = Box(1, 1, 11, 11, 0.9, "plate", "p@1536")
    c = Box(1, 1, 11, 11, 0.9, "face", "f")
    d = Box(50, 50, 60, 60, 0.4, "plate", "p@640")
    inner = Box(52, 52, 55, 55, 0.3, "plate", "parts")   # inside d -> same object
    merged = _merge_boxes([a, b, c, d, inner])
    assert len(merged) == 3
    assert Box(0, 0, 11, 11, 0.9, "plate", "p@1536+p@640") in merged
    assert Box(50, 50, 60, 60, 0.4, "plate", "p@640+parts") in merged
    assert c in merged


def _xyxy(boxes: list[Box]) -> np.ndarray:
    return np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], float).reshape(-1, 4)


def test_merge_boxes_covers_every_input_and_leaves_no_overlap() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        boxes = []
        for _ in range(int(rng.integers(1, 25))):
            (x, y), (w, h) = rng.uniform(0, 200, 2), rng.uniform(5, 60, 2)
            boxes.append(Box(x, y, x + w, y + h, float(rng.uniform()), str(rng.choice(["plate", "face"])), "s"))
        merged = _merge_boxes(boxes)
        for b in boxes:
            assert any(m.kind == b.kind and m.x1 <= b.x1 and m.y1 <= b.y1 and m.x2 >= b.x2 and m.y2 >= b.y2
                       for m in merged)
        for i, a in enumerate(merged):
            for b in merged[i + 1:]:
                if a.kind == b.kind:
                    o_iou, o_contain = _overlap(a, b)
                    assert o_iou < privacy.MERGE_IOU and o_contain < privacy.MERGE_CONTAIN


def test_merge_boxes_scales_to_many_detections() -> None:
    # Regression: the merge restarted its scan after every merge; 1600 boxes took ~87 s (now ~0.5 s).
    xy = np.random.default_rng(1).uniform(0, 4000, (800, 2))
    boxes = [Box(x, y, x + 40, y + 40, 0.9, "face", "a") for x, y in xy]
    boxes += [Box(x + 2, y + 2, x + 42, y + 42, 0.8, "face", "b") for x, y in xy]  # every face found twice
    t0 = time.perf_counter()
    merged = _merge_boxes(boxes)
    assert time.perf_counter() - t0 < 10
    assert len(merged) <= 800  # each pair (IoU 0.82) became one box
    a, m = _xyxy(boxes), _xyxy(merged)
    inside = (m[None, :, :2] <= a[:, None, :2]).all(-1) & (m[None, :, 2:] >= a[:, None, 2:]).all(-1)
    assert inside.any(axis=1).all()  # nothing lost


def test_tile_starts_cover_with_overlap() -> None:
    assert _tile_starts(1000) == [0]
    for n in (2600, 4284, 5712):
        starts = _tile_starts(n)
        assert starts[0] == 0 and starts[-1] + privacy.FACE_TILE == n
        assert all(0 < d <= int(privacy.FACE_TILE * (1 - privacy.FACE_TILE_OVERLAP)) for d in np.diff(starts))


def test_face_views_schedule() -> None:
    assert [v[0] for v in _face_views(1920, 1080)] == ["face@full", "face@400"]
    views = _face_views(5712, 4284)
    names = [v[0] for v in views]
    assert names[:2] == ["face@1600", "face@400"] and names.count("face@tiles") == 6 * 4
    assert views[0][1] == pytest.approx(1600 / 5712)


def fake_face_run(view: np.ndarray) -> np.ndarray | None:
    """Stands in for YuNet: 'detects' the bright green square, returns YuNet rows (x, y, w, h, 10 landmarks, score)."""
    ys, xs = np.nonzero(view[:, :, 1] > 200)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return np.array([[x0, y0, x1 - x0, y1 - y0, *([0.0] * 10), 0.99]], np.float32)


@pytest.mark.parametrize(
    "h, w, expected",
    [
        (1920, 1080, {"face@full": 1, "face@400": 1}),
        (3000, 4000, {"face@1600": 1, "face@400": 1, "face@tiles": 1}),
    ],
)
def test_face_boxes_map_back_to_full_resolution(h: int, w: int, expected: dict[str, int]) -> None:
    img = np.zeros((h, w, 3), np.uint8)
    img[700:800, 900:1000] = (0, 255, 0)  # a 100-px "face"
    boxes, passes = _detect_faces_multiscale(img, fake_face_run)
    assert passes == expected
    for b in boxes:
        scale = {"face@400": 400, "face@1600": 1600}.get(b.source, max(h, w)) / max(h, w)
        tol = 1.5 / scale + 0.5
        assert b.kind == "face"
        assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((900, 700, 1000, 800), abs=tol)


# --------------------------------------------------------------------------- parts-model plates


class _T:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def cpu(self) -> _T:
        return self

    def numpy(self) -> np.ndarray:
        return self.a


class _Boxes:
    def __init__(self, rows: list[list[float]]) -> None:
        r = np.asarray(rows, np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = _T(r[:, :4]), _T(r[:, 4]), _T(r[:, 5])

    def __len__(self) -> int:
        return len(self.xyxy.a)


class _Result:
    def __init__(self, rows: list[list[float]]) -> None:
        self.boxes = _Boxes(rows)


class FakePartsModel:
    """Mimics ultralytics YOLO.predict(); ignores `classes`, so the detector must filter by itself."""

    def __init__(self, names: dict[int, str] | list[str]) -> None:
        self.names = names
        self.calls: list[dict] = []

    def predict(self, img: np.ndarray, **kw: object) -> list[_Result]:
        self.calls.append(kw)
        return [_Result([[10, 20, 50, 40, 0.75, 1], [0, 0, 100, 100, 0.9, 0]])]


def test_parts_plate_detector_returns_plate_class_only() -> None:
    model = FakePartsModel({0: "hood", 1: "license-plate", 2: "front-bumper"})
    det = make_parts_plate_detector(model, device="cpu")
    boxes = det(noise(120, 160))
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2, b.kind, b.source) == (10, 20, 50, 40, "plate", "parts:license-plate")
    assert b.score == pytest.approx(0.75)
    assert model.calls[0]["classes"] == [0, 1] and model.calls[0]["imgsz"] == 768 and model.calls[0]["device"] == "cpu"

    out, rep = Anonymizer.from_detectors([det], pad=0.0).anonymize(noise(120, 160))
    assert rep.ok and rep.n_plates == 1 and rep.passes == {"parts:license-plate": 1, "plate:on_body_panel_dropped": 0}


class _NoBoxes:
    """What ultralytics returns for a classify or OBB model: a result whose boxes is None."""

    boxes = None


@pytest.mark.parametrize("results", [[_NoBoxes()], []])
def test_a_model_that_returns_no_boxes_fails_closed(results: list) -> None:
    # Regression: a result without boxes was read as "no plates", so every image passed unblurred.
    class NoBoxModel(FakePartsModel):
        def predict(self, img: np.ndarray, **kw: object) -> list:
            return results

    det = make_parts_plate_detector(NoBoxModel({0: "license-plate"}), device="cpu")
    out, rep = Anonymizer.from_detectors([det]).anonymize(noise())
    assert out is None and not rep.ok and rep.errors


def test_parts_plate_detector_fails_closed_on_a_wrong_class_map() -> None:
    with pytest.raises(PrivacyUnavailable):
        make_parts_plate_detector(FakePartsModel({0: "hood", 1: "door"}), device="cpu")
    with pytest.raises(PrivacyUnavailable):
        make_parts_plate_detector(object(), device="cpu")
    det = make_parts_plate_detector(FakePartsModel(["hood", "License_Plate"]), device="cpu")  # list + case
    assert det(noise(120, 160))[0].source == "parts:License_Plate"


# --------------------------------------------------------------------------- scripts/blur_raw.py (fake detectors)


def _load_blur_raw():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "blur_raw.py"
    spec = importlib.util.spec_from_file_location("blur_raw_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_raw(tmp_path: Path) -> Path:
    import cv2

    raw = tmp_path / "raw"
    for sub in ("video", "images", "vin"):
        (raw / sub).mkdir(parents=True)
    vw = cv2.VideoWriter(str(raw / "video" / "walk.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (180, 320))
    assert vw.isOpened()
    for i in range(25):
        vw.write(noise(320, 180, seed=i))
    vw.release()
    cv2.imwrite(str(raw / "images" / "CAR1.jpg"), noise(300, 400, seed=1))
    cv2.imwrite(str(raw / "images" / "CAR2.jpg"), noise(310, 400, seed=2))  # its size identifies it below
    cv2.imwrite(str(raw / "vin" / "VIN1.jpg"), noise(100, 100, seed=3))
    return raw


def test_blur_raw_writes_only_anonymised_copies(
    fake_raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from PIL import Image

    script = _load_blur_raw()

    def plates(img: np.ndarray) -> list[Box]:
        if img.shape[:2] == (310, 400):
            raise RuntimeError("detector crashed on CAR2")  # this photo must be dropped, never written
        return [Box(10, 10, 60, 40, 0.9, "plate", "fake")]

    monkeypatch.setattr(script, "build_anonymizer", lambda device: Anonymizer.from_detectors([plates]))
    out = tmp_path / "processed"
    rc = script.main(["--raw", str(fake_raw), "--out", str(out), "--device", "cpu"])
    assert rc == script.EXIT_PARTIAL  # one input dropped
    frames = sorted(p.name for p in (out / "frames").iterdir())
    assert frames == ["walk_0000.00s.jpg", "walk_0001.00s.jpg", "walk_0002.00s.jpg"]  # 25 frames at 10 fps -> 1 fps
    assert sorted(p.name for p in (out / "images").iterdir()) == ["CAR1.jpg"]
    assert not any("VIN" in p.name for p in out.rglob("*"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary"]["written"] == 4 and manifest["summary"]["dropped"] == 1
    assert [d["source"].endswith("CAR2.jpg") for d in manifest["dropped"]] == [True]
    for entry in manifest["outputs"]:
        assert entry["n_plates"] == 1 and len(entry["sha256"]) == 64 and len(entry["source_sha256"]) == 64
    with Image.open(out / "images" / "CAR1.jpg") as im:
        assert len(im.getexif()) == 0


def test_blur_raw_fails_closed_without_detectors(
    fake_raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_blur_raw()

    def unavailable(device: str) -> Anonymizer:
        raise PrivacyUnavailable("plate detector missing")

    monkeypatch.setattr(script, "build_anonymizer", unavailable)
    out = tmp_path / "processed"
    rc = script.main(["--raw", str(fake_raw), "--out", str(out), "--device", "cpu"])
    assert rc == script.EXIT_PRIVACY_UNAVAILABLE
    assert not out.exists()
    assert script.main(["--raw", str(fake_raw), "--out", str(out), "--dry-run"]) == script.EXIT_OK
    assert not out.exists()



# --------------------------------------------------------------------------- run-1 additions (main agent)

from avtopassport.pipeline.privacy import _gate_secondary_plate_boxes, _tiled_rows, make_person_head_detector  # noqa: E402


def _pb(x1: float, y1: float, x2: float, y2: float, score: float, source: str, kind: str = "plate") -> Box:
    return Box(x1, y1, x2, y2, score, kind, source)


def test_secondary_gate_keeps_primary_and_drops_unconfirmed_panel_boxes() -> None:
    long_side = 1920  # 15 % = 288 px; tile limit = 0.25 * 1280 = 320 px
    big_low = _pb(100, 800, 440, 1110, 0.27, "plate@1536")             # 340 px, low, alone -> dropped
    big_high = _pb(100, 100, 440, 300, 0.60, "plate@1536")             # large and unconfirmed -> dropped (any score)
    small_low = _pb(500, 500, 700, 560, 0.12, "plate@1536")            # small, plate-shaped -> kept
    tall_low = _pb(900, 500, 960, 600, 0.20, "plate@1536")             # small but taller than wide -> dropped
    tile_big = _pb(1000, 1000, 1700, 1400, 0.45, "plate@tiles")        # 700 px from a tile -> dropped
    tile_small = _pb(10, 10, 210, 70, 0.30, "plate@tiles")             # distant plate from a tile -> kept
    confirmed = _pb(600, 1300, 950, 1400, 0.15, "plate@1536")          # overlaps the parts box -> kept
    parts = _pb(620, 1310, 900, 1390, 0.80, "parts:license-plate")    # primary -> kept
    primary_tall = _pb(1800, 100, 1900, 300, 0.30, "plate@640")       # primary boxes are never gated
    tile_on_tile = _pb(1050, 1050, 1650, 1350, 0.40, "plate@1536")     # overlaps only another SECONDARY box -> dropped
    face = _pb(0, 0, 500, 500, 0.10, "face@full", kind="face")
    boxes = [big_low, big_high, small_low, tall_low, tile_big, tile_small, confirmed, parts, primary_tall, tile_on_tile, face]
    kept, dropped = _gate_secondary_plate_boxes(boxes, long_side, primary_sources=("plate@640",), hi_source="plate@1536")
    assert dropped == 5
    assert not {big_low, big_high, tall_low, tile_big, tile_on_tile} & set(kept)
    assert {small_low, tile_small, confirmed, parts, primary_tall, face} <= set(kept)


def test_anonymize_pixelates_detected_boxes_not_their_union() -> None:
    img = noise(200, 300)
    a = Box(10, 10, 60, 40, 0.9, "plate", "x")
    b = Box(20, 15, 90, 45, 0.9, "plate", "y")     # 67 % of a inside b -> one object; union = (10, 10, 90, 45)
    out, rep = Anonymizer.from_detectors([lambda im: [a, b]], pad=0.0).anonymize(img)
    assert rep.ok and rep.n_plates == 1             # counted as one object
    assert np.array_equal(out[10:15, 62:90], img[10:15, 62:90])  # inside the union, outside both boxes: untouched
    assert np.array_equal(out[41:45, 10:19], img[41:45, 10:19])
    assert not np.array_equal(out[10:40, 10:60], img[10:40, 10:60])


def test_blur_boxes_changes_only_the_padded_box_and_removes_detail() -> None:
    img = noise(200, 300)
    out = privacy.blur_boxes(img, [Box(50, 60, 150, 100, 0.9, "plate", "t")], pad=0.0)
    assert np.array_equal(out[:60], img[:60]) and np.array_equal(out[100:], img[100:])
    assert np.array_equal(out[:, :50], img[:, :50]) and np.array_equal(out[:, 150:], img[:, 150:])
    roi_before, roi_after = img[60:100, 50:150].astype(float), out[60:100, 50:150].astype(float)
    lap = lambda a: float(np.var(np.diff(a, n=2, axis=1)))  # second-derivative energy = fine detail
    assert lap(roi_after) < 0.01 * lap(roi_before)


def test_tiled_rows_map_tile_boxes_back_to_the_full_image() -> None:
    img = np.zeros((3000, 2600, 3), np.uint8)
    seen: list[tuple[int, int]] = []

    def run(view: np.ndarray) -> list[tuple[float, float, float, float, float, int]]:
        seen.append(view.shape[:2])
        return [(10.0, 20.0, 30.0, 40.0, 0.5, 0)]

    rows = _tiled_rows(run, img, tile=1280, overlap=0.15)
    assert all(h <= 1280 and w <= 1280 for h, w in seen)
    starts_y, starts_x = privacy._tile_starts(3000, 1280, 0.15), privacy._tile_starts(2600, 1280, 0.15)
    assert len(rows) == len(starts_y) * len(starts_x)
    assert sorted((r[0], r[1]) for r in rows) == sorted((x0 + 10.0, y0 + 20.0) for y0 in starts_y for x0 in starts_x)


def test_person_head_detector_returns_the_head_band_as_a_face() -> None:
    model = FakePartsModel({0: "person", 1: "bicycle"})  # returns a class-1 box and a class-0 box (0,0,100,100)
    det = make_person_head_detector(model, device="cpu")
    boxes = det(noise(120, 160))
    assert len(boxes) == 1
    b = boxes[0]
    # 100x100 person box is not taller than 2x its width -> upper-body rule: top half, widened by 10 % per side
    assert (b.kind, b.source) == ("face", "coco:person-head")
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.0, 0.0, 110.0, 50.0))
    assert model.calls[0]["classes"] == [0]
    out, rep = Anonymizer.from_detectors([det], pad=0.0).anonymize(noise(120, 160))
    assert rep.ok and rep.n_faces == 1 and rep.passes == {"coco:person-head": 1}


def test_person_head_detector_requires_a_person_class() -> None:
    with pytest.raises(PrivacyUnavailable):
        make_person_head_detector(FakePartsModel({0: "car"}), device="cpu")



class _Masks:
    def __init__(self, polys: list[np.ndarray]) -> None:
        self.xy = polys


class _SegResult:
    def __init__(self, rows: list[list[float]], polys: list[np.ndarray]) -> None:
        self.boxes = _Boxes(rows)
        self.masks = _Masks(polys)


class FakeSegPartsModel:
    """Parts model with a door (class 0) covering the left half and a plate (class 1) on the bumper."""

    names = {0: "left_front-door", 1: "license-plate", 2: "front-bumper"}

    def predict(self, img: np.ndarray, **kw: object) -> list[_SegResult]:
        door = np.array([[0, 0], [150, 0], [150, 200], [0, 200]], np.float32)
        return [_SegResult([[0, 0, 150, 200, 0.9, 0], [200, 150, 260, 170, 0.8, 1]], [door, door])]


def test_plate_boxes_on_body_panels_are_dropped_whatever_their_source() -> None:
    det = make_parts_plate_detector(FakeSegPartsModel(), device="cpu")
    img = noise(200, 300)
    on_door = Box(20, 40, 120, 90, 0.84, "plate", "plate@640")         # plate model on a door: dropped
    decal = Box(30, 100, 130, 130, 0.60, "plate", "parts:license-plate")  # parts model on its own door: dropped
    on_bumper = Box(205, 152, 255, 168, 0.70, "plate", "plate@640")    # on the bumper: kept
    background = Box(160, 10, 200, 25, 0.60, "plate", "plate@640")     # off the panels: kept
    anon = Anonymizer.from_detectors([lambda im: [on_door, decal, on_bumper, background], det], pad=0.0)
    boxes, passes = anon.detect(img)
    kinds = {(b.source, b.x1) for b in boxes}
    assert ("plate@640", 20) not in kinds and ("parts:license-plate", 30) not in kinds
    assert ("plate@640", 205) in kinds and ("plate@640", 160) in kinds and ("parts:license-plate", 200.0) in kinds
    assert passes["plate:on_body_panel_dropped"] == 2
