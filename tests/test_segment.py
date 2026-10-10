"""segment: YOLO-seg wrapper -> schema.Segment. Fake ultralytics-like model objects only (no weights, no
ultralytics import), synthetic images only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avtopassport.pipeline import segment
from avtopassport.pipeline.segment import YoloSegmenter, polygon_mask, resolve_spec


class FakeBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32)
        self.cls = np.asarray(cls, np.float32)

    def __len__(self) -> int:
        return len(self.cls)


class FakeMasks:
    def __init__(self, xy: list[np.ndarray]) -> None:
        self.xy = xy


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, masks: FakeMasks | None) -> None:
        self.boxes, self.masks = boxes, masks


class FakeModel:
    """Mimics ultralytics YOLO: .names and .predict(img, **kw) -> [Result]; ignores `classes` on purpose."""

    def __init__(self, results: list[FakeResult], names: dict[int, str] | None = None) -> None:
        self.names = names or {0: "scratch", 1: "dent", 2: "crack"}
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def predict(self, img: np.ndarray, **kw: Any) -> list[FakeResult]:
        self.calls.append(kw)
        return self.results


def square(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


IMG = np.zeros((60, 100, 3), np.uint8)   # H=60, W=100


def seg(model: FakeModel, **kw: Any) -> YoloSegmenter:
    return YoloSegmenter(Path("unused.pt"), imgsz=640, model=model, **kw)


def test_polygon_becomes_mask_at_input_resolution() -> None:
    boxes = FakeBoxes([[10, 20, 30, 40]], [0.9], [1])
    model = FakeModel([FakeResult(boxes, FakeMasks([square(10, 20, 29, 39)]))])
    out = seg(model).segment(IMG)
    assert len(out) == 1
    s = out[0]
    assert s.cls == "dent" and s.score == pytest.approx(0.9)
    assert s.box == pytest.approx((10, 20, 30, 40))
    assert s.mask is not None and s.mask.dtype == bool and s.mask.shape == (60, 100)
    assert s.mask[30, 20] and not s.mask[5, 5] and not s.mask[50, 90]
    assert s.area_px == int(s.mask.sum()) == 20 * 20   # fillPoly includes both edges


def test_predict_gets_explicit_settings() -> None:
    model = FakeModel([FakeResult(FakeBoxes([], [], []), None)])
    s = seg(model, conf=0.3, iou=0.5, device="cuda")
    s.segment(IMG)
    kw = model.calls[-1]
    assert kw["imgsz"] == 640 and kw["conf"] == 0.3 and kw["iou"] == 0.5 and kw["device"] == "cuda"
    assert kw["retina_masks"] is False and kw["verbose"] is False
    assert "classes" in kw and kw["classes"] is None   # explicit None resets a shared model's filter
    assert s.names == {0: "scratch", 1: "dent", 2: "crack"}


def test_class_filter_is_passed_and_enforced() -> None:
    boxes = FakeBoxes([[0, 0, 10, 10], [20, 20, 40, 40], [50, 10, 70, 30]], [0.5, 0.6, 0.7], [0, 1, 2])
    masks = FakeMasks([square(0, 0, 9, 9), square(20, 20, 39, 39), square(50, 10, 69, 29)])
    model = FakeModel([FakeResult(boxes, masks)])
    out = seg(model, classes=[1, 2]).segment(IMG)
    assert model.calls[-1]["classes"] == [1, 2]
    assert [s.cls for s in out] == ["dent", "crack"]


def test_empty_results() -> None:
    assert seg(FakeModel([FakeResult(FakeBoxes([], [], []), None)])).segment(IMG) == []
    assert seg(FakeModel([FakeResult(FakeBoxes([], [], []), FakeMasks([]))])).segment(IMG) == []


def test_wrong_model_output_raises() -> None:
    with pytest.raises(RuntimeError, match="no result"):
        seg(FakeModel([])).segment(IMG)
    with pytest.raises(RuntimeError, match="no boxes"):
        seg(FakeModel([FakeResult(None, None)])).segment(IMG)


def test_detect_model_gives_box_area_and_no_mask() -> None:
    model = FakeModel([FakeResult(FakeBoxes([[10, 10, 30, 20]], [0.8], [0]), None)])
    (s,) = seg(model).segment(IMG)
    assert s.mask is None and s.area_px == 200


def test_degenerate_polygon_is_dropped_and_box_clipped() -> None:
    boxes = FakeBoxes([[0, 0, 5, 5], [-5, -5, 200, 200]], [0.5, 0.6], [0, 0])
    masks = FakeMasks([np.zeros((0, 2), np.float32), square(-5, -5, 200, 200)])
    out = seg(FakeModel([FakeResult(boxes, masks)])).segment(IMG)
    assert len(out) == 1
    assert out[0].box == (0.0, 0.0, 100.0, 60.0)
    assert out[0].area_px == 60 * 100


def test_polygon_mask_helper() -> None:
    assert not polygon_mask(None, 4, 4).any()
    assert not polygon_mask(np.array([[0, 0], [3, 3]]), 4, 4).any()
    assert polygon_mask(square(0, 0, 3, 3), 4, 4).all()


def test_missing_weights_fail_before_any_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_models"):
        YoloSegmenter(tmp_path / "missing.pt", imgsz=640)
    with pytest.raises(ValueError):
        YoloSegmenter(tmp_path / "x.pt", imgsz=640, conf=1.5, model=FakeModel([]))


def test_resolve_spec_reads_model_config(tmp_path: Path) -> None:
    cfg = tmp_path / "models/damage/mitbersh_damage_inference_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"inference": {"imgsz": 896, "conf": 0.31, "iou": 0.55}}))
    s = resolve_spec("damage_mitbersh", root=tmp_path)
    assert (s["imgsz"], s["conf"], s["iou"]) == (896, 0.31, 0.55)
    assert s["path"] == tmp_path / "models/damage/mitbersh_damage_segmentation.pt"
    d = resolve_spec("damage_danishali", root=tmp_path)   # no config: ultralytics defaults
    assert (d["imgsz"], d["conf"], d["iou"]) == (640, 0.25, 0.7)
    assert resolve_spec("damage_harpreet", root=tmp_path)["imgsz"] == 1024
    assert resolve_spec("parts_mitbersh", root=tmp_path)["imgsz"] == 768
    with pytest.raises(KeyError):
        resolve_spec("nope", root=tmp_path)


def test_manifest_sha256(tmp_path: Path) -> None:
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"files": {"a": {"sha256": "abc"}}}))
    assert segment.manifest_sha256(["a", "b"], m) == {"a": "abc", "b": None}
    assert segment.manifest_sha256(["a"], tmp_path / "missing.json") == {"a": None}
