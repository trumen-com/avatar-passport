"""scripts/eval.py and scripts/blur_testset.py on synthetic data (CPU, offline, no model weights, no data/raw).

Covers: numpy polygon rasterisation, mask IoU, AP50 with known answers, F1-optimal threshold calibration,
part-recall rule, FP per clean car, car-level splits, refusal without annotations, marker replacement,
slices.csv merging and the notebook builder.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("avto_eval", "scripts/eval.py")
H, W = 60, 80


def rect_poly(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def rect_crop(x1: int, y1: int, x2: int, y2: int) -> "ev.CropMask":
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    cm = ev.CropMask.from_full(m)
    assert cm is not None
    return cm


def gt_image(file: str, instances: list[tuple[str, str | None, tuple[int, int, int, int]]], *, slice_: str = "day",
             car: str | None = None, brand: str = "", no_damage: bool = False) -> "ev.GtImage":
    return ev.GtImage(file=file, path=Path(file), width=W, height=H, slice=slice_, brand=brand,
                      car=car or Path(file).stem, no_damage=no_damage,
                      instances=[ev.GtInstance(c, ev.part_key(p), p, rect_crop(*b)) for c, p, b in instances])


def pred(cls: str, score: float, box: tuple[int, int, int, int], part: str | None = None) -> "ev.Pred":
    return ev.Pred(cls, score, rect_crop(*box), ev.part_key(part))


# ------------------------------------------------------------------------------------------- rasterisation

def test_rectangle_rasterises_to_exact_pixels() -> None:
    m = ev.rasterize_polygon(rect_poly(10, 20, 30, 40), H, W)
    expected = np.zeros((H, W), bool)
    expected[20:40, 10:30] = True
    assert np.array_equal(m, expected)


def test_triangle_area_and_pixel_centre_rule() -> None:
    m = ev.rasterize_polygon([0, 0, 40, 0, 0, 40], H, W)   # right triangle, area 800
    assert abs(int(m.sum()) - 800) <= 40
    assert m[0, 0] and m[0, 38] and not m[0, 39] and not m[39, 39]


def test_degenerate_and_clipped_polygons() -> None:
    assert not ev.rasterize_polygon([1, 1, 5, 5], H, W).any()              # < 3 points
    m = ev.rasterize_polygon(rect_poly(-10, -10, 5, 5), H, W)             # clipped to the image
    assert int(m.sum()) == 25 and m[:5, :5].all()


def test_multi_polygon_union_and_shared_edge() -> None:
    a, b = rect_poly(0, 0, 10, 10), rect_poly(10, 0, 20, 10)
    assert not (ev.rasterize_polygon(a, H, W) & ev.rasterize_polygon(b, H, W)).any()   # half-open edges
    assert int(ev.rasterize_polygons([a, b], H, W).sum()) == 200


def test_concave_polygon_even_odd() -> None:
    u = [0, 0, 30, 0, 30, 30, 20, 30, 20, 10, 10, 10, 10, 30, 0, 30]   # a "U": the notch is outside
    m = ev.rasterize_polygon(u, H, W)
    assert m[5, 15] and m[20, 5] and m[20, 25] and not m[20, 15]
    assert int(m.sum()) == 900 - 200


# ------------------------------------------------------------------------------------------- IoU

def test_mask_iou_known_values() -> None:
    a, b = rect_crop(0, 0, 10, 10), rect_crop(5, 0, 15, 10)
    assert ev.mask_iou(a, a) == pytest.approx(1.0)
    assert ev.mask_iou(a, b) == pytest.approx(50 / 150)
    assert ev.mask_iou(a, rect_crop(20, 20, 30, 30)) == 0.0
    assert ev.mask_iou(a, None) == 0.0


def test_crop_mask_roundtrip_and_box_hint() -> None:
    m = np.zeros((H, W), bool)
    m[7:19, 33:41] = True
    cm = ev.CropMask.from_full(m, box=(33, 7, 41, 19))
    assert cm is not None and (cm.y0, cm.x0, cm.area) == (7, 33, 96)
    assert np.array_equal(cm.full(H, W), m)
    cm2 = ev.CropMask.from_full(m, box=(0, 0, 5, 5))     # wrong hint: falls back to the whole image
    assert cm2 is not None and np.array_equal(cm2.full(H, W), m)
    assert ev.CropMask.from_full(np.zeros((H, W), bool)) is None


# ------------------------------------------------------------------------------------------- AP50

def test_ap_known_answers() -> None:
    assert ev.average_precision(np.array([True, True]), 2) == pytest.approx(1.0)
    assert ev.average_precision(np.array([False, True]), 1) == pytest.approx(0.5)
    # TP, FP, TP with 2 GT: precision envelope 1.0 up to recall 0.5 (51 points), 2/3 above (50 points)
    assert ev.average_precision(np.array([True, False, True]), 2) == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert ev.average_precision(np.array([], bool), 3) == 0.0
    assert ev.average_precision(np.array([True]), 0) is None
    assert ev.average_precision(np.array([True]), 2) == pytest.approx(51 / 101)   # recall stops at 0.5


def test_match_perfect_predictions_ap_is_one() -> None:
    imgs = [gt_image("a.jpg", [("scratch", "hood", (10, 10, 30, 20)), ("dent", "hood", (40, 30, 60, 50))]),
            gt_image("b.jpg", [("scratch", "trunk", (0, 0, 20, 20))])]
    preds = [[pred("scratch", 0.9, (10, 10, 30, 20)), pred("dent", 0.8, (40, 30, 60, 50))],
             [pred("scratch", 0.7, (0, 0, 20, 20))]]
    t = ev.match(imgs, preds)
    aps = ev.ap_table(t, ["dent", "scratch"], np.ones(2, bool))
    assert aps == {"dent": pytest.approx(1.0), "scratch": pytest.approx(1.0)}
    assert ev.mean_ap(aps) == pytest.approx(1.0)


def test_match_fp_ranked_first_gives_half() -> None:
    imgs = [gt_image("a.jpg", [("scratch", "hood", (10, 10, 30, 20))])]
    preds = [[pred("scratch", 0.95, (50, 40, 70, 55)),        # no overlap: FP, ranked first
              pred("scratch", 0.60, (10, 10, 30, 20))]]       # TP
    t = ev.match(imgs, preds)
    assert ev.ap_table(t, ["scratch"], np.ones(1, bool))["scratch"] == pytest.approx(0.5)


def test_match_class_must_agree_and_each_gt_matches_once() -> None:
    imgs = [gt_image("a.jpg", [("scratch", None, (10, 10, 30, 20))])]
    preds = [[pred("dent", 0.9, (10, 10, 30, 20)),             # right place, wrong class: FP
              pred("scratch", 0.8, (10, 10, 30, 20)),          # TP
              pred("scratch", 0.7, (10, 10, 30, 20))]]         # duplicate: FP
    t = ev.match(imgs, preds)
    assert t.p_cls == ["dent", "scratch", "scratch"]
    assert t.p_tp.tolist() == [False, True, False]


def test_match_iou_threshold() -> None:
    imgs = [gt_image("a.jpg", [("dent", None, (0, 0, 20, 10))])]
    t_lo = ev.match(imgs, [[pred("dent", 0.9, (10, 0, 30, 10))]])    # IoU 1/3 < 0.5
    t_hi = ev.match(imgs, [[pred("dent", 0.9, (0, 0, 15, 10))]])     # IoU 0.75
    assert t_lo.p_tp.tolist() == [False] and t_hi.p_tp.tolist() == [True]


# ------------------------------------------------------------------------------------------- calibration

def test_calibration_picks_f1_optimal_threshold() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    tp = np.array([True, True, False, True, False, False])
    # n_gt = 4: F1 at cut k = 2*TP_k / (k + 4): 0.4, 0.667, 0.571, 0.75, 0.667, 0.6 -> keep >= 0.6
    thr, why = ev.calibrate_threshold(scores, tp, 4)
    assert thr == pytest.approx(0.6) and "0.750" in why


def test_calibration_tie_goes_to_higher_threshold_and_equal_scores_stay_together() -> None:
    thr, _ = ev.calibrate_threshold(np.array([0.9, 0.5]), np.array([True, False]), 1)
    assert thr == pytest.approx(0.9)
    # two predictions at 0.7 (one TP, one FP) cannot be split by a threshold
    thr2, _ = ev.calibrate_threshold(np.array([0.7, 0.7, 0.3]), np.array([False, True, True]), 2)
    assert thr2 in (pytest.approx(0.7), pytest.approx(0.3))
    f1_07, f1_03 = 2 * 1 / (2 + 2), 2 * 2 / (3 + 2)
    assert thr2 == pytest.approx(0.3 if f1_03 > f1_07 else 0.7)


def test_calibration_without_gt_or_tp_keeps_default() -> None:
    assert ev.calibrate_threshold(np.array([0.9]), np.array([False]), 0, 0.25)[0] == 0.25
    assert ev.calibrate_threshold(np.array([0.9]), np.array([False]), 3, 0.25)[0] == 0.25


# ------------------------------------------------------------------------------------------- part recall + FP

def test_part_recall_rule_mapping_or_iou() -> None:
    imgs = [gt_image("a.jpg", [("scratch", "left_front-door", (0, 0, 40, 20)),
                               ("dent", "hood", (50, 40, 70, 60)),
                               ("crack", "front-bumper", (0, 40, 10, 50))])]
    preds = [[
        pred("dent", 0.9, (30, 25, 35, 30), part="left front door"),   # (a) mapped to the door, no overlap
        pred("crack", 0.8, (50, 40, 54, 44)),                          # (b) IoU 16/400 = 0.04 < 0.1: not found
        pred("scratch", 0.7, (0, 40, 10, 48), part=None),              # (b) IoU 80/100 = 0.8 with the bumper
    ]]
    t = ev.match(imgs, preds)
    m = ev.threshold_metrics(t, np.ones(3, bool), np.ones(1, bool), ["crack", "dent", "scratch"])
    pr = m["part_recall"]
    assert (pr["n_parts"], pr["found"], pr["found_by_mapping"]) == (3, 2, 1)
    assert pr["recall"] == pytest.approx(2 / 3)
    # with the mapped prediction dropped by a threshold, the door is no longer found
    keep = t.p_score < 0.85
    assert ev.threshold_metrics(t, keep, np.ones(1, bool), ["dent"])["part_recall"]["found"] == 1


def test_fp_per_clean_car_counts_cars_not_images() -> None:
    imgs = [gt_image("c1a.jpg", [], car="car1", no_damage=True),
            gt_image("c1b.jpg", [], car="car1", no_damage=True),
            gt_image("c2.jpg", [], car="car2", no_damage=True),
            gt_image("d.jpg", [("dent", "hood", (0, 0, 20, 20))])]
    preds = [[pred("scratch", 0.9, (0, 0, 5, 5))], [pred("dent", 0.3, (0, 0, 5, 5))], [],
             [pred("dent", 0.9, (0, 0, 20, 20)), pred("glass shatter", 0.5, (40, 40, 60, 60))]]
    t = ev.match(imgs, preds)
    base = ev.threshold_metrics(t, ev.keep_mask(t, {}, 0.25), np.ones(4, bool), ["dent", "scratch"])
    assert base["fp"]["clean_cars"] == 2 and base["fp"]["fp_on_clean"] == 2
    assert base["fp"]["fp_per_clean_car"] == pytest.approx(1.0)
    assert base["fp"]["fp_all"] == 3 and base["fp"]["fp_per_image"] == pytest.approx(0.75)
    high = ev.threshold_metrics(t, ev.keep_mask(t, {"dent": 0.5}, 0.25), np.ones(4, bool), ["dent"])
    assert high["fp"]["fp_on_clean"] == 1


def test_evaluate_model_slices_and_kfold_split() -> None:
    imgs = [gt_image(f"{s}{i}.jpg", [("dent", "hood", (0, 0, 20, 20))], slice_=s, brand="BYD" if i else "Chevrolet")
            for s in ("day", "night") for i in range(3)]
    preds = [[pred("dent", 0.9, (0, 0, 20, 20)), pred("dent", 0.2, (40, 40, 60, 60))] for _ in imgs]
    t = ev.match(imgs, preds)
    res = ev.evaluate_model(t, ["dent", "scratch"], ["dent"], split="kfold:3", seed=1)
    assert set(res["slices"]) >= {"all", "day", "night", "wet", "brand:BYD", "brand:Chevrolet"}
    assert res["slices"]["wet"]["n_images"] == 0 and res["slices"]["wet"]["map50"] is None
    assert res["slices"]["day"]["map50"] == pytest.approx(1.0)
    assert res["calibration"]["thresholds"]["dent"]["threshold"] == pytest.approx(0.9)
    assert res["calibration"]["thresholds"]["scratch"]["threshold"] == 0.25   # no GT: baseline kept
    sp = res["calibration"]["split"]
    assert sp["held_out_images"] == 6 and sp["n_folds"] == 3
    assert sp["calibrated_held_out"]["per_class"]["dent"]["f1"] == pytest.approx(1.0)
    assert res["slices"]["all"]["baseline"]["per_class"]["dent"]["fp"] == 0
    blocks = ev.render_blocks({
        "run": "test", "generated_utc": "t", "config": {"device": "cpu", "split": "kfold:3", "seed": 1,
                                                        "iou_match": 0.5, "iou_part": 0.1},
        "dataset": ev.dataset_summary(imgs), "models": {"m": res}, "latency": {},
        "vin": {"status": "skipped", "reason": "x"}, "gallery": {}})
    assert set(blocks) == set(ev.BLOCKS)


def test_split_folds_by_car_deterministic() -> None:
    cars = [f"c{i}" for i in range(10)] * 2
    folds = ev.split_folds(cars, "kfold:5", seed=3)
    assert len(folds) == 5 and set().union(*folds) == set(cars) and sum(map(len, folds)) == 10
    assert folds == ev.split_folds(cars, "kfold:5", seed=3)
    (held,) = ev.split_folds(cars, "holdout:0.3", seed=0)
    assert len(held) == 3
    assert ev.split_folds(cars, "none") == []
    with pytest.raises(ValueError):
        ev.split_folds(cars, "kfold:1")


# ------------------------------------------------------------------------------------------- COCO + refusal

def write_coco(root: Path, *, rle: bool = False) -> Path:
    img_dir = root / "images" / "night"
    img_dir.mkdir(parents=True)
    (img_dir / "car1.jpg").write_bytes(b"not decoded by parse_coco")
    (img_dir / "car2.jpg").write_bytes(b"x")
    seg: object = {"counts": "abc", "size": [H, W]} if rle else [rect_poly(10, 10, 30, 30)]
    coco = {
        "images": [{"id": 1, "file_name": "night/car1.jpg", "width": W, "height": H},
                   {"id": 2, "file_name": "car2.jpg", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "Glass_Shatter"}, {"id": 2, "name": "no_damage"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": seg, "iscrowd": 0,
             "attributes": {"part": "Left front-window", "occluded": False}},
            {"id": 2, "image_id": 2, "category_id": 2, "segmentation": [], "bbox": [0, 0, 5, 5], "iscrowd": 0}],
    }
    path = root / "annotations.json"
    path.write_text(json.dumps(coco))
    (root / "slices.csv").write_text("file,slice,brand,car,no_damage,notes\nnight/car1.jpg,night,Chevrolet,,,\n")
    return path


def test_parse_coco_polygons_parts_tags_and_slices(tmp_path: Path) -> None:
    ann = write_coco(tmp_path)
    ds = ev.parse_coco(json.loads(ann.read_text()), ev.read_slices(tmp_path / "slices.csv"), tmp_path / "images")
    a, b = ds.images
    assert ds.classes == ["glass shatter"]
    assert (a.file, a.slice, a.brand, a.car, a.no_damage) == ("night/car1.jpg", "night", "Chevrolet", "car1", False)
    assert a.instances[0].part == "left-front-window" and a.instances[0].mask.area == 400
    assert b.file == "night/car2.jpg" and b.no_damage and not b.instances and b.slice == "night"
    assert len(ev.usable_images(ds)) == 2


def test_rle_is_rejected(tmp_path: Path) -> None:
    ann = write_coco(tmp_path, rle=True)
    with pytest.raises(ev.DataError, match="RLE"):
        ev.parse_coco(json.loads(ann.read_text()), {}, tmp_path / "images")


def test_refuses_without_annotations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = tmp_path / "evaluation.md"
    doc.write_text("<!-- BEGIN:summary -->\n_not generated yet_\n<!-- END:summary -->\n")
    rc = ev.main(["--annotations", str(tmp_path / "annotations.json"), "--slices", str(tmp_path / "s.csv"),
                  "--images", str(tmp_path / "images"), "--out", str(tmp_path / "out"), "--doc", str(doc),
                  "--gallery-dir", str(tmp_path / "gal"), "--vin-gt", str(tmp_path / "v.csv"),
                  "--vin-dir", str(tmp_path / "vin")])
    err = capsys.readouterr()
    assert rc == ev.EXIT_NO_DATA == 3
    assert "MISSING" in err.out and "REFUSED" in err.err
    assert not (tmp_path / "out").exists() and not (tmp_path / "gal").exists()
    assert "_not generated yet_" in doc.read_text()


def test_refuses_with_only_unlabelled_images(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "x.jpg", "width": W, "height": H}], "categories": [], "annotations": []}))
    rc = ev.main(["--annotations", str(tmp_path / "annotations.json"), "--images", str(tmp_path / "images"),
                  "--slices", str(tmp_path / "s.csv"), "--out", str(tmp_path / "out"), "--check",
                  "--vin-gt", str(tmp_path / "v.csv"), "--vin-dir", str(tmp_path / "vin")])
    assert rc == 3


def test_replace_blocks_only_between_markers() -> None:
    text = "intro\n<!-- BEGIN:vin -->\nold\n<!-- END:vin -->\noutro\n<!-- BEGIN:latency -->\n"
    new, skipped = ev.replace_blocks(text, {"vin": "NEW", "latency": "X"})
    assert new.startswith("intro\n<!-- BEGIN:vin -->\nNEW\n<!-- END:vin -->\noutro\n")
    assert skipped == ["latency"]


def test_committed_evaluation_doc_has_every_block_once() -> None:
    text = (ROOT / "docs" / "research" / "evaluation.md").read_text(encoding="utf-8")
    for name in ev.BLOCKS:
        assert text.count(f"<!-- BEGIN:{name} -->") == 1 and text.count(f"<!-- END:{name} -->") == 1


# ------------------------------------------------------------------------------------------- blur_testset

def test_blur_testset_slices_merge_keeps_team_values(tmp_path: Path) -> None:
    bt = _load("avto_blur_testset", "scripts/blur_testset.py")
    csv_path = tmp_path / "slices.csv"
    csv_path.write_text("file,slice,brand,notes\nday/a.jpg,day,Toyota,scratch on bumper\nold/z.jpg,day,,\n")
    rows = bt.merge_slices(csv_path, [("day/a.jpg", "day"), ("wet/b.jpg", "wet")])
    by_file = {r["file"]: r for r in rows}
    assert by_file["day/a.jpg"]["brand"] == "Toyota" and by_file["day/a.jpg"]["notes"] == "scratch on bumper"
    assert by_file["wet/b.jpg"] == {"file": "wet/b.jpg", "slice": "wet", "brand": "", "car": "", "no_damage": "",
                                    "notes": ""}
    assert "old/z.jpg" in by_file   # rows of files not produced this run are kept (team data is never dropped)
    assert list(rows[0]) == list(bt.SLICES_FIELDS)


def test_blur_testset_dry_run_without_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bt = _load("avto_blur_testset", "scripts/blur_testset.py")
    rc = bt.main(["--dry-run", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")])
    assert rc == 3 and "no photos" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------- notebook

def test_notebook_builds_and_committed_copy_is_current(tmp_path: Path) -> None:
    out = tmp_path / "05.ipynb"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook_05.py"), "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    nb = nbformat.read(out, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(c.outputs == [] and c.execution_count is None for c in code)
    assert code[0].source.index('os.environ["YOLO_AUTOINSTALL"] = "false"') < code[0].source.find("import avtopassport")
    assert any("scripts/eval.py" in c.source or "eval.py" in c.source for c in code)
    committed = ROOT / "notebooks" / "05_eval_local_testset.ipynb"
    assert committed.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------- end to end (fake models)

class FakeSegmenter:
    """Stands in for segment.YoloSegmenter: fixed rectangles, scores filtered by self.conf."""

    def __init__(self, key: str) -> None:
        from avtopassport.schema import Segment

        self.key, self.conf, self.imgsz = key, 0.25, 64
        if key.startswith("parts"):
            self.names = {0: "hood", 1: "left_front-door"}
            self._segs = [("hood", 0.9, (0, 0, 40, 30)), ("left_front-door", 0.9, (40, 0, 80, 60))]
        else:
            self.names = {0: "dent", 1: "scratch", 2: "glass shatter"}
            self._segs = [("dent", 0.8, (10, 10, 30, 20)), ("scratch", 0.3, (50, 40, 70, 55)),
                          ("glass shatter", 0.05, (0, 50, 10, 60))]
        self._mk = Segment

    def segment(self, img: np.ndarray) -> list:
        h, w = img.shape[:2]
        out = []
        for cls, score, (x1, y1, x2, y2) in self._segs:
            if score < self.conf:
                continue
            m = np.zeros((h, w), bool)
            m[y1:y2, x1:x2] = True
            out.append(self._mk(cls, score, (float(x1), float(y1), float(x2), float(y2)), m, int(m.sum())))
        return out


def test_end_to_end_with_fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    import avtopassport.pipeline.segment as seg_mod

    monkeypatch.setattr(seg_mod, "load_segmenter", lambda key, device: FakeSegmenter(key))
    root = tmp_path / "lt"
    for s, name in (("day", "a"), ("night", "b"), ("wet", "c")):
        (root / "images" / s).mkdir(parents=True)
        cv2.imwrite(str(root / "images" / s / f"{name}.jpg"), np.full((H, W, 3), 128, np.uint8))
    coco = {"images": [{"id": i, "file_name": f, "width": W, "height": H}
                       for i, f in enumerate(["day/a.jpg", "night/b.jpg", "wet/c.jpg"], 1)],
            "categories": [{"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [rect_poly(10, 10, 30, 20)],
                             "attributes": {"part": "hood"}},
                            {"id": 2, "image_id": 2, "category_id": 2, "segmentation": [rect_poly(0, 0, 20, 8)],
                             "attributes": {"part": "hood"}}]}
    (root / "annotations.json").write_text(json.dumps(coco))
    (root / "slices.csv").write_text("file,slice,brand,car,no_damage,notes\nday/a.jpg,day,BYD,,,\n"
                                     "night/b.jpg,night,BYD,,,\nwet/c.jpg,wet,Chevrolet,,yes,\n")
    doc = tmp_path / "evaluation.md"
    doc.write_text((ROOT / "docs" / "research" / "evaluation.md").read_text(encoding="utf-8"))
    gal, out = tmp_path / "gal", tmp_path / "out"
    rc = ev.main(["--annotations", str(root / "annotations.json"), "--slices", str(root / "slices.csv"),
                  "--images", str(root / "images"), "--out", str(out), "--doc", str(doc), "--gallery-dir", str(gal),
                  "--device", "cpu", "--run", "t1", "--latency-n", "2", "--no-vin", "--split", "holdout:0.5",
                  "--models", "damage_a,damage_b"])
    assert rc == 0
    m = json.loads((out / "t1" / "metrics.json").read_text())
    a = m["models"]["damage_a"]["slices"]["all"]
    assert a["ap50"]["dent"] == pytest.approx(1.0) and a["ap50"]["scratch"] == pytest.approx(0.0)
    assert a["baseline"]["part_recall"] == {"n_parts": 2, "found": 2, "found_by_mapping": 2, "recall": 1.0}
    # wet/c is clean: dent 0.8 + scratch 0.3 kept at 0.25 -> 2 FP on 1 clean car
    assert a["baseline"]["fp"]["fp_per_clean_car"] == pytest.approx(2.0)
    assert m["dataset"]["per_slice"] == {"day": 1, "night": 1, "wet": 1}
    assert set(m["latency"]) == {"parts_mitbersh", "damage_a", "damage_b"}
    assert "split" in m["models"]["damage_a"]["calibration"]
    text = doc.read_text()
    assert "_not generated yet_" not in text and "run t1" in text
    shots = sorted(p.name for p in gal.glob("*.jpg"))
    assert shots and all(s.startswith(("damage_a_", "damage_b_")) for s in shots)
