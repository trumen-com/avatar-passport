"""recon3d.hq: viewer orientation from SfM cameras, 3D pins from SfM observations, PSNR parsing, availability and
build_hq() with a fake 3D python + fake Brush binary (no GPU, no pycolmap needed)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from avtopassport.recon3d import hq, worker


def _ring(n: int, center: np.ndarray, radius: float, height: float) -> np.ndarray:
    """OpenCV w2c [n, 3, 4] of cameras on a horizontal circle (world +z up) looking at `center`."""
    up_world = np.array([0.0, 0.0, 1.0])
    out = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), height])
        R = worker.look_at(eye, center, up_world)  # rows right, down, forward
        out.append(np.hstack([R, (-R @ eye)[:, None]]))
    return np.stack(out)


def test_view_from_cameras_recovers_centre_up_and_radius() -> None:
    center = np.array([1.0, -2.0, 0.5])
    w2cs = _ring(24, center, 3.0, 0.4)
    v = hq.view_from_cameras(w2cs)
    assert np.allclose(v["center"], center, atol=1e-6)
    assert np.allclose(v["up"], [0, 0, 1], atol=1e-6)  # sign from the cameras' own image-up
    assert v["radius"] == pytest.approx(3.0)
    assert np.allclose(v["forward0"], [1, 0, 0], atol=1e-6)  # horizontal direction to the first camera


def test_pins_from_observations_median_inside_box() -> None:
    xy = np.array([[10, 10], [12, 11], [14, 12], [100, 100]], float)
    xyz = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [9, 9, 9]], float)
    obs = {"f_00001.jpg": (xy, xyz), "f_00002.jpg": (xy, xyz + 0.01)}
    pins2d = [
        {"frame": "/some/dir/f_00001.jpg", "box": [0, 0, 20, 20], "damage": "dent", "part": "door", "score": 0.7},
        {"frame": "f_00002.png", "box": [20, 20, 0, 0], "damage": "dent", "part": "door", "score": 0.9},  # by stem
        {"frame": "f_00001.jpg", "box": [90, 90, 110, 110], "damage": "dent", "part": "hood", "score": 0.5},  # 1 pt
        {"frame": "missing.jpg", "box": [0, 0, 20, 20], "damage": "dent", "part": "door", "score": 0.5},
    ]
    pins = hq.pins_from_observations(obs, pins2d, merge_radius=0.5)
    assert len(pins) == 1  # the two door boxes merge; < 3 points and unknown frames are skipped
    p = pins[0]
    assert p["part"] == "door" and p["n_views"] == 2 and p["score"] == pytest.approx(0.9)
    assert np.allclose(p["xyz"], [0.005, 0.005, 2.005], atol=1e-6)  # median of the two views


def test_parse_psnr() -> None:
    assert hq.parse_psnr("step 1000 loss 0.1\neval PSNR: 23.41 dB\nfinal psnr=25.9") == pytest.approx(25.9)
    assert hq.parse_psnr("no metrics here") is None
    assert hq.parse_psnr("") is None


def _exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


FAKE_PYTHON = r'''
if [ "$1" = "-c" ]; then exit 0; fi   # "import pycolmap"
ds=""; while [ $# -gt 0 ]; do if [ "$1" = "--dataset" ]; then ds="$2"; fi; shift; done
n=$(ls "$ds/images" | wc -l)
cat > "$ds/sfm_report.json" <<JSON
{"stage": "sfm", "n_images": $n, "n_registered": $n, "n_points3d": 1234, "mean_reprojection_error_px": 0.6,
 "view": {"center": [0, 0, 0], "up": [0, 0, 1], "forward0": [1, 0, 0], "radius": 3.0}, "pins": [],
 "seconds": {"features_s": 0.1}}
JSON
'''

FAKE_BRUSH = r'''
out=""; name=""; while [ $# -gt 0 ]; do
  case "$1" in --export-path) out="$2";; --export-name) name="$2";; esac; shift; done
echo "training..."; echo "eval PSNR: 24.5"
printf 'ply\nformat binary_little_endian 1.0\nend_header\nSPLAT' > "$out/$name"
'''


def _frames(d: Path, n: int) -> Path:
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"f_{i:05d}.jpg").write_bytes(b"\xff\xd8 blurred frame")
    return d


def test_build_hq_ok_with_fakes(tmp_path: Path) -> None:
    py = _exe(tmp_path / "python", FAKE_PYTHON)
    brush = _exe(tmp_path / "brush_app", FAKE_BRUSH)
    frames = _frames(tmp_path / "frames", 12)
    res = hq.build_hq(frames, tmp_path / "out", steps=50, python=py, brush=brush,
                      pins2d=[{"frame": "f_00000.jpg", "box": [0, 0, 1, 1]}])
    assert res["ok"] is True and res["error"] is None, res
    splat = Path(res["splat"])
    assert splat.name == "splat.ply" and splat.read_bytes().endswith(b"SPLAT")
    rep = json.loads((tmp_path / "out" / "hq_report.json").read_text())
    assert rep == res["report"] and rep["ok"] is True
    assert rep["n_images"] == 12 and rep["n_registered"] == 12 and rep["mean_reprojection_error_px"] == 0.6
    assert rep["psnr"] == pytest.approx(24.5) and rep["steps"] == 50 and rep["view"]["radius"] == 3.0
    assert set(rep["seconds"]) == {"sfm", "brush", "crop", "total"} and rep["splat_bytes"] == splat.stat().st_size
    assert rep["crop"]["ok"] is False and "no car box" in rep["crop"]["error"]  # no masks -> full splat served
    assert res["splat_full"] == res["splat"] and rep["n_car_masks"] == 0
    assert json.loads((tmp_path / "out" / "pins2d.json").read_text())[0]["frame"] == "f_00000.jpg"
    assert sorted(p.name for p in (tmp_path / "out" / "images").iterdir())[0] == "f_00000.jpg"


def test_build_hq_failures_are_reported(tmp_path: Path) -> None:
    py = _exe(tmp_path / "python", FAKE_PYTHON)
    brush = _exe(tmp_path / "brush_app", FAKE_BRUSH)
    few = hq.build_hq(_frames(tmp_path / "few", 3), tmp_path / "o1", python=py, brush=brush)
    assert few["ok"] is False and "need >= 10" in few["error"] and few["splat"] is None
    bad_py = _exe(tmp_path / "bad_python", "echo 'SfM exploded' >&2; exit 3\n")
    sfm = hq.build_hq(_frames(tmp_path / "f2", 12), tmp_path / "o2", python=bad_py, brush=brush)
    assert sfm["ok"] is False and "SfM failed (exit 3)" in sfm["error"] and "exploded" in sfm["error"]
    bad_brush = _exe(tmp_path / "bad_brush", "echo 'no Vulkan device'; exit 1\n")
    br = hq.build_hq(_frames(tmp_path / "f3", 12), tmp_path / "o3", python=py, brush=bad_brush)
    assert br["ok"] is False and "Brush failed (exit 1)" in br["error"] and "no Vulkan device" in br["error"]
    assert json.loads((tmp_path / "o3" / "hq_report.json").read_text())["ok"] is False


def test_available_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    py = _exe(tmp_path / "python", FAKE_PYTHON)
    brush = _exe(tmp_path / "brush_app", FAKE_BRUSH)
    hq.available.cache_clear()
    ok, msg = hq.available(py, tmp_path / "missing_brush")
    assert not ok and "Brush binary missing" in msg
    monkeypatch.setattr(hq, "vulkan_gpu", lambda: (False, "no GPU Vulkan driver (ICD manifest) found"))
    hq.available.cache_clear()
    assert hq.available(py, brush) == (False, "no GPU Vulkan driver (ICD manifest) found")
    monkeypatch.setattr(hq, "vulkan_gpu", lambda: (True, "stub GPU"))
    hq.available.cache_clear()
    ok, msg = hq.available(py, brush)
    assert ok and "stub GPU" in msg
    hq.available.cache_clear()
    bad = _exe(tmp_path / "nopycolmap", "echo 'No module named pycolmap' >&2; exit 1\n")
    ok, msg = hq.available(bad, brush)
    assert not ok and "pycolmap import failed" in msg
    hq.available.cache_clear()


def test_eval_psnr_known_value(tmp_path: Path) -> None:
    from PIL import Image

    ev, im = tmp_path / "eval_10", tmp_path / "images"
    ev.mkdir()
    im.mkdir()
    gt = np.full((8, 8, 3), 100, np.uint8)
    Image.fromarray(gt).save(im / "f_00000.png")
    Image.fromarray(gt + 10).save(ev / "f_00000.png")  # error 10/255 everywhere
    Image.fromarray(gt).save(ev / "orphan.png")  # no ground truth -> ignored
    res = hq.eval_psnr(ev, im)
    assert res["n_views"] == 1 and res["mean_db"] == pytest.approx(20 * np.log10(255 / 10), abs=1e-3)
    assert hq.eval_psnr(tmp_path / "missing", im) is None


def test_view_extent_from_points() -> None:
    view = {"center": [0, 0, 0], "up": [0, 0, 1], "forward0": [1, 0, 0], "radius": 10.0}
    rng = np.random.default_rng(0)
    car = rng.uniform([-2, -1, 0], [2, 1, 1.5], (5000, 3))
    far = rng.uniform(-50, 50, (500, 3))
    far = far[np.linalg.norm(far[:, :2], axis=1) > 7]  # background outside 0.6 x radius
    ext = hq.view_extent(view, np.concatenate([car, far]))
    assert ext == pytest.approx(np.linalg.norm([3.6, 1.8, 1.35]) / 2, rel=0.05)  # 5-95 percentile box
    assert hq.view_extent(view, car[:5]) is None


# --------------------------------------------------------------------------------------------- car crop

def test_car_hull_and_point_in_mask() -> None:
    m = np.zeros((20, 20), bool)
    m[2:5, 2:18] = True  # a "C" shape: roof strip + two pillars (glass is not in body masks)
    m[2:18, 2:5] = True
    m[15:18, 2:18] = True
    hull = hq.car_hull(m)
    assert hull[10, 10] and hull[3, 3] and not hull[0, 0]
    xy = np.array([[10.5, 10.5], [0.5, 0.5], [40.0, 40.0], [21.0, 21.0]])
    assert hq.point_in_mask(hull, xy).tolist() == [True, False, False, False]
    # image coordinates at twice the mask resolution
    assert hq.point_in_mask(hull, np.array([[21.0, 21.0]]), image_wh=(40, 40)).tolist() == [True]


def _car_view() -> dict:
    return {"center": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "forward0": [1.0, 0.0, 0.0], "radius": 10.0}


def test_car_box_follows_the_car_axis_and_keeps_the_tyres() -> None:
    rng = np.random.default_rng(0)
    # a 4.8 x 1.8 x 1.4 "car" rotated 30 degrees in the ground plane, plus a few low outliers
    local = rng.uniform([-2.4, -0.9, 0.0], [2.4, 0.9, 1.4], (4000, 3))
    a = np.radians(30)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    pts = local @ R.T + [1.0, 2.0, 0.0]
    box = hq.car_box(_car_view(), pts, pct=(1, 99), margin=0.0, floor_pct=0.0)
    long_ax = np.asarray(box["axes"][0])
    assert abs(abs(long_ax @ R[:, 0]) - 1) < 0.02  # long axis = the car's own axis, not forward0
    assert box["size"][0] == pytest.approx(4.8 * 0.98, rel=0.03) and box["size"][1] == pytest.approx(1.8 * 0.98, rel=0.05)
    assert box["lo"][2] == pytest.approx(0.0, abs=0.01)  # floor = lowest car point
    assert np.allclose(box["center"], [1.0, 2.0, 0.7], atol=0.05)
    assert hq.in_box(np.array([[1.0, 2.0, 0.7], [9.0, 9.0, 0.7]]), box).tolist() == [True, False]


def _splat_ply(path: Path, means: np.ndarray, log_scales: np.ndarray) -> None:
    props = ["x", "y", "z", "f_dc_0", "opacity", "scale_0", "scale_1", "scale_2"]
    data = np.zeros(len(means), dtype=[(p, "<f4") for p in props])
    data["x"], data["y"], data["z"] = means.T
    data["f_dc_0"] = np.arange(len(means))
    for i in range(3):
        data[f"scale_{i}"] = log_scales
    header = ["ply", "format binary_little_endian 1.0", "comment Exported from Brush", "comment Vertical axis: y",
              f"element vertex {len(means)}", *[f"property float {p}" for p in props], "end_header"]
    hq.write_splat_ply(path, header, data)


def test_splat_ply_roundtrip_frame_check_and_crop(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    car = rng.uniform([-2, -1, 0], [2, 1, 1.4], (3000, 3))
    background = rng.uniform(-30, 30, (1000, 3))
    background = background[np.abs(background).max(axis=1) > 5]
    streak = np.array([[0.0, 0.0, 0.7]])  # centred on the car but huge
    edge = np.array([[2.1, 0.0, 0.7]])  # near the box edge, medium size: its 2-sigma extent leaves the box
    means = np.concatenate([car, background, streak, edge])
    log_scales = np.concatenate([np.full(len(car), np.log(0.01)), np.full(len(background), np.log(0.01)),
                                 [np.log(5.0)], [np.log(0.12)]])
    splat = tmp_path / "splat.ply"
    _splat_ply(splat, means, log_scales)
    header, data = hq.read_splat_ply(splat)
    assert "comment Vertical axis: y" in header and len(data) == len(means)
    assert np.allclose(hq.splat_means(data), means, atol=1e-5)

    # dataset files the crop reads: SfM points (car flagged) + car box in sfm_report.json
    sfm_xyz = car[::3] + rng.normal(0, 0.005, (len(car[::3]), 3))
    np.savez_compressed(tmp_path / "sfm_points.npz", xyz=sfm_xyz.astype(np.float32),
                        is_car=np.ones(len(sfm_xyz), bool))
    box = hq.car_box(_car_view(), car)
    (tmp_path / "sfm_report.json").write_text(json.dumps({"car_box": box}))
    res = hq.crop_splat(tmp_path, splat, tmp_path / "splat_car.ply")
    assert res["ok"] and res["transform"] == "identity" and res["frame_check"]["same_frame"]
    assert res["n_dropped_too_large"] == 1 and res["n_dropped_crossing_box"] >= 1
    _, kept = hq.read_splat_ply(tmp_path / "splat_car.ply")
    ids = set(kept["f_dc_0"].astype(int).tolist())
    assert len(car) - 5 <= len(ids & set(range(len(car))))  # (nearly) the whole car stays
    assert not ids & set(range(len(car), len(means)))  # background, streak and edge-crossing Gaussian dropped
    assert res["n_splats_car"] == len(kept)


def test_frame_check_detects_flipped_axes() -> None:
    rng = np.random.default_rng(2)
    pts = rng.uniform([-2, -1, 0], [2, 1, 1.5], (2000, 3)) + [3.0, 1.0, 2.0]
    flipped = pts * [1, -1, -1]  # what a y-down exporter would write
    res = hq.frame_check(pts[::4], flipped, scale=5.0)
    assert res["best"] == "flip_yz" and not res["same_frame"]
