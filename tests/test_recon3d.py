"""recon3d: pins back-projection, box scaling, merging, worker helpers, build() via a fake 3D python (CPU)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from avtopassport.recon3d import build as build_mod
from avtopassport.recon3d import worker
from avtopassport.recon3d.pins import (bbox_diagonal, box_to_processed, da3_resized_wh, project_pins,
                                       read_ply_xyz)

FRAME_WH = (720, 1280)  # portrait phone frames, as in data/processed/frames
PROCESS_RES = 504
HW = (504, 280)  # DA3 processed (h, w) for 720x1280 at 504 (measured by the DA3 smoke run)


def test_da3_resized_matches_upper_bound_resize() -> None:
    assert da3_resized_wh(720, 1280, 504) == (280, 504)
    assert da3_resized_wh(720, 1280, 756) == (420, 756)
    assert da3_resized_wh(1280, 720, 504) == (504, 280)
    assert da3_resized_wh(504, 504, 504) == (504, 504)
    w, h = da3_resized_wh(1000, 777, 518)
    assert w % 14 == 0 and h % 14 == 0


def test_box_scaling_full_frame_and_centre() -> None:
    assert box_to_processed([0, 0, 720, 1280], FRAME_WH, PROCESS_RES, HW) == pytest.approx((0, 0, 280, 504))
    x1, y1, x2, y2 = box_to_processed([360, 640, 360, 640], FRAME_WH, PROCESS_RES, HW)
    assert (x1, y1) == pytest.approx((140.0, 252.0))


def look_at_w2c(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3x4 world-to-camera (OpenCV) looking from eye at target with world -y up."""
    R = worker.look_at(eye, target, np.array([0.0, -1.0, 0.0]))
    return np.hstack([R, (-R @ eye)[:, None]])


def make_recon(tmp: Path, P: np.ndarray, eyes: list[np.ndarray], names: list[str], f: float = 200.0) -> list[tuple]:
    """Synthetic worker output: each camera sees a fronto-parallel plane through P. Returns P's pixel per frame."""
    h, w = HW
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    depth = np.zeros((len(eyes), h, w), np.float16)
    conf = np.ones((len(eyes), h, w), np.float16)
    cams, uv = [], []
    for i, (eye, name) in enumerate(zip(eyes, names)):
        w2c = look_at_w2c(eye, P)
        pc = w2c[:, :3] @ P + w2c[:, 3]
        depth[i] = pc[2]
        u, v = (K @ (pc / pc[2]))[:2]
        uv.append((u, v))
        cams.append({"frame": name, "orig_wh": list(FRAME_WH), "intrinsics": K.tolist(), "w2c": w2c.tolist()})
    (tmp / "cameras.json").write_text(json.dumps({"process_res": PROCESS_RES, "processed_hw": list(HW),
                                                  "frames": names, "cameras": cams}))
    np.savez_compressed(tmp / "depth.npz", depth=depth, conf=conf, valid=np.ones(depth.shape, np.uint8),
                        frames=np.array(names))
    cloud = np.random.default_rng(0).uniform(-5, 5, size=(1000, 3)).astype(np.float32)
    worker.write_ply(tmp / "points.ply", cloud, np.zeros((1000, 3), np.uint8))
    return uv


def orig_box(u: float, v: float, half: float = 3.0) -> list[float]:
    """Box in ORIGINAL frame pixels around processed pixel (u, v)."""
    sx, sy = FRAME_WH[0] / HW[1], FRAME_WH[1] / HW[0]
    return [(u - half) * sx, (v - half) * sy, (u + half) * sx, (v + half) * sy]


def test_pin_recovers_known_point_and_merges_views(tmp_path: Path) -> None:
    P = np.array([0.3, -0.2, 4.0])
    eyes = [np.array([0.0, 0.0, 0.0]), np.array([2.0, -0.5, 1.0])]
    names = ["f0.jpg", "f1.jpg"]
    uv = make_recon(tmp_path, P, eyes, names)
    pins2d = [
        {"frame": "some/dir/f0.jpg", "box": orig_box(*uv[0]), "damage": "dent", "part": "door", "score": 0.6},
        {"frame": "f1", "box": orig_box(*uv[1]), "damage": "dent", "part": "door", "score": 0.9},  # stem match
        {"frame": "f1.jpg", "box": orig_box(*uv[1]), "damage": "scratch", "part": "door", "score": 0.5},
        {"frame": "missing.jpg", "box": [0, 0, 10, 10], "damage": "dent", "part": "door", "score": 1.0},
    ]
    pins = project_pins(tmp_path, pins2d, FRAME_WH)
    assert len(pins) == 2  # dent merged over two views, scratch separate, missing frame ignored
    dent = pins[0]
    assert dent["damage"] == "dent" and dent["n_views"] == 2 and dent["frames"] == names
    assert dent["score"] == pytest.approx(0.9)
    assert np.allclose(dent["xyz"], P, atol=0.03)
    assert pins[1]["damage"] == "scratch" and np.allclose(pins[1]["xyz"], P, atol=0.03)


def test_pins_not_merged_beyond_radius_and_frame_size_fallback(tmp_path: Path) -> None:
    P = np.array([0.0, 0.0, 4.0])
    uv = make_recon(tmp_path, P, [np.zeros(3)], ["a.jpg"])
    u, v = uv[0]
    far = [u + 60, v + 60]
    pins2d = [{"frame": "a.jpg", "box": orig_box(u, v), "damage": "dent", "part": "hood", "score": 0.5},
              {"frame": "a.jpg", "box": orig_box(*far), "damage": "dent", "part": "hood", "score": 0.4}]
    assert len(project_pins(tmp_path, pins2d, None, merge_radius=0.1)) == 2  # orig_wh from cameras.json
    assert len(project_pins(tmp_path, pins2d, FRAME_WH, merge_radius=100.0)) == 1
    assert project_pins(tmp_path, [{"frame": "zzz.jpg", "box": [0, 0, 5, 5]}], FRAME_WH) == []


def test_ply_roundtrip_and_bbox(tmp_path: Path) -> None:
    pts = np.array([[0, 0, 0], [3, 4, 0], [1, 1, 1]], np.float32)
    worker.write_ply(tmp_path / "p.ply", pts, np.full((3, 3), 7, np.uint8))
    assert np.allclose(read_ply_xyz(tmp_path / "p.ply"), pts)
    assert bbox_diagonal(pts.astype(float), 0, 100) == pytest.approx(np.sqrt(9 + 16 + 1))


def test_worker_helpers_numpy_only() -> None:
    assert not {"torch", "depth_anything_3", "cv2"} & set(vars(worker))  # no heavy top-level imports
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    d = worker.dilate(m, 2)
    assert d.sum() == 25 and d[2:7, 2:7].all()
    assert worker.subsample(10, 20) is None and len(worker.subsample(100, 10)) == 10


def test_render_splats_depth_test() -> None:
    eye, target = np.array([0.0, 0.0, -5.0]), np.zeros(3)
    R = worker.look_at(eye, target, np.array([0.0, -1.0, 0.0]))
    pts = np.array([[0, 0, 0], [0, 0, -1], [1, 1, 0], [-1, -1, 0]], np.float64)  # 2nd is nearer, same pixel
    cols = np.array([[255, 0, 0], [0, 255, 0], [9, 9, 9], [9, 9, 9]], np.uint8)
    img = worker.render_splats(pts, cols, R, eye, size=(64, 48), fit=pts[2:])
    assert img.shape == (48, 64, 3)
    assert (img == [0, 255, 0]).all(axis=2).any() and not (img == [255, 0, 0]).all(axis=2).any()


def fake_python(tmp: Path, body: str) -> Path:
    """Executable stand-in for .venv-3d/bin/python running `body` (gets sys.argv like the worker)."""
    p = tmp / "fake_python"
    p.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


FAKE_OK = """
out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
(out / "points.ply").write_bytes(b"ply")
(out / "points_web.ply").write_bytes(b"ply")
(out / "result.json").write_text(json.dumps({"ok": True, "argv": sys.argv[1:],
                                             "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}))
"""


def test_build_ok_with_fake_worker(tmp_path: Path) -> None:
    py = fake_python(tmp_path, FAKE_OK)
    r = build_mod.build(tmp_path, tmp_path / "out", device="cpu", python=py, masks_dir=tmp_path, threads=2)
    assert r["ok"] and r["error"] is None and r["seconds_wall"] >= 0
    assert r["result"]["cuda_visible"] == ""
    argv = r["result"]["argv"]
    assert argv[:2] == ["-m", "avtopassport.recon3d.worker"] and "--masks-dir" in argv and "--threads" in argv
    assert "--process-res" not in argv  # worker picks 756 on cuda / 504 on cpu
    assert r["ply"].endswith("points.ply") and r["ply_web"].endswith("points_web.ply") and r["preview"] is None


@pytest.mark.parametrize("body,needle", [
    ("sys.stderr.write('recon3d worker failed: boom\\n'); sys.exit(3)", "boom"),
    ("pass", "no result.json"),
])
def test_build_failure_paths(tmp_path: Path, body: str, needle: str) -> None:
    r = build_mod.build(tmp_path, tmp_path / "out", python=fake_python(tmp_path, body))
    assert r["ok"] is False and r["result"] is None and needle in r["error"]


def test_build_missing_python_and_available(tmp_path: Path) -> None:
    r = build_mod.build(tmp_path, tmp_path / "out", python=tmp_path / "nope")
    assert r["ok"] is False and "cannot start" in r["error"]
    assert build_mod.available(tmp_path / "nope")[0] is False
    ok, msg = build_mod.available(fake_python(tmp_path, "pass"))
    assert ok and "importable" in msg


# --------------------------------------------------------------------------------------------- cloud clean-up

def test_depth_edges_marks_steps_not_smooth_slopes() -> None:
    z = np.full((20, 30), 2.0, np.float32)
    z[:, 15:] = 4.0  # silhouette: car at 2 m, background at 4 m
    e = worker.depth_edges(z, 0.015)
    assert e[:, 14].all() and e[:, 15].all()  # both sides of the jump (flying pixels on either side)
    assert not e[:, :13].any() and not e[:, 17:].any()
    slope = np.tile(np.linspace(2.0, 2.2, 30, dtype=np.float32), (20, 1))  # 0.3% per pixel: a slanted panel
    assert not worker.depth_edges(slope, 0.015).any()
    bad = np.full((5, 5), 3.0, np.float32)
    bad[2, 2] = np.nan
    eb = worker.depth_edges(bad, 0.015)
    assert eb[1:4, 1:4].all() and not eb[0, 0]  # invalid pixel and its neighbours


def _blob(rng: np.random.Generator, centre: tuple[float, float, float], n: int, size: float) -> np.ndarray:
    return (np.asarray(centre) + rng.uniform(-size, size, (n, 3))).astype(np.float32)


def test_voxel_filter_keeps_main_cluster_drops_noise_and_islands() -> None:
    rng = np.random.default_rng(0)
    car = _blob(rng, (0, 0, 0), 60_000, 1.0)
    island = _blob(rng, (5, 0, 0), 600, 0.05)  # dense but small and disconnected (< 5% of the car)
    noise = rng.uniform(-4, 4, (300, 3)).astype(np.float32)
    noise = noise[np.abs(noise).max(axis=1) > 1.3]  # floating points away from the car
    pts = np.concatenate([car, island, noise])
    keep, st = worker.voxel_filter(pts, voxel=0.2, min_points=6, component_min_frac=0.05)
    assert keep[: len(car)].mean() > 0.99
    assert not keep[len(car): len(car) + len(island)].any()
    assert not keep[len(car) + len(island):].any()
    assert st["n_in"] == len(pts) and st["n_out"] == int(keep.sum())
    assert st["n_sparse_dropped"] + st["n_disconnected_dropped"] == len(pts) - st["n_out"]
    assert st["n_components"] >= 2


def test_voxel_components_26_connectivity() -> None:
    keys = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [10, 10, 10], [11, 10, 10]])
    lab = worker._voxel_components(keys)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3]
    assert worker._voxel_components(np.zeros((0, 3), dtype=np.int64)).size == 0


def test_voxel_filter_edge_cases_and_speed() -> None:
    keep, st = worker.voxel_filter(np.zeros((0, 3), np.float32), 0.1)
    assert keep.size == 0 and st["n_out"] == 0
    pts = np.random.default_rng(1).normal(size=(1000, 3)).astype(np.float32)
    keep, _ = worker.voxel_filter(pts, 0.0)  # disabled
    assert keep.all()
    assert worker.robust_diagonal(np.zeros((0, 3))) == 0.0
    import time

    big = np.random.default_rng(2).normal(size=(1_500_000, 3)).astype(np.float32)
    t0 = time.perf_counter()
    keep, st = worker.voxel_filter(big, 0.005 * worker.robust_diagonal(big))
    assert time.perf_counter() - t0 < 10.0  # measured ~0.2-0.8 s on the dev laptop; generous for CI
    assert st["n_in"] == 1_500_000


def test_worker_cli_filter_defaults() -> None:
    a = worker.parse_args(["--frames-dir", "f", "--out-dir", "o"])
    assert a.conf_percentile == worker.DEFAULT_CONF_PERCENTILE == 40.0
    assert a.edge_threshold == worker.EDGE_REL_THRESHOLD and a.voxel_frac == worker.VOXEL_FRAC
    assert a.voxel_min_points == worker.VOXEL_MIN_POINTS and a.component_min_frac == worker.COMPONENT_MIN_FRAC
    b = worker.parse_args(["--frames-dir", "f", "--out-dir", "o", "--edge-threshold", "0", "--voxel-frac", "0"])
    assert b.edge_threshold == 0 and b.voxel_frac == 0
