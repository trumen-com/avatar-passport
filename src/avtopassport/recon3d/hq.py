"""High-quality 3D (GPU only): COLMAP structure-from-motion on the anonymised frames, then Gaussian splats (Brush).

The DA3 point cloud (worker.py) is the fast preview; this is the slow, accurate path. Calibrated camera poses come
from classic SfM (pycolmap, in the separate .venv-3d) over all the blurred frames the inspection saved, and Brush
(Apache-2.0, Vulkan, prebuilt binary, no compiler) trains 3D Gaussians on the calibrated views.
More frames = better coverage: the inspection keeps AVTO_MAX_FRAMES frames at AVTO_FPS (defaults 72 at 2.5 fps);
the offline demo that registered 115/115 views used ~4 fps (scripts/recon3d_hq.py).

Privacy: the inputs are the inspection's blurred frames only (never data/raw). Every number in the reports is
measured by this module. Top-level imports are stdlib + numpy only; pycolmap is imported inside run_sfm, which runs in
.venv-3d (`python -m avtopassport.recon3d.hq sfm --dataset D`).

Car crop (the background of a walkaround is under-observed and trains into "starburst" streaks):
- an SfM point is "car" when it has >= CAR_MIN_OBS observations in masked images and > 50% of them fall inside that
  image's car mask (convex hull of the inspection's body mask, <stem>.png);
- car box = principal axes of the car points in the ground plane + up, 2-98 percentile per axis + 8% margin, bottom
  extended to the 0.5th percentile of the car points (tyres);
- frame check: Brush exports in the COLMAP world frame (despite its "Vertical axis: y" header comment). Measured on
  the 115-frame demo: median distance from car SfM points to the nearest splat mean 0.0077 units with the identity
  vs 0.13 for the best axis flip (car box diagonal 8.2 units). crop_splat() re-checks this on every run and uses
  the best candidate transform;
- kept Gaussians: mean inside the box, largest scale <= 5% of the box half diagonal, mean +- 2 sigma inside the box.

Layout of out_dir: images/ (copies of the blurred frames), car_masks/, colmap/, sparse/0/, sfm_report.json,
sfm_points.npz, brush/ (splat.ply, eval_<steps>/ renders), splat.ply (full), splat_car.ply (crop), hq_report.json
(written last; its presence with "ok": true means success).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from avtopassport.paths import REPO_ROOT

BRUSH = REPO_ROOT / "tools" / "brush-dl" / "brush-app-x86_64-unknown-linux-gnu" / "brush_app"
DEFAULT_PYTHON = REPO_ROOT / ".venv-3d" / "bin" / "python"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 10
DEFAULT_STEPS = 30_000
VULKAN_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d")
_PSNR_RE = re.compile(r"psnr[^0-9\-]{0,20}(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _tail(text: str | None, n: int = 1500) -> str:
    return (text or "").strip()[-n:]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _torch_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def vulkan_gpu() -> tuple[bool, str]:
    """(ok, message): a Vulkan driver (ICD manifest) and a GPU device node are present."""
    icds = [p for d in VULKAN_ICD_DIRS if Path(d).is_dir() for p in Path(d).glob("*.json")
            if not p.name.startswith("lvp")]  # lavapipe = CPU software Vulkan, not a GPU
    if not icds:
        return False, "no GPU Vulkan driver (ICD manifest) found"
    nodes = sorted(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    nodes += [p for p in (Path("/dev/nvidia0"),) if p.exists()]
    if not nodes:
        return False, "no GPU device node (/dev/dri/renderD*, /dev/nvidia0)"
    return True, f"{len(icds)} Vulkan ICD manifest(s), GPU nodes: {', '.join(map(str, nodes))}"


@lru_cache(maxsize=8)
def available(python: Path = DEFAULT_PYTHON, brush: Path = BRUSH) -> tuple[bool, str]:
    """(ok, message): Brush binary + Vulkan GPU + pycolmap importable in the 3D venv. Cached."""
    brush, python = Path(brush), Path(python)
    if not brush.is_file() or not os.access(brush, os.X_OK):
        return False, f"Brush binary missing or not executable: {brush}"
    ok, msg = vulkan_gpu()
    if not ok:
        return False, msg
    if not python.exists():
        return False, f"3D python not found: {python}"
    try:
        proc = subprocess.run([str(python), "-c", "import pycolmap"], capture_output=True, text=True, timeout=120,
                              env=_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {python}: {exc}"
    if proc.returncode != 0:
        return False, f"pycolmap import failed: {_tail(proc.stderr, 300)}"
    return True, f"Brush + Vulkan GPU + pycolmap ready ({msg})"


def view_from_cameras(w2cs: np.ndarray) -> dict:
    """Viewer orientation from SfM cameras (OpenCV w2c [N, 3, 4]): {center, up, forward0, radius}.

    center = least-squares point closest to all optical axes (the walkaround looks at the car); up = normal of the
    camera-centre plane, signed by the cameras' own image-up; forward0 = horizontal direction from the centre to the
    first camera; radius = median horizontal camera distance. Same convention as worker.scene_frame / cameras.json.
    """
    w2cs = np.asarray(w2cs, dtype=np.float64)
    R, t = w2cs[:, :3, :3], w2cs[:, :3, 3]
    C = -np.einsum("nji,nj->ni", R, t)
    d = R[:, 2, :]  # optical axis (camera +z) in world
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, v in zip(C, d):
        P = np.eye(3) - np.outer(v, v)
        A += P
        b += P @ c
    center = np.linalg.lstsq(A, b, rcond=None)[0] if len(C) >= 2 else C.mean(axis=0)
    up = (-R[:, 1, :]).mean(axis=0)
    if len(C) >= 3:
        _, s, vt = np.linalg.svd(C - C.mean(axis=0), full_matrices=False)
        if s[1] > 1e-9 and s[2] < 0.5 * s[1]:
            up = vt[2] * (1.0 if np.dot(vt[2], up) >= 0 else -1.0)
    up = up / (np.linalg.norm(up) + 1e-12)
    f0 = C[0] - center
    f0 = f0 - np.dot(f0, up) * up
    if np.linalg.norm(f0) < 1e-9:
        f0 = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    f0 = f0 / np.linalg.norm(f0)
    h = C - center
    h = h - np.outer(h @ up, up)
    radius = float(np.median(np.linalg.norm(h, axis=1)))
    return {"center": center.tolist(), "up": up.tolist(), "forward0": f0.tolist(), "radius": max(radius, 1e-6)}


def view_extent(view: dict, xyz: np.ndarray, frac: float = 0.6) -> float | None:
    """Half the diagonal of the 5-95 percentile box (axes forward0, side, up) of the SfM points within
    frac x walkaround radius (horizontally) of the centre: the car (plus some ground) for framing the viewer."""
    if len(xyz) < 20:
        return None
    c, up, f0 = (np.asarray(view[k], dtype=np.float64) for k in ("center", "up", "forward0"))
    q = np.asarray(xyz, dtype=np.float64) - c
    q = np.stack([q @ f0, q @ np.cross(up, f0), q @ up], axis=1)
    near = np.linalg.norm(q[:, :2], axis=1) <= frac * float(view["radius"])
    if near.sum() < 20:
        return None
    lo, hi = np.percentile(q[near], 5, axis=0), np.percentile(q[near], 95, axis=0)
    return float(np.linalg.norm(hi - lo) / 2)


def pins_from_observations(obs: dict[str, tuple[np.ndarray, np.ndarray]], pins2d: list[dict],
                           merge_radius: float) -> list[dict]:
    """2D damage boxes -> 3D pins in the SfM frame: median of the triangulated points observed inside each box.

    obs: image name -> (xy [M, 2] in original pixels, xyz [M, 3]) of that image's 2D-3D correspondences.
    pins2d: {"frame", "box" [x1, y1, x2, y2] in the same pixels, "damage", "part", "score"}. Boxes with fewer than
    3 triangulated points are skipped (no guessed position). Same-(part, damage) pins within merge_radius merge.
    """
    from avtopassport.recon3d.pins import _merge  # numpy-only module

    by_stem = {Path(n).stem: n for n in obs}
    singles = []
    for item in pins2d or []:
        key = Path(str(item.get("frame", ""))).name
        name = key if key in obs else by_stem.get(Path(key).stem)
        if name is None or "box" not in item:
            continue
        xy, xyz = obs[name]
        x1, y1, x2, y2 = (float(v) for v in item["box"])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        inside = (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)
        if inside.sum() < 3:
            continue
        singles.append({"xyz": np.median(xyz[inside], axis=0).tolist(), "damage": item.get("damage"),
                        "part": item.get("part"), "score": float(item.get("score") or 0.0), "frames": [name],
                        "n_points": int(inside.sum())})
    return _merge(singles, float(merge_radius)) if singles else []


def run_sfm(dataset: Path, *, camera_model: str = "SIMPLE_RADIAL", max_image_size: int = 1920,
            pins2d: list[dict] | None = None, masks_dir: Path | None = None) -> dict:
    """pycolmap SfM over dataset/images -> dataset/sparse/0 + sfm_report.json (runs in .venv-3d)."""
    import pycolmap

    images = dataset / "images"
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if len(names) < MIN_IMAGES:
        raise RuntimeError(f"need >= {MIN_IMAGES} anonymised images in {images}, found {len(names)}")
    work = dataset / "colmap"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    t["features_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pycolmap.match_exhaustive(db)  # 360-degree walkaround: exhaustive pairs close the loop
    t["matching_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    recs = pycolmap.incremental_mapping(db, images, work / "sparse")
    t["mapping_s"] = time.perf_counter() - t0
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction")
    rec = recs[max(recs, key=lambda k: recs[k].num_reg_images())]
    final = dataset / "sparse" / "0"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True)
    rec.write(final)
    # Brush scans the dataset folder for COLMAP models. The mapper's raw output (work/sparse/<k>) can hold extra,
    # degenerate side-models (seen: a 2-image model next to the 115-image one); Brush picked it up and trained on two
    # views. Keep only the selected model in dataset/sparse/0.
    shutil.rmtree(work / "sparse", ignore_errors=True)
    cam = next(iter(rec.cameras.values()))

    report = {
        "stage": "sfm", "n_images": len(names), "n_registered": rec.num_reg_images(),
        "n_points3d": rec.num_points3D(), "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "n_models": len(recs), "camera": {"model": str(cam.model), "width": cam.width, "height": cam.height,
                                            "params": [float(x) for x in cam.params]},
        "seconds": {k: round(v, 2) for k, v in t.items()}, "pycolmap": pycolmap.__version__,
    }
    report.update(analyse_model(dataset, rec, pins2d=pins2d, masks_dir=masks_dir))
    (dataset / "sfm_report.json").write_text(json.dumps(report, indent=2))
    return report


def analyse_model(dataset: Path, rec, *, pins2d: list[dict] | None = None,  # noqa: ANN001 - pycolmap object
                  masks_dir: Path | None = None) -> dict:
    """From a pycolmap Reconstruction: viewer orientation, 3D damage pins, and (with car masks) the car box.

    Writes dataset/sfm_points.npz (xyz, is_car) for the splat frame check / crop done later in the main venv."""
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    w2cs, obs = [], {}
    pid_index = {pid: i for i, pid in enumerate(rec.points3D)}
    xyz_all = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    votes_in = np.zeros(len(xyz_all))
    votes_all = np.zeros(len(xyz_all))
    masks = load_car_masks(masks_dir, [im.name for im in imgs]) if masks_dir else {}
    for im in imgs:
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        w2cs.append(np.hstack([cfw.rotation.matrix(), np.asarray(cfw.translation).reshape(3, 1)]))
        pts = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        if not pts:
            continue
        xy = np.array([q[0] for q in pts], dtype=np.float64)
        idx = np.array([pid_index[q[1]] for q in pts], dtype=np.int64)
        obs[im.name] = (xy, xyz_all[idx])
        m = masks.get(im.name)
        if m is not None:
            cam = rec.cameras[im.camera_id]
            inside = point_in_mask(m, xy, (int(cam.width), int(cam.height)))
            np.add.at(votes_all, idx, 1.0)
            np.add.at(votes_in, idx, inside.astype(np.float64))
    out: dict = {"view": None, "pins": [], "car_box": None,
                 "pins_method": "median of SfM points observed inside each 2D damage box"}
    if not w2cs:
        return out
    view = view_from_cameras(np.stack(w2cs))
    view["extent"] = view_extent(view, xyz_all)
    is_car = (votes_all >= CAR_MIN_OBS) & (votes_in > 0.5 * votes_all)
    box = car_box(view, xyz_all[is_car]) if is_car.sum() >= 50 else None
    if box is not None:
        box.update(n_car_points=int(is_car.sum()), n_points_with_mask_votes=int((votes_all > 0).sum()),
                   n_masks=len(masks), rule=f"SfM point is car if it has >= {CAR_MIN_OBS} observations in masked "
                                         "images and > 50% of them fall inside that image's car mask (convex hull)")
        view["camera_axes_center"] = view["center"]
        view["center"], view["extent"] = box["center"], box["extent"]  # orbit around the car itself
        view["box"] = {"axes": box["axes"], "half": [v / 2 for v in box["size"]]}  # for framing in the viewer
    out.update(view=view, car_box=box)
    if pins2d:
        out["pins"] = pins_from_observations(obs, pins2d, merge_radius=0.1 * view["radius"])
    np.savez_compressed(Path(dataset) / "sfm_points.npz", xyz=xyz_all.astype(np.float32), is_car=is_car)
    return out


def car_hull(mask: np.ndarray) -> np.ndarray:
    """Convex hull fill of a binary mask (part / body masks have holes at glass, wheels, lights)."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    pts = cv2.findNonZero(m)
    out = np.zeros_like(m)
    if pts is not None and len(pts) >= 3:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def load_car_masks(masks_dir: Path | None, names: list[str]) -> dict[str, np.ndarray]:
    """name -> convex-hull car mask (bool, mask resolution) for the images that have <stem>.png in masks_dir."""
    from PIL import Image

    out: dict[str, np.ndarray] = {}
    if masks_dir is None or not Path(masks_dir).is_dir():
        return out
    for name in names:
        p = Path(masks_dir) / f"{Path(name).stem}.png"
        if p.is_file():
            with Image.open(p) as im:
                out[name] = car_hull(np.asarray(im.convert("L")) > 127)
    return out


def point_in_mask(mask: np.ndarray, xy: np.ndarray, image_wh: tuple[int, int] | None = None) -> np.ndarray:
    """bool per (x, y) pixel coordinate (scaled from image_wh to the mask size when they differ)."""
    h, w = mask.shape
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    if image_wh is not None and tuple(image_wh) != (w, h):
        x, y = x * w / image_wh[0], y * h / image_wh[1]
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    res = np.zeros(len(xy), dtype=bool)
    res[ok] = mask[yi[ok], xi[ok]]
    return res


CAR_MIN_OBS = 3  # a point needs >= 3 observations for its car vote (1-2 view points are mostly mis-matches)
CAR_BOX_PCT = (2.0, 98.0)  # 1-99 kept mis-voted ground / background tails (demo: 4.5 vs 2.6 units across)
CAR_BOX_MARGIN = 0.08
CAR_FLOOR_PCT = 0.5  # "lowest car point" = 0.5th percentile along up (the minimum was an outlier 16 units down)


def car_box(view: dict, xyz_car: np.ndarray, *, pct: tuple[float, float] = CAR_BOX_PCT,
            margin: float = CAR_BOX_MARGIN, floor_pct: float = CAR_FLOOR_PCT) -> dict:
    """Box around the car points: up = view up; horizontal axes = principal axes of the car points (long, short),
    so the box is tight for a car parked at any angle. Per axis pct percentiles + margin x size, and the bottom
    extended down to the floor_pct percentile of the car points so the tyres stay. Returns axes (rows: long, short,
    up), origin, lo, hi, center (world), extent (half diagonal)."""
    c, up, f0 = (np.asarray(view[k], dtype=np.float64) for k in ("center", "up", "forward0"))
    side = np.cross(up, f0)
    rel = np.asarray(xyz_car, dtype=np.float64) - c
    h = np.stack([rel @ f0, rel @ side], axis=1)
    core = np.all((h > np.percentile(h, 5, axis=0)) & (h < np.percentile(h, 95, axis=0)), axis=1)
    _, vecs = np.linalg.eigh(np.cov(h[core if core.sum() >= 10 else slice(None)].T))
    a = vecs[:, 1]  # largest eigenvalue: the car's long axis in the (f0, side) plane
    long_ax = a[0] * f0 + a[1] * side
    axes = np.stack([long_ax, np.cross(up, long_ax), up])
    q = rel @ axes.T
    lo, hi = np.percentile(q, pct[0], axis=0), np.percentile(q, pct[1], axis=0)
    size = hi - lo
    lo, hi = lo - margin * size, hi + margin * size
    lo[2] = min(lo[2], float(np.percentile(q[:, 2], floor_pct)))
    mid = (lo + hi) / 2
    return {"axes": axes.tolist(), "origin": c.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "size": (hi - lo).tolist(), "center": (c + mid @ axes).tolist(),
            "extent": float(np.linalg.norm(hi - lo) / 2), "margin": margin, "percentiles": list(pct),
            "floor_percentile": floor_pct}


def in_box(points: np.ndarray, box: dict) -> np.ndarray:
    """bool per point: inside the car box (world coordinates)."""
    axes = np.asarray(box["axes"], dtype=np.float64)
    q = (np.asarray(points, dtype=np.float64) - np.asarray(box["origin"], dtype=np.float64)) @ axes.T
    lo, hi = np.asarray(box["lo"]), np.asarray(box["hi"])
    return np.all((q >= lo) & (q <= hi), axis=1)


def parse_psnr(text: str) -> float | None:
    """Last PSNR value Brush printed (None when it printed none)."""
    vals = [float(m.group(1)) for m in _PSNR_RE.finditer(text or "")]
    return vals[-1] if vals else None


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2)) if a.size else float("nan")
    return round(float(10.0 * np.log10(1.0 / max(mse, 1e-12))), 3)


def eval_psnr(eval_dir: Path, images_dir: Path, masks_dir: Path | None = None) -> dict | None:
    """PSNR (dB) of Brush's renders of its eval-split views (eval_dir/<stem>.png; --eval-split-every N selects every
    Nth frame) against the blurred input frames (images_dir/<stem>.*), over the whole image and, when masks_dir has
    <stem>.png, inside the car (convex hull of the mask). None when there is nothing to compare."""
    from PIL import Image

    eval_dir, images_dir = Path(eval_dir), Path(images_dir)
    if not eval_dir.is_dir():
        return None
    gt_by_stem = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS} if images_dir.is_dir() else {}
    rows = []
    for render in sorted(eval_dir.glob("*.png")):
        gt = gt_by_stem.get(render.stem)
        if gt is None:
            continue
        with Image.open(render) as a, Image.open(gt) as b:
            ra = np.asarray(a.convert("RGB"), dtype=np.float64) / 255.0
            bb = b.convert("RGB")
            if bb.size != a.size:
                bb = bb.resize(a.size, Image.BICUBIC)
            rb = np.asarray(bb, dtype=np.float64) / 255.0
        row = {"frame": render.stem, "psnr_db": _psnr(ra, rb), "psnr_car_db": None}
        mp = Path(masks_dir) / f"{render.stem}.png" if masks_dir else None
        if mp is not None and mp.is_file():
            with Image.open(mp) as mi:
                m = mi.convert("L")
                if m.size != a.size:
                    m = m.resize(a.size, Image.NEAREST)
                hull = car_hull(np.asarray(m) > 127)
            if hull.any():
                row["psnr_car_db"] = _psnr(ra[hull], rb[hull])
                row["car_fraction"] = round(float(hull.mean()), 4)
        rows.append(row)
    if not rows:
        return None
    car = [r["psnr_car_db"] for r in rows if r["psnr_car_db"] is not None]
    return {"mean_db": round(float(np.mean([r["psnr_db"] for r in rows])), 3), "n_views": len(rows),
            "mean_car_db": round(float(np.mean(car)), 3) if car else None, "n_views_car": len(car), "rows": rows,
            "method": "Brush renders of its eval-split views (--eval-split-every N) vs. the blurred input frames; "
                      "car = convex hull of the car mask"}


# ------------------------------------------------------------------ splat PLY: read / frame check / car crop

def read_splat_ply(path: Path) -> tuple[list[str], np.ndarray]:
    """(header lines, structured vertex array) of a binary little-endian 3DGS PLY (all properties kept)."""
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4",
             "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    header, props, n, fmt = [], [], 0, ""
    with Path(path).open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"no end_header in {path}")
            line = raw.decode("ascii", "replace").rstrip("\n").rstrip("\r")
            header.append(line)
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                if tok[1] != "vertex":
                    raise ValueError(f"unexpected element {tok[1]!r} in {path}")
                n = int(tok[2])
            elif tok[0] == "property":
                props.append((tok[-1], types[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt!r}")
        data = np.fromfile(f, dtype=np.dtype(props), count=n)
    return header, data


def write_splat_ply(path: Path, header: list[str], data: np.ndarray) -> None:
    """Same header (vertex count updated) + the given structured records."""
    lines = [f"element vertex {len(data)}" if ln.startswith("element vertex") else ln for ln in header]
    with Path(path).open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(data.tobytes())


def splat_means(data: np.ndarray) -> np.ndarray:
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


FRAME_CANDIDATES = {  # tried when checking that Brush exported in the COLMAP world frame
    "identity": np.eye(3), "flip_x": np.diag([-1.0, 1, 1]), "flip_y": np.diag([1.0, -1, 1]),
    "flip_z": np.diag([1.0, 1, -1]), "flip_yz": np.diag([1.0, -1, -1]),
    "swap_yz": np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
}


def frame_check(sfm_xyz: np.ndarray, means: np.ndarray, scale: float) -> dict:
    """Median distance from SfM points to the nearest splat mean under each candidate axis transform (applied to
    the SfM points). The splats are in the COLMAP frame when 'identity' wins by a clear factor and is tiny vs scale."""
    from scipy.spatial import cKDTree

    tree = cKDTree(means)
    res = {}
    for name, T in FRAME_CANDIDATES.items():
        d, _ = tree.query(np.asarray(sfm_xyz, dtype=np.float64) @ T.T, k=1)
        res[name] = round(float(np.median(d)), 6)
    best = min(res, key=res.get)
    second = sorted(res.values())[1]
    return {"median_nn": res, "best": best, "scale": round(float(scale), 6),
            "best_ratio_to_scale": round(res[best] / max(scale, 1e-12), 6),
            "best_vs_second": round(res[best] / max(second, 1e-12), 4),
            "same_frame": bool(best == "identity" and res[best] < 0.02 * scale)}


SPLAT_MAX_SCALE_FRAC = 0.05  # drop Gaussians whose largest axis (1 sigma) exceeds 5% of the car-box half diagonal
SPLAT_INSIDE_SIGMAS = 2.0  # ... or whose mean +- 2 sigma (largest axis) leaves the car box: the "starburst" streaks


def splat_keep(data: np.ndarray, box: dict, T: np.ndarray, *, max_scale_frac: float = SPLAT_MAX_SCALE_FRAC,
               inside_sigmas: float = SPLAT_INSIDE_SIGMAS) -> tuple[np.ndarray, dict]:
    """bool keep per Gaussian: mean inside the car box, largest scale <= max_scale_frac x box extent, and the
    mean +- inside_sigmas x largest scale still inside the box. Scales are log-scales (standard 3DGS PLY)."""
    means = splat_means(data) @ T  # T^-1 = T for the candidate transforms
    axes = np.asarray(box["axes"], dtype=np.float64)
    q = (means - np.asarray(box["origin"], dtype=np.float64)) @ axes.T
    lo, hi = np.asarray(box["lo"], dtype=np.float64), np.asarray(box["hi"], dtype=np.float64)
    inside = np.all((q >= lo) & (q <= hi), axis=1)
    names = [n for n in ("scale_0", "scale_1", "scale_2") if n in (data.dtype.names or ())]
    if len(names) == 3:
        smax = np.exp(np.max(np.stack([data[n] for n in names], axis=1).astype(np.float64), axis=1))
    else:
        smax = np.zeros(len(data))
    small = smax <= max_scale_frac * float(box["extent"])
    r = (inside_sigmas * smax)[:, None]
    contained = np.all((q - r >= lo) & (q + r <= hi), axis=1)
    keep = inside & small & contained
    stats = {"n_mean_inside_box": int(inside.sum()), "n_dropped_too_large": int((inside & ~small).sum()),
             "n_dropped_crossing_box": int((inside & small & ~contained).sum()), "max_scale_frac": max_scale_frac,
             "inside_sigmas": inside_sigmas}
    return keep, stats


def crop_splat(dataset: Path, splat: Path, out: Path) -> dict:
    """Keep the car's Gaussians (splat_keep) using the car box from sfm_report.json and sfm_points.npz."""
    dataset = Path(dataset)
    sfm = json.loads((dataset / "sfm_report.json").read_text())
    box = sfm.get("car_box")
    if not box:
        return {"ok": False, "error": "no car box (no car masks or too few car points)"}
    header, data = read_splat_ply(splat)
    means = splat_means(data)
    with np.load(dataset / "sfm_points.npz") as z:
        xyz, is_car = z["xyz"].astype(np.float64), z["is_car"].astype(bool)
    check = frame_check(xyz[is_car] if is_car.sum() >= 50 else xyz, means, 2 * float(box["extent"]))
    keep, stats = splat_keep(data, box, FRAME_CANDIDATES[check["best"]])
    write_splat_ply(out, header, data[keep])
    return {"ok": True, "frame_check": check, "transform": check["best"], "n_splats": int(len(data)),
            "n_splats_car": int(keep.sum()), **stats, "car_ply_bytes": Path(out).stat().st_size, "car_box": box}


def run_brush(dataset: Path, *, steps: int, eval_every_n: int = 10, max_splats: int | None = None,
              sh_degree: int | None = None, name: str = "splat", brush: Path = BRUSH,
              timeout_s: int | None = None, masks_dir: Path | None = None) -> dict:
    """Train Gaussian splats with the Brush binary on dataset (COLMAP layout) -> dataset/brush/<name>.ply.

    Brush prints no PSNR, so the report's psnr comes from eval_psnr() on the eval renders it saves to disk."""
    brush = Path(brush)
    if not brush.exists():
        raise RuntimeError(f"Brush binary missing: {brush}")
    dataset = Path(dataset).resolve()  # Brush runs with cwd=out, so relative paths would break
    out = dataset / "brush"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(brush), str(dataset), "--total-steps", str(steps), "--export-every", str(steps),
           "--export-path", str(out), "--export-name", f"{name}.ply", "--eval-split-every", str(eval_every_n),
           "--eval-every", str(steps), "--eval-save-to-disk"]
    if max_splats:
        cmd += ["--max-splats", str(max_splats)]
    if sh_degree is not None:
        cmd += ["--sh-degree", str(sh_degree)]
    t0 = time.perf_counter()
    log = out / f"{name}.log"
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=out, timeout=timeout_s)
    secs = time.perf_counter() - t0
    text = log.read_text(errors="replace")
    psnr_lines = [line for line in text.splitlines() if "psnr" in line.lower()]
    try:
        psnr_eval = eval_psnr(out / f"eval_{steps}", dataset / "images", masks_dir)
    except Exception as exc:  # noqa: BLE001 - a metric failure must not fail the build
        psnr_eval = {"error": f"{type(exc).__name__}: {exc}"}
    plys = sorted(out.glob("*.ply"))
    report = {"stage": "brush", "exit_code": proc.returncode, "seconds": round(secs, 1), "steps": steps,
              "eval_split_every": eval_every_n, "max_splats": max_splats, "sh_degree": sh_degree,
              "ply": [str(p.relative_to(dataset)) for p in plys],
              "ply_mb": [round(p.stat().st_size / 1e6, 1) for p in plys], "psnr_lines": psnr_lines[-10:],
              "psnr": (psnr_eval or {}).get("mean_db", parse_psnr("\n".join(psnr_lines))),
              "psnr_car": (psnr_eval or {}).get("mean_car_db"), "psnr_eval": psnr_eval,
              "log_tail": _tail(text, 800), "command": " ".join(cmd)}
    (dataset / f"brush_report_{name}.json").write_text(json.dumps(report, indent=2))
    return report


def prepare_images(frames_dir: Path, dataset: Path, masks_dir: Path | None = None) -> tuple[int, int]:
    """Copy the blurred frames into dataset/images and their car masks (<stem>.png) into dataset/car_masks
    (not "masks", which Brush would read as training masks). Returns (n_images, n_masks)."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise RuntimeError(f"frames dir not found: {frames_dir}")
    images, cmasks = dataset / "images", dataset / "car_masks"
    for d in (images, cmasks):
        if d.exists():
            shutil.rmtree(d)
    images.mkdir(parents=True)
    n = n_masks = 0
    for p in sorted(frames_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.copy2(p, images / p.name)
            n += 1
            mp = Path(masks_dir) / f"{p.stem}.png" if masks_dir else None
            if mp is not None and mp.is_file():
                cmasks.mkdir(exist_ok=True)
                shutil.copy2(mp, cmasks / mp.name)
                n_masks += 1
    return n, n_masks


WEB_MAX_SPLATS = 500_000  # phone-friendly cap for the served splat (uncalibrated default)


def _read_ply_header(path: Path) -> tuple[int, list[tuple[str, str]], int]:
    """(vertex count, [(type, name)], byte offset of the data) for a binary little-endian PLY."""
    with open(path, "rb") as fh:
        head = b""
        while b"end_header\n" not in head:
            chunk = fh.read(4096)
            if not chunk:
                raise ValueError(f"no end_header in {path}")
            head += chunk
    text, _ = head.split(b"end_header\n", 1)
    offset = len(text) + len(b"end_header\n")
    lines = text.decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary_little_endian PLY is supported")
    n = next(int(line.split()[2]) for line in lines if line.startswith("element vertex"))
    props = [(line.split()[1], line.split()[2]) for line in lines if line.startswith("property")]
    return n, props, offset


def web_splat(src: Path, dst: Path, *, max_splats: int = WEB_MAX_SPLATS, keep_sh: bool = False) -> dict:
    """Phone-friendly copy of a 3DGS PLY: drop the higher-order SH colour terms (f_rest_*) unless keep_sh, and keep
    at most max_splats splats ranked by importance = sigmoid(opacity) * exp(mean(log scales)) (opaque, large splats
    first). Returns counts and sizes. The full-quality file is left untouched."""
    n, props, offset = _read_ply_header(Path(src))
    np_type = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    dtype = np.dtype([(name, np_type[t]) for t, name in props])
    data = np.fromfile(src, dtype=dtype, count=n, offset=offset)
    names = [name for _, name in props]
    keep_names = [x for x in names if keep_sh or not x.startswith("f_rest_")]
    sel = np.arange(n)
    if n > max_splats and {"opacity", "scale_0", "scale_1", "scale_2"} <= set(names):
        op = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
        size = np.exp((data["scale_0"].astype(np.float64) + data["scale_1"] + data["scale_2"]) / 3.0)
        sel = np.sort(np.argpartition(-(op * size), max_splats - 1)[:max_splats])
    out = np.empty(len(sel), dtype=np.dtype([(x, dtype[x]) for x in keep_names]))
    for x in keep_names:
        out[x] = data[x][sel]
    inv = {v: k for k, v in np_type.items()}
    header = ["ply", "format binary_little_endian 1.0", "comment AvtoPassport web export (SH degree 0)" if not keep_sh
              else "comment AvtoPassport web export", f"element vertex {len(out)}"]
    header += [f"property {inv[dtype[x].str if dtype[x].str in inv else '<f4']} {x}" for x in keep_names]
    header.append("end_header")
    dst = Path(dst)
    with open(dst, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(out.tobytes())
    return {"n_in": int(n), "n_out": int(len(out)), "sh_kept": keep_sh, "bytes_in": Path(src).stat().st_size,
            "bytes_out": dst.stat().st_size, "max_splats": max_splats}


def finalize(dataset: Path, splat: Path, report: dict) -> Path:
    """Car crop of a trained splat (dataset/splat_car.ply) when the SfM stage found a car box; updates report.
    Returns the file to serve (the crop, else the full splat)."""
    dataset = Path(dataset)
    out = dataset / "splat_car.ply"
    try:
        crop = crop_splat(dataset, splat, out)
    except Exception as exc:  # noqa: BLE001 - the full splat is still a valid result
        crop = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["crop"] = {k: v for k, v in crop.items() if k != "car_box"}
    if crop.get("ok") and crop.get("n_splats_car"):
        serve = out
    else:
        out.unlink(missing_ok=True)
        serve = Path(splat)
    # Phones: a 2M-splat SH3 model is several hundred MB. Serve a web copy; the full-quality files stay on disk.
    web = dataset / "splat_web.ply"
    try:
        report["web"] = web_splat(serve, web)
        report["web"]["source"] = serve.name
        return web
    except Exception as exc:  # noqa: BLE001 - fall back to the full file
        report["web"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        web.unlink(missing_ok=True)
        return serve


def build_hq(frames_dir: Path, out_dir: Path, *, steps: int = DEFAULT_STEPS, max_splats: int | None = None,
             pins2d: list[dict] | None = None, masks_dir: Path | None = None, python: Path = DEFAULT_PYTHON,
             brush: Path = BRUSH, sfm_timeout_s: int = 3600, brush_timeout_s: int = 4 * 3600,
             parts3d: bool = True) -> dict:
    """SfM (in .venv-3d) + Brush + car crop -> out_dir/splat.ply, splat_car.ply, hq_report.json. Never raises.

    Returns {"ok", "error", "splat" (car crop when available, else full), "splat_full", "report", "seconds_wall",
    "out_dir"}; report has the measured seconds per stage, n_images, n_registered, mean_reprojection_error_px,
    psnr / psnr_car (eval views), view (orbit around the car box), car_box, crop (frame check + counts), pins.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hq_report.json").unlink(missing_ok=True)
    res: dict = {"ok": False, "error": None, "splat": None, "splat_full": None, "report": None, "seconds_wall": 0.0,
                 "out_dir": str(out_dir)}
    t_all = time.perf_counter()
    report: dict = {"ok": False, "steps": steps, "max_splats": max_splats, "seconds": {}}
    try:
        n_img, n_masks = prepare_images(frames_dir, out_dir, masks_dir)
        report.update(n_images=n_img, n_car_masks=n_masks)
        if n_img < MIN_IMAGES:
            raise RuntimeError(f"need >= {MIN_IMAGES} blurred frames for SfM, found {n_img}")
        cmd = [str(python), "-m", "avtopassport.recon3d.hq", "sfm", "--dataset", str(out_dir)]
        if n_masks:
            cmd += ["--masks-dir", str(out_dir / "car_masks")]
        if pins2d:
            pins_path = out_dir / "pins2d.json"
            pins_path.write_text(json.dumps(pins2d))
            cmd += ["--pins-json", str(pins_path)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=sfm_timeout_s, env=_env(),
                              cwd=str(REPO_ROOT))
        report["seconds"]["sfm"] = round(time.perf_counter() - t0, 1)
        (out_dir / "sfm.log").write_text(f"$ {' '.join(cmd)}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
        sfm_path = out_dir / "sfm_report.json"
        if proc.returncode != 0 or not sfm_path.exists():
            raise RuntimeError(f"SfM failed (exit {proc.returncode}): {_tail(proc.stderr, 500)}")
        sfm = json.loads(sfm_path.read_text())
        report.update(n_registered=sfm.get("n_registered"), n_points3d=sfm.get("n_points3d"),
                      mean_reprojection_error_px=sfm.get("mean_reprojection_error_px"), view=sfm.get("view"),
                      car_box=sfm.get("car_box"), pins=sfm.get("pins") or [], sfm_seconds_detail=sfm.get("seconds"))
        if parts3d:  # the parts 3D model needs only the SfM: build it now (~30 s), before the long Brush training
            from avtopassport.recon3d.parts3d import build_parts3d  # noqa: PLC0415

            p3 = build_parts3d(out_dir, device=_torch_device())
            report["parts3d"] = {k: p3.get(k) for k in ("ok", "error", "seconds")}
            report["seconds"]["parts3d"] = p3.get("seconds")
        br = run_brush(out_dir, steps=steps, max_splats=max_splats, name="splat", brush=brush,
                       timeout_s=brush_timeout_s, masks_dir=out_dir / "car_masks" if n_masks else None)
        report["seconds"]["brush"] = br["seconds"]
        report.update(psnr=br.get("psnr"), psnr_car=br.get("psnr_car"), psnr_eval=br.get("psnr_eval"),
                      psnr_lines=br.get("psnr_lines"), brush_exit_code=br["exit_code"])
        src = out_dir / "brush" / "splat.ply"
        if br["exit_code"] != 0 or not src.is_file():
            raise RuntimeError(f"Brush failed (exit {br['exit_code']}): {br.get('log_tail', '')[-500:]}")
        full = out_dir / "splat.ply"
        shutil.copyfile(src, full)
        report["splat_bytes"] = full.stat().st_size
        t0 = time.perf_counter()
        serve = finalize(out_dir, full, report)
        report["seconds"]["crop"] = round(time.perf_counter() - t0, 1)
        report["ok"] = True
        res.update(ok=True, splat=str(serve), splat_full=str(full))
    except subprocess.TimeoutExpired as exc:
        res["error"] = report["error"] = f"timed out after {exc.timeout} s: {' '.join(map(str, exc.cmd))[:200]}"
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised
        res["error"] = report["error"] = f"{type(exc).__name__}: {exc}"
    report["seconds"]["total"] = round(time.perf_counter() - t_all, 1)
    res["seconds_wall"] = report["seconds"]["total"]
    res["report"] = report
    (out_dir / "hq_report.json").write_text(json.dumps(report, indent=2))
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI stages: sfm / analyse (in .venv-3d: pycolmap), brush, crop (main venv), all (build_hq)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sfm", "analyse", "brush", "crop", "all"],
                    help="analyse = view / car box / pins from an existing sparse/0; crop = PSNR + car crop of a "
                         "trained splat")
    ap.add_argument("--dataset", type=Path, required=True, help="dataset / output dir (images/ for sfm)")
    ap.add_argument("--frames-dir", type=Path, default=None, help="for 'all': blurred frames to copy into images/")
    ap.add_argument("--masks-dir", type=Path, default=None, help="car masks <stem>.png (255 = car), any resolution")
    ap.add_argument("--splat", type=Path, default=None, help="for 'crop': trained splat .ply (default brush/<name>.ply)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--max-splats", type=int, default=None)
    ap.add_argument("--sh-degree", type=int, default=None)
    ap.add_argument("--name", default="splat")
    ap.add_argument("--pins-json", type=Path, default=None, help="2D damage boxes to place as 3D pins (sfm)")
    a = ap.parse_args(argv)
    ds = a.dataset if a.dataset.is_absolute() else REPO_ROOT / a.dataset
    pins2d = json.loads(a.pins_json.read_text()) if a.pins_json else None
    if a.stage == "sfm":
        rep = run_sfm(ds, pins2d=pins2d, masks_dir=a.masks_dir)
    elif a.stage == "analyse":
        import pycolmap

        rec = pycolmap.Reconstruction(str(ds / "sparse" / "0"))
        path = ds / "sfm_report.json"
        rep = json.loads(path.read_text()) if path.exists() else {"stage": "sfm"}
        rep.update(analyse_model(ds, rec, pins2d=pins2d, masks_dir=a.masks_dir))
        path.write_text(json.dumps(rep, indent=2))
    elif a.stage == "brush":
        rep = run_brush(ds, steps=a.steps, max_splats=a.max_splats, sh_degree=a.sh_degree, name=a.name,
                        masks_dir=a.masks_dir)
    elif a.stage == "crop":
        splat = a.splat or ds / "brush" / f"{a.name}.ply"
        splat = splat if splat.is_absolute() else REPO_ROOT / splat
        t0 = time.perf_counter()
        rep = {"splat": str(splat), "psnr_eval": eval_psnr(ds / "brush" / f"eval_{a.steps}", ds / "images",
                                                            a.masks_dir)}
        rep["serve"] = str(finalize(ds, splat, rep))
        rep["seconds"] = round(time.perf_counter() - t0, 1)
        (ds / f"crop_report_{splat.stem}.json").write_text(json.dumps(rep, indent=2))
    else:
        if a.frames_dir is None:
            ap.error("'all' needs --frames-dir")
        rep = build_hq(a.frames_dir, ds, steps=a.steps, max_splats=a.max_splats, masks_dir=a.masks_dir,
                       pins2d=pins2d)
    print(json.dumps(rep, indent=2, default=str))
    return 0 if (a.stage != "all" or rep.get("ok")) else 1


if __name__ == "__main__":
    sys.exit(main())
