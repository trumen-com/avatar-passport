"""3D reconstruction worker: Depth Anything 3 on anonymised frames -> point cloud, cameras, depth, previews.

Runs ONLY in the separate 3D venv (DA3 pins numpy<2); the main app calls it through `build.build()`:
    .venv-3d/bin/python -m avtopassport.recon3d.worker --frames-dir D --out-dir O --device cuda

Writes into O (result.json is written last, so its presence means success):
- points.ply      binary PLY, xyz float32 + rgb uchar, <= --max-points (full resolution cloud)
- points_web.ply  same cloud decimated to <= 600k points (for phones)
- cameras.json    process_res, processed_hw [h, w], frames, per-frame intrinsics / w2c (OpenCV convention), up, center
- depth.npz       depth / conf float16 [N, h, w], valid uint8 [N, h, w] (sky / car-mask / finite), frames
- preview.jpg     2x2 grid of z-buffer splat renders; preview_<view>.jpg singles
- result.json     every number measured by this run

Top-level imports are numpy + stdlib only (helpers are unit-tested in the main venv); torch / DA3 / PIL are
imported inside the functions that need them. Never import cv2-dependent avtopassport modules here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MODEL_REVISIONS = {  # pinned 2026-10-11 from https://huggingface.co/api/models/<repo> (both Apache-2.0)
    "depth-anything/DA3-BASE": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
    "depth-anything/DA3-SMALL": "e08cab65ca0ec38e7826075418411ab90cab4da3",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_PROCESS_RES = {"cuda": 756, "cpu": 504}
OOM_FALLBACK_RES = 504
WEB_MAX_POINTS = 600_000
MASK_DILATE_FRAC = 0.02
PREVIEW_SIZE = (960, 640)  # w, h of each view
# name -> (azimuth deg, elevation deg). Azimuth 0 = the first frame's viewpoint (the car's own front is unknown,
# so views are named by angle, not by "front" / "rear").
PREVIEW_VIEWS = {"az045": (45.0, 15.0), "az090": (90.0, 15.0), "az180": (180.0, 15.0), "top": (45.0, 60.0)}
PREVIEW_CROP = 1.5  # keep points within PREVIEW_CROP x walkaround radius of the car centre
PREVIEW_EYE = 2.5  # virtual camera distance, x walkaround radius (outside the crop sphere)
PREVIEW_BG = (24, 24, 28)


class WorkerError(RuntimeError):
    """A failure with a message meant for the caller."""


def select_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    """Sorted image files, evenly spread to at most max_frames."""
    if not frames_dir.is_dir():
        raise WorkerError(f"frames dir not found: {frames_dir}")
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise WorkerError(f"no frames in {frames_dir}")
    if len(files) > max_frames:
        idx = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    return files


def backproject(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points (float32 [M, 3]) for the pixels in mask; pixel centres at (u+0.5, v+0.5)."""
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    pix = np.stack([u + 0.5, v + 0.5, np.ones(len(u))], axis=0)
    cam = (np.linalg.inv(np.asarray(K, dtype=np.float64)) @ pix) * z
    w2c = np.asarray(w2c, dtype=np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3:4]
    return (R.T @ (cam - t)).T.astype(np.float32)


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    """Binary little-endian PLY with float xyz and uchar rgb."""
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r+1)^2 square, separable, numpy only."""
    out = mask.astype(bool)
    if r <= 0:
        return out
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad)
        acc = np.zeros_like(out)
        for s in range(2 * r + 1):
            acc |= np.take(p, np.arange(s, s + n), axis=axis)
        out = acc
    return out


def load_car_mask(path: Path, hw: tuple[int, int], dilate_frac: float = MASK_DILATE_FRAC) -> np.ndarray:
    """Car mask PNG (255 = car, any resolution) -> bool [h, w] at the processed size, dilated."""
    from PIL import Image

    h, w = hw
    with Image.open(path) as im:
        m = np.asarray(im.convert("L").resize((w, h), Image.NEAREST)) > 127
    return dilate(m, int(round(dilate_frac * max(h, w))))


def subsample(n: int, k: int, seed: int = 0) -> np.ndarray | None:
    """Sorted random indices of k out of n, or None when n <= k."""
    if n <= k:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def camera_centers(w2cs: np.ndarray) -> np.ndarray:
    """Camera centres in world coordinates, C = -R^T t."""
    R, t = w2cs[:, :3, :3], w2cs[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def scene_frame(pts: np.ndarray, w2cs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(center, up, forward0, radius): car centre = median of points; up = normal of the camera-centre plane
    (sign from the cameras' own up), forward0 = direction to the first camera; radius = median camera distance."""
    center = np.median(pts, axis=0).astype(np.float64)
    w2cs = np.asarray(w2cs, dtype=np.float64)
    C = camera_centers(w2cs)
    cam_up = -w2cs[:, 1, :3]  # world direction of each camera's -y (image up)
    up = cam_up.mean(axis=0)
    if len(C) >= 3:
        _, s, vt = np.linalg.svd(C - C.mean(axis=0), full_matrices=False)
        if s[1] > 1e-6 and s[2] < 0.5 * s[1]:  # cameras spread in a plane -> walkaround circle
            up = vt[2] * (1.0 if np.dot(vt[2], up) >= 0 else -1.0)
    up = up / (np.linalg.norm(up) + 1e-12)
    f0 = C[0] - center
    f0 = f0 - np.dot(f0, up) * up
    if np.linalg.norm(f0) < 1e-9:
        f0 = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    f0 = f0 / np.linalg.norm(f0)
    d = C - center
    d = d - np.outer(d @ up, up)
    radius = float(np.median(np.linalg.norm(d, axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 50))
    return center, up, f0, max(radius, 1e-6)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera rotation (rows: right, down, forward; OpenCV convention)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:  # looking straight along up
        r = np.cross(f, [1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.cross(f, [0.0, 1.0, 0.0])
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


def render_splats(pts: np.ndarray, cols: np.ndarray, R: np.ndarray, eye: np.ndarray,
                  size: tuple[int, int] = PREVIEW_SIZE, splat: int = 2, fit: np.ndarray | None = None,
                  bg: tuple[int, int, int] = PREVIEW_BG) -> np.ndarray:
    """Pinhole z-buffer render: each point is a splat x splat px square, nearest wins. Focal length and principal
    point are fitted so the 1-99 percentile box of `fit` points (default: all) is centred and fills ~84% of the
    image. Returns uint8 [h, w, 3]."""
    w, h = size
    cam = (pts.astype(np.float64) - eye) @ R.T
    fit_cam = cam if fit is None else (fit.astype(np.float64) - eye) @ R.T
    near = 1e-3 * float(np.median(np.abs(fit_cam[:, 2]))) + 1e-9 if len(fit_cam) else 1e-9
    fc = fit_cam[fit_cam[:, 2] > near]
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = bg
    if len(fc) == 0:
        return img
    a_lo, a_hi = np.percentile(fc[:, 0] / fc[:, 2], [1, 99])
    b_lo, b_hi = np.percentile(fc[:, 1] / fc[:, 2], [1, 99])
    focal = 0.84 * min(w / max(a_hi - a_lo, 1e-6), h / max(b_hi - b_lo, 1e-6))
    cx, cy = w / 2 - focal * (a_lo + a_hi) / 2, h / 2 - focal * (b_lo + b_hi) / 2
    keep = cam[:, 2] > near
    cam, c = cam[keep], cols[keep]
    u = np.floor(focal * cam[:, 0] / cam[:, 2] + cx).astype(np.int64)
    v = np.floor(focal * cam[:, 1] / cam[:, 2] + cy).astype(np.int64)
    order = np.argsort(cam[:, 2], kind="stable")  # nearest first
    u, v, c = u[order], v[order], c[order]
    lins, srcs = [], []
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            lins.append(vv[ok] * w + uu[ok])
            srcs.append(np.nonzero(ok)[0])
    lin, src = np.concatenate(lins), np.concatenate(srcs)
    o = np.argsort(src, kind="stable")  # back to depth order across the splat offsets
    lin, src = lin[o], src[o]
    uniq, first = np.unique(lin, return_index=True)  # first occurrence = nearest point
    img.reshape(-1, 3)[uniq] = c[src[first]]
    return img


def write_previews(out_dir: Path, pts: np.ndarray, cols: np.ndarray, w2cs: np.ndarray) -> dict[str, str]:
    """Render PREVIEW_VIEWS around the car centre; save singles + a 2x2 grid (JPEG q92). Returns file names."""
    from PIL import Image, ImageDraw

    center, up, f0, radius = scene_frame(pts, w2cs)
    side = np.cross(up, f0)
    core = np.linalg.norm(pts - center, axis=1) <= PREVIEW_CROP * radius  # drops far background
    p, c = pts[core], cols[core]
    if len(p) == 0:
        p, c = pts, cols
    files: dict[str, str] = {}
    tiles = []
    for name, (az, el) in PREVIEW_VIEWS.items():
        a, e = np.radians(az), np.radians(el)
        direction = np.cos(e) * (np.cos(a) * f0 + np.sin(a) * side) + np.sin(e) * up
        eye = center + PREVIEW_EYE * radius * direction
        img = render_splats(p, c, look_at(eye, center, up), eye)
        Image.fromarray(img).save(out_dir / f"preview_{name}.jpg", quality=92)
        files[name] = f"preview_{name}.jpg"
        tile = Image.fromarray(img)
        label = f"azimuth {az:.0f} deg from first frame, elevation {el:.0f} deg"
        ImageDraw.Draw(tile).text((12, 10), label, fill=(235, 235, 235))
        tiles.append(tile)
    w, h = PREVIEW_SIZE
    grid = Image.new("RGB", (2 * w, 2 * h), PREVIEW_BG)
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % 2) * w, (i // 2) * h))
    grid.save(out_dir / "preview.jpg", quality=92)
    files["grid"] = "preview.jpg"
    return files


def _is_cuda_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run(args: argparse.Namespace) -> dict:
    """Load DA3, infer, filter, write all outputs; returns the result dict (also written to result.json)."""
    t_start = time.perf_counter()
    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError("device cuda requested but torch.cuda.is_available() is False")
    if args.threads:
        torch.set_num_threads(args.threads)
    if device == "cpu":
        # DA3 picks fp16 autocast when CUDA bf16 is unavailable; on CPU use bf16 autocast instead.
        torch.cuda.is_bf16_supported = lambda *a, **k: True  # type: ignore[assignment]
    from depth_anything_3.api import DepthAnything3

    if args.model not in MODEL_REVISIONS:
        raise WorkerError(f"model {args.model} not allowed (Apache-2.0 only): {sorted(MODEL_REVISIONS)}")
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "result.json"
    if stale.exists():
        stale.unlink()
    frames = select_frames(args.frames_dir, args.max_frames)
    res_req = args.process_res or DEFAULT_PROCESS_RES["cuda" if device.startswith("cuda") else "cpu"]

    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model, revision=MODEL_REVISIONS[args.model]).to(device)
    t_load = time.perf_counter() - t0

    process_res, oom_retry = res_req, None
    t0 = time.perf_counter()
    while True:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            pred = model.inference([str(p) for p in frames], process_res=process_res)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            break
        except Exception as exc:  # noqa: BLE001 - retry once on CUDA OOM only
            if not (device.startswith("cuda") and oom_retry is None and process_res > OOM_FALLBACK_RES
                    and _is_cuda_oom(exc)):
                raise
            oom_retry = {"from_res": process_res, "to_res": OOM_FALLBACK_RES, "error": str(exc)[:300]}
            process_res = OOM_FALLBACK_RES
            t0 = time.perf_counter()
    t_infer = time.perf_counter() - t0
    peak_cuda = round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.startswith("cuda") else None

    t0 = time.perf_counter()
    depth = np.asarray(pred.depth, dtype=np.float32)
    n, h, w = depth.shape
    conf = np.asarray(pred.conf, dtype=np.float32) if pred.conf is not None else np.ones_like(depth)
    thr = float(np.percentile(conf, args.conf_percentile))
    w2cs = np.asarray(pred.extrinsics, dtype=np.float64)[:, :3, :4]
    Ks = np.asarray(pred.intrinsics, dtype=np.float64)
    images = np.asarray(pred.processed_images)
    valid = np.isfinite(depth) & (depth > 0)
    if pred.sky is not None:
        valid &= ~(np.asarray(pred.sky) > 0.5)
    n_masks = 0
    if args.masks_dir is not None:
        for i, f in enumerate(frames):
            mp = args.masks_dir / f"{f.stem}.png"
            if mp.exists():
                valid[i] &= load_car_mask(mp, (h, w))
                n_masks += 1
    all_pts, all_cols, cams = [], [], []
    for i, f in enumerate(frames):
        keep = valid[i] & (conf[i] >= thr)
        all_pts.append(backproject(depth[i], Ks[i], w2cs[i], keep))
        all_cols.append(images[i][keep].astype(np.uint8))
        with _open_image(f) as im:
            orig_wh = list(im.size)
        cams.append({"frame": f.name, "orig_wh": orig_wh, "intrinsics": Ks[i].tolist(), "w2c": w2cs[i].tolist()})
    pts, cols = np.concatenate(all_pts), np.concatenate(all_cols)
    n_all = len(pts)
    if n_all == 0:
        raise WorkerError("no points survived the confidence / sky / car-mask filters")
    sel = subsample(n_all, args.max_points)
    if sel is not None:
        pts, cols = pts[sel], cols[sel]
    write_ply(out / "points.ply", pts, cols)
    sel_web = subsample(len(pts), WEB_MAX_POINTS, seed=1)
    pts_web, cols_web = (pts, cols) if sel_web is None else (pts[sel_web], cols[sel_web])
    write_ply(out / "points_web.ply", pts_web, cols_web)
    center, up, f0, radius = scene_frame(pts, w2cs)
    (out / "cameras.json").write_text(json.dumps({
        "process_res": process_res, "processed_hw": [h, w], "frames": [f.name for f in frames],
        "convention": "OpenCV camera (x right, y down, z forward); w2c = world-to-camera [3x4]; "
                      "intrinsics in processed pixels, pixel centres at (u+0.5, v+0.5); world = R^T (cam - t)",
        "center": center.tolist(), "up": up.tolist(), "forward0": f0.tolist(), "walkaround_radius": radius,
        "cameras": cams,
    }))
    np.savez_compressed(out / "depth.npz", depth=depth.astype(np.float16), conf=conf.astype(np.float16),
                        valid=valid.astype(np.uint8), frames=np.array([f.name for f in frames]))
    t_post = time.perf_counter() - t0

    t0 = time.perf_counter()
    previews = write_previews(out, pts, cols, w2cs) if not args.no_preview else {}
    t_prev = time.perf_counter() - t0

    import importlib.metadata as md

    result = {
        "ok": True, "device": device, "model": args.model, "revision": MODEL_REVISIONS[args.model],
        "licence": "Apache-2.0 (code and these weights)", "frames_dir": str(args.frames_dir),
        "n_frames": n, "process_res_requested": res_req, "process_res": process_res, "oom_retry": oom_retry,
        "processed_hw": [h, w], "masks_dir": str(args.masks_dir) if args.masks_dir else None,
        "n_masks_used": n_masks, "conf_percentile": args.conf_percentile, "conf_threshold": thr,
        "n_points_after_filter": int(n_all), "n_points": int(len(pts)), "n_points_web": int(len(pts_web)),
        "files": {"points": "points.ply", "points_web": "points_web.ply", "cameras": "cameras.json",
                  "depth": "depth.npz", "previews": previews},
        "ply_bytes": (out / "points.ply").stat().st_size, "ply_web_bytes": (out / "points_web.ply").stat().st_size,
        "seconds": {"model_load": round(t_load, 3), "inference": round(t_infer, 3),
                    "postprocess_write": round(t_post, 3), "preview": round(t_prev, 3),
                    "total": round(time.perf_counter() - t_start, 3)},
        "peak_cuda_mem_mb": peak_cuda,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "torch_threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "loadavg_1m": os.getloadavg()[0],
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                     "depth_anything_3": md.version("depth-anything-3")},
        "is_metric": str(getattr(pred, "is_metric", None)),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _open_image(path: Path):  # noqa: ANN202 - PIL image context manager
    from PIL import Image

    return Image.open(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments (see module docstring)."""
    ap = argparse.ArgumentParser(description="DA3 3D reconstruction worker (run with .venv-3d python)")
    ap.add_argument("--frames-dir", type=Path, required=True, help="anonymised frames only (never data/raw)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--process-res", type=int, default=None, help="default 756 on cuda, 504 on cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--masks-dir", type=Path, default=None, help="optional <stem>.png car masks (255 = car)")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    ap.add_argument("--no-preview", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: exit 0 and print result.json on success, exit 1 with a message on failure."""
    args = parse_args(argv)
    try:
        result = run(args)
    except WorkerError as exc:
        print(f"recon3d worker failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report anything to the caller
        traceback.print_exc(file=sys.stderr)
        print(f"recon3d worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
