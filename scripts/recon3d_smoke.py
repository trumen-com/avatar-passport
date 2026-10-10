"""3D reconstruction smoke test: Depth Anything 3 on the anonymised walkaround frames.

Runs in the separate 3D environment (DA3 pins numpy<2):
    .venv-3d/bin/python scripts/recon3d_smoke.py --device cuda --run R3D-gpu
    CUDA_VISIBLE_DEVICES= .venv-3d/bin/python scripts/recon3d_smoke.py --device cpu --run R3D-cpu

Input: data/processed/frames/*.jpg (already blurred; never data/raw). Output: outputs/recon3d/<run>/
- points.ply      coloured point cloud (binary PLY), the format the web viewer loads
- cameras.json    per-frame intrinsics / world-to-camera extrinsics (for projecting damage pins later)
- preview.jpg     three orthographic views of the point cloud (for the gate check "recognisable car")
- results.json    timings, sizes, versions, model revision - every number here is measured by this script
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = {  # pinned 2026-10-11 from https://huggingface.co/api/models/<repo> (both Apache-2.0)
    "depth-anything/DA3-BASE": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
    "depth-anything/DA3-SMALL": "e08cab65ca0ec38e7826075418411ab90cab4da3",
}


def select_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no frames in {frames_dir} (run scripts/blur_raw.py first)")
    if len(files) > max_frames:  # evenly spread over the walkaround
        idx = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    return files


def backproject(depth: np.ndarray, conf: np.ndarray, K: np.ndarray, w2c: np.ndarray, rgb: np.ndarray,
                conf_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours for the pixels in conf_mask. w2c is 3x4 or 4x4 world-to-camera."""
    h, w = depth.shape
    v, u = np.nonzero(conf_mask)
    z = depth[v, u]
    pix = np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=np.float64)], axis=0)
    cam = (np.linalg.inv(K) @ pix) * z
    R, t = w2c[:3, :3], w2c[:3, 3:4]
    world = (R.T @ (cam - t)).T
    return world.astype(np.float32), rgb[v, u].astype(np.uint8)


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def preview(path: Path, pts: np.ndarray, cols: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = pts - np.median(pts, axis=0)
    # principal axes so the views are top / side / front regardless of the camera convention
    _, _, vt = np.linalg.svd(c[np.random.default_rng(0).choice(len(c), min(len(c), 50000), replace=False)],
                             full_matrices=False)
    p = c @ vt.T
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    keep = np.all((p >= lo) & (p <= hi), axis=1)
    p, col = p[keep], cols[keep] / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j), name in zip(axes, [(0, 1), (0, 2), (1, 2)], ["axes 1-2 (largest)", "axes 1-3", "axes 2-3"]):
        ax.scatter(p[:, i], p[:, j], c=col, s=0.3, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "frames")
    ap.add_argument("--model", default="depth-anything/DA3-BASE", choices=sorted(MODEL_REVISIONS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--run", default=None)
    args = ap.parse_args(argv)

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cpu":
        # DA3 picks fp16 autocast when CUDA bf16 is unavailable; on CPU use bf16 autocast instead.
        torch.cuda.is_bf16_supported = lambda *a, **k: True  # type: ignore[assignment]
    from depth_anything_3.api import DepthAnything3

    run = args.run or f"R3D-{args.device}"
    out = REPO_ROOT / "outputs" / "recon3d" / run
    out.mkdir(parents=True, exist_ok=True)
    frames = select_frames(args.frames_dir, args.max_frames)

    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model, revision=MODEL_REVISIONS[args.model]).to(args.device)
    t_load = time.perf_counter() - t0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pred = model.inference([str(p) for p in frames], process_res=args.process_res)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0

    t0 = time.perf_counter()
    conf = pred.conf if pred.conf is not None else np.ones_like(pred.depth)
    thr = float(np.percentile(conf, args.conf_percentile))
    all_pts, all_cols, cams = [], [], []
    for i in range(len(frames)):
        mask = (conf[i] >= thr) & np.isfinite(pred.depth[i]) & (pred.depth[i] > 0)
        if pred.sky is not None:
            mask &= ~(pred.sky[i] > 0.5)
        pts, cols = backproject(pred.depth[i], conf[i], pred.intrinsics[i], pred.extrinsics[i],
                                pred.processed_images[i], mask)
        all_pts.append(pts)
        all_cols.append(cols)
        cams.append({"frame": frames[i].name, "intrinsics": pred.intrinsics[i].tolist(),
                     "w2c": np.asarray(pred.extrinsics[i]).tolist()})
    pts, cols = np.concatenate(all_pts), np.concatenate(all_cols)
    n_all = len(pts)
    if n_all > args.max_points:
        sel = np.random.default_rng(0).choice(n_all, args.max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    write_ply(out / "points.ply", pts, cols)
    (out / "cameras.json").write_text(json.dumps({"process_res": args.process_res, "cameras": cams}))
    preview(out / "preview.jpg", pts, cols, f"{run}: DA3 {args.model.split('/')[1]} on {len(frames)} anonymised frames")
    t_post = time.perf_counter() - t0

    res = {
        "run": run, "device": args.device, "model": args.model, "revision": MODEL_REVISIONS[args.model],
        "licence": "Apache-2.0 (code and these weights)", "frames_dir": str(args.frames_dir.relative_to(REPO_ROOT)),
        "n_frames": len(frames), "process_res": args.process_res, "processed_hw": list(pred.depth.shape[1:]),
        "conf_percentile": args.conf_percentile, "conf_threshold": thr,
        "n_points_after_conf": int(n_all), "n_points_written": int(len(pts)),
        "ply_bytes": (out / "points.ply").stat().st_size,
        "seconds": {"model_load": round(t_load, 3), "inference": round(t_infer, 3), "backproject_write": round(t_post, 3)},
        "peak_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if args.device.startswith("cuda") else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "torch_threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "loadavg_1m": os.getloadavg()[0],
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
        "is_metric": str(pred.is_metric),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
