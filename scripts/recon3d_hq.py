"""High-quality 3D: COLMAP structure-from-motion on many anonymised video frames, then Gaussian splats (Brush).

The DA3 point cloud (src/avtopassport/recon3d) is the fast preview. This is the slow, accurate path the user asked
for. Calibrated camera poses come from classic SfM over ~4 fps of the walkaround video; Brush (Apache-2.0, Vulkan,
no compiler) trains photo-realistic 3D Gaussians on the calibrated views and renders held-out views for a real
accuracy number (PSNR).

Stages (run from the repo root):
    .venv-3d/bin/python scripts/recon3d_hq.py sfm   --dataset outputs/recon3d/hq   # needs images/ (anonymised)
    .venv/bin/python   scripts/recon3d_hq.py brush --dataset outputs/recon3d/hq --steps 30000
Inputs must already be anonymised (they are written by the inspection pipeline / blur step); this script never
reads data/raw. Outputs stay under the dataset directory (gitignored outputs/). Every number in the JSON reports is
measured by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BRUSH = REPO_ROOT / "tools" / "brush-dl" / "brush-app-x86_64-unknown-linux-gnu" / "brush_app"


def run_sfm(dataset: Path, *, camera_model: str = "SIMPLE_RADIAL", max_image_size: int = 1920) -> dict:
    import pycolmap

    images = dataset / "images"
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(names) < 10:
        raise SystemExit(f"need >= 10 anonymised images in {images}, found {len(names)}")
    work = dataset / "colmap"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    t = {}
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
        raise SystemExit("SfM failed: no reconstruction")
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
    final = dataset / "sparse" / "0"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True)
    rec.write(final)
    cam = next(iter(rec.cameras.values()))
    report = {
        "stage": "sfm", "n_images": len(names), "n_registered": rec.num_reg_images(),
        "n_points3d": rec.num_points3D(), "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "n_models": len(recs), "camera": {"model": str(cam.model), "width": cam.width, "height": cam.height,
                                            "params": [float(x) for x in cam.params]},
        "seconds": {k: round(v, 2) for k, v in t.items()}, "pycolmap": pycolmap.__version__,
    }
    (dataset / "sfm_report.json").write_text(json.dumps(report, indent=2))
    return report


def run_brush(dataset: Path, *, steps: int, eval_every_n: int = 10, max_splats: int | None = None,
              sh_degree: int | None = None, name: str = "splat") -> dict:
    if not BRUSH.exists():
        raise SystemExit(f"Brush binary missing: {BRUSH}")
    out = dataset / "brush"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(BRUSH), str(dataset), "--total-steps", str(steps), "--export-every", str(steps),
           "--export-path", str(out), "--export-name", f"{name}.ply", "--eval-split-every", str(eval_every_n),
           "--eval-every", str(steps), "--eval-save-to-disk"]
    if max_splats:
        cmd += ["--max-splats", str(max_splats)]
    if sh_degree is not None:
        cmd += ["--sh-degree", str(sh_degree)]
    t0 = time.perf_counter()
    log = out / f"{name}.log"
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=out)
    secs = time.perf_counter() - t0
    text = log.read_text(errors="replace")
    psnr = [line for line in text.splitlines() if "psnr" in line.lower()]
    plys = sorted(out.glob("*.ply"))
    report = {"stage": "brush", "exit_code": proc.returncode, "seconds": round(secs, 1), "steps": steps,
              "eval_split_every": eval_every_n, "max_splats": max_splats, "sh_degree": sh_degree,
              "ply": [str(p.relative_to(dataset)) for p in plys],
              "ply_mb": [round(p.stat().st_size / 1e6, 1) for p in plys], "psnr_lines": psnr[-10:],
              "command": " ".join(cmd)}
    (dataset / f"brush_report_{name}.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sfm", "brush"])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--max-splats", type=int, default=None)
    ap.add_argument("--sh-degree", type=int, default=None)
    ap.add_argument("--name", default="splat")
    a = ap.parse_args(argv)
    ds = a.dataset if a.dataset.is_absolute() else REPO_ROOT / a.dataset
    rep = run_sfm(ds) if a.stage == "sfm" else run_brush(ds, steps=a.steps, max_splats=a.max_splats,
                                                         sh_degree=a.sh_degree, name=a.name)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
