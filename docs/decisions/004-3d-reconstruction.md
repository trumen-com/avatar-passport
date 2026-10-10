# ADR 004: 3D reconstruction of the car (asynchronous, slow is acceptable)

**Status:** accepted on 2026-10-11 by the user. It supersedes ADR 002. The method is provisional until the 3D smoke test passes.

## Context
CLAUDE.md section 7 originally ruled 3D reconstruction out of scope, because it is too slow for a live CPU demo. On
2026-10-11 the user moved it into scope:
- It is fine for 3D inference to be slow in the demo, as long as it really works.
- The passport itself must still come back first. The 3D model is an extra that appears when ready.

Constraints:
- No compiler or CUDA toolkit on the dev machine.
- The deploy target is a 2-vCPU CPU host.
- Licences must be documented, with non-commercial ones flagged.
- The "no faked demos" rule means a precomputed model must never be presented as live.

## Decision
- **Method:** Depth Anything 3, the Apache-2.0 checkpoints DA3-BASE / DA3-SMALL. It is a feed-forward model: from
  ~30 anonymised walkaround frames it predicts per-pixel depth, confidence and camera poses in one pass. The output is a
  coloured point cloud (`.ply`) plus damage pins, shown with three.js. Candidates and licences are in
  `docs/research/recon3d_candidates.md`.
- **Damage pins:** 2D damage masks are back-projected through DA3's depth and poses, so pins and cloud share one frame.
- **Placement in the phases:**
  - Phase 0: research (done).
  - Phase 1: `notebooks/07_3d_smoke_test.ipynb`, run on GPU and CPU and timed. Pass condition: a recognisable car.
  - Phase 3: `src/avtopassport/recon3d/`.
  - Phase 4: an asynchronous job in FastAPI; the passport page shows "3D model building…" and then the viewer.
- **Privacy:** reconstruct only from anonymised frames.
- **Fallbacks:** the 360° spin viewer stays as the fast view. If DA3 fails, try MapAnything-apache on GPU, or pycolmap
  sparse SfM on CPU.
- **Optional:** Gaussian splats through the Brush binary on a GPU (no compiler needed). Not on the CPU Space.

## Consequences
- **Where it runs is decided by the smoke test's measured CPU time.** On the free CPU Space it may take minutes. If it
  does not fit at all, the alternatives are a GPU Space (paid) or the laptop GPU as a worker. Paid options need the
  user's approval at the Run 2 decision point.
- **New dependencies** (depth-anything-3, three.js) go into `DATA_SOURCES.md` with their licences. The non-commercial
  DA3 checkpoints (GIANT, NESTED, LARGE, LARGE-1.1) are not used.
- **More engineering than the spin viewer:** job state, storage of the `.ply`, and a viewer page.

## Update 2026-10-11 (overnight): high-quality path implemented

The user reviewed the first point cloud and asked for more accuracy and detail, using more frames from the video and
better calibration. Changes made in response:
- **Point cloud (fast path):** a depth-edge filter, a voxel-density filter and a connected-cluster filter remove the
  "flying pixel" streaks. The viewer opens on an upright 3/4 view using the orientation saved in `cameras.json`.
- **High-quality path (`src/avtopassport/recon3d/hq.py`):**
  1. Anonymised frames at about 4 fps, full resolution.
  2. COLMAP structure-from-motion: one shared camera, exhaustive matching to close the walkaround loop.
  3. Brush Gaussian splats on the GPU.
  4. A car crop. An SfM point counts as car when most of its observations fall inside the parts-model car masks; a
     robust, car-aligned box is fitted to those points, and Gaussians inside it are kept.
  5. Held-out evaluation. Every 10th frame is never trained on, and PSNR is measured against it over the whole image
     and inside the car mask.
  
  Measured numbers are generated into `docs/research/recon3d_smoke_results.md`.
- **Where it runs:** only with a GPU worker (`AVTO_HQ3D=1`). The UI labels it as unavailable on the CPU-only public
  deployment, and the fast DA3 cloud remains the CPU path.
- **Licences added:** pycolmap (BSD-3), Brush (Apache-2.0), Spark (MIT). See `DATA_SOURCES.md`.
