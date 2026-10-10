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
