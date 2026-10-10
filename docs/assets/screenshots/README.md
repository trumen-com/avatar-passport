# App screenshots (2026-10-11, local run on the dev laptop)

Captured from a real inspection of the team's demo car with `scripts/deck_screenshots.py`, using headless Firefox
against the app on `localhost:8000`.

- **The VIN is masked.** Positions 12–17 are replaced in the page before capture.
- **Images in the page are the anonymised frames.** Plates and faces are blurred.
- **The demo car has no damage.** Every damage finding shown is a false positive from the uncalibrated damage models:
  reflections on the windshield read as "glass shatter", and decals read as scratches or dents. Real accuracy needs the
  Phase 2 test set (`docs/research/evaluation.md`).
- **The high-quality 3D was built on the laptop GPU.** Its timings and PSNR are shown on the card and in
  `docs/research/recon3d_smoke_results.md`.

`mobile/` uses a 430 px wide viewport; `desktop/` is 1280 px.
