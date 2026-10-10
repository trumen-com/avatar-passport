# Morning hand-off (overnight build, 2026-10-11)

Read this first. Kept up to date during the night; see `git log` for the exact order of work.

## What is done (committed)

- **Phase 0 research:** models, licences, VIN rules, the WMI table, prices, deploy options and 3D options.
  See `docs/research/`.
- **Phase 1 smoke test**, run locally on the RTX 5050 GPU and on the CPU (`docs/research/smoke_test_results.md`):
  - All five models run.
  - The VIN is read from all three VIN photos, with a valid check digit.
  - The privacy pipeline was fixed after your review: smooth blur, no blur on car panels, and a person-head backstop.
  - A three-reviewer visual audit of the committed images found no blockers.
- **Phase 3 pipeline package** (`src/avtopassport/`), with tests:
  - frames, quality filter and privacy (fail-closed);
  - damage and parts segmentation, damage-to-part mapping, merging across frames, severity rules;
  - cost in UZS from the observed price table;
  - VIN OCR, validation and decoding;
  - passport store (SQLite), "new since last inspection" diff, and a PDF with frame hashes;
  - paint try-on, and the 3D point cloud with damage pins.
- **Phase 4 app** (`src/avtopassport/api/`, `web/`) runs on this laptop. Upload a video and a VIN photo, and a passport
  comes back in about 20–25 s on the GPU:
  - VIN card, 360° spin viewer with damage pins, damage list with costs;
  - diff and history, try-on, the 3D viewer, nearby shops (OpenStreetMap), and the PDF;
  - the synthetic registry sandbox page, with the SYNTHETIC DEMO DATA badge.
  - Two uploads of the same video correctly give "no new damage since the last inspection".
- **3D, new scope:** a fast DA3 point cloud in the app, and a high-quality path that adds COLMAP calibration and Gaussian
  splats (Brush) on the GPU. See `docs/decisions/004-3d-reconstruction.md`.
- **Repeat-inspection diff:**
  - Changes are judged only on confirmed findings: score >= 0.5 and seen in at least 2 frames. The uncalibrated models
    produce different low-confidence false positives on every run, and those were showing up as phantom "new damage".
  - Low-confidence differences are listed separately as "possible changes - please verify".
- **High-quality 3D, verified inside the app** on a fresh inspection with all fixes:
  - 115 frames at 4 fps, all 115 calibrated by COLMAP (0.53 px), then 30k-step Brush splats on the GPU, 36 min in total.
  - Held-out views reach 25.1 dB PSNR on the whole image and 24.8 dB on the car.
  - The car crop keeps 911k of 2.1M Gaussians, and phones get a 28 MB copy.
  - Enable it with `AVTO_HQ3D=1 AVTO_FPS=4 AVTO_MAX_FRAMES=120`; it needs the Brush binary in `tools/`.
  - A bug where Brush picked up a degenerate 2-image COLMAP side-model was found and fixed. Both runs are recorded in
    `docs/research/recon3d_smoke_results.md`.
- **Speed on the HF CPU Basic proxy, measured:** a full app inspection with the server pinned to 2 threads and the GPU
  hidden takes 138.8 s for 72 frames. That is minutes, not seconds, but it works. See `docs/pitch/cost_model.md`.
- **Screenshots for the deck** (VIN masked, demo car undamaged): `docs/assets/screenshots/`.
- **Docker:** not built, because docker is not on this laptop. Both of the image's Python environments were installed
  from scratch in clean venvs, and the full test suite passes there on CPU only.
- **Phase 2 tooling:** see `docs/research/labelling_guide.md` and `scripts/eval.py`. It produces no numbers until
  labelled data exists.
- **Pitch docs** (`docs/pitch/`): cost model, ROI model (illustrative, every assumption listed), deck outline, Q&A prep.

## What you need to do (in this order)

1. **Look at the app on the laptop and your phone.** Start it with the command in the README. Open
   `http://192.168.1.11:8000` from a phone on the same Wi-Fi.
2. **GitHub:** create a repository, then run `git remote add origin <url>` and `git push -u origin main`. All commits are
   local. Then set `REPO_URL` in `web/app.js` for the AGPL source link.
3. **Colab run** of `notebooks/01_smoke_test_colab.ipynb`. The brief requires it. Paste the generated tables back with
   `scripts/render_results.py`.
4. **HF PRO decision** ($9/month) once you are happy with the local app. Then create the Docker Space, add the
   `HF_TOKEN` / `HF_SPACE` secrets and run the manual `deploy.yml` workflow. The hub-sync inputs were checked against the action's v0.3.0 `action.yml` on 2026-10-11.
5. **VIN ground truth:** put the true VIN in `data/raw/vin/ground_truth.csv`, which is gitignored and never committed.
6. **Test set (Phase 2):**
   - Photograph damaged and undamaged cars across the day, night and wet slices.
   - Run `scripts/blur_testset.py`, label the blurred images (`docs/research/labelling_guide.md`), then run `scripts/eval.py`.
   - The current demo car has no damage, so every damage detection so far is a false positive.
7. **Prices:** phone 4–6 body shops and add rows (see the template in `scripts/build_repair_prices.py`).
8. **Before you submit:**
   - Test on 3 phones.
   - Record a backup demo video.
   - Make sure the deployed commit is the submitted one.

## Known limitations (honest list)

- **False positives:**
  - The damage models are uncalibrated. On the undamaged demo car they report reflections on the windshield as
    "glass shatter", and decals as scratches or dents.
  - The UI groups low-confidence findings as "please verify".
  - Real accuracy needs the Phase 2 test set.
- **Speed on the free CPU Space:**
  - A full inspection will take minutes there, not seconds (see `docs/pitch/cost_model.md`).
  - The high-quality 3D path needs a GPU.
- **Licences:**
  - Two of the three damage models were trained on CarDD (non-commercial), and the third's training data is unknown,
    so all are demo/research only.
  - The Ultralytics stack is AGPL. See `DATA_SOURCES.md`.
- **Privacy:** CAR8's front plate is seen almost edge-on, so no detector finds it. The auditor could read no
  characters, so it is left as is.
