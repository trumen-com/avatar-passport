# Deck outline (12 slides)

Number rule for the deck: every figure below is copied from a file in this repo, and the path is given. Before the pitch:
1. Re-run `python scripts/roi_calc.py`.
2. Re-copy the cost figures from `cost_model.md` and `roi_model.md`.
3. Replace laptop or proxy numbers with the Space measurement once Phase 4 writes it.

Labels to keep on the slides:
- "measured on the dev laptop (RTX 5050 / Ryzen 7 260)";
- "2-thread pinned proxy, optimistic lower bound";
- "ASSUMPTION" and "SYNTHETIC DEMO DATA".

## 1. Problem: "that scratch wasn't there"

- **Today.** Vehicle intake and trade-in at Royal Motors after-sales is a manual paper walkaround (CLAUDE.md section 1).
- **What goes wrong:**
  - missed damage;
  - disputes at hand-back ("that scratch wasn't there");
  - lost repair and accessory upsell;
  - no history per VIN.
- **Visual:** a paper damage sketch next to a phone.

## 2. Solution: a phone video becomes a condition passport

- **Input.** A 30 to 60 s walkaround video plus a VIN photo (CLAUDE.md section 1).
- **What the passport shows:**
  - decoded VIN card;
  - spin viewer with damage pins;
  - damage by part, with severity, action and a "from" price in UZS;
  - history and diff per VIN;
  - paint try-on;
  - 3D model;
  - nearby body shops;
  - PDF with a SHA-256 hash of the source frames.
- **Owner:** the after-sales unit (service reception, body shop, accessories).

## 3. What runs where

| stage | where | status |
|---|---|---|
| development | dev laptop (RTX 5050 / Ryzen 7 260) | in use |
| public demo | HF Docker Space, CPU Basic 2 vCPU / 16 GB | pending HF PRO, $9/month (ADR 003) |
| 3D model | async after the passport; DA3-BASE on CPU, or the laptop GPU as a worker | ADR 004 |

- **Models:** ONNX on CPU, PyTorch on GPU. No paid API.
- **Free external services:** NHTSA vPIC, OpenStreetMap Overpass (`cost_model.md` section 5).

## 4. Live demo script (about 3 minutes)

1. **Record.** A jury member films the car on their own phone, 30 to 60 s, plus one VIN photo.
2. **Upload.** Open the public URL and upload. The progress view polls (no SSE; ADR 003).
3. **Passport.**
   - VIN card: WMI from `data/reference/wmi_table.csv`, plus vPIC.
   - Damage list with "from" prices and their source and date.
   - Spin viewer.
4. **3D.** "3D model building..." changes to the point cloud with pins when ready.
5. **Repeat inspection.** Upload a second video of the same VIN. The diff shows "new since last inspection".
6. **PDF.** Download the PDF and show the frame hash and timestamp.

**Fallbacks:**
- Open the Space before the pitch, because cold start after 48 h idle has not been measured.
- Keep a laptop-plus-tunnel backup ready.
- Never show a precomputed result as live (CLAUDE.md section 2).

## 5. Speed (measured)

| configuration | time per inspection | label |
|---|---|---|
| GPU, full pipeline, 72 frames | 25.3 s | measured on the dev laptop (RTX 5050 / Ryzen 7 260); `outputs/e2e/run1/result.json` |
| CPU, 16 threads, ONNX Runtime (24 frames + 9 photos) | 53.9 s | measured on the dev laptop; `outputs/smoke/R2-local-cpu/estimate.csv` |
| HF CPU Basic proxy, 2 threads | 106.4 s; planning figure x2 = 212.7 s | 2-thread pinned proxy, optimistic lower bound |
| 3D, DA3-BASE, 29 frames | GPU module 13.3 s; CPU 2 threads 96.6 s | `outputs/recon3d/module-gpu/result.json`, `outputs/recon3d/R3D-cpu2-pinned/results.json` |

- **Honest line:** "The 10 to 20 s in the brief is our target. On the GPU we are at 25 s. On a free 2-vCPU host it is minutes, mostly privacy blurring."

## 6. Accuracy so far (honest)

- **The only test car was undamaged.** Every damage detection on it is a false positive (`docs/research/smoke_test_results.md`, Observations 3).
- **The full pipeline on that car** reported 11 findings, all false (`outputs/e2e/run1/result.json`). The recurring false positives are "glass shatter" on windshields reflecting sky and trees, and "scratch" or "dent" on the door decals (Observations 3).
- **VIN.** RapidOCR read all 3 VIN photos with a valid check digit; EasyOCR read none (`outputs/smoke/R1-local-gpu/summary.md`). The exact match is pending ground truth.
- **Phase 2 is pending:** recall, false positives per car, and day / night / rain slices. No accuracy number is claimed yet.
- **Visual:** a false-positive example (windshield reflection) with the caption "why we calibrate thresholds".

## 7. Privacy by design

- **Blur first.** No pixel is shown or saved before plates and faces are blurred (`docs/research/smoke_test_results.md`).
- **Fail closed.** If a detector fails, the frame is dropped.
- **Audited.** In run1, 72 of 72 frames were anonymised and 0 dropped (`outputs/e2e/run1/result.json`). The W2 visual audit was done by 3 reviewers. Its known limitation, an edge-on plate, is stated.
- **3D uses only anonymised frames** (ADR 004). VINs are masked in docs and logs.

## 8. Registry connector: sandbox

- **Synthetic data only.** Owner, accident and insurance history is synthetic (`scripts/generate_synthetic.py`, fixed seed). Every record has `"synthetic": true`, and the UI shows a **SYNTHETIC DEMO DATA** badge.
- **Synthetic VINs only.** Synthetic history is never attached to a real scanned VIN.
- **Why:** we have no access to the police or insurer registries, and the rules forbid real personal data (CLAUDE.md section 2).

## 9. Cost per inspection

- **Marginal cost is $0.** CPU Basic costs $0/hour, and there is no paid API.
- **Fixed cost:** HF PRO $9/month.
- **Per inspection:** at the base **ASSUMPTION** of 240 inspections a month, $0.0375 or 446 UZS. At 120 a month, 892 UZS. Generated in `cost_model.md` section 5.
- **Break-even:** fewer than two upsold "from"-priced repairs a month cover hosting (`roi_model.md`, illustrative). Team time is not costed.

## 10. Pilot proposal

- **Scope:** 1 branch (up to the 3 Royal Motors service desks), 3 months. These are **ASSUMPTIONS**, to agree with the partner.
- **Cash cost:**
  - PRO for 3 months: $27;
  - optional T4 GPU during opening hours: $288 for 3 months (`cost_model.md` section 6).
- **Measure during the pilot:**
  - dispute count;
  - damage found per intake;
  - body-shop conversion;
  - advisor minutes;
  - false positives per car.
- **Success criterion:** the partner's own numbers replace every ASSUMPTION row in `roi_model.md`.

## 11. Roadmap

1. **Real registry integration.** Replace the sandbox connector with the dealer DMS, then insurer and police data, under agreements.
2. **Retrain on dealer data.** Use the dealer's own labelled body-shop photos, so we no longer depend on the non-commercial CarDD-trained weights.
3. **Calibrated thresholds.** Per class and per condition (night, rain, reflections), from the Phase 2 test set.
4. **Speed.** Fewer frames, privacy at 1280 px and a GPU worker (`docs/research/smoke_test_results.md`, Observations 6).
5. **Timestamp and PDF.** An RFC 3161 trusted timestamp, and Toyota colour data for the try-on.

## 12. Licensing, and what we need

**Licences**
- The repo is AGPL-3.0 (Ultralytics). The app links to its source.
- The damage weights are CarDD-trained and non-commercial: demo and research only.
- The parts weights declare no licence.
- The plate model's training-data licence is unverified.
- YuNet is MIT but trained on WIDER FACE (CC BY-NC-ND).
- DA3-BASE is Apache-2.0; the non-commercial DA3 checkpoints are not used.
- OSM data is ODbL, with attribution shown.
- Sources: ADR 001, `docs/research/model_candidates.md`, `docs/research/recon3d_candidates.md`.

**Team asks**
1. Partner data for the ROI: volumes, disputes, margins, accessory prices, advisor cost.
2. Licensed damaged-car photos for retraining and for the Phase 2 test set.
3. A pilot branch.
4. A decision between an Ultralytics Enterprise licence and a non-AGPL stack.
5. Approval for hosting: HF PRO, and the optional GPU.
