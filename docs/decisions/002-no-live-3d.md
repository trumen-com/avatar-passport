# ADR 002: No live 3D reconstruction; a spin viewer instead

- **Status:** Accepted. This is set by the brief (CLAUDE.md section 7).
- **Date:** 2026-10-10

## Context

- **What the passport shows.** CLAUDE.md section 1 asks for a "360° spin viewer built from the video frames, with damage pins".
- **What is out of scope.** CLAUDE.md section 7 rules out real 3D reconstruction: "Real 3D reconstruction (Gaussian splatting / photogrammetry) — too slow for a live demo on CPU. The spin viewer replaces it."
- **Constraints.**
  - Inference on the deploy runs on CPU (ADR 003).
  - An inspection has a target of about 10 to 20 s. That figure is a target from the brief, not a measured result.

## Decision

- **No reconstruction.** We build no Gaussian splatting, photogrammetry or other 3D reconstruction, and no 3D rims, body kits or spoilers.
- **The spin viewer** shows the walkaround frames as an image sequence:
  - The frames are the quality-filtered and already blurred ones from the pipeline (frames, then quality, then privacy).
  - They are ordered by time.
  - The user scrubs through them by drag or swipe on the phone.
- **Damage pins** are drawn on the frames where the damage was detected. A pin links to the damage-list entry, which holds the part, severity, action and cost.
- **Optional "AI preview"** of accessories (section 7), only if time remains:
  - It must be labelled as an AI preview.
  - It uses an image-editing model API, so it is a paid API: ask before adding it, and add its cost to `docs/pitch/cost_model.md` first (CLAUDE.md section 9).
- **The roadmap slide** lists real 3D reconstruction as future work.

## Consequences

- **The viewer is 2D, frame by frame.** "360°" covers only what the video filmed. Gaps in the walkaround show up as jumps between frames.
- **Pins belong to frames, not to a 3D model.** The same damage seen in several frames is merged into one entry by `dedupe.merge`. The viewer can show the pin on each frame where that damage appears.
- **Cheap to run.** No GPU and no extra model are needed, and no extra inference time is added beyond what the pipeline already does.
- **Privacy comes for free.** The viewer shows only frames that have already been blurred, so it meets the "blurred in everything we store, show or commit" rule (section 2).
- **Smoothness depends on the source frames.** It depends on the frame rate kept by the quality filter (2 to 3 fps extraction per section 3) and on how steadily the advisor walks around the car.
