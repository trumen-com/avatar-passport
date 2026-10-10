# Phase 1 smoke test results

<!-- Hand-written text lives outside the marker pairs. Everything between a BEGIN/END marker pair is generated
by scripts/render_results.py from outputs/smoke/<run>/ (gitignored): never edit it by hand, re-run the script.
`python scripts/render_results.py --check` exits 1 when the generated tables are stale. -->

## Purpose and pass condition

Phase 1 checks that the candidate models, the privacy step and the VIN reader work on our own media before any app code is written. From CLAUDE.md, section 3:

> ### Phase 1 — Colab smoke test (`notebooks/01_smoke_test_colab.ipynb`)
> Must run top-to-bottom in Google Colab (GPU and CPU runtime):
> - Install cell, GPU check, load inputs from `data/raw/` (upload or Google Drive mount).
> - Extract frames from `data/raw/video/*` at 2–3 fps; quality filter (Laplacian blur variance, brightness).
> - Run every candidate damage model + parts model on the frames and the photos in `data/raw/images/`; save annotated outputs to `outputs/` (gitignored).
> - OCR on `data/raw/vin/*`, validate, decode.
> - Print per-frame latency on CPU and GPU.
>
> Pass condition: annotated images look sensible, VIN read correctly or failure understood, latencies recorded in `docs/research/smoke_test_results.md`.

User-approved deviations that apply here:

- The notebook is executed locally on the RTX 5050 laptop (GPU and CPU). The Colab runs are a user step.
- RapidOCR replaces PaddleOCR. EasyOCR runs as the comparison.
- SAM 3 is dropped.
- The parts model is the pretrained `mitbersh/car-parts-segmentation`.
- Deploy CPU latency is a pinned 2-thread local proxy, labelled optimistic.

Privacy rule for every run: no pixels are displayed or saved before anonymisation. Plates and faces are pixelated and the step fails closed, so an input whose detectors fail is dropped. `outputs/` is gitignored, and only anonymised contact sheets are copied to `docs/assets/smoke/`. VINs are masked after position 11 in everything the runs write and in this file.

## Hardware and software disclosure

The notebook's environment cell records the machine and library versions of every run (`env.json`, or `env` in `results.json`). Latency rows also record the 1-minute load average. The ONNX Runtime rows record the CPU affinity in force and the number of physical cores it spans.

<!-- BEGIN:runs_env -->
_not generated yet_
<!-- END:runs_env -->

## Run matrix

| id | where | device | purpose |
|---|---|---|---|
| R0 | local laptop | `cpu` | Dry run with small input caps (`AVTO_MAX_FRAMES`, `AVTO_MAX_IMAGES`, `AVTO_MAX_VINS`). It checks that the notebook runs top to bottom and writes every output file. |
| R1 | local laptop, NVIDIA RTX 5050 Laptop GPU | `cuda` | Full run with every model, plus the `gpu` latency tier. |
| R2 | local laptop, CUDA hidden (`CUDA_VISIBLE_DEVICES=""`) | `cpu` | Full run on CPU, plus the `cpu_torch`, `cpu_ort16` and `cpu_ort2` latency tiers. |
| Colab GPU | Google Colab GPU runtime | `cuda` | User step: the same notebook on Colab. |
| Colab CPU | Google Colab CPU runtime | `cpu` | User step: the same notebook on a Colab CPU runtime. |

Each run writes to `outputs/smoke/<run>/`, with the run id taken from `AVTO_RUN`. A run that is missing from the generated tables has not been executed, or its outputs are not on this machine.

The generated tables round the values in the run files half up: milliseconds to 1 decimal, load averages to 2, and confidences, scores and IoU to 3. Counts are integers.

## Latency protocol

- **Inputs:** a fixed list of `AVTO_LAT_FRAMES` anonymised working frames, saved to `outputs/smoke/<run>/latency_frames/`, so every tier times the same images.
- **Runs:** each model and tier gets one untimed setup call (model or session load plus the first inference), then `AVTO_LAT_WARMUP` untimed warm-ups, then up to `AVTO_LAT_RUNS` timed single-image runs cycling the frames.
- **Cap:** the timed phase stops early once it has taken longer than `AVTO_LAT_CAP_S` seconds. The row then shows `cap hit` = yes, and `timed runs` says how many runs completed. The values used are in each row's protocol string.
- **Statistics:** median and p90 over the timed runs. p90 interpolates linearly between the closest ranks.
- **Tiers:**
  - `gpu`: PyTorch on CUDA, every loaded model (R1).
  - `cpu_torch`: PyTorch on CPU, in the notebook process (R2).
  - `cpu_ort16`: ONNX Runtime with 16 intra-op threads, unpinned (R2). It runs in its own process via `scripts/bench_cpu.py`.
  - `cpu_ort2`: ONNX Runtime with 2 intra-op threads, pinned with `taskset -c 0,8` via `scripts/bench_cpu.py`. On this laptop, CPUs 0 and 8 are the two hardware threads of one physical core.
- **The deploy proxy:** `cpu_ort2` stands in for deploy hardware and is labelled the **optimistic lower bound for HF CPU Basic (2 vCPU)**. It is not a measurement of Hugging Face hardware. The assumption, not measured, is that an idle laptop core is at least as fast as a shared 2-vCPU cloud slice, so deploy latency should be equal or higher.
- **Kinds** (the `kind` column, which is the first token of the protocol when that token is a measurement kind; rows that record a failed step, such as an ONNX export, leave it empty):
  - `e2e-ultralytics`: preprocess + inference + postprocess, from ultralytics `r.speed`. `infer median ms` is inference only, and `wall median ms` is the wall time of `predict()`.
  - `raw-ort`: ONNX Runtime `session.run` only, on a letterboxed float32 NCHW input. `wall median ms` adds the letterbox step.
- **Threads:** the ONNX Runtime tiers set intra-op = threads and inter-op = 1, including for the session ultralytics creates internally. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are exported before numpy, OpenCV or torch is imported. Importing ultralytics makes OpenCV single-threaded (`cv2.setNumThreads(0)`), and its CPU device setup resets torch to `min(8, os.cpu_count() - 1)` threads, so `scripts/bench_cpu.py` sets torch back to the tier's thread count after the setup call (ultralytics 8.4.175, read in `ultralytics/utils/__init__.py` and `select_device`). Each bench row's protocol string records the ONNX Runtime, torch and OpenCV thread counts in force during the timed runs.
- **Pinning:** in bench rows, `pinned` = yes only when the process's CPU affinity really is restricted. The bench checks the caller's `--pinned 1` against the affinity, and records a mismatch in the protocol string.
- **ONNX files:** each deploy candidate is exported once per image size, to `models/onnx/<key>_<imgsz>.onnx`.
- **Load:** `load 1m` is the 1-minute load average sampled just before the timed phase. Read rows with a high value with care.

## Models

<!-- BEGIN:models -->
_not generated yet_
<!-- END:models -->

## Detections

All thresholds are uncalibrated defaults; calibration happens in Phase 2. Detection counts say nothing about correctness. The visual check is G3 below.

<!-- BEGIN:detections -->
_not generated yet_
<!-- END:detections -->

## Privacy

<!-- BEGIN:privacy -->
_not generated yet_
<!-- END:privacy -->

## VIN OCR

<!-- BEGIN:vin -->
_not generated yet_
<!-- END:vin -->

## Latency

<!-- BEGIN:latency -->
_not generated yet_
<!-- END:latency -->

## Per-inspection estimate

<!-- BEGIN:estimate -->
_not generated yet_
<!-- END:estimate -->

## Artifacts

<!-- BEGIN:artifacts -->
_not generated yet_
<!-- END:artifacts -->

## Gate checklist

Phase 1 passes only when every item is ticked with evidence.

- [ ] **G1** `YOLO_AUTOINSTALL=false .venv/bin/python -m pytest -q tests` is green.
- [ ] **G2** R1 and R2 exit 0, and the Models table records every model as `ok`, or as `load_failed` with its error.
- [ ] **G3** Anonymised, annotated contact sheets for each model are in `docs/assets/smoke/`, and the user has filled in the verdict table below.
- [ ] **G4** Every VIN image is either `read` and compared with the ground truth (`data/raw/vin/ground_truth.csv`, never committed), or `could_not_read` with the failure understood and written up under Observations.
- [ ] **G5** The tables in this file, and `measured_latency` in `model_candidates.md`, are generated: `python scripts/render_results.py --check` exits 0.
- [ ] **G6** The W2 audits (privacy and numbers) pass.

### G3 verdicts (filled in by the user)

| model | contact sheet | annotations look sensible? (Y/N) | notes |
|---|---|---|---|
| `damage_mitbersh` | | | |
| `damage_danishali` | | | |
| `damage_harpreet` | | | |
| `parts_mitbersh` | | | |
| `coco_yolo26n` | | | |
| privacy: plates and faces pixelated on the frames contact sheet? | | | |

## Observations

_Left for the main agent: what the tables show, failures and their causes, and decisions for Phase 2._
