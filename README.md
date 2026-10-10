---
title: AvtoPassport
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: Phone walkaround video to a verified car condition passport
---

# AvtoPassport

AvtoPassport turns a phone walkaround video of a car, plus a photo of its VIN plate, into a digital condition passport tied to the VIN. The passport holds a decoded VIN card, a damage list by car part with repair action and cost, and a comparison with earlier inspections of the same car.
It is a hackathon prototype for dealer after-sales: vehicle intake and trade-in inspection. The project brief is [CLAUDE.md](CLAUDE.md).

## Status (2026-10-11, overnight build)

| Phase | Status | Where |
|---|---|---|
| Phase 0: research | Done | `docs/research/phase0_findings_*.md` (frozen), `model_candidates.md`, `vin_decoding.md`, `market_prices.md`, `recon3d_candidates.md` |
| Phase 1: smoke test | Done locally (GPU + CPU runs); Colab run by the team still to do | `docs/research/smoke_test_results.md` |
| Phase 2: test-set evaluation | Tooling ready; needs the team's labelled photos | `scripts/eval.py`, `docs/research/labelling_guide.md`, `docs/research/evaluation.md` |
| Phase 3: pipeline package | Done | `src/avtopassport/` |
| Phase 4: app | Runs locally (laptop + phones on the same Wi-Fi); public deploy waits for the HF PRO decision | `src/avtopassport/api/`, `web/`, `Dockerfile`, `.github/workflows/deploy.yml` |
| 3D model (scope change 2026-10-11) | Done. Fast DA3 point cloud in the app (CPU or GPU). High-quality COLMAP + Gaussian splats inside the app with a GPU worker; held-out PSNR about 25 dB | `src/avtopassport/recon3d/`, `docs/research/recon3d_smoke_results.md`, `docs/decisions/004-3d-reconstruction.md` |

**Phase 1 smoke test.** `notebooks/01_smoke_test_colab.ipynb` runs the candidate damage, parts, privacy and VIN-OCR models on our own media. We run it in three places:
- the dev laptop's NVIDIA RTX 5050 Laptop GPU
- the dev laptop's CPU
- Google Colab

**Where results go.** Measured numbers are written only by `scripts/render_results.py`, from the run outputs, into the generated blocks of `docs/research/smoke_test_results.md` and `docs/research/model_candidates.md`. Each number names its run id.

**Speed target.** The brief's goal is a passport about 10-20 s after upload. That is a target, not a measured result.

**How numbers are labelled in this repository**
- **measured (run id):** produced by our code in a named run. Written by scripts, never typed by hand.
- **reported (URL):** a third-party figure.
- **estimate / assumption:** our own number, labelled as such.
- **target:** a goal from the brief.

## Quick start (Linux, Python 3.12)

These are the commands used on the dev machine (uv 0.13.0, Python 3.12, NVIDIA RTX 5050 Laptop GPU):

```bash
git clone <REPO_URL> avtopassport && cd avtopassport

curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin; open a new shell afterwards
uv python install 3.12
uv venv --python 3.12 .venv

# 1. PyTorch from the CUDA 13.0 index (the build we use for the RTX 50xx / Blackwell GPU)
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 2. Everything else. requirements-exclude.txt drops opencv-python; we use opencv-python-headless (same cv2 module)
uv pip install --python .venv/bin/python -r requirements-dev.txt --excludes requirements-exclude.txt
# 3. This package, editable
uv pip install --python .venv/bin/python --no-deps -e .

# Model weights: pinned revisions, size and SHA-256 checked, recorded in models/manifest.json (gitignored)
.venv/bin/python scripts/download_models.py
YOLO_AUTOINSTALL=false .venv/bin/python -m pytest -q
```

- **No NVIDIA GPU:** install torch from `https://download.pytorch.org/whl/cpu` instead. We have not tested this.
- **`YOLO_AUTOINSTALL=false`** stops Ultralytics from pip-installing packages at run time. Set it for every command that imports `ultralytics`.
- **`HF_TOKEN`** is optional. Setting it gives faster Hugging Face downloads.
- **Regenerating data:** the synthetic sandbox comes from `.venv/bin/python scripts/generate_synthetic.py`, and the repair price table from `.venv/bin/python scripts/build_repair_prices.py`.

## Run the smoke test locally

1. **Add your media** to `data/raw/`. The folder is gitignored and never committed.
   - `data/raw/video/`: walkaround videos (`.mp4`, `.mov`)
   - `data/raw/images/`: car photos (`.jpg`, `.heic`)
   - `data/raw/vin/`: close-up photos of the VIN plate or sticker
2. **Run the notebook headless** from the repository root.
   - `AVTO_RUN` is the run id that results are cited by. Use a new id for every run; outputs go to `outputs/smoke/<AVTO_RUN>/` (gitignored).
   - `AVTO_LATENCY=1` also records latency. CPU timings come from `scripts/bench_cpu.py`, which runs in its own process.
   - The notebook's first cell lists every `AVTO_*` variable, including caps for a quick dry run.

   ```bash
   # R1: GPU (RTX 5050)
   YOLO_AUTOINSTALL=false AVTO_RUN=R1-local-gpu AVTO_LATENCY=1 \
     .venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_smoke_test_colab.ipynb \
     --ExecutePreprocessor.timeout=-1 \
     --output-dir outputs/smoke/R1-local-gpu --output 01_smoke_test_colab.executed.ipynb

   # R2: CPU only (CUDA hidden)
   CUDA_VISIBLE_DEVICES= YOLO_AUTOINSTALL=false AVTO_RUN=R2-local-cpu AVTO_LATENCY=1 \
     .venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_smoke_test_colab.ipynb \
     --ExecutePreprocessor.timeout=-1 \
     --output-dir outputs/smoke/R2-local-cpu --output 01_smoke_test_colab.executed.ipynb
   ```

3. **Write the measured tables into the docs:**

   ```bash
   .venv/bin/python scripts/render_results.py           # rewrites only the generated blocks
   .venv/bin/python scripts/render_results.py --check   # exit 1 if the docs are out of date
   ```

## Run the smoke test in Google Colab

1. **Open the notebook.** Open `notebooks/01_smoke_test_colab.ipynb` in Colab, either with File > Upload notebook or from GitHub once the repository is public.
2. **Choose the runtime.** Use a GPU runtime for the GPU run and a CPU runtime for the CPU run, with a new run id for each.
3. **Get the code in.** In cell 1a, either set `REPO_URL` so the notebook clones the repository, or upload a zip of the repository to Google Drive and set `DRIVE_ZIP` to it.
4. **Get the media in.** In cell 1a, either set `UPLOAD = True` and upload the files when asked, or set `DRIVE_RAW_DIR` to a Drive folder with `video/`, `images/` and `vin/` subfolders. On Colab, `AVTO_*` settings go into `OVERRIDES` in the same cell.
5. **Run all.** Use Runtime > Run all. The install cell uses `requirements-colab.txt`, which leaves Colab's preinstalled torch, torchvision, numpy and OpenCV alone.
6. **Bring the results back.** The last step zips the run's outputs and offers the download. Unzip it so the run folder sits at `outputs/smoke/<run id>/` in the repository, then run `scripts/render_results.py`.

## Data policy (summary)

- **Raw media stays out of git.** It lives in `data/raw/`, which is gitignored. Camera originals (`*.heic`, `*.heif`, `*.mov`, `*.mp4`, `*.dng`) are ignored anywhere in the tree.
- **Plates and faces are pixelated first.** This happens before any image is saved, shown or committed. The privacy step fails closed: if a detector cannot run, the image is dropped. Saved JPEGs carry no EXIF or GPS metadata. Only blurred copies may be committed (`data/processed/`).
- **No full real VINs.** A real car's full VIN is never committed. Where one must appear, positions 12-17 are masked.
- **Registry-type data is synthetic.** Owners, accidents, insurance claims and customer records are generated with a fixed seed. Every record has `"synthetic": true`, synthetic VINs start with `0AP`, and the UI marks the data SYNTHETIC DEMO DATA. Synthetic history is never attached to a VIN read from a real car (see `data/synthetic/README.md`).
- **Model weights are not in git.** `scripts/download_models.py` fetches them.

Details are in CLAUDE.md section 5 and [DATA_SOURCES.md](DATA_SOURCES.md).

## Repository layout

```
CLAUDE.md              project brief: scope, rules, phases
DATA_SOURCES.md        every model, dataset, library and reference source, with licences
LICENSE                AGPL-3.0 (code)
pyproject.toml         package metadata (src layout); dependencies are in requirements*.txt
requirements*.txt      runtime (requirements.txt), dev, Colab; requirements-exclude.txt drops opencv-python
data/
  raw/                 GITIGNORED originals: video/, images/, vin/
  processed/           blurred, metadata-free copies only
  reference/           wmi_table.csv (CC BY-SA 4.0), repair_prices.csv, colours.json
  synthetic/           seeded synthetic sandbox (flagged), README.md = method
  local_testset/       Phase 2 test set (blurred images + COCO annotations)
docs/
  research/            Phase 0 findings, model candidates, VIN decoding, prices, smoke-test results
  decisions/           short ADRs (001 model choice, 002 no live 3D, 003 deploy)
  pitch/, assets/      deck material, smoke-test images, failure cases
notebooks/             01_smoke_test_colab.ipynb (generated by scripts/build_notebook_01.py)
scripts/               download_models.py, build_notebook_01.py, render_results.py, bench_cpu.py,
                       blur_raw.py, generate_synthetic.py, build_repair_prices.py
src/avtopassport/      pipeline/ (frames, quality, privacy), vin/ (ocr, validate, decode),
                       synthetic.py, io.py, timing.py; api/, cost/, passport/, tryon/ come later
tests/                 pytest
models/, outputs/      GITIGNORED: downloaded weights, run outputs
web/                   mobile frontend (Phase 4)
```

## Licensing

- **Code:** GNU AGPL-3.0 (`LICENSE`; `pyproject.toml` declares `AGPL-3.0-only`). We use Ultralytics YOLO, and Ultralytics-trained weights are AGPL-3.0 by default.
- **Exceptions by path:** `data/reference/wmi_table.csv`, `docs/research/phase0_findings_vin.md` and section 8 of `docs/research/vin_decoding.md` are CC BY-SA 4.0, because they adapt Wikipedia and Wikibooks content. The attribution is in [`data/reference/LICENSE-wmi_table.md`](data/reference/LICENSE-wmi_table.md).
- **Model weights** are not in this repository and keep their own licences (see [DATA_SOURCES.md](DATA_SOURCES.md)).
  - The CarDD-trained damage weights are for non-commercial research only.
  - The Danishali damage model is used for research comparison only.
  - Commercial use would need damage models trained on commercially licensed data, plus AGPL compliance or an Ultralytics Enterprise licence.
- **Libraries** keep their own licences. Some wheels bundle GPL or LGPL components (FFmpeg, x264/x265, libheif); see DATA_SOURCES.md.
- **Team photos and video:** the licence for the blurred copies in `data/processed/` has not been decided yet.
- **Network use (AGPL-3.0 section 13):** once the app is deployed, anyone who uses it over the network must be offered the complete source of the running version. The deployed app must therefore link to this repository at the deployed commit.

## Run the app locally (laptop + phones)

```bash
# once: the separate 3D environment (Depth Anything 3 pins numpy<2)
uv venv --python 3.12 .venv-3d
uv pip install --python .venv-3d/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-3d/bin/python depth-anything-3==0.1.1 addict
uv pip install --python .venv-3d/bin/python --no-deps -e .

# the app (models load on start; first request may wait for the warm-up)
YOLO_AUTOINSTALL=false AVTO_DATA_DIR=var .venv/bin/uvicorn avtopassport.api.app:app --host 0.0.0.0 --port 8000
```

- Open `http://localhost:8000` on the laptop, or `http://<laptop LAN IP>:8000` on a phone on the same Wi-Fi.
- Phones can record and upload over plain HTTP. "Find nearby body shops" with GPS needs HTTPS, so on plain HTTP use the
  "Tashkent centre" button. A free `cloudflared` quick tunnel gives a temporary HTTPS URL if needed.
- Data goes to `var/` (gitignored): blurred frames, passports (`var/passports.sqlite`), PDFs and 3D models. Raw uploads
  are deleted after each inspection.
- Optional high-quality 3D on the GPU: see `scripts/recon3d_hq.py` (COLMAP + Brush; Brush binary goes to `tools/`).

## Deploy

Not deployed yet. The Dockerfile and a manual-only GitHub Actions workflow are ready, but the public Space needs HF PRO
(see `docs/decisions/003-deploy-hf-docker-local-first.md`). That decision is the user's, after testing locally.

### Persistence on Hugging Face (optional)

The Space disk is not persistent, so a restart would wipe the passport history and break the "repeat inspection shows a
diff" demo. `src/avtopassport/api/persist.py` can keep it in a **private** Hugging Face dataset repo. It is off unless
both variables below are set; with them unset nothing changes.

| Variable | Where | Meaning |
|---|---|---|
| `AVTO_SNAPSHOT_REPO` | Space variable | dataset repo id, e.g. `<user>/avtopassport-data`; created as a private dataset on the first snapshot |
| `HF_TOKEN` | Space **secret** | token with write access to that repo |
| `AVTO_SNAPSHOT_EVERY_MIN` | Space variable, optional | snapshot interval in minutes (default 5) |

- **At startup**, if there is no local `passports.sqlite`, the app downloads `snapshots/latest.tar.gz` and restores it.
  A local DB is never overwritten. If the download fails, the app logs a warning and starts with an empty history.
- **Snapshots** run every N minutes, about 30 s after each finished inspection, and once at shutdown. Nothing is uploaded
  when the content has not changed. The archive holds a consistent copy of the DB (SQLite backup API), plus
  `state.json`, `report.json` and `report.pdf` of each finished job, and a manifest with a SHA-256 for every file.
  The app refuses to upload to a repo that is not private.
- **Not stored:** blurred frames, masks, try-on images, 3D models and uploads. They are large (tens of MB per
  inspection), and keeping the minimum is the privacy default, even though the frames are blurred. A restored
  inspection therefore shows its findings, diff and PDF, and says that its frames and 3D model were not kept.
- **Privacy:** the PDFs contain blurred frames and the reports contain VINs, so the dataset repo must stay private.
  Delete it after the demo.
- **Loss window:** anything written after the last snapshot is lost on a restart (ADR 003).
- `/api/health` returns `persistence: {enabled, last_snapshot_utc, last_error, last_restore, every_min}`.
