# syntax=docker/dockerfile:1
# AvtoPassport - Docker image for a Hugging Face Docker Space (CPU Basic: 2 vCPU, 16 GB RAM) or any Docker host.
#
# PREPARED, NOT DEPLOYED. Hugging Face Docker Spaces need a paid plan (HF PRO, $9/month - ADR 003,
# docs/decisions/003-deploy-hf-docker-local-first.md). The user decides after the local end-to-end test.
#
# Local use:
#   docker build -t avtopassport .                          # full image (passport + async 3D)
#   docker build -t avtopassport --build-arg ENABLE_3D=0 .  # no 3D env: 3D status is "unavailable"
#   docker run --rm -p 7860:7860 avtopassport               # open http://localhost:7860
#
# Layout:
#   /opt/venv       main app env: CPU torch + torchvision, requirements.txt, ultralytics (opencv-python excluded,
#                   opencv-python-headless provides cv2), the app itself (editable install of /home/user/app)
#   /opt/venv-3d    optional Depth Anything 3 env (DA3 pins numpy<2, so it cannot share the main env). The app
#                   runs it as a subprocess: recon3d.build -> <repo>/.venv-3d/bin/python, a symlink to /opt/venv-3d.
#   models/         weights downloaded at build time by scripts/download_models.py (pinned revisions + SHA-256)
#
# Image size impact of ENABLE_3D=1 (not measured as a whole yet - run `docker images avtopassport` after a build):
#   - a second CPU torch + torchvision (the CUDA build is 1.2 GB for torch alone on the dev venv; the CPU wheel is
#     much smaller but still the biggest item),
#   - DA3 0.1.1 dependencies, measured on the dev .venv-3d: pycolmap 134 MB, scipy 89 MB, opencv 79 MB,
#     sympy 43 MB, pandas 41 MB, plotly 39 MB, dash 37 MB, matplotlib 30 MB,
#   - the DA3-BASE weights (Apache-2.0) pre-downloaded into the HF cache,
#   - excluded on purpose: open3d (903 MB on the dev venv; DA3 imports it only in its benchmark code),
#     xformers (CUDA-only; DA3 imports it inside try/except) and opencv-python (headless build instead).
# 3D on CPU, measured by scripts/recon3d_smoke.py (outputs/recon3d/R3D-cpu2-pinned/results.json, gitignored):
# 29 frames, DA3-BASE, 2 torch threads on the dev laptop -> 92.1 s inference, 3.5 GB peak RSS. The 2-vCPU Space
# has not been measured; the passport is returned first and the 3D job runs after it in its own worker.
# opencv-python-headless 4.11.0.86 in the 3D env: same version as the dev .venv-3d, built against numpy<2.

FROM python:3.12-slim

ARG ENABLE_3D=1
# deployed commit for /api/version; HF builds pass no build args, there the CI-written COMMIT_SHA file is used
ARG COMMIT_SHA=unknown
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
ARG ULTRALYTICS_VERSION=8.4.175
ARG DA3_VERSION=0.1.1
ARG DA3_MODEL=depth-anything/DA3-BASE
# models the pipeline loads (pipeline/segment.py MODEL_SPECS + privacy.py); empty = every model in the script
ARG MODEL_KEYS=damage_mitbersh,damage_mitbersh_config,parts_mitbersh,parts_mitbersh_config,parts_mitbersh_catalog,plate_n_pt,plate_n_onnx,face_yunet_2026may,coco_yolo26n

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# fonts-dejavu-core: Cyrillic glyphs in the PDF (passport/report.py uses DejaVuSans).
# libglib2.0-0: needed by OpenCV's highgui/gthread symbols even in the headless wheel on slim images.
# libgomp1: OpenMP runtime for onnxruntime / scipy / torch CPU kernels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core libglib2.0-0 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv: same version as the dev machine (supports --excludes)
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR /home/user/app

# ---------------------------------------------------------------- main env (cached layer: only requirements)
COPY --chown=user requirements.txt requirements-exclude.txt ./
RUN uv venv --python /usr/local/bin/python3.12 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --index-url "$TORCH_INDEX" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
 && uv pip install --python /opt/venv/bin/python -r requirements.txt "ultralytics==${ULTRALYTICS_VERSION}" \
        --excludes requirements-exclude.txt \
 && /opt/venv/bin/python -c "import cv2, torch, ultralytics, onnxruntime, fastapi; print('main env ok', cv2.__version__, torch.__version__)"

# ---------------------------------------------------------------- optional 3D env (Depth Anything 3, CPU torch)
RUN if [ "$ENABLE_3D" = "1" ]; then \
      printf 'open3d\nxformers\nopencv-python\ngradio\n' > /tmp/exclude-3d.txt \
   && uv venv --python /usr/local/bin/python3.12 /opt/venv-3d \
   && uv pip install --python /opt/venv-3d/bin/python --index-url "$TORCH_INDEX" \
          "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
   && uv pip install --python /opt/venv-3d/bin/python "depth-anything-3==${DA3_VERSION}" addict \
          "opencv-python-headless==4.11.0.86" --excludes /tmp/exclude-3d.txt \
   && /opt/venv-3d/bin/python -c "import depth_anything_3, numpy, torch; print('3D env ok', numpy.__version__, torch.__version__)"; \
    else echo "ENABLE_3D=$ENABLE_3D: no 3D env (recon3d status will be 'unavailable')"; fi

# ---------------------------------------------------------------- app source + models
COPY --chown=user . .
RUN uv pip install --python /opt/venv/bin/python --no-deps -e . \
 && if [ "$ENABLE_3D" = "1" ]; then ln -sfn /opt/venv-3d /home/user/app/.venv-3d; fi \
 && if [ "$COMMIT_SHA" != "unknown" ]; then echo "$COMMIT_SHA" > /home/user/app/COMMIT_SHA; fi \
 && mkdir -p /home/user/app/models && chown -R user:user /home/user/app

# RapidOCR keeps its ONNX models inside its package folder (root-owned /opt/venv): fetch them now, as root.
RUN python -c "from avtopassport.vin.ocr import make_engine; make_engine(); print('rapidocr models ready')"

USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=true \
    MPLCONFIGDIR=/tmp/matplotlib \
    AVTO_DATA_DIR=/tmp/avtopassport \
    AVTO_DEVICE=cpu

RUN if [ -n "$MODEL_KEYS" ]; then python scripts/download_models.py --only "$MODEL_KEYS"; \
    else python scripts/download_models.py; fi
RUN if [ "$ENABLE_3D" = "1" ]; then \
      PYTHONPATH=/home/user/app/src /opt/venv-3d/bin/python -c "from huggingface_hub import snapshot_download; from avtopassport.recon3d.worker import MODEL_REVISIONS as R; m='${DA3_MODEL}'; print(snapshot_download(m, revision=R[m]))"; \
    fi

# The Space disk is not persistent (ADR 003): AVTO_DATA_DIR=/tmp/avtopassport holds jobs, frames and SQLite and is
# lost on restart. To keep passports across restarts set AVTO_SNAPSHOT_REPO (private HF dataset) + HF_TOKEN:
# src/avtopassport/api/persist.py snapshots the SQLite DB and job reports there and restores them at startup.
EXPOSE 7860
CMD ["uvicorn", "avtopassport.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
