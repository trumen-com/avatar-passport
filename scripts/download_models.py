"""Download every model used by AvtoPassport into models/ and write models/manifest.json.

Each file is pinned to an exact source revision and verified by size and SHA-256
(the expected values come from the Hugging Face / GitHub APIs on 2026-10-10).
A file that fails verification is deleted and the script exits non-zero.

Usage:
    python scripts/download_models.py            # default set (smoke test)
    python scripts/download_models.py --only parts_s,plate_n
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.json"
MIN_BYTES = 1024  # anything smaller is an LFS pointer or an error page


@dataclass(frozen=True)
class ModelFile:
    key: str            # stable key used by the code (e.g. "damage_mitbersh")
    role: str           # damage | parts | plate | face | coco | config
    source: str         # "hf:<repo>@<revision>" or a URL
    filename: str       # file name inside the source repo
    local: str          # path relative to models/
    size: int | None    # expected size in bytes (None = record on first download)
    sha256: str | None  # expected sha256 (None = trust on first use, recorded)
    licence_declared: str
    notes: str = ""


def hf(repo: str, rev: str) -> str:
    return f"hf:{repo}@{rev}"


MODELS: list[ModelFile] = [
    # --- damage segmentation candidates (all three are run in the smoke test) ---
    ModelFile("damage_mitbersh", "damage", hf("mitbersh/car-damage-segmentation", "3242b6c580e5f45fdb7b436c4296a78a16683a94"),
              "damage_segmentation.pt", "damage/mitbersh_damage_segmentation.pt", 54504241,
              "7c4a870b16bf642924d0b7473bb1a7df00903aa2a055bf1b7d8f50b905308561",
              "none declared on HF (checkpoint metadata: AGPL-3.0)", "YOLO26m-seg, imgsz 896, trained on CarDD (non-commercial)"),
    ModelFile("damage_mitbersh_config", "config", hf("mitbersh/car-damage-segmentation", "3242b6c580e5f45fdb7b436c4296a78a16683a94"),
              "damage_inference_config.json", "damage/mitbersh_damage_inference_config.json", 446, None, "none declared"),
    ModelFile("damage_danishali", "damage", hf("Danishali273/Car-Damage-Type-Segmentation", "12f084adf567d0870c8c487cef4e29fa27fd45ef"),
              "best.pt", "damage/danishali_best.pt", 63442309,
              "671554b49b8de0acdb187c06a4e524418725ea13286a718ad0648ecd01047cc0",
              "card: Apache-2.0 (checkpoint metadata: AGPL-3.0)", "YOLO26l-seg, imgsz 640, training data unknown"),
    ModelFile("damage_harpreet", "damage", hf("harpreetsahota/car-dd-segmentation-yolov11", "f455e7b592b64958e343f457de3d0ba2c79df884"),
              "best.pt", "damage/harpreet_best.pt", 124885473,
              "ab7d4e440104b8ef529d6fb7a5306ec93d191a97f454b9202dbf9d8eb8b2ab2a",
              "AGPL-3.0 (card)", "YOLO11x-seg, imgsz 1024, trained on CarDD (non-commercial)"),
    # --- car parts ---
    ModelFile("parts_mitbersh", "parts", hf("mitbersh/car-parts-segmentation", "05e58a396aefbe46d86ab9d17bbc945817163577"),
              "parts_segmentation.pt", "parts/mitbersh_parts_segmentation.pt", 23413021,
              "92228a116cbd4b57da083cb2de9b4da408fa5ae1bf56da213dd36816a96ae122",
              "none declared on HF", "YOLO26s-seg, imgsz 768, 33 side-aware classes incl. license-plate; HITL car-parts data (CC0)"),
    ModelFile("parts_mitbersh_config", "config", hf("mitbersh/car-parts-segmentation", "05e58a396aefbe46d86ab9d17bbc945817163577"),
              "parts_inference_config.json", "parts/mitbersh_parts_inference_config.json", 327, None, "none declared"),
    ModelFile("parts_mitbersh_catalog", "config", hf("mitbersh/car-parts-segmentation", "05e58a396aefbe46d86ab9d17bbc945817163577"),
              "parts_catalog.json", "parts/mitbersh_parts_catalog.json", 2372, None, "none declared"),
    # --- privacy ---
    ModelFile("plate_n_pt", "plate", hf("morsetechlab/yolov11-license-plate-detection", "251a30d7daedca065f56e04b0af04052c907c68f"),
              "license-plate-finetune-v1n.pt", "plate/license-plate-finetune-v1n.pt", 5465235,
              "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2", "AGPL-3.0", "YOLO11n, for torch/CUDA use"),
    ModelFile("plate_n_onnx", "plate", hf("morsetechlab/yolov11-license-plate-detection", "251a30d7daedca065f56e04b0af04052c907c68f"),
              "license-plate-finetune-v1n.onnx", "plate/license-plate-finetune-v1n.onnx", 10481682,
              "693133a1db97a3ba1e90068986f80afb72c3fcddb681e57181a89a9a3dc351d6", "AGPL-3.0", "dynamic input size; CPU / onnxruntime only"),
    ModelFile("face_yunet_2026may", "face", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
              "face_detection_yunet_2026may.onnx", "face/face_detection_yunet_2026may.onnx", 229738,
              "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0", "MIT (opencv_zoo face_detection_yunet LICENSE)", "for OpenCV >= 5; WIDER FACE training data is CC BY-NC-ND"),
    ModelFile("face_yunet_2023mar", "face", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "face_detection_yunet_2023mar.onnx", "face/face_detection_yunet_2023mar.onnx", 232589, None, "MIT (opencv_zoo face_detection_yunet LICENSE)", "for OpenCV 4.x (e.g. Colab's opencv-contrib)"),
    # --- generic COCO detector (car / person sanity check) ---
    ModelFile("coco_yolo26n", "coco", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
              "yolo26n.pt", "coco/yolo26n.pt", 5544453,
              "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef", "AGPL-3.0", "COCO 2017 (CC BY 4.0 annotations)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(m: ModelFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if m.source.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        repo, rev = m.source[3:].split("@", 1)
        tmp = Path(hf_hub_download(repo_id=repo, filename=m.filename, revision=rev, local_dir=dest.parent / ".hf"))
        tmp.replace(dest)
    else:
        req = urllib.request.Request(m.source, headers={"User-Agent": "avtopassport-download/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)


def verify(m: ModelFile, dest: Path) -> tuple[bool, str, int, str]:
    size = dest.stat().st_size
    digest = sha256_file(dest)
    if size < MIN_BYTES and m.role != "config":
        return False, f"too small ({size} B) - probably an LFS pointer", size, digest
    if m.size is not None and size != m.size:
        return False, f"size {size} != expected {m.size}", size, digest
    if m.sha256 is not None and digest != m.sha256:
        return False, f"sha256 {digest} != expected {m.sha256}", size, digest
    return True, "ok" if m.sha256 else "ok (sha256 recorded on first use)", size, digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated keys to download")
    args = ap.parse_args(argv)
    wanted = set(args.only.split(",")) if args.only else None

    manifest: dict = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    failures = 0
    for m in MODELS:
        if wanted and m.key not in wanted:
            continue
        dest = MODELS_DIR / m.local
        if dest.exists() and verify(m, dest)[0]:
            ok, msg, size, digest = verify(m, dest)
            status = "cached"
        else:
            t0 = time.time()
            try:
                fetch(m, dest)
                ok, msg, size, digest = verify(m, dest)
            except Exception as exc:  # network / HTTP errors
                ok, msg, size, digest = False, f"download failed: {exc}", 0, ""
            status = f"downloaded in {time.time() - t0:.1f}s"
            if not ok and dest.exists():
                dest.unlink()
        print(f"[{'OK ' if ok else 'ERR'}] {m.key:24s} {status:24s} {msg}")
        if not ok:
            failures += 1
            continue
        manifest["files"][m.key] = {**asdict(m), "local": str(Path("models") / m.local), "size": size,
                                     "sha256": digest, "verified": msg}
    MODELS_DIR.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {MANIFEST.relative_to(REPO_ROOT)} ({len(manifest['files'])} files), failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
