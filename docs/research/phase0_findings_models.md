<!-- Phase 0 desk research, collected 2026-10-10 by read-only research agents (web/API fetches). Verbatim summary; claims cite the URLs fetched. Third-party figures here are REPORTED, not measured by us. -->
<!-- Correction 2026-10-10: the YuNet model files are MIT-licensed (opencv_zoo models/face_detection_yunet LICENSE, Copyright (c) 2020 Shiqi Yu), not Apache-2.0 as section 3 below says; the opencv_zoo repo itself is Apache-2.0. -->

I extracted and checked findings for all three areas. The three agents left no written conclusions, so everything below comes from their tool results (fetched pages, HF/PyPI/GitHub API output, checkpoint headers). I also re-checked the top HF repos live today: all six exist and are ungated. The DAMAGE transcript also recorded this machine: Python 3.14.4, RTX 5050 Laptop GPU (compute capability 12.0), driver 595.91.07.

## 1. Damage segmentation

| HF id | Arch / trained on | Weights | Classes (checkpoint order) | imgsz | Licence model / data | Commercial |
|---|---|---|---|---|---|---|
| mitbersh/car-damage-segmentation | YOLO26m-seg / ultralytics 8.4.43 | damage_segmentation.pt (54.5 MB) | dent, scratch, crack, glass shatter, lamp broken, tire flat | 896 | none declared (HF and GitHub); checkpoint stamped AGPL-3.0 / CarDD | No |
| Danishali273/Car-Damage-Type-Segmentation | YOLO26l-seg / 8.4.83 | best.pt (63.4 MB) | crack, dent, glass shatter, lamp broken, scratch, tire flat | 640 | card says Apache-2.0 but checkpoint stamped AGPL-3.0 / unknown data ("damage-type-1") | Unclear |
| harpreetsahota/car-dd-segmentation-yolov11 | YOLO11x-seg / 8.3.162 | best.pt (124.9 MB) | same as Danishali | 1024 (card) | AGPL-3.0 (card) / CarDD | No |
| shanexf/car-damage-yolo11m-ft | YOLO11m, boxes only | damage_best.pt (40.5 MB) | dent, scratch, crack, shattered_glass, broken_lamp, flat_tire | 640 | none / CarDD | No |

- **mitbersh: confirmed.** YOLO26-m-seg, 6 classes, `damage_segmentation.pt`, imgsz 896. Its `damage_inference_config.json` sets conf 0.25, iou 0.7, max_det 300, retina_masks true. Validation masks scored mAP50 0.754 and mAP50-95 0.581.
  - **Watch the class order:** the checkpoint's order differs from the list in the GitHub README. Always read `model.names` at runtime.
- **Danishali: confirmed.** It is actually YOLO26-**l**; the card only says "YOLOv26". Masks scored mAP50 0.749 and mAP50-95 0.571.
- **CarDD data:** 4,000 images and you must submit a licence form; the licence PDF wasn't machine-readable. The HF re-upload card says "intended for non-commercial research and educational purposes".
- **Minimum ultralytics version:** YOLO26 heads first appear in 8.4.0 (8.4.0 has them, 8.3.253 doesn't). Use at least 8.4.83, or the latest 8.4.175, which lists Python 3.14. Since 8.4.142, `nms=None` means external NMS (the default) and `nms=False` means the NMS-free head.
- **Smoke-test ranking:**
  1. **mitbersh.** Best documented; it pairs with a parts model from the same author and a `base_areas.json` that maps damage types to part groups.
  2. **Danishali.**
  3. **harpreetsahota.** YOLO11, so use it if YOLO26 checkpoints won't load.
  - **Optional:** shanexf can filter false positives. On clean cars it flags damage 4% of the time, versus 75% before its fine-tune.

## 2. Car-part segmentation

- **carparts-seg dataset:**
  - **Classes:** 23 — back_bumper, back_door, back_glass, back/front_left/right_door, back/front_left/right_light, back_light, front_bumper, front_door, front_glass, front_light, hood, left/right_mirror, object, tailgate, trunk, wheel.
  - **Size:** 3,833 images (3,156 / 401 / 276, checked against the zip's file list); zip is 139.5 MB.
  - **Source and licence:** Roboflow `gianmarco-russo-vt9xr/car-seg-un1pm` (Gianmarco Russo, 2023), CC BY 4.0 (Ultralytics docs plus an archived copy of the Roboflow page).
  - **Missing parts:** no fender, quarter-panel, roof or plate class.
- **mitbersh/car-parts-segmentation:** YOLO26s-seg, imgsz 768, `parts_segmentation.pt` (23.4 MB). A `-m` variant uses YOLO26m at 896 (54.6 MB).
  - **Classes:** 33, side-aware — the 21 HITL parts, with left_/right_ versions of 12 paired parts, plus a `license-plate` class.
  - **Data:** Humans in the Loop car-parts dataset, CC0 1.0. The model itself declares no licence.
- **Others:** shanexf/car-panel-yolo11s-seg-scratch has 21 classes without sides and mask mAP50 0.886 (no licence). konst22 (no licence) and Majorburn (AGPL, mAP50 71%) are about 6 MB YOLO11n models with the 23 carparts-seg classes.
- **Recommendation: download mitbersh parts-s.** Train only as a fallback: YOLO26n- or s-seg on carparts-seg, imgsz 640, 30–50 epochs.
  - **Set `fliplr=0.0`.** Ultralytics' horizontal flip mirrors the masks but never swaps left/right class IDs (the swap mapping only applies to keypoints), so the default 0.5 corrupts side labels.
- **Training time:** the Ultralytics notebook trained YOLO26n-seg on a T4 at batch 32 for 10 epochs in 0.229 h, using 6.2 GB, reaching mask mAP50 0.686. I found no RTX 5050 benchmark (it has 2,560 CUDA cores and 8 GB). My estimate is about 1–1.5 min per epoch for n at batch 16 and roughly double for s.
- **Torch for Blackwell (sm_120):** plain `pip install torch torchvision` works. PyPI torch 2.14.1 for Linux pulls CUDA 13.0 and has Python 3.14 wheels; the `/whl/cu130` and `cu132` indexes work too. CUDA 13 needs driver ≥580 (this machine has 595.91). Avoid `cu126`: sm_120 support arrived with CUDA 12.8 / PyTorch 2.7.

## 3. Privacy

- **Plates:**
  - **Primary:** morsetechlab/yolov11-license-plate-detection, AGPL-3.0, in .pt and .onnx (the n model is 5.5 / 10.5 MB). Its card warns its metrics are inflated by train/test overlap.
    - **Tested on Uzbek plates** by IzzatulloOne/plate-recognition-uz: it found 37/40 at imgsz 640, and 80/80 with a second pass (imgsz 1536, conf 0.10), at about 66 ms per frame on CPU.
  - **Non-AGPL option:** `open-image-models` 0.6.0 — MIT code, ONNX via onnxruntime, Python 3.10–3.14 — with `yolo-v9-t-*-license-plate-end2end` models (about 7.8 MB).
    - **Weight licence is unclear:** they were trained with the author's fork of the GPL-3.0 YOLOv9 repo, and the author won't confirm MIT for the weights (issues 14, 20, 22).
  - **Weaker options:** fast-alpr only declares support up to Python 3.13, and its OCR doesn't cover Uzbekistan. LPD-YuNet was trained on Chinese plates only. EgoBlur Gen2 is Apache-2.0 but needs an email form and supports only Python 3.10–3.12.
- **Faces:**
  - **Recommended: YuNet through OpenCV `FaceDetectorYN`** (opencv_zoo, Apache-2.0).
    - **File:** `face_detection_yunet_2026may.onnx`; OpenCV 5.x needs this dynamic-input version.
    - **Caveat:** the training data, WIDER FACE, is CC BY-NC-ND 4.0.
  - **MediaPipe 1.1.0** (Apache-2.0) has removed the old `mp.solutions` API, so only the Tasks API works. Its recall in the EgoBlur paper was 0.533, versus 0.905 for RetinaFace.
- **SAM 3:** `facebook/sam3` is gated behind manual approval. It uses the "SAM License" (19 Nov 2025), which grants a royalty-free right to use, distribute and modify; I didn't review the rest. `sam3.pt` is 3.45 GB, and Ultralytics measured 2,921 ms per image on an RTX PRO 6000. **Skip it.**
- **COCO detector:** use YOLO26n (AGPL, 5.5 MB), since the stack is Ultralytics anyway. Apache-licensed alternatives are RF-DETR Nano–Large (its Atto/Femto/Pico/XL/2XL sizes are PML 1.0 instead), D-FINE nano, and RT-DETRv2 r18.

## 4. Ultralytics AGPL

- **Licence page:** "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default."
- **What it requires:** open-source the whole project under AGPL-3.0, or buy an Enterprise licence. SaaS and APIs are explicitly listed as needing Enterprise.
- **Checkpoints:** every one inspected has `license: AGPL-3.0` embedded.
- **For the hackathon:** publish the repo under AGPL. The non-AGPL alternative is RF-DETR + open-image-models + YuNet.

## 5. Gaps / unverified

- The Kaggle training notebooks returned only their titles, so there are no per-class metrics.
- The mitbersh parts checkpoint wasn't opened, so its class order comes from `dataset.yaml` and its training ultralytics version is unknown.
- The CarDD licence wording comes from a re-upload's card, not the original PDF.
- Danishali's training data is unknown.
- The RTX 5050 training times are my estimate.
- The BlazeFace licence and the Roboflow plate dataset's licence weren't confirmed (Roboflow returned 403).

Transcripts are in `/home/akkomron/.claude/projects/-home-akkomron-Projects-avtopassport-skeleton/63e51715-48a9-4d2a-967e-a37c00625496/subagents/workflows/wf_4f8d5e33-171/`.
