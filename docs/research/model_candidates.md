# Model candidates

Phase 0 output required by CLAUDE.md section 3: name, link, licence, classes, size, CPU latency and notes for every model.
Compiled on 2026-10-10 from the frozen Phase 0 files (`phase0_findings_models.md`, `phase0_findings_vin.md`, `phase0_findings_deploy_market.md`) and a few read-only spot-checks made the same day.

How to read the provenance column:
- **reported (URL)**: a statement or figure from a third party. We have not checked it.
- **header**: read from the metadata of the pinned file on 2026-10-10 (pickle opcodes of the `.pt` checkpoint, `metadata_props` of the `.onnx`). Nothing was run.
- **measured (run id)**: produced by our own code. There are no measured numbers yet. They will appear only in the generated block below.

Exact byte sizes, source revisions and SHA-256 hashes are pinned in `scripts/download_models.py`. The download verifies them and records them in `models/manifest.json`.

## Shortlisted (run in the smoke test)

| name | link | role | declared licence (card) | checkpoint licence | training data (licence) | classes | size | CPU latency | notes | provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| damage_mitbersh | [mitbersh/car-damage-segmentation][m-dmg] @3242b6c | damage segmentation, candidate 1 | none declared (HF card or GitHub) | AGPL-3.0 (Ultralytics stamp) | CarDD (non-commercial research and education; licence form) | 6: dent, scratch, crack, glass shatter, lamp broken, tire flat (checkpoint order) | 54.5 MB | see measured table below | YOLO26m-seg, imgsz 896. Repo config: conf 0.25, iou 0.7, max_det 300, retina masks, plus damage-to-part matching settings. The checkpoint's class order differs from the GitHub README. | reported ([card][m-dmg], [training notebook][m-dmg-nb], [CarDD][cardd], [CarDD re-upload][cardd-hf]); header |
| damage_danishali | [Danishali273/Car-Damage-Type-Segmentation][d-dmg] @12f084a | damage segmentation, candidate 2 | Apache-2.0 | AGPL-3.0 (contradicts the card) | unknown (the checkpoint's data path is a Kaggle "damage-type-1" folder), so treated as research-only | 6: crack, dent, glass shatter, lamp broken, scratch, tire flat | 63.4 MB | see measured table below | YOLO26l-seg (the card only says "YOLOv26"), imgsz 640. | reported ([card][d-dmg]); header |
| damage_harpreet | [harpreetsahota/car-dd-segmentation-yolov11][h-dmg] @f455e7b | damage segmentation, candidate 3 (YOLO11 fallback if the YOLO26 checkpoints fail to load) | AGPL-3.0 (card text: "falls under the AGPL-3.0 License"; no licence metadata field) | AGPL-3.0 | CarDD (non-commercial) | 6: crack, dent, glass shatter, lamp broken, scratch, tire flat | 124.9 MB | see measured table below | YOLO11x-seg. The card says imgsz 1024, but the checkpoint's training arguments say 1280 (header); the smoke test runs it at the card's 1024. The largest file in the set. | reported ([card][h-dmg], [CarDD][cardd]); header |
| parts_mitbersh | [mitbersh/car-parts-segmentation][m-parts] @05e58a3 | car-part segmentation | none declared | AGPL-3.0 | Humans in the Loop car parts (CC0 1.0), with left/right tags added by the model author (the derived HF dataset declares no licence) | 33: 21 part types, 12 of them split into left_/right_. Includes fender, quarter-panel, rocker-panel, roof, grille and license-plate (full list below). | 23.4 MB | see measured table below | YOLO26s-seg, imgsz 768. Repo config: conf 0.25, iou 0.7, max_det 100, retina masks. The header shows ultralytics 8.4.41. | reported ([card][m-parts], [dataset.yaml][m-parts-ds], [HITL dataset][hitl]); header |
| plate | [morsetechlab/yolov11-license-plate-detection][plate] @251a30d (`license-plate-finetune-v1n.pt` / `.onnx`) | plate detection for blurring | AGPL-3.0 | AGPL-3.0 in both files (.pt stamp: ultralytics 8.3.122; .onnx metadata: ultralytics 8.3.123) | Roboflow `license-plate-recognition-rxg4e`, 10,125 images (licence unverified: Roboflow returned 403) | 1: licence plate | 5.5 MB (.pt), 10.5 MB (.onnx) | see measured table below | YOLO11n, trained at 640. The card warns that train/test contamination inflates its metrics. The .pt is for torch/GPU; the .onnx (dynamic input) is for CPU. | reported ([card][plate], [dataset][plate-ds]); header |
| face | [opencv_zoo face_detection_yunet][yunet] (`2026may`, `2023mar`) | face detection for blurring | MIT (the model folder's LICENSE, Shiqi Yu 2020); the opencv_zoo repo as a whole is Apache-2.0 | n/a (ONNX, not an Ultralytics checkpoint) | WIDER FACE (CC BY-NC-ND 4.0) | 1: face | 0.23 MB each | see measured table below | `2026may` has dynamic input and is for OpenCV 5.x. `2023mar` has fixed input and is for OpenCV 4.x (Colab). It detects faces of roughly 10x10 to 300x300 px. | reported ([README][yunet], [model LICENSE][yunet-lic], [WIDER FACE][wider]) |
| coco_yolo26n | [ultralytics/assets v8.4.0 `yolo26n.pt`][yolo26n-file] | generic car and person detection (sanity and privacy checks) | AGPL-3.0 (Ultralytics licence page) | AGPL-3.0 | COCO 2017 (annotations CC BY 4.0, COCO Consortium; images under the Flickr Terms of Use) | 80 COCO classes; we use car and person | 5.5 MB | see measured table below | The smallest YOLO26 detector; same stack as the other models. | reported ([Ultralytics licence][ul-lic], [YOLO26 docs][yolo26-docs], [COCO terms][coco], read in the [site source][coco-src] on 2026-10-10); header |

Measured CPU latency (generated; do not edit by hand):

<!-- BEGIN:measured_latency -->
_not measured yet - generated by scripts/render_results.py_
<!-- END:measured_latency -->

Planned content of the generated table (Run 1 smoke test): median and p90 latency for each model and runtime. One runtime is ONNX Runtime CPU pinned to 2 threads, labelled as a proxy for the 2-vCPU Hugging Face CPU Basic host. That proxy is not a deploy-hardware measurement (see ADR 001).

Notes on the table:
- **Classes** come from the Phase 0 research (cards, checkpoint headers and `dataset.yaml`). The smoke test records every model's runtime `model.names` in `docs/research/smoke_test_results.md`. Where the two differ, the runtime value wins.
- **parts_mitbersh class order** (from `dataset.yaml`, ids 0 to 32): back-bumper, back-windshield, front-bumper, grille, hood, left_back-door, left_back-wheel, left_back-window, left_fender, left_front-door, left_front-wheel, left_front-window, left_headlight, left_mirror, left_quarter-panel, left_rocker-panel, left_tail-light, license-plate, right_back-door, right_back-wheel, right_back-window, right_fender, right_front-door, right_front-wheel, right_front-window, right_headlight, right_mirror, right_quarter-panel, right_rocker-panel, right_tail-light, roof, trunk, windshield. Reported ([dataset.yaml][m-parts-ds]).
- **Checkpoint licence**: all six Ultralytics `.pt` files carry the string `AGPL-3.0 (https://ultralytics.com/license)` in their header. The plate `.onnx` carries `AGPL-3.0 License (https://ultralytics.com/license)` in its metadata; the two YuNet `.onnx` files carry no metadata. Read on 2026-10-10.
- **Local ONNX exports**: `models/onnx/damage_mitbersh_896.onnx` and `models/onnx/parts_mitbersh_768.onnx` were exported on the laptop with ultralytics 8.4.175 on 2026-10-10 (their metadata says so). They are derived from the `.pt` files and keep the same licences and flags.
- **Ultralytics version**: YOLO26 heads first appear in ultralytics 8.4.0, and we pin 8.4.175 (`requirements*.txt`). Reported (`phase0_findings_models.md` section 1).
- **VIN OCR engines.** These are libraries, not model rows.
  - **RapidOCR 3.10.0** is the primary VIN reader on both Colab and the CPU deploy. It is Apache-2.0, bundles the PP-OCRv6 small detection and recognition models as ONNX, and runs on onnxruntime. It has no character allowlist, so `vin.validate` filters its output afterwards.
  - **EasyOCR 1.7.2** runs as a comparison. It is Apache-2.0, needs torch, and downloads 77.3 MB + 14.0 MB of models on first run. We compare it because it supports `readtext(..., allowlist=...)`.
  - Reported ([RapidOCR][rapidocr], [EasyOCR][easyocr], `phase0_findings_vin.md` section 1).

## Considered, not shortlisted

| name | link | licence | reason not shortlisted |
|---|---|---|---|
| shanexf/car-damage-yolo11m-ft | [HF][shanexf-dmg] | none declared; fine-tuned on CarDD (licence agreement) | Boxes only, with no masks, so it gives no damage area for severity. CarDD-trained. It could later filter false positives: the card reports clean-car false alarms falling from 75.0% to 4.0% at conf 0.25. |
| mitbersh/car-parts-segmentation-m | [HF][m-parts-m] | none declared (same author and data as parts-s) | YOLO26m-seg at 896, 54.6 MB against 23.4 MB for parts-s, trained on the same data. Revisit only if parts-s masks are too coarse on our frames. |
| shanexf/car-panel-yolo11s-seg-scratch | [HF][shanexf-panel] | none declared | No licence. It has 21 classes without left/right and was trained from scratch on 415 original photos (augmented 7x). |
| konst22/yolo11n-carparts-seg | [HF][konst22] | none declared | Fallback only. It uses the 23 carparts-seg classes, which have no fender, quarter-panel, roof or plate class, and it has no licence. |
| Majorburn/yolov11-carparts-seg | [HF][majorburn] | AGPL-3.0 | Fallback only. Same 23 carparts-seg classes; the card gives only "mAP50: 71.0%". |
| open-image-models plate detector (`yolo-v9-t-*-license-plate-end2end`) | [GitHub][oim] | code MIT; weights unclear | The weights were trained with the author's fork of the GPL-3.0 YOLOv9 repo, and the author has not confirmed MIT for them ([#14][oim-14], [#20][oim-20], [#22][oim-22]). |
| fast-alpr | [GitHub][fastalpr], [PyPI][fastalpr-pypi] | MIT | PyPI declares Python only up to 3.13. Its OCR does not cover Uzbek plates. It uses the open-image-models detector, so it has the same weights question. |
| EgoBlur Gen2 | [project page][egoblur] | Apache-2.0 (Gen 2) | The weights sit behind an email form, and the package supports only Python 3.10 to 3.12. |
| MediaPipe face detector | [docs][mediapipe] | Apache-2.0 | Lower recall is reported: the EgoBlur paper gives AR 0.533 against 0.905 for RetinaFace on Aria Pilot RGB ([paper][egoblur-paper], Table VII). Version 1.1.0 removed the old `mp.solutions` API. |
| SCRFD / InsightFace | [GitHub][insightface] | code MIT; models non-commercial | The README says models trained with its data "are available for non-commercial research purposes only". |
| SAM 3 (text prompt "license plate") | [HF][sam3] | SAM License | Gated behind manual approval, and `sam3.pt` is 3.45 GB. [Ultralytics][sam3-ul] reports 2,921 ms per image on an RTX PRO 6000 GPU. It needs a GPU, which the CPU deploy does not have (CLAUDE.md section 3). |
| PaddleOCR 3.x | [GitHub][paddleocr], [paddlepaddle on PyPI][paddle-pypi] | Apache-2.0 | Heavy: the paddlepaddle wheel is 194.8 MB, plus paddlex. RapidOCR ships PP-OCRv6 small models as ONNX in a 27.3 MB wheel. |
| EasyOCR 1.7.2 | [GitHub][easyocr] | Apache-2.0 (its CRAFT weights are MIT) | Kept as a comparison only. It needs torch, downloads 77.3 + 14.0 MB of models on first run, and has had no release since 2024-09. |
| Roboflow Universe hosted parts models | [pricing][roboflow] | per model; hosted API | A paid API with an API key: free tier 10 credits/month, Core $39/month, on-demand credits $6 with a linked card (pricing page, 2026-10-10). Not needed, because a self-hosted pretrained parts model exists. |

## Third-party reported metrics

**These figures were reported by the model authors or other third parties. We did not measure them.** They come from different datasets and splits, so they cannot be compared with each other or with our Phase 2 numbers.

| model | metric | value | source URL |
|---|---|---|---|
| mitbersh damage (YOLO26m-seg) | mask mAP50 / mAP50-95, author's validation split of a CarDD-based dataset (`train_metrics` stored in the checkpoint) | 0.754 / 0.581 | https://huggingface.co/mitbersh/car-damage-segmentation |
| Danishali damage (YOLO26l-seg) | mask mAP50 / mAP50-95, author's validation split (dataset unknown; `train_metrics` in the checkpoint) | 0.749 / 0.571 | https://huggingface.co/Danishali273/Car-Damage-Type-Segmentation |
| harpreetsahota damage (YOLO11x-seg) | mask mAP50 / mAP50-95, all classes (571 images, 1,247 instances) | 0.792 / 0.576 | https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11 |
| harpreetsahota damage (YOLO11x-seg) | mask mAP50 for crack / dent / scratch, its weakest classes | 0.518 / 0.612 / 0.680 | https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11 |
| shanexf damage (YOLO11m, boxes) | clean-car false-alarm rate before and after fine-tuning (conf 0.25) | 75.0% to 4.0% | https://huggingface.co/shanexf/car-damage-yolo11m-ft |
| shanexf damage (YOLO11m, boxes) | box mAP50 on its CarDD test split | 0.899 | https://huggingface.co/shanexf/car-damage-yolo11m-ft |
| shanexf panels (YOLO11s-seg) | val mask mAP50 | 0.886 | https://huggingface.co/shanexf/car-panel-yolo11s-seg-scratch |
| Majorburn carparts (YOLO11n-seg) | mAP50 | 71.0% | https://huggingface.co/Majorburn/yolov11-carparts-seg |
| YOLO26n-seg trained on carparts-seg (Ultralytics notebook, T4, 10 epochs) | mask mAP50 (reference for our training fallback) | 0.686 | https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-carparts-segmentation-dataset.ipynb |
| morsetechlab plate v1n (third-party test on Uzbek plates) | plates found: 40-photo set at imgsz 640 with no second pass; 80-photo set with a second pass (imgsz 1536, conf 0.10) | 37/40; 80/80 (the README notes one of the 80 regions found was a "MAN" badge, not a plate) | https://github.com/IzzatulloOne/plate-recognition-uz |
| YuNet | AP easy / medium / hard on the WIDER Face validation set | 0.834 / 0.824 / 0.708 | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| MediaPipe vs RetinaFace vs EgoBlur (faces) | average recall on Aria Pilot RGB (all about 0.99 on CCV2) | 0.533 / 0.905 / 0.938 | https://arxiv.org/abs/2308.13093 |

The morsetechlab card's own metrics are left out on purpose: the card says they were computed on a contaminated test split.

## Licensing summary

1. **Ultralytics AGPL-3.0.** "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" ([licence page][ul-lic]), and all six of our Ultralytics checkpoints carry the AGPL-3.0 stamp.
   - Using them without an Enterprise licence means publishing the whole project under AGPL-3.0, so the repo is AGPL-3.0 (the user's decision; see `LICENSE`).
   - Section 13 requires the deployed app to offer its source to network users (ADR 003).
2. **The damage weights are demo/research only.**
   - mitbersh and harpreetsahota were trained on CarDD. The [CarDD site][cardd] requires a signed licence form, and the HF re-upload card says the dataset "is intended for non-commercial research and educational purposes" ([re-upload][cardd-hf]).
   - Danishali's training data is unknown, and its card (Apache-2.0) contradicts its checkpoint stamp (AGPL-3.0).
   - Treat all three as unusable in a commercial product for Royal Motors.
3. **A production pilot needs a damage model retrained on licensed data.**
   - Candidate data: the dealer's own body-shop photos (consented, plates and faces blurred) and CC0 data such as the 814 damage images in the [Humans in the Loop dataset][hitl] (classes: dent, cracked, scratch, flaking, broken part, paint chip, missing part, corrosion).
   - A model retrained with Ultralytics is still AGPL. A closed-source product needs either an Ultralytics Enterprise licence or a non-AGPL stack.
   - Phase 0 named RF-DETR (its Apache-2.0 sizes) + open-image-models + YuNet as the non-AGPL path (`phase0_findings_models.md` sections 3 and 4). The open-image-models weight licence is itself unclear (see above).
4. **Parts model.** The training data is CC0, but the author declares no licence for the weights. The only licence statement is the AGPL-3.0 stamp that Ultralytics writes into every checkpoint. Flag it in `DATA_SOURCES.md`. Asking the author, or retraining on the CC0 HITL data, removes the doubt.
5. **Plate model.** AGPL-3.0 is declared, but the licence of its Roboflow training data is unverified. Treat it as research-only until that is checked.
6. **Face model.** The YuNet model is MIT, but it was trained on WIDER FACE (CC BY-NC-ND 4.0). Flag the non-commercial training data.
7. **COCO detector.** AGPL-3.0 (Ultralytics). COCO 2017 terms, read on 2026-10-10 in the [site source][coco-src] because the [terms page][coco] is rendered by JavaScript: the annotations "belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License"; the Consortium "does not own the copyright of the images", whose use "must abide by the Flickr Terms of Use".
8. **OCR.** RapidOCR and EasyOCR are Apache-2.0 and onnxruntime is MIT (`phase0_findings_vin.md` section 5).

[m-dmg]: https://huggingface.co/mitbersh/car-damage-segmentation
[m-dmg-nb]: https://www.kaggle.com/code/brshtskmit/train-car-damage-segmentation-yolov26-m-cardd
[d-dmg]: https://huggingface.co/Danishali273/Car-Damage-Type-Segmentation
[h-dmg]: https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11
[m-parts]: https://huggingface.co/mitbersh/car-parts-segmentation
[m-parts-ds]: https://huggingface.co/datasets/mitbersh/car-parts-segmentation-yolo
[m-parts-m]: https://huggingface.co/mitbersh/car-parts-segmentation-m
[hitl]: https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/
[cardd]: https://cardd-ustc.github.io/
[cardd-hf]: https://huggingface.co/datasets/harpreetsahota/CarDD
[plate]: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
[plate-ds]: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11
[yunet]: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
[yunet-lic]: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE
[wider]: https://huggingface.co/datasets/CUHK-CSE/wider_face
[yolo26n-file]: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
[yolo26-docs]: https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo26.md
[ul-lic]: https://www.ultralytics.com/license
[coco]: https://cocodataset.org/#termsofuse
[coco-src]: https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm
[rapidocr]: https://github.com/RapidAI/RapidOCR
[easyocr]: https://github.com/JaidedAI/EasyOCR
[shanexf-dmg]: https://huggingface.co/shanexf/car-damage-yolo11m-ft
[shanexf-panel]: https://huggingface.co/shanexf/car-panel-yolo11s-seg-scratch
[konst22]: https://huggingface.co/konst22/yolo11n-carparts-seg
[majorburn]: https://huggingface.co/Majorburn/yolov11-carparts-seg
[oim]: https://github.com/ankandrew/open-image-models
[oim-14]: https://github.com/ankandrew/open-image-models/issues/14
[oim-20]: https://github.com/ankandrew/open-image-models/issues/20
[oim-22]: https://github.com/ankandrew/open-image-models/issues/22
[fastalpr]: https://github.com/ankandrew/fast-alpr
[fastalpr-pypi]: https://pypi.org/project/fast-alpr/
[egoblur]: https://www.projectaria.com/tools/egoblur/
[egoblur-paper]: https://arxiv.org/abs/2308.13093
[mediapipe]: https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
[insightface]: https://github.com/deepinsight/insightface
[sam3]: https://huggingface.co/facebook/sam3
[sam3-ul]: https://docs.ultralytics.com/models/sam-3/
[paddleocr]: https://github.com/PaddlePaddle/PaddleOCR
[paddle-pypi]: https://pypi.org/project/paddlepaddle/
[roboflow]: https://roboflow.com/pricing
