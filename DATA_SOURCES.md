# Data sources and licences

This file lists every model, dataset, library and reference source AvtoPassport uses, with its licence. It reflects checks made on 2026-10-10 (Run 1) unless a row gives another date.

Model weights are not stored in this repository. `scripts/download_models.py` downloads them into `models/` (gitignored) at pinned revisions and checks each file's size and SHA-256.

**How to read the tables**
- **Card:** the licence the model card or source repository declares.
- **Checkpoint stamp:** the licence string, Ultralytics version and date stored inside the weight file. We read these on 2026-10-10 without loading the files: from the pickle opcodes for `.pt` files and from `metadata_props` for `.onnx` files.
- **Third-party claims** are as reported by the linked source. We have not verified them.
- **Not legal advice:** this is our own reading of the terms.

| Flag | Meaning |
|---|---|
| NON-COMMERCIAL | The licence, or the terms of the training data, rule out commercial use. |
| RESEARCH-ONLY | We use it only for research comparisons, not in the app. |
| LICENCE UNCLEAR | No licence is declared, or the sources disagree. |
| AGPL | AGPL-3.0 copyleft. This is why the repository is AGPL-3.0. Keeping the code closed, including when it runs as a hosted service, would need an Ultralytics Enterprise licence. |

Ultralytics states: "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" (reported, https://www.ultralytics.com/license).

## Models

### Files downloaded by `scripts/download_models.py`

| Key | File | Source (repo @ revision, or URL) | Card | Checkpoint stamp | Training data (licence) | Flags |
|---|---|---|---|---|---|---|
| `damage_mitbersh` | `damage_segmentation.pt` | [mitbersh/car-damage-segmentation](https://huggingface.co/mitbersh/car-damage-segmentation) @ `3242b6c580e5f45fdb7b436c4296a78a16683a94` | none declared | AGPL-3.0 (ultralytics 8.4.43, 2026-04-29) | CarDD (non-commercial research) | NON-COMMERCIAL, LICENCE UNCLEAR, AGPL |
| `damage_mitbersh_config` | `damage_inference_config.json` | same repo and revision | none declared | not applicable (JSON config) | not applicable | LICENCE UNCLEAR |
| `damage_danishali` | `best.pt` | [Danishali273/Car-Damage-Type-Segmentation](https://huggingface.co/Danishali273/Car-Damage-Type-Segmentation) @ `12f084adf567d0870c8c487cef4e29fa27fd45ef` | Apache-2.0 | AGPL-3.0 (ultralytics 8.4.83, 2026-07-01) | unknown: the checkpoint names a dataset `damage-type-1` that is not published | RESEARCH-ONLY, LICENCE UNCLEAR, AGPL |
| `damage_harpreet` | `best.pt` | [harpreetsahota/car-dd-segmentation-yolov11](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11) @ `f455e7b592b64958e343f457de3d0ba2c79df884` | AGPL-3.0 | AGPL-3.0 (ultralytics 8.3.162, 2025-07-09) | CarDD (non-commercial research) | NON-COMMERCIAL, AGPL |
| `parts_mitbersh` | `parts_segmentation.pt` | [mitbersh/car-parts-segmentation](https://huggingface.co/mitbersh/car-parts-segmentation) @ `05e58a396aefbe46d86ab9d17bbc945817163577` | none declared | AGPL-3.0 (ultralytics 8.4.41, 2026-04-25) | Humans in the Loop car-parts data (CC0 1.0), with left/right tags added by the model author ([raw](https://huggingface.co/datasets/mitbersh/car-parts-segmentation-raw) and [yolo](https://huggingface.co/datasets/mitbersh/car-parts-segmentation-yolo) HF datasets, no licence declared) | LICENCE UNCLEAR, AGPL |
| `parts_mitbersh_config` | `parts_inference_config.json` | same repo and revision | none declared | not applicable (JSON config) | not applicable | LICENCE UNCLEAR |
| `parts_mitbersh_catalog` | `parts_catalog.json` | same repo and revision | none declared | not applicable (JSON part list) | not applicable | LICENCE UNCLEAR |
| `plate_n_pt` | `license-plate-finetune-v1n.pt` | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) @ `251a30d7daedca065f56e04b0af04052c907c68f` | AGPL-3.0 | AGPL-3.0 (ultralytics 8.3.122, 2025-05-01) | Roboflow `license-plate-recognition-rxg4e` v11 (licence unverified; the card warns of train/test contamination) | AGPL, LICENCE UNCLEAR (data) |
| `plate_n_onnx` | `license-plate-finetune-v1n.onnx` | same repo and revision | AGPL-3.0 | AGPL-3.0 in ONNX metadata (ultralytics 8.3.123, 2025-05-02) | same as above | AGPL, LICENCE UNCLEAR (data) |
| `face_yunet_2026may` | `face_detection_yunet_2026may.onnx` | [opencv/opencv_zoo `models/face_detection_yunet`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), `main` branch (no commit pinned; SHA-256 pinned in the script) | MIT ("All files in this directory are licensed under MIT License"; Copyright (c) 2020 Shiqi Yu) | none (not an Ultralytics model; ONNX metadata empty) | WIDER FACE (CC BY-NC-ND 4.0) | NON-COMMERCIAL (training data) |
| `face_yunet_2023mar` | `face_detection_yunet_2023mar.onnx` | same directory, `main` branch (no commit pinned; SHA-256 recorded on first download) | MIT (as above) | none (ONNX metadata empty) | WIDER FACE (CC BY-NC-ND 4.0) | NON-COMMERCIAL (training data) |
| `coco_yolo26n` | `yolo26n.pt` | [ultralytics/assets release v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) (`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`) | AGPL-3.0 (Ultralytics) | AGPL-3.0 (ultralytics 8.3.222, 2025-12-15) | COCO 2017 (annotations CC BY 4.0; images under Flickr terms) | AGPL |

**Notes**
- **Damage and CarDD.** CarDD's terms allow only non-commercial research and educational use. We use the two CarDD-trained damage models (`damage_mitbersh`, `damage_harpreet`) in a non-commercial hackathon prototype on that basis. A commercial pilot would need a damage model trained on commercially licensed data.
- **Danishali conflict.** The card says Apache-2.0 but the checkpoint says AGPL-3.0, and the training data is unknown. We apply the stricter licence and use the model only for comparison.
- **mitbersh models.** Neither the damage nor the parts model declares a licence on its Hugging Face card, so we rely on the AGPL-3.0 checkpoint stamp. Asking the author for an explicit licence is an open item.
- **Plate model.** The card says the upstream dataset has train/test contamination, so its published metrics are inflated (reported, https://huggingface.co/morsetechlab/yolov11-license-plate-detection). We judge it by our own runs on our images, not by the card's numbers.
- **YuNet (MIT).** If the ONNX files are redistributed, for example inside the Docker image, the MIT notice must go with them. (`scripts/download_models.py` and `docs/research/phase0_findings_models.md` still say Apache-2.0; the model folder's own README and LICENSE say MIT.)
- **Local ONNX exports.** `models/onnx/damage_mitbersh_896.onnx` and `models/onnx/parts_mitbersh_768.onnx` were exported on the dev laptop with ultralytics 8.4.175 on 2026-10-10. Their metadata says `AGPL-3.0 License (https://ultralytics.com/license)`. They are derived from the `.pt` files above and keep the same licences and flags. Like all weights, they are gitignored.

### Models that come with libraries (not in `download_models.py`)

| Component | Files | Where they come from | Licence | Training data | Flags |
|---|---|---|---|---|---|
| RapidOCR 3.10.0 (VIN OCR) | `PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx`, `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | Bundled in the `rapidocr` wheel (listed in its `RECORD`, checked locally) | Apache-2.0 (rapidocr; the PP-OCR models come from PaddleOCR, Apache-2.0) | Not documented by RapidOCR | none |
| EasyOCR 1.7.2 (OCR comparison) | `craft_mlt_25k.pth`, plus the recognition model for the requested languages (`english_g2.pth` for `['en']`) | Downloaded on first use to `~/.EasyOCR/model/` from the JaidedAI/EasyOCR GitHub releases (`pre-v1.1.6/craft_mlt_25k.zip`, `v1.3/english_g2.zip`, URLs in `easyocr/config.py`) | EasyOCR Apache-2.0; CRAFT weights MIT (NAVER, reported in Phase 0) | Not documented | RESEARCH-ONLY (comparison) |

**Considered and not used in Run 1** (see `docs/research/phase0_findings_models.md`)
- **SAM 3** (`facebook/sam3`): gated behind approval, under the SAM License, and too large and slow for the CPU deploy.
- **Roboflow-hosted parts models:** they need an API key, and the team decided not to use the Roboflow API.
- **Training our own parts model** on carparts-seg: kept as a fallback only.
- **`shanexf/car-damage-yolo11m-ft`:** not downloaded.
- **`open-image-models` plate detectors:** the licence of the weights is unclear.

## Datasets

We did not download any of these. They matter because the models above were trained on them.

| Dataset | Our use | Licence / terms | Source | Flags |
|---|---|---|---|---|
| CarDD (Car Damage Detection, USTC) | Training data of `damage_mitbersh` and `damage_harpreet` | The data is released only after you submit the CarDD licence form (https://cardd-ustc.github.io/docs/CarDD_license.pdf); its text could not be extracted when we checked. A Hugging Face re-upload card says the dataset "is intended for non-commercial research and educational purposes" and that the images fall under Flickr and Shutterstock terms (https://huggingface.co/datasets/harpreetsahota/CarDD). | https://cardd-ustc.github.io/ | NON-COMMERCIAL |
| Humans in the Loop "Car parts and car damages" | Training data of `parts_mitbersh`, through the author's re-labelled copies on Hugging Face | CC0 1.0: "dedicated to the public domain by Humans in the Loop under CC0 1.0 license" | https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/ and https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages | none (the HF copies declare no licence: LICENCE UNCLEAR) |
| carparts-seg (Ultralytics), from Roboflow `gianmarco-russo-vt9xr/car-seg-un1pm` | Not used: fallback only, if we ever train our own parts model | CC BY 4.0. If used, attribution is required, and weights trained with Ultralytics are AGPL-3.0. | https://docs.ultralytics.com/datasets/segment/carparts-seg/ and https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm | none |
| COCO 2017 | Training data of `coco_yolo26n` | Annotations: CC BY 4.0 (COCO Consortium). Images: not owned by COCO; Flickr Terms of Use apply. Read on 2026-10-10 in the site source (https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm), because the terms page is rendered by JavaScript. | https://cocodataset.org/#termsofuse | none |
| WIDER FACE | Training data of YuNet | CC BY-NC-ND 4.0 (dataset card) | https://huggingface.co/datasets/CUHK-CSE/wider_face | NON-COMMERCIAL |
| Roboflow Universe "License Plate Recognition" (`license-plate-recognition-rxg4e`, version 11) | Training data of the plate detector | Not verified: Roboflow returned HTTP 403 when we checked on 2026-10-10. The model card warns of train/test contamination. | https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11 | LICENCE UNCLEAR |

## Libraries

**Status:** this is a snapshot of the dev venv taken on 2026-10-10 with `importlib.metadata`. It will be re-generated from `importlib.metadata` for the Docker image in Phase 4.

The **In** column says which requirements file lists the package: R = `requirements.txt`, D = `requirements-dev.txt`, C = `requirements-colab.txt`. "Separate" means the package is installed on its own (see README). Licences are as declared in each package's PyPI metadata.

| Package | Pin (dev venv version) | Licence | In | Link |
|---|---|---|---|---|
| numpy | unpinned (2.5.3) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | R | https://pypi.org/project/numpy/ |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (bundles FFmpeg, see below) | R | https://pypi.org/project/opencv-python-headless/5.0.0.93/ |
| pillow | unpinned (12.3.0) | MIT-CMU | R | https://pypi.org/project/pillow/ |
| pillow-heif | 1.8.0 | BSD-3-Clause for the source; the binary wheel is GPL-2.0 (see below) | R, C | https://pypi.org/project/pillow-heif/1.8.0/ |
| av (PyAV) | 19.0.1 | BSD-3-Clause for the bindings (bundles FFmpeg, see below) | R, C | https://pypi.org/project/av/19.0.1/ |
| onnxruntime | 1.31.0 | MIT | R, C | https://pypi.org/project/onnxruntime/1.31.0/ |
| rapidocr | 3.10.0 | Apache-2.0 | R, C | https://pypi.org/project/rapidocr/3.10.0/ |
| huggingface-hub | 2.2.0 | Apache-2.0 | R, C | https://pypi.org/project/huggingface-hub/2.2.0/ |
| requests | unpinned (2.34.2) | Apache-2.0 | R | https://pypi.org/project/requests/ |
| ultralytics | 8.4.175 | AGPL-3.0 (flag: AGPL) | D, C | https://pypi.org/project/ultralytics/8.4.175/ |
| onnx | 1.23.2 | Apache-2.0 | D, C | https://pypi.org/project/onnx/1.23.2/ |
| onnxslim | 0.1.98 | MIT | D, C | https://pypi.org/project/onnxslim/0.1.98/ |
| easyocr | 1.7.2 | Apache-2.0 | D, C | https://pypi.org/project/easyocr/1.7.2/ |
| pandas | unpinned (3.0.6) | BSD-3-Clause | D | https://pypi.org/project/pandas/ |
| matplotlib | unpinned (3.11.2) | Matplotlib licence (PSF-based) | D | https://pypi.org/project/matplotlib/ |
| pytest | unpinned (9.1.1) | MIT | D | https://pypi.org/project/pytest/ |
| nbformat | unpinned (5.11.1) | BSD-3-Clause | D | https://pypi.org/project/nbformat/ |
| nbclient | unpinned (0.11.0) | BSD-3-Clause | D | https://pypi.org/project/nbclient/ |
| nbconvert | unpinned (7.17.1) | BSD-3-Clause | D | https://pypi.org/project/nbconvert/ |
| ipykernel | unpinned (7.4.0) | BSD-3-Clause | D | https://pypi.org/project/ipykernel/ |
| torch | separate (2.14.1+cu130 from https://download.pytorch.org/whl/cu130; preinstalled on Colab) | BSD-3-Clause (project licence). The wheel metadata also lists bundled parts: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | separate | https://pytorch.org |
| torchvision | separate (0.29.1+cu130; preinstalled on Colab) | BSD-3-Clause | separate | https://github.com/pytorch/vision |

- **Excluded on purpose:** `opencv-python` (Apache-2.0) is pulled in by ultralytics but removed through `requirements-exclude.txt`. We use `opencv-python-headless`, which provides the same `cv2` module.
- **Colab:** the notebook uses the runtime's preinstalled torch, torchvision, numpy and OpenCV, which are not pinned. Their versions depend on the Colab runtime. It also imports `google.colab` (part of the Colab runtime, not installed by us).
- **IPython** (9.17.1 in the dev venv, BSD-3-Clause) is imported by the notebook and comes in through `ipykernel`.

**Notable transitive dependencies** (dev venv, 2026-10-10)

| Package | Licence | Pulled in by |
|---|---|---|
| `ultralytics-thop` 2.2.2 | AGPL-3.0 | ultralytics |
| `ultralytics-platform` 0.1.83 | AGPL-3.0-only | ultralytics |
| `python-bidi` 0.6.11 | LGPL | easyocr |
| `tqdm` 4.70.1 | MPL-2.0 AND MIT | various |
| `certifi` | MPL-2.0 | requests |
| `shapely` 2.2.0 | BSD-3-Clause; bundles GEOS, LGPL-2.1 | rapidocr, easyocr |
| `scipy`, `scikit-image`, `pyclipper`, `omegaconf`, `polars`, `triton` | permissive (BSD / MIT) | various |
| NVIDIA CUDA wheels (`nvidia-*`, `cuda-toolkit`, `cuda-bindings`) | NVIDIA proprietary licences | the cu130 torch build |

The NVIDIA CUDA wheels are on the dev machine only. The CPU deploy image must not contain them.

### Bundled native components

**Status:** to be re-generated from `importlib.metadata` and the wheels' licence and SBOM files (`*.dist-info/licenses`, `*.dist-info/sboms`) in Phase 4. We checked the facts below locally on the dev venv on 2026-10-10. For FFmpeg we called `avutil_license()` and `avutil_configuration()` on each bundled `libavutil`.

| Wheel | Bundled component | Licence | Notes |
|---|---|---|---|
| opencv-python-headless 5.0.0.93 | FFmpeg 8.1.1 (`libavcodec`, `libavformat`, `libavutil`, `libswresample`, `libswscale`) | LGPL-2.1-or-later | `avutil_license()`: "LGPL version 2.1 or later"; configure flags `--enable-openssl --enable-libvpx --enable-shared --enable-pic`, with no `--enable-gpl` and no `--enable-nonfree` |
| opencv-python-headless 5.0.0.93 | OpenSSL 1.1.1k (`libssl`, `libcrypto`), OpenBLAS, `libgfortran`, `libquadmath`, libvpx, libaom, libavif, libpng, libdrm | various | Full texts are in the wheel's `LICENSE-3RD-PARTY.txt`. Points to note: OpenSSL 1.1.1 uses the OpenSSL/SSLeay licence; `libgfortran` is GPL-3.0-or-later with the GCC Runtime Library Exception; `libquadmath` is LGPL-2.1-or-later. |
| av 19.0.1 | FFmpeg 9.0.2 | Reports "LGPL version 3 or later" (`--enable-version3`), but GPL code is linked in | Configured with `--enable-libx264 --enable-libx265`. x264 and x265 are GPL-2.0-or-later upstream (to verify), and both `.so` files ship in `av.libs/`. Treat this wheel as containing GPL code. It also bundles gnutls (LGPL-2.1+), nettle, gmp and libunistring (LGPL-3.0+ or GPL-2.0+), alsa-lib (LGPL-2.1+) and LAME (`--enable-libmp3lame`, LGPL), plus permissively licensed dav1d, SVT-AV1, libvpx, opus, libvmaf, libvpl, libwebp and libxcb. |
| pillow-heif 1.8.0 | libheif 1.23.4, libde265 1.1.3, x265 4.2 encoder | libheif and libde265: LGPL-3.0; x265: GPL-2.0 | The wheel's own `LICENSES_bundled.txt` says: "License for pillow-heif binary wheels: GPLv2, due to base library licenses." We only decode HEIC (libde265); the x265 encoder ships with the wheel but we never call it. |
| numpy 2.5.3 | scipy-openblas64 (OpenBLAS and LAPACK), `libgfortran`, `libquadmath` | BSD-3-Clause; BSD-3-Clause-Open-MPI; GPL-3.0-or-later WITH GCC-exception-3.1; LGPL-2.1-or-later | As listed in numpy's `LICENSE.txt` |
| pillow 12.3.0, scipy, shapely, torchvision, pyzmq | libjpeg, libpng, libtiff, libwebp, libavif, freetype, harfbuzz, lcms2, openjpeg, zstd, brotli, xz, OpenBLAS, GEOS, libzmq and others | various | Not itemised yet |

**Phase 4 action.** A Docker image that ships these wheels redistributes GPL and LGPL binaries. Before building it, either include the licence texts and a source offer, or drop pillow-heif and PyAV from the image (for example by converting HEIC and HDR video another way).

## Reference-data sources

All sources below were retrieved on 2026-10-10. Facts were copied with their URL and date, and no page content was redistributed except the CC BY-SA material noted.

### VIN and WMI

These feed `data/reference/wmi_table.csv`, `docs/research/phase0_findings_vin.md` and `docs/research/vin_decoding.md`. The table, the Phase 0 VIN findings and section 8 of `vin_decoding.md` are CC BY-SA 4.0 (see `data/reference/LICENSE-wmi_table.md`).

| Source | Used for | Terms |
|---|---|---|
| English Wikipedia, "Vehicle identification number", revision 1378888660: https://en.wikipedia.org/w/index.php?oldid=1378888660 | WMI rows, VIN structure | CC BY-SA 4.0. Share-alike applies to the table (attribution: `data/reference/LICENSE-wmi_table.md`). |
| Wikibooks, "Vehicle Identification Numbers (VIN codes)/World Manufacturer Identifier (WMI)", revision 4672643: https://en.wikibooks.org/w/index.php?oldid=4672643 | WMI rows | CC BY-SA 4.0 |
| Wikibooks, "Vehicle Identification Numbers (VIN codes)/Toyota/VIN Codes", revision 4673541: https://en.wikibooks.org/w/index.php?oldid=4673541 | Toyota WMI rows | CC BY-SA 4.0. Phase 0 read the unversioned page (https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Toyota/VIN_Codes) on 2026-10-10; 4673541 (2026-10-01) was the current revision that day, and the table cites it. |
| Russian Wikipedia, "Идентификационный номер транспортного средства", revision 152804777: https://ru.wikipedia.org/w/index.php?oldid=152804777 | Uzbek WMI rows (XWB, NS4, NST), XTA | CC BY-SA 4.0 |
| carinfo.kiev.ua WMI dictionary: https://carinfo.kiev.ua/cars/vin/wmi/dic?country=UZBEKISTAN | Uzbek WMI rows (XWB, ULV, NSA, UMS) | No licence stated; facts only |
| NHTSA vPIC API: https://vpic.nhtsa.dot.gov/api/ (`DecodeWMI`, `DecodeVinValues`) | WMI facts in the table; optional VIN decoding at run time (cached) | The FAQ (https://vpic.nhtsa.dot.gov/api/home/index/faq) says it is free for public use with no registration. No licence is stated, and data.gov lists it as "unknown-license" (https://catalog.data.gov/dataset/nhtsa-product-information-catalog-and-vehicle-listing-vpic-vin-decoder-19da5). |
| vin17.com vehicle listing (search snippet only, page not fetched) | The `LW4` (Li Auto) example in the research notes | Terms unknown. The URL is left out here because it contains a full real VIN. |
| vininfo package README on PyPI: https://pypi.org/pypi/vininfo/json | Two published example VINs (Renault, AvtoVAZ) used in `docs/research/vin_decoding.md` section 3.4, shown there with positions 12-17 masked | Cited only |
| ISO 3780 WMI world codes chart, rev. 2021-04-13: https://standards.iso.org/iso/3780/ed-4/en/Current%20WMI%20World%20Codes%20chart%204.13.21.pdf | Row "0" = "Spare Codes", which is why synthetic VINs start with `0` | ISO copyright. Cited only; the PDF is not redistributed. |
| GB 16735 approval draft, WTO TBT notification: https://members.wto.org/crnattachments/2019/TBT/CHN/19_3960_00_x.pdf | China: check digit is mandatory (clause 4.3.3, Annex A) | Cited only. We did not read the final GB 16735-2019 text, which is paywalled. |
| 49 CFR Part 565: https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-565 (read through the eCFR API: https://www.ecfr.gov/api/versioner/v1/full/2026-10-07/title-49.xml?part=565) | US VIN rules, VIN alphabet, check digit | US federal regulation; cited |
| Canada Motor Vehicle Safety Regulations, C.R.C., c. 1038, Schedule IV s.115: https://laws-lois.justice.gc.ca/eng/regulations/C.R.C.,_c._1038/FullText.html | Check digit is mandatory in Canada | Cited |
| Regulation (EU) 2021/535, Art. 12: https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R0535 | EU phase-in of the check digit | Cited |
| Uzbekistan Cabinet of Ministers Resolution No. 237 (2017): https://www.lex.uz/acts/3180907 | Uzbek VIN rules; kit-assembled cars keep the manufacturer's VIN | Cited |

### Prices, dealers and colours

These feed `data/reference/repair_prices.csv` (built by `scripts/build_repair_prices.py`), `data/reference/colours.json`, `docs/research/market_prices.md` and `docs/research/phase0_findings_deploy_market.md`.

| Source | Used for | Terms |
|---|---|---|
| https://auto.ustabor.uz/ru/price/body_work | "From" prices for painting, dent, scratch, panel replacement and polishing | Site terms not reviewed. Prices are quoted as facts with URL and date. |
| olx.uz listings: https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-i-t-d-ID1YwdF.html, https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-shumo-tsiya-kostoprav-ID22hsA.html, https://www.olx.uz/d/obyavlenie/polirovka-avto-kuzova-za-lyuboy-mashina-600000-sum-600-tsyach-sum-ID4bVtP.html, https://www.olx.uz/d/obyavlenie/avto-vakuum-pachok-tugirlash-vakum-ID3goDG.html, https://www.olx.uz/d/obyavlenie/proverka-avto-po-kuzovu-i-probegu-po-nizkim-tsenam-vyezdnoy-osmotr-ID3Vjj4.html | Paint, dent, polishing and inspection prices | Public classified ads. We keep the price, URL and date only, never seller names or phone numbers. |
| https://www.cenotavr.uz/tashkent/remont-kuzova-pokraska-polirovka-kuzova-90 | One advert price whose service is not stated (kept as context, excluded from cost estimates) | Site terms not reviewed. We keep the price, URL and date only, never the contact names shown on the page. |
| https://t.me/byd_chigatoy | BYD dealer promotions (kept as context, not used as repair costs) and dealer cities | Public channel posts |
| Central Bank of Uzbekistan rates: https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/2026-10-10/ | USD to UZS conversion (11897.90 UZS per USD, dated 2026-10-10) | Official published rate |
| https://bydauto.uz (model pages `/ru/chazor`, `/en/song-dmi-champion`, `/ru/yuan-up-dmi`, `/en/ti7`) | Colour swatches in `colours.json` (approximate website colours, not factory paint codes); BYD model prices in the research notes | Site terms not reviewed |
| https://byd-rm.uz/about | BYD Chigatoy belongs to the Royal Motors group (research notes) | Site terms not reviewed |
| https://www.toyota-centralasia.com/ru/dealers | Toyota dealer list (research notes) | Site terms not reviewed |
| https://t.me/toyota_rm | Toyota dealer channel (research notes) | Public channel posts |

### Maps and timestamps (shops lookup used by the app since 2026-10-11; timestamps not used yet)

| Source | Planned use | Terms |
|---|---|---|
| OpenStreetMap data through the Overpass API (https://overpass-api.de/api/interpreter; mirror https://maps.mail.ru/osm/tools/overpass/api/interpreter) | Nearby body and paint shops | ODbL 1.0. Show "© OpenStreetMap contributors" linked to https://www.openstreetmap.org/copyright, or print that URL in a PDF. |
| Nominatim geocoding | Optional address lookup | At most 1 request per second, cache results, no autocomplete (https://operations.osmfoundation.org/policies/nominatim/) |
| freetsa.org: https://freetsa.org/tsr | Optional RFC 3161 timestamps for passport PDFs | Free, with no SLA or rate limit stated. Commercial users are asked to contact the operator. |

### Documentation cited in `docs/research` (facts only, no data reused)

- **Hugging Face Spaces:** https://huggingface.co/docs/hub/spaces-gpus, https://huggingface.co/docs/hub/spaces-overview, https://huggingface.co/docs/hub/spaces-github-actions
- **Render:** https://render.com/docs/free, https://render.com/docs/compute-plans
- **Ultralytics licence:** https://www.ultralytics.com/license, https://www.ultralytics.com/request-license
- **PyTorch wheel index:** https://download.pytorch.org/whl/cu130
- **Package versions:** PyPI project pages for the packages listed above
- **HDR video handling in `src/avtopassport/pipeline/frames.py`:** SMPTE ST 2084 (PQ) constants and the 203 cd/m2 reference white reported in ITU-R BT.2408 (https://www.itu.int/pub/R-REP-BT.2408)

## Added on 2026-10-11 (Phase 3/4 and 3D)

### Web app and passport report (main venv, `requirements.txt`)

| Component | Version | Licence | Link | Use |
|---|---|---|---|---|
| FastAPI | 0.143.0 | MIT | https://pypi.org/project/fastapi/ | API |
| uvicorn[standard] | 0.54.0 | BSD-3-Clause | https://pypi.org/project/uvicorn/ | ASGI server |
| python-multipart | 0.0.32 | Apache-2.0 | https://pypi.org/project/python-multipart/ | uploads |
| httpx | 0.28.1 | BSD-3-Clause | https://pypi.org/project/httpx/ | tests (FastAPI TestClient) |
| reportlab | 5.0.1 | BSD | https://pypi.org/project/reportlab/ | PDF passport |
| DejaVu Sans font (system package `fonts-dejavu-core`) | system | Bitstream Vera / DejaVu font licence (free, redistributable) | https://dejavu-fonts.github.io/License.html | Cyrillic text in the PDF |
| three.js | 0.186.1 (CDN, jsdelivr) | MIT | https://github.com/mrdoob/three.js/blob/dev/LICENSE | 3D point-cloud viewer |
| Spark (`@sparkjsdev/spark`) | 2.3.1 (CDN, jsdelivr) | MIT | https://github.com/sparkjsdev/spark | Gaussian-splat viewer (high-quality 3D, when available) |

### 3D reconstruction (separate venv `.venv-3d`, see `docs/research/recon3d_candidates.md`)

| Component | Version | Licence | Link | Flags / use |
|---|---|---|---|---|
| Depth Anything 3 (code, PyPI `depth-anything-3`) | 0.1.1 | Apache-2.0 | https://github.com/ByteDance-Seed/Depth-Anything-3 | fast 3D point cloud from anonymised frames |
| DA3-BASE weights `depth-anything/DA3-BASE` @f4a6c9b3 | — | Apache-2.0 (HF card) | https://huggingface.co/depth-anything/DA3-BASE | used. DA3 GIANT / NESTED / LARGE / LARGE-1.1 are CC BY-NC or conflicting and are NOT used |
| DA3-SMALL weights `depth-anything/DA3-SMALL` @e08cab65 | — | Apache-2.0 (HF card) | https://huggingface.co/depth-anything/DA3-SMALL | optional lighter model |
| addict (undeclared DA3 dependency) | 2.4.0 | MIT | https://github.com/mewwts/addict | |
| xformers (pulled in by DA3, falls back without its CUDA extensions here) | 0.0.35 | BSD-3-Clause (repo licence) | https://github.com/facebookresearch/xformers | not required |
| open3d, moviepy, … (pulled in by DA3, not used by our code) | 0.20.0, 1.0.3 | MIT | https://pypi.org/project/open3d/, https://pypi.org/project/moviepy/ | to be excluded from the Docker 3D env |
| pycolmap (COLMAP structure-from-motion) | 4.2.1 | BSD-3-Clause | https://github.com/colmap/colmap | calibrated camera poses for high-quality 3D |
| Brush (Gaussian-splat trainer, release binary v0.3.0, sha256 4f0f9a87…) | 0.3.0 | Apache-2.0 | https://github.com/ArthurBrussee/brush | high-quality 3D on a GPU (Vulkan); downloaded to `tools/` (gitignored) |

## Our own data

| Data | Real or synthetic | Where | Rules |
|---|---|---|---|
| Walkaround video, car photos and VIN-plate photos of the team's own car | Real, consented | Originals in `data/raw/`, which is gitignored and never committed | Plates and faces are pixelated before anything is saved or shown, and the privacy step fails closed. Saved JPEGs carry no metadata. Only blurred copies may be committed (`data/processed/`). A real car's full VIN is never committed; if one must appear, positions 12-17 are masked. Licence for the committed copies: not decided yet. |
| Local test set (Phase 2, not collected yet) | Real, team-collected | `data/local_testset/` | Same rules, and no people in frame |
| Repair prices | Real public prices (phone quotes to follow, each with source and date) | `data/reference/repair_prices.csv` | `is_estimate=true` marks our own estimates (none so far) |
| Registry-type records: owners, accidents, insurance claims, customers | Synthetic | `data/synthetic/`, generated by `scripts/generate_synthetic.py` | See below |
| Run outputs (annotated frames, timings) | Derived from the above | `outputs/` (gitignored) | Only images that have passed the privacy step |

**Synthetic registry records**
- Generated with a fixed seed (`SEED` in `src/avtopassport/synthetic.py`).
- Synthetic VINs use WMI `0AP` (ISO 3780 row "0" = spare codes) and have a valid check digit.
- Every record has `"synthetic": true`, and the UI shows a SYNTHETIC DEMO DATA badge.
- The method is described in `data/synthetic/README.md`.
- Synthetic history is never attached to a VIN read from a real car.
- **Inputs taken from public sources (names only, no prices or records):**
  - Toyota model names from https://uzbekistan.toyota-centralasia.com/ru/ and BYD model names from https://bydauto.uz.
  - Part names from the parts model's `parts_catalog.json` (`parts_mitbersh_catalog` above).
  - The 12 Tashkent city district names from English Wikipedia, "Tashkent", revision 1376716206 (https://en.wikipedia.org/w/index.php?title=Tashkent&oldid=1376716206; the page is CC BY-SA 4.0, and only the names are used).
  - Emails use the reserved `.invalid` domain (RFC 2606, https://www.rfc-editor.org/rfc/rfc2606.html).
