# Labelling guide: local test set (Phase 2)

How the team photographs, blurs and labels the local test set that `scripts/eval.py` scores (CLAUDE.md section 3,
Phase 2). Method and metrics: [evaluation.md](evaluation.md).

## Privacy rule (read first)

- **Only blurred images are uploaded to the labelling tool.** Upload files from `data/local_testset/images/` (output
  of `scripts/blur_testset.py`), never from `data/raw/`. Check each image for an unblurred plate or face before you
  upload it; if you see one, delete the image from the upload and tell the person who runs the blur script.
- Photograph cars in parking lots with no people in frame. Plates are blurred by the script, but do not rely on it:
  avoid framing other people's plates when you can.
- Real VINs are never committed and never uploaded: VIN photos go to `data/raw/testset/vin/` and the true VINs to
  `data/raw/testset/vin_ground_truth.csv` (both under `data/raw/`, which is gitignored). `scripts/eval.py` reports
  only the exact-match rate.

## 1. Photograph

| folder | what |
|---|---|
| `data/raw/testset/day/` | daylight photos |
| `data/raw/testset/night/` | night / low light (street lamps, flash off) |
| `data/raw/testset/wet/` | rain or a wet car |
| `data/raw/testset/vin/` | VIN plate / sticker close-ups (not blurred, not labelled, never copied) |

- 1-3 photos per car, each showing a whole side or corner (front 3/4, side, rear 3/4), as the app's walkaround frames
  would. Damaged cars: also one closer photo of the damage, still showing the surrounding panels.
- **Include cars with no damage.** False positives per car (reflections, dirt, stickers) can only be measured on
  clean cars.
- File names: anything without personal data, e.g. `car017_front34.jpg`, and unique across day / night / wet
  (labelling tools often export the bare file name). Do not put plates or VINs in file names.

## 2. Blur

```bash
YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_testset.py --dry-run
YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_testset.py --device auto
```

This writes `data/local_testset/images/<slice>/<name>.jpg` (long side 1600 px, no EXIF) and
`data/local_testset/slices.csv`. Then fill in `slices.csv` (re-running the script keeps what you typed):

| column | fill with |
|---|---|
| `brand` | e.g. `Chevrolet`, `Toyota`, `BYD` (one spelling per brand) |
| `car` | the same id for all photos of one car (e.g. `car017`); empty = each photo is its own car |
| `no_damage` | `yes` when the car has no visible damage at all |
| `notes` | free text (e.g. "dirty", "low sun"); never plates, VINs or owner details |

## 3. Label (CVAT recommended, Roboflow possible)

Create one project/task with these **polygon labels** (exactly these names, the damage models' vocabulary):

`scratch`, `dent`, `crack`, `glass shatter`, `lamp broken`, `tire flat`

and one more label `no_damage` (see below).

Give every damage label a **text or select attribute `part`** whose values are the parts model's class names
(`parts_mitbersh`, `docs/research/model_candidates.md`). Left / right must follow the parts model's own convention:
before labelling, look at its overlay in `docs/assets/smoke/R1_contact_parts_mitbersh.jpg`, agree on the convention as
a team and use it for every image (a wrong side is only rescued by the IoU rule of part recall):

`back-bumper`, `back-windshield`, `front-bumper`, `grille`, `hood`, `left_back-door`, `left_back-wheel`,
`left_back-window`, `left_fender`, `left_front-door`, `left_front-wheel`, `left_front-window`, `left_headlight`,
`left_mirror`, `left_quarter-panel`, `left_rocker-panel`, `left_tail-light`, `license-plate`, `right_back-door`,
`right_back-wheel`, `right_back-window`, `right_fender`, `right_front-door`, `right_front-wheel`,
`right_front-window`, `right_headlight`, `right_mirror`, `right_quarter-panel`, `right_rocker-panel`,
`right_tail-light`, `roof`, `trunk`, `windshield`

Rules:

- **One polygon per damage instance**, tight around the visible damage (not the whole panel). A long scratch is one
  polygon; two separate dents are two polygons.
- Damage that crosses two parts: one polygon per part, each with its own `part`.
- Damage you are not sure about (dirt vs scratch, reflection vs dent): do not label it, and write it in `notes`.
- Damage of a type not in the list (paint chip, rust, missing part): skip it and write it in `notes`; it is outside
  the models' vocabulary.
- **Cars with no damage:** tag the image `no_damage` (CVAT: an image-level tag; Roboflow: draw one `no_damage` box
  anywhere) **and** write `yes` in the `no_damage` column of `slices.csv`. COCO exports often drop image-level tags,
  so the CSV column is what `scripts/eval.py` relies on. Images with neither a polygon nor the `no_damage` tag are
  treated as not yet labelled and are skipped.
- Roboflow has no per-polygon attributes: instead name the class `<damage>__<part>` (two underscores), e.g.
  `dent__left_front-door`; `scripts/eval.py` splits it.

## 4. Export

- CVAT: Export task dataset, format **COCO 1.0**, without images. Polygons are exported as polygons (do not use the
  brush / mask tool: RLE masks are rejected).
- Roboflow: Export, format **COCO Segmentation** (polygons).
- Save the JSON as `data/local_testset/annotations.json`. The `file_name` of each image must match the path under
  `data/local_testset/images/` (`night/car017_front34.jpg`) or at least its file name.
- Check it: `YOLO_AUTOINSTALL=false .venv/bin/python scripts/eval.py --check`

## 5. VIN ground truth (optional, never committed)

`data/raw/testset/vin_ground_truth.csv`:

```csv
file,vin
<name of the photo in data/raw/testset/vin/>,<the 17-character VIN read by a person>
```

Read the VIN from the car's documents or by eye, not from our OCR output.

## 6. How much data (our recommendation, not a standard)

To make per-slice numbers worth quoting, aim for at least:

| per slice (day / night / wet) | minimum |
|---|---|
| cars | 20 |
| damage instances (all classes) | 30 |
| clean cars (`no_damage`) | 5 |
| VIN photos (whole set) | 10 |

Per-class AP for a class with fewer than about 10 instances is noise; report it with its count or not at all. These
thresholds are our own guidance for a hackathon-sized set; `scripts/eval.py` prints the counts next to every metric so
the reader can judge.
