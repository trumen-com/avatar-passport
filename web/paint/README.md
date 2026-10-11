# Trumen Paint Check

Repaint screening for trade-in cars with a phone camera. The inspector photographs each of the 14 gaps between body panels.
The app compares the paint on the two sides of each gap **in the same photo, under the same light**, and marks panels
that differ from all their neighbours as likely repainted.

Live: `https://trumen-com.github.io/road-vision/paint/` (static, runs fully in the browser; photos never leave the phone
unless you export them).

## How it works (`engine.js`)

1. **Find the gap.** Near the guide line, the gap is the thin dark line with the strongest across-gap gradient. If it
   can't find a clear one, it asks the user to tap the gap.
2. **Sample both sides.** It takes narrow strips (10% of the photo width) right next to the gap, at three heights. In
   each strip it drops blown-out pixels, reflections (top 40% of brightness) and shadow and dirt (bottom 15%).
3. **Compare.** For each pair of facing strips it measures:
   - CIEDE2000 colour difference at equal brightness;
   - the brightness difference;
   - the colour direction (Δa\*, Δb\*).

   The joint takes the median over the three strip pairs.
4. **Score.** `score = ΔE / (1 + ΔL/30)`: more colour tolerance when one panel catches more light. The score is divided
   by 1.8 at hood and trunk edges and by 1.4 at bumpers, since those panels face the light differently. Joints read
   "same paint" below 1.0, "small difference" up to 1.6, and "different paint" above.
5. **Panels.** A panel is "likely repainted" when it has two or more measured joints, every joint shows at least a small
   difference, and its colour is off in the **same direction** against every neighbour (smallest projection onto the mean
   direction ≥ 0.45). Lighting differences point in random directions; a repaint does not. Otherwise a panel with a
   "different" joint is "check with a gauge".
6. **Photo checks.** No clear gap, too few clean paint pixels, a box covering trim or a sticker, glare over most of a
   box, too dark, or blurry → "Retake photo".

Texture and gloss are still measured and shown, but not scored: on the real photos below they added noise, not signal.

## Results on real photos

No public dataset has repainted panels with gauge readings, so we tested on what exists. The source is the
[Car Parts and Damages](https://huggingface.co/datasets/DrBimmer/car-parts-and-damage-dataset) dataset (DrBimmer, MIT
licence): 998 real car photos with every panel outlined.

- **Joints.** `eval/extract_joints.py` cuts out every seam between two neighbouring painted panels. It keeps only crops
  whose sample boxes are ≥ 92% clean paint, using the lights, glass, plates, grille and wheel outlines. That gives
  **1,766 real joints from 853 photos.**
- **Split.** Photos were split in half: settings were tuned on 845 joints and checked on 921 others.
- **What the photos show.** Nearly all panels are original paint, so they measure **false alarms**.
- **Simulated repaints.** We shifted the colour of one panel by a known amount, by the same amount at all of its joints,
  as a real repaint would be. That measures **detection**. These are real photos with simulated repaints, not real
  repaints.

| Held-out photos (921 joints, 133 panels with 2+ joints) | Result |
|---|---|
| Original paint flagged "different" at a joint | 26% (first version: about 64%) |
| Simulated repaint caught at a side joint (door, fender, quarter), 2 ΔE / 3 ΔE | 80% / 94% |
| Original panel called "likely repainted" | 13% |
| Simulated 2 ΔE panel repaint called "likely repainted" / at least "check with a gauge" | 65% / 90% |
| Simulated 3 ΔE panel repaint called "likely repainted" / at least "check with a gauge" | 75% / 95% |
| Automatic gap finding within 3% of the outlined seam | 69% |

**Caveats.**
- These are whole-car web photos of often damaged cars, with seams about 100 px long. The app asks for close-ups from
  half a metre, which should do better, but that is not yet measured.
- Some "original" panels in the dataset may in fact be repainted, which would make the false-alarm figures pessimistic.
- 49% of the dataset crops were too small for the narrow sample strips ("retake"). Full-size phone photos are about 10×
  larger.

Reproduce:

```
python3 -I eval/extract_joints.py <dataset> <joints>
node eval/sweep.mjs <joints> <ids> <out>       # features per sampling setting
python3 eval/fit_score.py <out>                # fit the score
node eval/panels.mjs ...                       # panel-level features
python3 eval/panel_rules.py ...                # compare panel rules
node eval/final_eval.mjs <joints> <test ids> <out>
```

## Data collection and evaluation

- In the app, the **Collect data** tab stores each joint photo with the make, model, colour and the paint-thickness gauge
  reading of every panel. Inside Claude it uploads to the artifact's private store; on the public site it keeps the data
  on the phone and exports one `.zip` (`labels.json` + `photos/`).
- Ground truth: a panel reading **over 200 µm** counts as repainted. Factory paint is usually 80–150 µm.
- `node eval/run.mjs <unzipped export> --out results.json` reports:
  - joint-level ROC AUC with a 95% bootstrap CI, resampling whole cars;
  - AUC per feature;
  - caught / false flags at the shipped threshold, and with the threshold chosen leaving one car out;
  - panel-level detection;
  - the retake rate.
- `node eval/make_synthetic.mjs <dir>` makes a synthetic set **only to test the pipeline**. Its numbers are not results.
- `node eval/selftest.mjs` checks CIEDE2000 against the Sharma et al. reference pairs and runs synthetic joints.

## Limits

- Screening, not a verdict: confirm flagged panels with a paint gauge before pricing.
- A well-blended professional repaint can match in colour and texture; a full respray of the whole car has no "odd one
  out".
- Hard cases are metallic and pearl paints, very dark colours, and dirty or wet cars.
