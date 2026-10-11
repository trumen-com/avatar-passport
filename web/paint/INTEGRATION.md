# Paint Check inside Trumen: integration guide

## 1. The core concept, and why Paint Check belongs to it

Trumen Guard's concept: **the phone that looks after the car at every Royal Motors after-sales step.** It runs on the
phone, nothing is uploaded, and every number is measured on real data before it is shown.

Paint Check is not a separate product. It is the same concept at a different step of the cycle:

| After-sales step | Who holds the phone | Trumen module | What it decides |
|---|---|---|---|
| **Car comes in** (trade-in intake) | Trade-in advisor | **Paint Check** | Which panels need a paint-gauge check before the price is set |
| **Car goes out** (new, serviced or traded-in car handed over) | Customer | **Guard** | When to warn the driver about a collision |
| **Service and claims** | Service advisor | **Scan** | Which parts are on this exact car, and their size and position |

The same principles hold across all three modules:

- **On-device.** Analysis runs in the phone's browser. Photos and video are not uploaded. Paint Check's Collect mode
  keeps photos on the phone and exports them only when the user taps Export (on the public site; the Claude artifact
  build stores Collect photos in its cloud store instead).
- **Measured before shown.** Every figure on the site comes from a reproducible evaluation script in the repository:
  Guard on 667 Nexar crash clips, Paint Check on 921 held-out real panel joints.
- **Assist, don't decide.** Guard warns but does not brake. Paint Check says "check with a gauge", not "this car was in
  an accident". Scan's part searches are text queries, not verified part numbers.
- **Inside Royal Motors after-sales.** One owner, one pilot, one budget line.

### One-sentence pitch line

> "Trumen sees the car at every after-sales step: Paint Check when it comes in, Guard when it goes out, Scan when it
> comes back for service."

## 2. What changed in the system (this commit)

| File | Change |
|---|---|
| `guard/app/index.html` | "Trade-in paint check" added to the site navigation, next to "3D parts scan" |
| `guard/app/src/pages.ts` | Business page gains "One phone at every after-sales step": the three steps as cards, Paint Check's measured results, and a Paint Check trial at the trade-in desks in pilot month 2 |
| `paint/app.html` | Header links back to the Trumen family site |
| `web/public/paint/*` | Built Paint Check, served at `/road-vision/paint/` (the deploy workflow already copies it) |
| `paint/` | Engine, app, evaluation scripts, README with results |

Unchanged: Guard's engine, models, results, deck file and Scan. Pricing and payback figures are unchanged too: Paint
Check adds no revenue line until the pilot measures it.

## 3. Technical contract (for building it into another app)

Paint Check is three plain ES modules with no dependencies and no build step: `engine.js`, `car.js`, `app.js`. The
engine works in a browser or in Node.

```js
import { analyzeJoint, judgePanels, DEFAULTS } from "./engine.js";
import { JOINTS, PANELS, jointFactor } from "./car.js";

// one photo of one gap -> one joint result
const r = analyzeJoint(imageData,              // {data: RGBA Uint8ClampedArray, width, height}, e.g. canvas getImageData
  { orientation: "vertical",                   // or "horizontal": which way the gap runs in the photo
    gapX: null,                                // null = find the gap; 0..1 = the user tapped it
    factor: jointFactor(JOINTS[2]) });         // 1 side joints, 1.4 bumpers, 1.8 hood/trunk edges
// r.verdict: "match" | "check" | "differs"; r.reliable (false = retake); r.score, r.dE, r.dL, r.dab, r.warnings

// all joints of one car -> panel statuses
const panels = judgePanels(JOINTS.map((j) => ({ a: j.a, b: j.b, factor: jointFactor(j), result: results[j.id] }))
  .filter((j) => j.result));
// panels.lf_door = { status: "original" | "check" | "suspect" | "repainted", joints, worst, consistency }
```

- **Speed:** about 0.15 s per joint photo in Node on our test machine; on phones it varies with the phone. A photo is downscaled to 720 px on its long side first.
- **Thresholds:** in `DEFAULTS` (`checkAt` 1.0, `flagAt` 1.6, `panelAt` 0.45, `angleK` 30) and `car.js` (`BUMPER_FACTOR`
  1.4, `LID_FACTOR` 1.8). They were tuned on real photos (`eval/`). Re-tune with `eval/run.mjs` when real repaint data
  with gauge readings arrives.
- **Data export (Collect mode):** `labels.json`
  `{format: "trumen-paintcheck-1", cars: [{id, make, model, colour, gauge: {panelId: µm}}], shots: [{car, joint, a, b, orientation, gapX, photo}]}`
  plus `photos/*.jpg`.
- **Panel ids:** `f_bumper, hood, lf_fender, rf_fender, lf_door, rf_door, lr_door, rr_door, lr_quarter, rr_quarter,
  trunk, r_bumper, roof`. Left and right are as seen from the driver's seat.
- **Hosting:** static files. On the team site the deploy workflow copies `web/public/paint` to `/paint/`.

### Hooks into Royal Motors systems (pilot, not built)

- **Trade-in record:** the panel statuses and the copied report go into the trade-in appraisal form as "panels to
  gauge". Paint Check's result is an input to the appraiser, never the price itself.
- **Scan:** a car that comes in for a trade-in can also be scanned. The two use different names, so map them
  with the table below (Paint Check ids → Scan `key`; left/right are the car's own left/right in both) and a flagged
  door can be highlighted on the 3D model. The table is built from `guard/app/public/scan/parts.json`; check the
  left/right convention on one real scan before relying on it.

  | Paint Check id | Scan key | | Paint Check id | Scan key |
  |---|---|---|---|---|
  | f_bumper | front-bumper | | rr_door | right_back-door |
  | hood | hood | | lr_quarter | left_quarter-panel |
  | lf_fender | left_fender | | rr_quarter | right_quarter-panel |
  | rf_fender | right_fender | | trunk | trunk |
  | lf_door | left_front-door | | r_bumper | back-bumper |
  | rf_door | right_front-door | | roof | roof |
  | lr_door | left_back-door | | | |
- **Guard:** a traded-in car that is resold leaves with Guard at handover, which completes the cycle.

## 4. Deck slide (copy for the business section of the defence deck)

**Title:** One phone at every after-sales step

- **Car comes in:** Paint Check marks panels to gauge before the trade-in price is set.
  - 26% of original-paint joints flagged; 13% of original panels wrongly called repainted.
  - 90% of simulated 2 ΔE panel repaints flagged for a check.
  - Measured on the held-out half (425 of 853 real photos, 921 joints); real repaints come in the pilot.
- **Car goes out:** Guard gives a collision warning at handover.
  - mAP 0.69 on 667 real crash clips.
- **Service:** Scan rebuilds the car in 3D and splits it into parts.
- **Same rules everywhere:** on the phone, nothing uploaded, measured before shown.

**Speaker note:** "We built Guard for the car leaving the dealership. The same idea works when a car comes in for
trade-in. A hidden accident repair lowers what a car is worth, and gauging every panel of every trade-in takes time,
so the phone tells the appraiser which panels to gauge first. Our first version false-alarmed on more than half the joints. We tuned it on real photos and cut that
to one in four, and we show you the numbers."

## 5. Honest limits to say out loud

- Real repaints have not been measured yet: detection numbers come from simulated repaints on real photos.
- The test photos were small web photos. Close-up phone photos should do better, but that is not yet measured.
- A whole-car respray has no "odd one out", so Paint Check cannot see it.
