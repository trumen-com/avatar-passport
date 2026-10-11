# Prompt for the pitch deck (paste into the slide tool)

Create a 7-minute investor and jury pitch deck for **AvtoPassport**: a phone video of a car becomes a verified condition
passport tied to its VIN. Build 11 slides. Keep each one very brief: one idea per slide, at most about 25 words of body
text, and one big number where possible. Put the talk track in the speaker notes, with the timing.

## Timing (7:00 total)
| Part | Time | Slides |
|---|---|---|
| Problem and user | 1:00 | 1–2 |
| Live demo | 2:00 | 3, then the live website |
| Technology | 2:00 | 4–6 |
| Budget and ROI | 1:30 | 7–9 |
| Next steps | 0:30 | 10 |
| Team | end | 11 |

## How the jury scores (design every slide for this)
- Technical complexity (25%): show the architecture, data pipeline, measured quality, speed, cost per request and
  known failure cases.
- Innovation (25%): say what another team could not rebuild in a weekend.
- Impact (25%): who the user is, who pays, the evidence, and a pilot within 3 months.
- Presentation (15%): clear story, live demo, honest numbers.
- Business (10%): pilot budget, unit economics and payback, with every assumption stated.

## Honesty rules (must hold on every slide)
- Every number has a source, or is labelled "estimate" or "assumption".
- Do **not** claim the app reveals a car's full accident history. It finds **visible damage now** and **screens for
  repaints**. Full history needs the registry (next steps).
- Say "check with a paint gauge", not "this car was in an accident".

## Slide by slide

**1. Problem and user (0:30)**
- Title: "Buying a used car in Uzbekistan is a gamble"
- Big numbers:
  - about 60% of passenger-car sales are second-hand: 57,400 of 95,400 in October 2025 (CERR).
  - about 690,000 used cars a year change hands. This is an estimate: about 57,000 a month × 12.
- Seller tricks: cosmetic cover-ups and repaints over crash repairs, rolled-back odometers, paperwork games, and
  "strategic silence".
- The state now fines hiding odometer changes: Article 178-2, 10–15 BCA, about 4.1–6.2 million UZS.
- Customer validation: "We interviewed 2 drivers who bought used cars."
  - Placeholders: [Quote 1, first name, city]; [Quote 2, first name, city].
  - Add more interviews if possible.

**2. Why today's checks fail (0:30)**
- Buyers rely on:
  - the seller's word;
  - a mechanic friend;
  - a paint-thickness gauge, which is a separate device, rarely owned, and needs know-how;
  - paid reports: the my.gov.uz history check costs 37,080 UZS (about $2.90); foreign VIN reports such as
    carVertical or BidCar cost €5–15, and cover imports only.
- None of them shows the car's **current condition** on the buyer's own phone in under a minute.

**3. Live demo (2:00), presenter switches to the website**
- Script:
  1. Open the site and tap "Use demo car". The passport arrives in about 20 s: VIN, damage list with repair prices in
     UZS, and a 360° view.
  2. Car tab: the 3D parts model with damaged parts in red, then "Take apart".
  3. Paint check: "Use demo photos" shows which panels to gauge.
  4. Switch the language to UZ or RU and download the PDF passport, which carries a SHA-256 fingerprint.
- Backup: a recorded video of the same flow.

**4. How it works: architecture (0:45)**
- Diagram: phone (video + VIN photo) → frames → quality filter → privacy blur (plates and faces, before anything is
  stored).
- Two branches, then a merge:
  - damage segmentation and car-parts segmentation (33 side-aware parts) → damage-to-part mapping → merge across
    frames → severity → repair cost from a local UZS price table;
  - VIN OCR → ISO 3779 check digit → decode with our own maker table for Uzbek and Chinese plants, plus NHTSA.
- Both branches feed: passport store by VIN → "new since last inspection" diff → PDF with SHA-256.
- In the background:
  - a 3D parts model: COLMAP camera poses + Depth Anything V2 depth + multi-view fusion;
  - optional Gaussian splats.
- Paint check runs in the browser, on panel-gap photos.

**5. Measured, not promised (0:45).** All figures were measured on our laptop GPU (RTX 5050) with a 29-second
walkaround of a real car.

| What | Measured |
|---|---|
| Passport from video | 17.3–17.5 s (115 frames) |
| 3D parts model | about 3.5 min (camera poses 180 s + 23 s) |
| Detailed 3D | 36 min; held-out image quality 25.1 dB PSNR |
| Camera poses | 115 of 115 frames placed, 0.53 px error |
| VIN reader, real photos | 3 of 3 read, valid check digit (one car) |
| VIN reader, synthetic | 60 of 60 exact, 0.3 s each |
| CPU-only server | 139 s per passport |
| Paint check (Trumen eval, 921 real held-out joints) | 26% false flags on original paint; simulated 2 ΔE repaint at least "check" on 90% of panels |
| Parts-model accuracy on the demo car | wheelbase +2 mm, width +16 mm, height +4 mm vs published (tuned on that car) |

- Cost per passport: about $0.003 of GPU time (17.5 s at an assumed $0.60/h cloud GPU). With the 3D parts model it is
  about $0.04.

**6. What stops a weekend clone (0:30)**
- It is a pipeline, not one API call: 10 stages and 3 models, plus fail-closed privacy, VIN validation and a tamper-proof
  PDF.
- Local data moat:
  - a UZS repair-price table from Tashkent shops;
  - a VIN maker table for Uzbek and Chinese plants;
  - local photos for privacy tuning;
  - next, paint-gauge readings collected at trade-in desks.
- Physics-based paint check, needing no training data, combined with per-part 3D.
- Works in Uzbek, Russian and English.
- Known failure cases, shown honestly:
  - windshield reflections read as "glass shatter";
  - wrap decals read as scratches;
  - hood-edge paint false alarms;
  - a full respray is invisible;
  - night and rain are not tested yet.

**7. Pricing (0:30)**
- Free: 1 quick check.
- Single report: 29,000 UZS (about $2.30), below the 37,080 UZS government check.
- Buyer Pass: 49,000 UZS (about $3.80) for 30 days of unlimited checks. Buyers usually look at several cars.
- Pro (dealers, bazaar inspectors, trade-in desks): 990,000 UZS (about $77) per location per month.
- Exchange rate used: 12,800 UZS per USD.

**8. Market: TAM / SAM / SOM (0:30)**
- TAM: 690,000 purchases × 3 cars checked × 29,000 UZS ≈ 60 billion UZS ≈ $4.7 M a year.
- SAM: smartphone buyers in Tashkent and the 3 largest regions, assumed 50% ≈ $2.3 M a year.
- SOM in year 3, assumed:
  - 10% of SAM buyers on a Buyer Pass ≈ $130 k;
  - 150 Pro locations ≈ $140 k;
  - total ≈ $270 k a year.
- Context:
  - 4.39 M private passenger cars, growing by about 0.5 M a year (stat.uz, Oct 2025);
  - 1.4 M+ used-car listings online (IMRI, Apr 2026).

**9. Budget and payback (0:30). All figures are assumptions; state them on the slide.**

| 3-month pilot | Cost |
|---|---|
| 3 engineers × $1,800/mo + 1 BD/ops × $1,000/mo | $19,200 |
| GPU and CPU servers hosted in Uzbekistan (data-localisation law) | $1,600 |
| Data: 1,000 labelled local damage photos, 2 paint gauges, gauge readings on 200 cars | $1,200 |
| Model licence (commercial Ultralytics) and retraining on licensed data | $5,000 |
| Marketing: Telegram and Instagram ads, bazaar presence | $3,000 |
| Legal and company setup | $1,500 |
| Contingency 15% | $4,725 |
| **Total** | **≈ $36,000** |

- First year: about $140,000, a run-rate of about $11,500 a month.
- Break-even: about 5,100 single reports a month, which is about 3% of the roughly 172,000 checks a month
  (57,400 purchases × 3).
- Gross margin above 95%, because inference costs well under 1% of the price.

**10. Next steps (0:30)**
1. A 3-month pilot at a Royal Motors trade-in desk. It fits the existing intake process; Paint check tells the appraiser
   what to gauge.
2. A partnership with the Ministry of Internal Affairs for VIN registry access: accidents, owners and mileage in the same
   passport.
3. Build our own labelled test set, covering day, night, rain and local cars, and calibrate the damage models on it.
4. Spare-part check: photograph a part to detect defects.

**11. Team (end).** Leave three input fields, each with a photo placeholder:
- [Member 1: name, role, photo]
- [Member 2: name, role, photo]
- [Member 3: name, role, photo]

Then the team name, large: **[TEAM NAME]**.

## Sources (cite on the slides as short footnotes)
- CERR via anhor.uz: October 2025, 57.4 k used out of 95.4 k passenger-car sales. https://anhor.uz/news/situation-4/
- kun.uz, Nov 2025: about 52 k used light vehicles a month; total market 101.1 k.
  https://kun.uz/en/news/2025/12/23/car-sales-in-uzbekistan-up-15-percent-year-on-year
- kun.uz (CERR, 2020): 400–500 k used-car contracts a year, market about $2 B.
  https://kun.uz/ru/news/2020/11/23/obyem-vtorichnogo-rynka-avtomobiley-v-uzbekistane-otsenili-v-2-mlrd-dollarov
- stat.uz: 4.39 M passenger cars owned by individuals, Oct 1, 2025.
  https://stat.uz/en/press-center/news-of-committee/65603-zbekistonda-zhismonij-shakhslarga-tegishli-avtotransport-vositalari-soni-4-7-mlndan-oshdi-4
- zamin.uz: the car count rose by almost half a million in one year.
  https://zamin.uz/en/auto/156672-the-number-of-cars-increased-by-almost-half-a-million-in-one-year.html
- anhor.uz: the Senate approved fines for hiding odometer changes (Article 178-2). https://anhor.uz/news/senat-uzbekistana/
- kun.uz (IMRS): used Cobalt about $12,000 and Onix about $12,170–13,500 (2025).
  https://kun.uz/en/news/2025/03/18/uzbekistans-secondary-car-market-sees-price-drops
- carVertical transparency index (global, 25 countries, not Uzbekistan): 43.8% of checked cars have damage history;
  odometer fraud can raise a price by up to 25%. https://www.carvertical.com/gb/transparency-index/metrics
- my.gov.uz: official vehicle history check, 37,080 UZS. https://my.gov.uz/en/news/b5abc9ee-0339-4642-b054-d5b05d0c1983
