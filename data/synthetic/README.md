# Synthetic sandbox data (SYNTHETIC DEMO DATA)

Everything in this folder is **generated, not collected**. It stands in for data that in reality lives in police,
insurer and dealer systems we have no access to (CLAUDE.md sections 2 and 5).

| File | Stands in for | Presented as |
|---|---|---|
| `registry.json` | vehicle registry history: owners, accidents, insurance claims, odometer readings | **registry connector - sandbox** |
| `customers.json` | dealer CRM: customers, their vehicles, service visits | **dealer CRM - sandbox** |

Regenerate (stdlib only, no network, byte-identical output on every run):

```bash
python scripts/generate_synthetic.py
```

The script also rewrites the two tables at the end of this file (between the `BEGIN`/`END` markers), so every
parameter and count shown there comes from code. `tests/test_synthetic.py` fails if the committed files differ
from a fresh run.

## Rules for anything that uses this data

1. **Badge.** Every screen, PDF, API response or slide that shows anything derived from these files carries a
   visible **SYNTHETIC DEMO DATA** badge. Each file and each record carries `"synthetic": true` so code can tell.
2. **Presentation.** Registry data is shown as "registry connector - sandbox" and CRM data as
   "dealer CRM - sandbox" (the `source` field of every record), never as real history and never as police,
   insurer or dealer data.
3. **Refusal.** Synthetic history is never attached to a VIN read from a real car or photo.
   `avtopassport.synthetic.lookup(vin, origin="scan")` always returns
   `RegistryRefusal(vin, "synthetic registry is never attached to a scanned VIN")`, even when the scanned string
   equals a sandbox VIN, and it does not read the registry at all. Only `origin="sandbox"` (a VIN picked from the
   sandbox list) returns a record, or `None` if there is none; any other origin raises `ValueError`.
4. **Loading.** `avtopassport.synthetic.load_registry()` raises `ValueError` if the file or any record is not
   flagged `"synthetic": true`, if a record's VIN is not a synthetic VIN, or if a VIN repeats.

## Method

- **Seed and reproducibility.** The seed is the fixed `SEED` in `src/avtopassport/synthetic.py` (also shown in the
  summary table). Every draw is built on `random.Random.random()` alone, the one method whose sequence Python
  promises to keep for a given seed across versions
  (https://docs.python.org/3/library/random.html#notes-on-reproducibility). JSON is written with sorted keys,
  indent 2, UTF-8 and `\n` line endings, and contains no timestamps.
- **Vehicles.** Make by `make_share`; model uniform from that make's dealer line-up; model year uniform from
  `max(range start, first_model_year[model])` to the range end; colour by `colour_share`. First registration is a
  uniform day of the model year.
- **Owners.** Consecutive holding periods drawn from `ownership_years`, starting at first registration. The next
  owner's `from` equals the previous owner's `to`; the last owner is current (`to: null`). Owners are IDs only
  (`SYN-OWNER-0001`, ...).
- **Accidents.** One Bernoulli trial per `event_window_days` window since first registration, with
  p = `accident_rate_per_year` x window / 365.25. Severity, impact zone and number of parts come from the tables;
  parts are drawn without repetition from the zone's list; the description is generated from those fields.
- **Insurance claims.** Filed for an accident with `claim_probability[severity]`, `claim_delay_days` after it,
  with one of three fictitious insurers. `amount_uzs` is the claimed amount: one draw per damaged part from
  `claim_amount_per_part_uzs[severity]`, summed and rounded. A claim can be `open` only if it was filed within
  `claim_open_window_days` of the window end; otherwise it is `rejected` with `claim_rejected_probability`, else
  `paid`. Each claim carries the `accident_id` it belongs to.
- **Mileage.** A first reading at first registration, then one every `mileage_interval_days`; the distance added
  is `annual_km` x interval / 365.25 x `annual_km_variation`. Readings never decrease.
- **Customers.** Registry VINs are shuffled and dealt out, one or two per customer
  (`two_vehicle_customer_probability`), each VIN to at most one customer, so some registry vehicles have no CRM
  customer. `district` is a coarse area: a Tashkent city district or "Tashkent region (synthetic)". Service visits
  are generated per vehicle from its current owner's `from` date (the CRM customer is taken to be the current
  owner): one Bernoulli trial per window with `service_visits_per_year`, type by `visit_type_share` (warranty only
  within `warranty_years` of first registration), amount from `visit_amount_uzs`. Each visit carries the `vin` it
  belongs to.

## VIN scheme

| Position | Content |
|---|---|
| 1-3 | `0AP` (`SYNTHETIC_WMI`; "AP" for AvtoPassport) |
| 4-8 | 5 random characters from the VIN alphabet (no I, O, Q) |
| 9 | check digit from `avtopassport.vin.checkdigit.compute_check_digit` (ISO 3779 / 49 CFR 565.15 weights) |
| 10 | model-year code (2016 = G ... 2025 = S) |
| 11 | random plant character from the VIN alphabet |
| 12-17 | random 6-digit serial |

Why a leading `0`: the ISO 3780 WMI country chart (revised 2021-04-13) marks row "0" as spare codes with no
country assigned (https://standards.iso.org/iso/3780/ed-4/en/Current%20WMI%20World%20Codes%20chart%204.13.21.pdf;
ISO copyright, cited and not redistributed). So no assigned real VIN starts with `0` today. "Spare" is not reserved
forever, so every record is also flagged `"synthetic": true`, and the code checks both (`is_synthetic_vin` and the
flag). VINs are unique, and a VIN that contains `998` (Uzbekistan's calling code) is redrawn so phone-number
scanners never flag the sandbox. Background: `docs/research/phase0_findings_vin.md` section 2.

## Privacy

- No licence plates of any format, no phone numbers, no street addresses, no person names (customers are
  "Synthetic Customer NNN", owners are IDs) and no real insurer names ("Sandbox Insurer A/B/C").
- Emails use the reserved `.invalid` top-level domain only (`custNNN@example.invalid`). RFC 2606 section 2 reserves
  `.invalid` for names that are "sure to be invalid" (https://www.rfc-editor.org/rfc/rfc2606.html), so no mail can
  ever be delivered.
- All dates fall inside `date_window`. All UZS amounts are synthetic assumptions, not prices; real repair prices
  with their sources belong in `data/reference/repair_prices.csv`.
- `tests/test_synthetic.py` scans every string in both files for Uzbek plate patterns, `998` / `+` phone patterns,
  runs of 9 or more digits, address words and e-mail addresses outside `example.invalid`.

## Sources

- **Make and model line-ups:** `docs/research/phase0_findings_deploy_market.md` section 6. Toyota: models on sale at
  https://uzbekistan.toyota-centralasia.com/ru/ (dealer channel: https://t.me/toyota_rm). BYD: models listed at
  https://bydauto.uz. Only the model names are used; no prices.
- **Part names:** the parts catalog of the parts model, mitbersh/car-parts-segmentation
  (https://huggingface.co/mitbersh/car-parts-segmentation/blob/05e58a396aefbe46d86ab9d17bbc945817163577/parts_catalog.json),
  written in snake_case with a `left_` / `right_` prefix for paired parts (`fender` is the front wing,
  `quarter_panel` the rear wing, `back` means rear). Sandbox history therefore uses the parts model's vocabulary.
  Wheels, side windows, roof and the licence-plate class are not used.
- **District names:** the district table of https://en.wikipedia.org/w/index.php?title=Tashkent&oldid=1376716206
  (CC BY-SA 4.0).
- **VIN rules:** `docs/research/phase0_findings_vin.md`.

## Limitations

- Every distribution below is an assumption, not calibrated to Uzbek registry, insurer or dealer statistics. The
  sandbox exists to exercise the UI, the badge and the refusal rule. Do not quote any statistic from it.
- `first_model_year` values are approximate market introductions, chosen only to avoid obvious anachronisms; they
  are not verified against manufacturer data.
- The registry, the CRM and real AvtoPassport inspections are independent: CRM body-repair visits are not tied to
  registry accidents, and nothing in the sandbox is ever linked to a real inspection.

## Assumptions (rendered by `scripts/generate_synthetic.py`)

<!-- BEGIN:assumptions -->
| Parameter | Value | Basis | Meaning |
|---|---|---|---|
| `n_vehicles` | 40 | assumption | registry records, one synthetic VIN each |
| `n_customers` | 25 | assumption | dealer-CRM customer records |
| `date_window` | 2016-01-01 to 2026-09-30 | assumption | every generated date falls inside this window |
| `model_years` | 2016 to 2025 | assumption | model-year range (VIN position 10 codes G to S) |
| `make_share` | Toyota: 0.5; BYD: 0.5 | assumption | share of vehicles per make; the model is then drawn uniformly from that make's line-up |
| `lineup_toyota` | Camry, RAV4, Land Cruiser Prado, Land Cruiser 300 | https://uzbekistan.toyota-centralasia.com/ru/ | models on sale (docs/research/phase0_findings_deploy_market.md section 6) |
| `lineup_byd` | Chazor, Song Plus DM-i, Seal 6, Sealion 7, Yuan Up, e2, Seagull | https://bydauto.uz | models listed with UZS prices (docs/research/phase0_findings_deploy_market.md section 6) |
| `first_model_year` | Song Plus DM-i: 2021; e2: 2020; Seagull: 2023; Land Cruiser 300: 2022; Chazor: 2024; Yuan Up: 2024; Seal 6: 2025; Sealion 7: 2025 | assumption | earliest model year allowed per model, so the sandbox avoids obvious anachronisms; approximate, not verified against manufacturer data; other models start at the range start |
| `colour_share` | white: 0.3; black: 0.2; silver: 0.15; grey: 0.15; blue: 0.08; red: 0.05; green: 0.04; beige: 0.03 | assumption | generic colour names; not market data |
| `ownership_years` | 1.5 to 7 | assumption | holding period per owner, uniform; the last owner is the current one (to = null) |
| `event_window_days` | 30 | assumption | accidents and service visits: one Bernoulli trial per window, p = annual rate x window / 365.25 |
| `accident_rate_per_year` | 0.12 | assumption | accidents per vehicle-year since first registration |
| `severity_share` | minor: 0.6; moderate: 0.3; severe: 0.1 | assumption | accident severity |
| `impact_zone_share` | front: 0.4; rear: 0.3; left: 0.15; right: 0.15 | assumption | side of the car that takes the impact |
| `impact_zone_parts` | front: front_bumper, grille, hood, windshield, left_headlight, right_headlight, left_fender, right_fender; rear: back_bumper, trunk, back_windshield, left_tail_light, right_tail_light, left_quarter_panel, right_quarter_panel; left: left_front_door, left_back_door, left_fender, left_quarter_panel, left_mirror, left_rocker_panel; right: right_front_door, right_back_door, right_fender, right_quarter_panel, right_mirror, right_rocker_panel | assumption | parts an impact in each zone can damage; names follow the parts-model catalog (README, Sources) |
| `parts_per_accident` | minor: 1; moderate: 1 to 2; severe: 2 to 4 | assumption | damaged parts per accident, uniform, drawn without repetition from the zone |
| `claim_probability` | minor: 0.35; moderate: 0.75; severe: 0.95 | assumption | P(insurance claim) by accident severity |
| `claim_delay_days` | 1 to 30 | assumption | days from accident to claim, uniform |
| `claim_amount_per_part_uzs` | minor: 300,000 to 1,500,000; moderate: 1,000,000 to 5,000,000; severe: 4,000,000 to 20,000,000 | assumption | claimed amount = sum of one uniform draw per damaged part; synthetic, not price data |
| `insurers` | Sandbox Insurer A, Sandbox Insurer B, Sandbox Insurer C | assumption | fictitious insurer labels, drawn uniformly |
| `claim_open_window_days` | 120 | assumption | only claims filed this many days or fewer before the window end can be open |
| `claim_open_probability` | 0.6 | assumption | P(open) for a claim inside the open window |
| `claim_rejected_probability` | 0.1 | assumption | P(rejected) for a closed claim; otherwise paid |
| `odometer_at_registration_km` | 5 to 50 | assumption | first reading, at first registration |
| `annual_km` | 8,000 to 25,000 | assumption | annual distance per vehicle, uniform |
| `annual_km_variation` | 0.8 to 1.2 | assumption | multiplier on the annual distance per interval |
| `mileage_interval_days` | 300 to 430 | assumption | days between odometer readings, uniform |
| `two_vehicle_customer_probability` | 0.2 | assumption | P(customer has two registry vehicles); each VIN goes to at most one customer |
| `tashkent_region_probability` | 0.2 | assumption | P(district = 'Tashkent region (synthetic)'); otherwise a city district, uniform |
| `tashkent_districts` | Bektemir, Chilanzar, Yashnobod, Mirobod, Mirzo Ulugbek, Sergeli, Shayxontoxur, Olmazor, Uchtepa, Yakkasaray, Yunusabad, Yangihayot | https://en.wikipedia.org/w/index.php?title=Tashkent&oldid=1376716206 | the 12 Tashkent city districts as spelled in the district table; a coarse area, never an address |
| `service_visits_per_year` | 1.5 | assumption | dealer visits per vehicle-year since the current owner's 'from' date |
| `visit_type_share` | maintenance: 0.6; body_repair: 0.12; warranty: 0.13; accessories: 0.15 | assumption | visit type; warranty is dropped (and the rest renormalised) after the warranty period |
| `warranty_years` | 3 | assumption | warranty visits only within this many years of first registration |
| `visit_amount_uzs` | maintenance: 600,000 to 3,000,000; body_repair: 1,000,000 to 15,000,000; warranty: 0; accessories: 300,000 to 8,000,000 | assumption | amount billed to the customer, uniform; warranty work is billed at 0; synthetic, not price data |
| `amount_rounding_uzs` | 10,000 | assumption | every UZS amount is rounded to this step |
<!-- END:assumptions -->

## Generated data summary (computed by `scripts/generate_synthetic.py`)

<!-- BEGIN:summary -->
| Item | Value |
|---|---|
| Seed | 20261010 |
| Registry vehicles | 40 (BYD: 15, Toyota: 25) |
| Models | BYD Seagull: 2, BYD Sealion 7: 5, BYD Song Plus DM-i: 6, BYD e2: 2, Toyota Camry: 3, Toyota Land Cruiser 300: 4, Toyota Land Cruiser Prado: 10, Toyota RAV4: 8 |
| Model years | 2016 to 2025 |
| Owners | 62; vehicles by owner count: 1 owner: 23, 2 owners: 12, 3 owners: 5 |
| Accidents | 15 (minor: 8, moderate: 4, severe: 3) on 11 vehicles |
| Insurance claims | 7 (open: 1, paid: 6) |
| Mileage readings | 187 |
| CRM customers | 25 (Tashkent city: 20, Tashkent region: 5) |
| Registry VINs linked to a customer | 30 of 40 |
| Service visits | 108 (accessories: 13, body_repair: 13, maintenance: 75, warranty: 7) |
<!-- END:summary -->
