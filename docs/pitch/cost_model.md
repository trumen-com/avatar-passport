# Cost model (stub)

- **Status:** stub, 2026-10-10.
- **Nothing has been purchased.**
- **Number rules:** every number is either sourced (with a URL and date), derived from sourced numbers (formula shown), or marked as an **assumption**. Measured values are written only by scripts.
- **Paid services:** per CLAUDE.md section 9, every paid service or API call is listed here, and none is added without asking the user first.

## 1. Hosting

| item | cost | source | status |
|---|---|---|---|
| Hugging Face PRO, needed to create a Docker Space | **$9 / month** | [HF pricing][hf-pricing], retrieved 2026-10-10. The requirement is in the [Spaces overview][hf-overview] and was added by [hub-docs PR #2624][hf-2624], merged 2026-07-21. | **not purchased - pending local test** (Run 2 decision point, ADR 003) |
| the same, in UZS | 107,081 UZS / month | derived: 9 x 11,897.90 = 107,081.1, rounded (CBU rate of 2026-10-10, [cbu.uz][cbu]) | depends on the purchase |
| CPU Basic hardware (2 vCPU, 16 GB RAM, 50 GB non-persistent disk) | $0 / hour | [Spaces overview][hf-overview] | used once the Space exists |
| CPU Upgrade (8 vCPU, 32 GB), only if CPU Basic is too slow | $0.03 / hour | [HF pricing][hf-pricing] | not planned; ask first |
| SQLite snapshots and blurred images in a private HF dataset repo | $0 (**assumption**: the data stays within the included private storage) | Free accounts get 100 GB of private storage (`docs/research/phase0_findings_deploy_market.md` section 1). | check once the storage per inspection is measured |

## 2. Cost per inspection

- **Marginal compute cost per inspection: $0** on CPU Basic. This is derived from two facts:
  - The hardware has no hourly charge ([Spaces overview][hf-overview]).
  - The pipeline calls no paid API.
- **Paid APIs: none.** The external services the app uses are free but rate-limited:
  - NHTSA vPIC (VIN decoding; "free for use by the public").
  - OpenStreetMap Overpass and Nominatim (usage policies apply).
  - freetsa.org RFC 3161 timestamps (no SLA).
  - Sources: `docs/research/phase0_findings_vin.md` section 3 and `docs/research/phase0_findings_deploy_market.md` sections 5 and 8.
- **Not used:** Roboflow hosted inference (paid credits), so it adds no cost. See `docs/research/model_candidates.md`.
- **Fixed cost per inspection** = $9 / inspections per month. This is a formula; its input is in section 4.

## 3. Inspection time on deploy hardware

<!-- BEGIN:measured_inspection_time -->

| metric | value | source |
|---|---|---|
| full inspection time on the deploy hardware (HF CPU Basic) | not measured yet | Phase 4 measurement on the Space |
| local proxy: laptop, ONNX Runtime CPU pinned to 2 threads (tier `cpu_ort2`) | not measured yet | Run 1 smoke test (`outputs/smoke/<run id>/latency.csv`, rendered into `docs/research/smoke_test_results.md`) |

<!-- END:measured_inspection_time -->

Capacity per Space, as a formula with no number until measured: inspections per hour = 3600 / (measured inspection time in seconds), processing one inspection at a time.

The brief's figure of about 10 to 20 s per inspection is a **target** (CLAUDE.md section 1), not a result.

## 4. Inputs still needed

| input | status | who / where |
|---|---|---|
| measured full inspection time on the deployed Space | not measured | Phase 4, written by script |
| inspections per month | **assumption needed**; ask the partner for monthly trade-in and service-intake volume | team and Royal Motors after-sales |
| pilot size: service points, advisors, months | **assumption needed** | team |
| storage per inspection (blurred frames, PDF, database row) | not measured | Phase 4 |
| HF PRO purchase | pending the local end-to-end test | user, Run 2 decision point |
| optional paid "AI preview" image-editing API (CLAUDE.md section 7) | not planned; ask before adding | user |

## 5. Reference prices (context, not our costs)

- **Competitor price:** a manual mobile pre-purchase inspection is advertised from 300,000 UZS (reported: one [OLX listing](https://www.olx.uz/d/obyavlenie/proverka-avto-po-kuzovu-i-probegu-po-nizkim-tsenam-vyezdnoy-osmotr-ID3Vjj4.html), refreshed 2026-10-04, retrieved 2026-10-10; `docs/research/market_prices.md` section 3). One listing, not a market average.
- **Body-shop prices:** "from" prices for body work are listed in `docs/research/market_prices.md`.
- **Where they belong:** these feed `roi_model.md`, not this file.

[hf-pricing]: https://huggingface.co/pricing
[hf-overview]: https://huggingface.co/docs/hub/spaces-overview
[hf-2624]: https://github.com/huggingface/hub-docs/pull/2624
[cbu]: https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/2026-10-10/
