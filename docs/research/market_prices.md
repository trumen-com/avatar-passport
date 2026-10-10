# Market prices: Tashkent body work, competitor inspection, dealer context

Status: desk research only, collected 2026-10-10. No phone calls have been made yet.

Every price below was **reported** by the source named in its row. USD prices are converted with the CBU rate in section 1, and any figure we derived says so. The evidence comes from `phase0_findings_deploy_market.md` sections 6 and 7. Listing dates come from the OLX offers API output captured in the Phase 0 transcripts; the cenotavr.uz page text and the two BYD Telegram posts were checked in the same transcripts. The CBU rate was spot-checked again on 2026-10-10.

## 1. Exchange rate used

- **Rate:** 1 USD = **11,897.90 UZS**, the official Central Bank of Uzbekistan rate dated 10.10.2026.
  - Source: https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/2026-10-10/
  - Retrieved 2026-10-10, with the day's change listed as +51.33.
- **Conversion:** price_uzs = price_usd x 11,897.90. In the tables below the result is rounded to the nearest 1,000 UZS. For example, $55 gives 654,384.5, shown as 654,000.
- **In the CSV:** `repair_prices.csv` keeps the original price, currency and rate, so the conversion can be recomputed (section 6).

## 2. Body-shop prices seen (public web listings, Tashkent)

| # | service (as listed) | price as listed | UZS | kind | source | listing dates | retrieved |
|---|---|---|---|---|---|---|---|
| 1 | painting (Покраска) | from 295,000 сўм | 295,000 | from | [auto.ustabor.uz body-work prices][ustabor] | page undated | 2026-10-10 |
| 2 | dent straightening (Рихтовка вмятин) | from 500,000 сўм | 500,000 | from | [ustabor][ustabor] | page undated | 2026-10-10 |
| 3 | scratch removal (Удаление царапин) | from 325,000 сўм | 325,000 | from | [ustabor][ustabor] | page undated | 2026-10-10 |
| 4 | body part replacement (Замена детали кузова) | from 1,200,000 сўм | 1,200,000 | from | [ustabor][ustabor] | page undated | 2026-10-10 |
| 5 | polishing (Полировка) | from 1,004,167 сўм | 1,004,167 | from | [ustabor][ustabor] (odd figure, copied exactly as printed) | page undated | 2026-10-10 |
| 6 | paint one panel (Покраска 1 детали) | from $55 | 654,000 | from | [olx ID1YwdF][olx-1] | created 2020-01-26, refreshed 2026-09-25 | 2026-10-10 |
| 7 | paintless dent repair (Вакум без покраски) | from $88 | 1,047,000 | from | [olx ID1YwdF][olx-1] | created 2020-01-26, refreshed 2026-09-25 | 2026-10-10 |
| 8 | paint one panel (Покрас 1 детали) | from $77 | 916,000 | from | [olx ID22hsA][olx-2] | created 2020-03-27, refreshed 2026-10-03 | 2026-10-10 |
| 9 | body polishing, any car size | 600,000 сум | 600,000 | flat (advertised as "акция", i.e. a promo) | [olx ID4bVtP][olx-3] | created 2025-10-22, refreshed 2026-10-06 | 2026-10-10 |
| 10 | stone-chip dent repair without paint | from 100,000 сум | 100,000 | from | [olx ID3goDG][olx-4] | created 2023-05-20, refreshed 2026-10-10 | 2026-10-10 |
| 11 | body repair / paint listing (Nexia, Lacetti, Matiz, foreign cars) | "Цена: 200 000 сум" | 200,000 | unclear: one price on an advert that lists body repair, scratch removal, bumper repair, polishing and painting, without saying which service it is for | [cenotavr.uz listing No. 90][cenotavr] | page shows "16 октября" with no year, so no later than 2025-10-16 | 2026-10-10 |
| 12 | dealer polishing promo, any BYD | 2,000,000 | 2,000,000 | promo, valid until 1 March 2025 (ended) | [t.me/byd_chigatoy][byd-tg] | post 2025-02-15 | 2026-10-10 |
| 13 | wheel alignment promo, BYD (normally 200,000) | 100,000 | 100,000 | promo, valid until 30 April 2026 (ended) | [t.me/byd_chigatoy][byd-tg] | post 2026-04-08 | 2026-10-10 |

Caveats on these prices:
- **Lower bounds, not quotes.** Almost every price is a "from" minimum. None states the car class, the panel, the paint type or the damage size.
- **ustabor is a marketplace summary page.** It has no date and no car class. It also lists anti-corrosion treatment from 1,000,000 and "other" from 106,000, which we do not use.
- **Rows 6 to 8 are probably one business.** Both OLX listings use near-identical text and the same business name ("Автосервис Детельниг"), though they give different locations. Treat them as one source, not two data points.
- **Rows 12 and 13 are time-limited dealer promotions**, not price lists, and both have ended.
- **Row 11 says nothing about what 200,000 UZS buys.** It is kept as context only (`service_category=unspecified` in the CSV).
- **Detailing is mixed in.**
  - Rows 1 to 4, 6 to 8 and 10 are repair actions: paint, dent repair, replace, small dents.
  - Rows 9 and 12 are full-body polishing, a detailing service. Row 5 does not say what is polished; it is probably the same service. None of them is the per-panel "polish" action in CLAUDE.md section 1.

## 3. Competitor price: manual pre-purchase inspection

| service | price | kind | source | listing dates | retrieved |
|---|---|---|---|---|---|
| mobile inspection: body/paint check, mileage, computer diagnostics, suspension, engine ("Выездной осмотр ... Цена от 300.000 сум") | 300,000 UZS | from | [olx ID3Vjj4][olx-5] | created 2025-01-30, refreshed 2026-10-04 | 2026-10-10 |

This is the closest public price we found for a manual condition check, the job AvtoPassport partly automates. It is a single listing, not a market average.

## 4. Dealer context

**Toyota**
- **Distributor:** Toyota Central Asia (listed as ИП ООО "Toyota Central Asia"), not Royal Motors ([Uzbekistan site][toyota-uz]).
- **Dealer list:** the [official list][toyota-dealers] has two Tashkent entries of interest:
  - **"Toyota Royal Motors":** "Официальный дилер Toyota в Ташкенте: продажа новых автомобилей и сервис по стандартам бренда".
  - **"Royal Motors Qushbegi":** a new dealer centre with a showroom and a service zone.
  - Other Uzbekistan entries: Toyota A-Auto, Toyota Otto Motors (Samarkand) and Uzinavto Service Center.
- **Telegram:** the channel [t.me/toyota_rm][toyota-tg] (phone 1853) refers to "Toyota RM Tashkent City" and "Toyota RM Qushbegi".
- **Line-up:** Camry and Camry HEV, RAV4 HEV, Land Cruiser Prado (250), Land Cruiser 300 and 300 HEV ([Uzbekistan site][toyota-uz], [t.me/toyota_rm][toyota-tg]).
- **Body work:** none of the dealer entries mentions body or paint repair. On Yandex Maps, the Tashkent City showroom lists polishing and film, but not body or paint repair.

**BYD**
- **Royal Motors group:** [byd-rm.uz/about][byd-rm] says "BYD Чигатой … Входим в группу компаний Royal Motors".
  - Address: Обводная Нурафшон 7. Phone: +998 78 140 1111.
  - The same site has a news item about "BYD Royal Motors" car-loan terms.
- **Locations:** [t.me/byd_chigatoy][byd-tg] lists Tashkent, Namangan, Kokand and Samarkand.
- **Models on byd-rm.uz:** Chazor Champion, e2, Han, Song Plus DM-i, Song Plus EV, Song Pro DM-i Champion, Seagull, Sealion and Yuan Up.
- **List prices on [bydauto.uz][bydauto]** (the BYD Uzbekistan site; footer "© 2022 BYD Europe B.V."), retrieved 2026-10-10:

  | model | price (UZS) |
  |---|---|
  | Seagull | 208,300,000 |
  | e2 | 236,600,000 |
  | Chazor | 255,900,000 |
  | Yuan Up | 266,700,000 |
  | Seal 6 | 364,400,000 |
  | Song Plus DM-i | 374,100,000 |
  | Sealion 7 | 451,100,000 |
  | Ti 7 AWD 5 Seat Luxe | 513,500,000 |
  | Atto 8 | 529,800,000 |

**What this means for us**
- **Customer:** the partner's after-sales unit runs service reception. Neither dealer publishes a body-shop price list, and it is not public whether they run their own body shop. Ask the partner.
- **Car classes:** BYD list prices alone run from 208.3M to 529.8M UZS (table above). A single "any" repair price will therefore be wrong for many cars, so the phone survey collects prices by car class (section 5).

## 5. Gaps and the phone survey

**Gaps**
- **No per-panel prices by car class.** Neither dealer publishes body-shop prices, and no shop gives per-panel prices by class.
- **Class prices seen only in a snippet.** Polishing prices by car class from car-driver.uz (750k to 2.5M) appeared only in a search snippet, so they are not used.
- **No prices for new OEM parts**, which "replace" actions need.
- **Web prices are minimums.** Real jobs depend on damage size, paint type (metallic, pearl) and the number of panels.

**Phone survey plan.** The Run 1 plan has the team calling from 09:00 on 2026-10-11. CLAUDE.md section 5 requires repair prices "collected by phone from Tashkent body shops (source + date per row)".
- **Who to call:** target 5 to 10 Tashkent body shops (our target, not a result), found through OSM name matches, 2GIS or ustabor. Also the Royal Motors service desks: Toyota RM Tashkent City, Royal Motors Qushbegi and BYD Chigatoy.
- **Proposed car classes** (the main agent or CSV owner fixes the final names):
  - small: BYD Seagull, e2
  - sedan: BYD Chazor, Toyota Camry
  - crossover/SUV: BYD Song Plus, Sealion 7, Toyota RAV4
  - large SUV: Land Cruiser Prado, Land Cruiser 300
- **Minimum questions per shop**, for the sedan and crossover/SUV classes, to keep calls short:
  1. Paint one panel (покраска одной детали): bumper, door, fender, hood.
  2. Dent repair plus paint (рихтовка + покраска): door, fender.
  3. Paintless dent removal, small dent (удаление вмятин без покраски).
  4. Polish one panel (полировка детали) and full-body polish.
  5. Replace a bumper (замена бампера): labour only. Ask separately whether the part price is included.
  6. For every answer: is it a "from" price or a fixed price, are paint and materials included, and how many days does the job take?
- **Call log template.** Fill one row per answer. Record only business names; do not record staff names or personal phone numbers.

  | shop (business name) | public listing URL | call date | car class | part | action | price as said | currency | from / flat / range | paint and materials included? | caller initials |
  |---|---|---|---|---|---|---|---|---|---|---|
  | | | 2026-10-11 | | | | | | | | |

- **Converting to CSV:** add each log row as a `PriceRow` in `scripts/build_repair_prices.py` (the template is in its docstring) and re-run the script. Phone rows use `source_type=phone`, `source_url` = `phone: <shop business name>, <call date>` (the script enforces this form; the shop's public listing URL stays in the call log), `listing_date` and `retrieved_on` = the call date, and `is_estimate=false`. A value we estimate ourselves, for example by interpolating between classes, gets `is_estimate=true` (CLAUDE.md section 5).

## 6. How `data/reference/repair_prices.csv` is structured

The CSV is generated by `scripts/build_repair_prices.py` (run it after any change; `--check` fails if the CSV is stale). The script's docstring is the authoritative column definition; this is a summary. The 14 rows of 2026-10-10 are the web listings and Telegram posts of sections 2 and 3, and phone rows follow.

| column | meaning |
|---|---|
| `service` | what the source offered, in plain English, e.g. "paint one panel" |
| `service_category` | `paint`, `dent`, `polish`, `scratch`, `panel_replacement`, `inspection`, `promo` or `unspecified` |
| `part` | `any` (the source does not say), `panel` (one body panel), `whole_body` or `wheels` |
| `car_class` | as the source states it, or `any` when it names none |
| `price_original` | the price as listed, in the listed currency |
| `currency` | `UZS` or `USD` |
| `fx_rate_uzs`, `fx_date`, `fx_source` | the conversion used, only for USD rows: 11897.90, 2026-10-10, the dated cbu.uz JSON URL; empty for UZS rows |
| `price_uzs` | integer UZS: UZS rows unchanged; USD rows `price_original` x `fx_rate_uzs`, rounded to the nearest 1,000 |
| `price_kind` | `from` (a lower bound), `flat` (one price without "from") or `promo` (time-limited dealer offer) |
| `source_url` | the listing URL, or `phone: <shop business name>, <call date>` for phone rows |
| `source_type` | `web_listing`, `telegram_post`, `phone` or `dealer` |
| `listing_date` | OLX: the listing's last refresh date (creation dates are in section 2); Telegram: the post date; phone: the call date; empty when the source shows no full date |
| `retrieved_on` | when we saw the listing or made the call |
| `is_estimate` | `true` only for our own estimates (CLAUDE.md section 5); `false` on every row so far |

Proposal for `cost.estimate` (an **assumption** until the estimator is written and reviewed):
- **Exclude** `inspection`, `promo` and `unspecified` rows: they are context, not repair costs.
- **Map to the brief's actions** (CLAUDE.md section 1: polish / paint / repair / replace): `polish` -> polish; `paint` -> paint; `dent` and `scratch` -> repair; `panel_replacement` -> replace (labour only; part prices are a gap, section 5). The web `polish` rows are full-body (or unstated) detailing prices, so they should not price a one-panel polish; that needs phone quotes.
- **Choosing a row:** prefer phone rows that match the action, `part` and `car_class`. Fall back to `any` rows.
- **What the UI shows:** the price kind (for example "from 295,000 UZS"), the source and the date, never a bare exact-looking number.

[ustabor]: https://auto.ustabor.uz/ru/price/body_work
[olx-1]: https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-i-t-d-ID1YwdF.html
[olx-2]: https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-shumo-tsiya-kostoprav-ID22hsA.html
[olx-3]: https://www.olx.uz/d/obyavlenie/polirovka-avto-kuzova-za-lyuboy-mashina-600000-sum-600-tsyach-sum-ID4bVtP.html
[olx-4]: https://www.olx.uz/d/obyavlenie/avto-vakuum-pachok-tugirlash-vakum-ID3goDG.html
[olx-5]: https://www.olx.uz/d/obyavlenie/proverka-avto-po-kuzovu-i-probegu-po-nizkim-tsenam-vyezdnoy-osmotr-ID3Vjj4.html
[cenotavr]: https://www.cenotavr.uz/tashkent/remont-kuzova-pokraska-polirovka-kuzova-90
[byd-tg]: https://t.me/byd_chigatoy
[byd-rm]: https://byd-rm.uz/about
[bydauto]: https://bydauto.uz
[toyota-uz]: https://uzbekistan.toyota-centralasia.com/ru/
[toyota-dealers]: https://www.toyota-centralasia.com/ru/dealers
[toyota-tg]: https://t.me/toyota_rm
