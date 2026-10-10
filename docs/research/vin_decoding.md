# VIN decoding: structure, validation, vPIC and our WMI table

> **Licence.** Section 8 summarises our WMI table, which is derived from Wikipedia and Wikibooks (CC BY-SA 4.0; the revisions are listed in 8.5). Section 8 is therefore licensed CC BY-SA 4.0, like `data/reference/wmi_table.csv` and `docs/research/phase0_findings_vin.md`; attribution is in `data/reference/LICENSE-wmi_table.md`. The rest of this file is under the repository licence (AGPL-3.0).

Written on 2026-10-10. The sources are the Phase 0 research (`docs/research/phase0_findings_vin.md`) and six NHTSA vPIC look-ups made on 2026-10-10 (section 6.4).

How to read the numbers:
- Facts from laws, standards and websites are **reported**, and each one names its source (full links in section 10).
- Values produced by our code are marked **computed**.
- Nothing here measures our system.

Real VINs appear only with positions 12-17 masked as `******`.

## 1. VIN structure

A VIN has 17 characters (49 CFR 565.13(b); Uzbekistan Resolution 237, VIN clause 2).

| Positions | ISO 3779 section | US rule (49 CFR 565.15) | Uzbek rule (Resolution 237) |
|---|---|---|---|
| 1-3 | WMI (world manufacturer identifier) | manufacturer identifier | WMI. Character 1 = zone, 1-2 = country (assigned under ISO rules), 3 = manufacturer (assigned by the national authority) (clause 4) |
| 4-9 | VDS (vehicle descriptor section) | 4-8 vehicle attributes, 9 check digit | VDS: six characters chosen by the manufacturer (clause 5) |
| 10-17 | VIS (vehicle indicator section) | 10 model year, 11 plant, 12-17 serial number | VIS: eight characters. The last four must be digits and unused positions are zeros (clause 7) |

- **Small manufacturers:** a `9` in position 3 can mean the manufacturer code continues in positions 12-14 (49 CFR 565.15(a), low-volume manufacturers; Resolution 237 clause 4, makers of fewer than 500 vehicles a year). Our WMI lookup reads positions 1-3 only.
- **Marking in Uzbekistan:** the VIN may be stamped on one line or on two (Resolution 237 clause 18).
  - On two lines, characters 1-9 go on the first line and 10-17 on the second. A separator chosen by the manufacturer (for example `*`) marks the start and end of each line (clause 19).
  - In documents, the VIN is written on one line with no spaces or separators (clause 12).
  - So OCR must join the two lines and drop the separators. `ocr.py` stitches lines and `validate.py` strips `*`.

## 2. Allowed characters

- **Allowed:** the digits 0-9 and the letters A-Z except I, O and Q (49 CFR 565.13(g); Resolution 237 clause 2). In code this is `checkdigit.VIN_RE = ^[A-HJ-NPR-Z0-9]{17}$`.
- **Position 10:** in addition, U, Z and 0 never appear as year codes there (section 4).

## 3. Check digit (position 9)

### 3.1 Algorithm (49 CFR 565.15(c); implemented in `src/avtopassport/vin/checkdigit.py`)

1. Replace each letter with its value. Digits keep their own value.

| A | B | C | D | E | F | G | H | J | K | L | M | N | P | R | S | T | U | V | W | X | Y | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

2. Multiply each value by the weight of its position. Position 9 has weight 0, so the check digit does not count towards its own sum.

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Weight | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 10 | 0 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 |

3. Add the products and divide by 11. A remainder of 0-9 is the check digit; a remainder of 10 is written as `X`.

Phase 0 reports that this algorithm reproduces the check digits of the published samples in 49 CFR 565, the GB 16735 draft and Wikipedia.

**Worked example (computed with `checkdigit.py`).** The VIN is illustrative and synthetic (section 9); it is not taken from the generator's output.

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Character | 0 | A | P | S | Y | N | T | H | ? | T | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Value | 0 | 1 | 7 | 2 | 8 | 5 | 3 | 8 | - | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Product | 0 | 7 | 42 | 10 | 32 | 15 | 6 | 80 | 0 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

The products add up to 221, and 221 mod 11 = 1. The check digit is therefore `1`, and the full VIN is `0APSYNTH1T0000001`.

### 3.2 Where the check digit is mandatory

| Jurisdiction | Status | Source |
|---|---|---|
| United States | Mandatory: "A check digit shall be part of each VIN", in position 9 | 49 CFR 565.13(c) |
| Canada | Mandatory | Motor Vehicle Safety Regulations, Schedule IV, s.115(3)(c), (7) and (8) |
| China | Mandatory. Clause 4.3.3 of the GB 16735 approval draft filed with the WTO: position 9 is a check digit computed as in Annex A, with the same letter values. The foreword makes all technical content mandatory. The final GB 16735-2019 text is paywalled and was not read. | WTO TBT file 19_3960 |
| European Union | Phasing in. Regulation (EU) 2021/535, Art. 12: required for new vehicle types from 2022-07-06, and registration is refused without it from 2026-07-07 (Phase 0 summary). | EUR-Lex 32021R0535 |
| Uzbekistan | No check-digit clause in the VIN section of Cabinet of Ministers Resolution No. 237 (2017-04-25), as far as Phase 0 read it. The whole act was not read. Clause 15: vehicles assembled in Uzbekistan from large kits keep the manufacturer's VIN. | lex.uz act 3180907 |

### 3.3 What this means for our validator

- **When a bad check digit rejects the VIN:** only when the VIN starts with 1-5 (North America), L or H (China), giving status `invalid`. Before rejecting, the validator tries single-character confusion swaps (section 5).
- **Every other first character:** a bad check digit only produces a warning (status `valid_unverified`). Real European, Korean and Russian VINs fail the check (3.4), and we found no Uzbek requirement for it.
- **A passing check digit is not proof:** a wrong VIN still has a 1 in 11 chance of passing (reported, Wikipedia VIN article). Phase 0 therefore recommends that the app always shows the VIN to the user to confirm.

### 3.4 Real VINs that fail the check digit

The full strings are in the cited sources; we mask positions 12-17. The "computed" column was produced with `checkdigit.py` on the full published strings on 2026-10-10.

| VIN (masked) | Maker, country | Position 9 | Computed | Published in |
|---|---|---|---|---|
| `WP0ZZZ99ZTS******` | Porsche, Germany | Z | 8 | en.wikipedia "Vehicle identification number", rev. 1378888660 |
| `KLATF08Y1VB******` | Daewoo / GM Korea, South Korea | 1 | 4 | same article ("GM-T body" image) |
| `VF1LM1B0H36******` | Renault, France | H | 8 | vininfo README (PyPI) |
| `XTAGFK330JY******` | AvtoVAZ (Lada), Russia | 0 | 6 | vininfo README (PyPI) |

The Wikipedia article itself says the VINs in its Porsche and GM-T body images "do not pass the North American check-digit verification".

`validate()` returns `valid_unverified` for all four (computed, 2026-10-10).

We have no Uzbek example yet. If the team's own consented car becomes one, write in the docs only whether the check passed, never the full VIN.

## 4. Model year (position 10) and plant (position 11)

Position 10 codes repeat every 30 years (49 CFR 565.15; Resolution 237 clause 9 gives the same codes for 2001-2040). I, O, Q, U, Z and 0 are never year codes. This matches `validate.YEAR_CODES`.

| Code | A | B | C | D | E | F | G | H | J | K | L | M | N | P | R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cycle 1 | 1980 | 1981 | 1982 | 1983 | 1984 | 1985 | 1986 | 1987 | 1988 | 1989 | 1990 | 1991 | 1992 | 1993 | 1994 |
| Cycle 2 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |

| Code | S | T | V | W | X | Y | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cycle 1 | 1995 | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 |
| Cycle 2 | 2025 | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | 2032 | 2033 | 2034 | 2035 | 2036 | 2037 | 2038 | 2039 |

**Rules as implemented** (`validate.py`):
- `model_year_candidates` returns the candidates from both cycles, within 1980-2039. `validate()` fills them only for `valid` and `valid_unverified` results.
- Where the check digit is mandatory (first character 1-5, L or H), position 10 must be a year code; otherwise the VIN is `invalid` unless one unique swap repairs it.
- For North American VINs (first character 1-5), a letter in position 7 selects the 2010-2039 cycle.
  - Basis: 49 CFR 565.15(b) requires position 7 to be a letter on passenger cars, multipurpose vehicles and trucks up to 4,536 kg GVWR.
  - Phase 0 reads this as "a letter in position 7 means 2010-2039".

**Outside North America:**
- **China:** a year code in position 10 is required (GB 16735 draft, Phase 0 summary).
- **Uzbekistan:** a year code and a plant code in the VIS are permitted, not required. The year should go in the first VIS position and the plant in the second (Resolution 237 clause 8).
- **Elsewhere:** position 10 is often not a year code at all. Phase 0 found it is often just `0`, which is never a year code.
- **Consequence in `decode.py`:**
  - `model_year_candidates` comes from the position-10 rules.
  - `model_year` is set only from vPIC, and only when vPIC's year is one of those candidates.
  - We never guess a single year.

**Position 11:** the assembly-plant code is defined by each manufacturer (49 CFR 565.15(d)(2); GB 16735 clause 4.4.3). We do not decode plant codes ourselves. vPIC's PlantCountry and PlantCity are used when vPIC returns them.

## 5. OCR confusion handling (as implemented)

These are the rules in the `src/avtopassport/vin/validate.py` docstring, in order:

1. **Normalise:** uppercase; strip spaces, `*`, `-`, `.` and `:`; map I->1, O->0 and Q->0, since I, O and Q are never valid VIN letters.
2. **Positions 14-17 must be digits:** map S->5, B->8, G->6, Z->2 and D->0, in those positions only.
3. **WMI swaps:** 1<->L, 0<->D, 5<->S and 8<->B are applied only if the result is a known WMI (`known_wmis`, from `data/reference/wmi_table.csv`).
4. **When the check digit is mandatory:** only when the first character is 1-5 (North America), L or H (China). Anywhere else a failing check digit is only a warning.
5. **Mandatory check digit fails:**
   - Try single-character confusion swaps: 0/O, 1/I, 8/B, 5/S, 2/Z, 6/G.
   - Accept a swap only if exactly one candidate passes.
   - Otherwise the status is `invalid`, with the candidates listed.
6. **Never guess:** anything that is not 17 VIN-alphabet characters after normalisation is `unreadable` or `invalid`.

**Implementation notes** (from the same docstring):
- **Unicode first:** normalisation applies Unicode NFKC before anything else, so full-width OCR output such as `Ｘ` or `：` becomes ASCII.
- **Year code where the check digit is mandatory:** for VINs starting with 1-5, L or H, position 10 must also be a model-year code.
- **A swap may not break the alphabet:** its result must be a VIN letter, a digit or `X` in position 9, a digit in positions 14-17 and, where the check digit is mandatory, a year code in position 10.
- **When a WMI swap is skipped:**
  - When the VIN as read already passes a mandatory check digit. The read is verified; the WMI is just missing from our table.
  - When several swaps give different known WMIs. Those VINs are listed in `candidates` instead.
- **`vin` can be set on an invalid result:** it is set whenever the input normalises to 17 VIN-alphabet characters, even when the status is `invalid`. Always check `status` before using it.

**What the validator returns:**

| Status | When |
|---|---|
| `valid` | The check digit is verified, directly or after one unique confusion swap |
| `valid_unverified` | The check digit fails where it is not mandatory. This is the warning case of 3.3. |
| `invalid` | More than 17 characters; non-digits left in positions 14-17; or a mandatory check digit or model-year code that no unique swap repairs. This is the rejection case of 3.3. |
| `unreadable` | Characters outside the VIN alphabet, or fewer than 17 usable characters |

Every change is recorded in `VinValidation.corrections`, for example "pos 3: O->0".

**Why these rules are safe:**
- I, O and Q can never be VIN characters.
- Positions 14-17 are digits for all vehicles (Canada s.115(3)(h); Resolution 237 clause 7).
- WMI swaps are limited to WMIs we know.
- Check-digit swaps happen only where the check digit is mandatory, and only when the result is unique.

**How `ocr.py` uses the validator:**
- **Engine:** RapidOCR.
- **Candidates:** it builds 17-character windows from the OCR lines, and also from two lines stitched together, which covers two-line plates.
- **Ranking:** a check digit that passes where it is mandatory comes first. Then a known WMI, then a single-token read, then the OCR score.
- **Result:** `read`, `ambiguous` or `could_not_read`.

## 6. NHTSA vPIC

### 6.1 Endpoints (base `https://vpic.nhtsa.dot.gov/api/vehicles/`)

- **`DecodeVinValues/{VIN}?format=json&modelyear=YYYY`:** returns one flat record of 154 fields (reported).
  - Useful fields: Make, Model, ModelYear, Manufacturer, ManufacturerId, VehicleType, BodyClass, PlantCountry, PlantCity, FuelTypePrimary, ElectrificationLevel, ErrorCode, ErrorText, SuggestedVIN and PossibleValues.
  - Partial VINs work, with `*` for each missing character.
  - NHTSA recommends always sending the model year.
- **`DecodeVin/{VIN}`:** the same data as variable/value rows.
- **`DecodeVinValuesExtended/{VIN}`:** adds NHTSA statistics fields.
- **`DecodeWMI/{WMI}?format=json`:** Make, ManufacturerName, VehicleType, CommonName, CreatedOn and UpdatedOn.
- **`GetWMIsForManufacturer/{name}?format=json`:** lists a manufacturer's WMIs.
- **Batch:** `POST DecodeVINValuesBatch/` with the body `format=json&data=VIN,YEAR;VIN`, up to 50 VINs (reported).
- **Offline:** the database dump `vPICList_lite_2026_09` (PostgreSQL 17+, 67.8 MB, reported) from the downloads page. Query it with `SELECT * FROM vpic.spVinDecode('...')`.

### 6.2 ErrorCode

ErrorCode can hold several codes separated by commas, for example `6,7,11`. Reported from the vPIC error-code list.

| Code | Meaning |
|---|---|
| 0 | Clean decode |
| 1 | Bad check digit |
| 2-4 | VIN corrected |
| 5 | Errors in several positions |
| 6 | Incomplete VIN |
| 7 | Manufacturer not registered with NHTSA |
| 8 | No detailed data |
| 11 | Bad model-year code |
| 14 | Some positions not decodable |
| 400 | Invalid characters |

### 6.3 Usage limits and licence

**Limits:** the sources disagree.
- The FAQ says vPIC is free, needs no registration and has no daily limit, and that it handles 1,000-2,000 requests a minute (reported).
- The API page mentions "automated traffic rate control", and the downloads page mentions "quotas and rate limits".
- Phase 0 saw no rate-limit headers.

**How our code copes:**
- `decode.py` caches responses in `outputs/cache/vpic/`.
- It treats vPIC as optional: failures and timeouts are written to `notes` and never raised.

**Licence:** the FAQ says vPIC is "free for use by the public", and data.gov lists it as "unknown-license". Calling it public domain because it is US-government work is an inference; no source states it.

### 6.4 Behaviour for cars sold in Uzbekistan

**Phase 0 `DecodeWMI` results:**
- **Found:** LGX, LC0 and LPE (BYD; LC0 listed as Bus), JTD/JTE/JTM/JTH/JTJ (Toyota/Lexus), NMT, KNA/KND, KMH/KM8, KL1/KL7/KL8/KLA, and L6T (listed as Zeekr's US company).
- **Nothing returned:** XWB, NSA/NS2/NS4/NST, LVV/LVT/LVU/LNN, LGW/XZG, LB3, LS4-LS6, LW4/HLX, Z94, XTA, XW7, MR0, MHF, LFM, LVG and MZB.

**Decoding a bare 3-character prefix:**
- XWB, Z94, XTA, LVV, LGW, NST, NS4 and NSA return ErrorCode `6,7,11` (manufacturer not registered).
- LGX, LC0, JTD and JTM return `6,8,11` (registered, no detailed data).

**Spot-check on 2026-10-10** (`DecodeWMI/{wmi}?format=json`, values as returned):

| WMI | Make | ManufacturerName | VehicleType |
|---|---|---|---|
| LGX | BYD | BYD AUTO CO., LTD. | Passenger Car |
| LC0 | BYD | BYD AUTO INDUSTRY COMPANY LIMITED | Bus |
| JTM | TOYOTA, SUBARU | TOYOTA MOTOR CORPORATION | Multipurpose Passenger Vehicle (MPV) |
| LSG | CADILLAC | GENERAL MOTORS LLC | Passenger Car |
| KNC | (no result) | | |
| XWB | (no result) | | |

**Conclusion:** for cars sold in Tashkent, vPIC gives at most the make. For Uzbek-built and most Chinese-market cars it gives nothing, so our own WMI table (section 8) is the primary source.

## 7. How decoding combines the sources (`decode.py` design)

**As implemented** (`decode.py` docstring and code):
- **Source priority:** our WMI table gives `manufacturer` and `country`. It is the primary source for Uzbek and Chinese cars.
- **vPIC is optional and cached:**
  - It adds make, model, body class, fuel type, plant and the model year.
  - `modelyear` is sent only when position 10 gives a single candidate.
  - Successful responses are cached as `outputs/cache/vpic/<vin>.json`.
  - Network, HTTP and JSON failures become notes, never exceptions.
- **Never guess:** a field that no source provides stays empty, and every filled field records its source (`wmi_table`, `vpic` or `vin_rules`).
- **Low-confidence rows:** a row rated below `medium` is still used, but a note quotes its caveat from `plant_or_notes`, for example the LSG conflict.
- **How the table is read:** the loader matches header names regardless of case and spacing, and ignores extra columns. It skips rows whose WMI is not 3 VIN-alphabet characters. For a duplicate WMI it keeps the higher-confidence row.

**Recommendation, not yet in the code:** do not send synthetic VINs (section 9) to vPIC.

## 8. Our WMI table (`data/reference/wmi_table.csv`)

### 8.1 How it was built

**Sources.** The Phase 0 research collected WMIs for the brands sold or built in Uzbekistan from these sources:
- W: en.wikipedia "Vehicle identification number", rev. 1378888660
- B: en.wikibooks "Vehicle Identification Numbers (VIN codes)/World Manufacturer Identifier (WMI)", rev. 4672643
- T: en.wikibooks "Vehicle Identification Numbers (VIN codes)/Toyota/VIN Codes", rev. 4673541 (the current revision when Phase 0 read the page on 2026-10-10)
- R: ru.wikipedia "Идентификационный номер транспортного средства", rev. 152804777
- C: the carinfo.kiev.ua WMI dictionary for Uzbekistan
- N: NHTSA vPIC `DecodeWMI`

**Columns.** Each row lists the sources that support it (`source_url`, separated by ` | `), their licence, a confidence level and the check date. The columns are `wmi, manufacturer, country, plant_or_notes, source_url, source_licence, confidence, verified_on`. The code depends on these names.

**Review on 2026-10-10 (this file):**
- All 48 Phase 0 rows are present, in Phase 0 order, with the same codes and sources. Two exceptions: the LW4 vin17.com URL, explained below, and the Toyota Wikibooks page, which is now cited by its revision (oldid 4673541) instead of the unversioned URL.
- Six vPIC spot-checks were run (6.4).
- Four rows changed:
  - LSG moved from medium to low, under the rule that a conflict means low.
  - The LW4 note no longer contains a full real VIN. Its vin17.com evidence URL also contained that VIN, so the URL is not stored.
  - The JTM and KNC notes now record the vPIC result.

**Confidence.** The levels are the Phase 0 judgement, not a formula:
- **high:** vPIC agrees with the encyclopedic sources. XWB is the exception: it is high on four sources without vPIC.
- **medium:** one to three sources, none of them vPIC, and no conflict. NS2, LVT, LVU, LNN, XZG, LS6 and LW4 rest on Wikibooks alone.
- **low:** listed in 8.3.

### 8.2 Coverage

48 WMIs: 19 high, 22 medium and 7 low (computed from the CSV).

By country: China 15, South Korea 8, Uzbekistan 7, Japan 7, Russia 4, Thailand 2, India 2, Indonesia 1, Turkey 1, Kazakhstan 1.

`*` marks a conflict between sources.

| Group | high | medium | low |
|---|---|---|---|
| Built in Uzbekistan | XWB | NSA, NS2 | ULV*, NS4, NST, UMS* |
| Toyota / Lexus | JTD, JTE, JTM, JTN, JTH, JTJ, NMT | JTF, MR0, MHF, LFM, LVG, XW7 | - |
| BYD | LGX, LC0, LPE | - | MRX |
| Chery group | - | LVV, LVT, LVU, LNN | - |
| Great Wall / Haval | - | LGW, XZG | - |
| Geely group | L6T | LB3 | - |
| Changan | - | LS6 | - |
| Li Auto | - | LW4 | - |
| Kia | KNA, KND | MZB | KNC* |
| Hyundai | KMH, KM8 | MAL, Z94, MX3 | - |
| GM Korea / Daewoo | KL1, KL7, KLA | - | - |
| SAIC-GM | - | - | LSG* |
| AvtoVAZ (Lada) | - | XTA | - |

### 8.3 Conflicts and other low-confidence rows

| WMI | Why low |
|---|---|
| ULV | C gives GM-Uzbekistan (Daewoo, before 2004), but B places "UL" in Denmark's range (UH-UM) |
| UMS | C gives UzOtoyol JV, but B places "UM" in Denmark's range |
| KNC | W says Kia cars, B says Kia trucks; vPIC returned nothing (2026-10-10) |
| LSG | W and B say SAIC-GM (Chevrolet); vPIC says make Cadillac, manufacturer General Motors LLC (2026-10-10) |
| NS4 | Single source (R) |
| NST | Single source (R) |
| MRX | Single source (W) |

### 8.4 Gaps and how to close them

**Gaps:**
- **BYD Uzbekistan (Jizzakh):** no evidence of its own WMI. Cars assembled from kits may keep the Chinese LGX or LC0 VIN (Resolution 237 clause 15); this is unconfirmed.
- **ADM-Jizzakh:** only NST, from a single source. What ADM-built cars actually carry is unconfirmed.
- **Other UzAuto plants:** there is evidence only for Asaka (XWB).
- **KL8:** Phase 0 found it in vPIC, but it is not in our table yet. It needs its own vPIC look-up before a row is added.

**How to close a gap.** Use public sources only, and record the 3-character WMI only:
1. **Display car in a public showroom:** read the VIN plate and note the first three characters. Do not store the full VIN or a photo of the plate, and keep people and number plates out of any photo.
2. **Public classified ad that shows the VIN:** record the ad URL and the WMI. If the URL itself contains the VIN, do not store the URL.

Never use customer documents such as technical passports: they are personal data.

**Adding the row.** Use the ad URL, or `observed: <showroom>, <YYYY-MM-DD>, display car VIN plate`, as `source_url`. Keep `confidence` low until a second, independent public source agrees.

### 8.5 Attribution (CC BY-SA 4.0 sources)

- Wikipedia, "Vehicle identification number", revision 1378888660 (2026-10-06): https://en.wikipedia.org/w/index.php?oldid=1378888660
- Wikibooks, "Vehicle Identification Numbers (VIN codes)/World Manufacturer Identifier (WMI)", revision 4672643 (2026-09-27): https://en.wikibooks.org/w/index.php?oldid=4672643
- Wikibooks, "Vehicle Identification Numbers (VIN codes)/Toyota/VIN Codes", revision 4673541 (2026-10-01), the current revision on 2026-10-10: https://en.wikibooks.org/w/index.php?oldid=4673541
- Wikipedia (ru), "Идентификационный номер транспортного средства", revision 152804777 (2026-04-21): https://ru.wikipedia.org/w/index.php?oldid=152804777

Revision dates are from the MediaWiki API. Full attribution and the list of changes: `data/reference/LICENSE-wmi_table.md`.

Other sources:
- carinfo.kiev.ua (no licence stated; facts only): https://carinfo.kiev.ua/cars/vin/wmi/dic?country=UZBEKISTAN
- NHTSA vPIC ("free for use by the public"): https://vpic.nhtsa.dot.gov/api/vehicles/DecodeWMI/{wmi}?format=json

## 9. Synthetic VINs

**Why they start with "0":**
- The ISO 3780 WMI country chart (revised 2021-04-13) labels row "0" "Spare Codes", with no country assigned. The chart is ISO copyright: we cite it and do not copy the PDF into the repository.
- "Spare" is not a permanent reservation, so synthetic VINs are also marked in other ways.

**How they are marked** (`src/avtopassport/synthetic.py`):
- They use the WMI `0AP` (`SYNTHETIC_WMI`) and carry a valid check digit.
- Every synthetic record has `"synthetic": true`.
- The generator uses the fixed seed `20261010` (`SEED`).

**Guard:** `lookup(..., origin="scan")` always refuses. Synthetic history is therefore never attached to a VIN read from a real car, even if the strings match.

**How our VIN code treats them:**
- The first character, `0`, is outside 1-5/L/H, so the check digit is not mandatory. Synthetic VINs pass it anyway.
- `0AP` is not in the WMI table, so nothing is decoded from it.
- Anything shown for them carries the SYNTHETIC DEMO DATA badge.

## 10. Sources

- 49 CFR 565.13 (eCFR, version 2026-10-07): https://www.ecfr.gov/api/versioner/v1/full/2026-10-07/title-49.xml?part=565&section=565.13
- 49 CFR 565.15 (eCFR, version 2026-10-07): https://www.ecfr.gov/api/versioner/v1/full/2026-10-07/title-49.xml?part=565&section=565.15
- Canada, Motor Vehicle Safety Regulations (C.R.C., c. 1038), Schedule IV s.115: https://laws-lois.justice.gc.ca/eng/regulations/C.R.C.,_c._1038/FullText.html
- GB 16735 approval draft, WTO TBT notification attachment (cite only): https://members.wto.org/crnattachments/2019/TBT/CHN/19_3960_00_x.pdf
- Regulation (EU) 2021/535: https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R0535
- Uzbekistan, Cabinet of Ministers Resolution No. 237 of 2017-04-25 (general technical regulation on the safety of wheeled vehicles): https://www.lex.uz/acts/3180907
- ISO 3780 WMI country chart, rev. 2021-04-13 (ISO copyright, cite only): https://standards.iso.org/iso/3780/ed-4/en/Current%20WMI%20World%20Codes%20chart%204.13.21.pdf
- NHTSA vPIC API: https://vpic.nhtsa.dot.gov/api/
- vPIC FAQ: https://vpic.nhtsa.dot.gov/api/home/index/faq
- vPIC downloads: https://vpic.nhtsa.dot.gov/downloads/
- vPIC error codes: https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableValuesList/Error%20Code?format=json
- data.gov catalogue search for vPIC: https://catalog.data.gov/api/3/action/package_search?q=vPIC&rows=5
- vininfo package README (example VINs): https://pypi.org/pypi/vininfo/json
- WMI sources: section 8.5
