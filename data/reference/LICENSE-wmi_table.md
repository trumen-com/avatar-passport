# Licence and attribution: WMI reference table

The following files are licensed under the **Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0)**, https://creativecommons.org/licenses/by-sa/4.0/:

- `data/reference/wmi_table.csv`
- `docs/research/phase0_findings_vin.md`, which contains the same table and the notes it was built from
- section 8 ("Our WMI table") of `docs/research/vin_decoding.md`, which summarises the table; the rest of that file is under the repository licence
- this file

This licence is separate from the repository's code licence. The code is licensed under AGPL-3.0 (`LICENSE` in the repository root). The files listed above are not covered by AGPL-3.0.

## Attribution (TASL)

| | |
|---|---|
| **Title** | AvtoPassport WMI reference table (World Manufacturer Identifier to manufacturer, country and plant notes) |
| **Authors** | Wikipedia contributors; Wikibooks contributors. Adapted by the AvtoPassport hackathon team. |
| **Source** | The four page revisions listed below |
| **Licence** | CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ |

The sources were retrieved on 2026-10-10. Page titles and revision dates come from the MediaWiki API.

1. English Wikipedia, "Vehicle identification number", revision 1378888660 (2026-10-06): https://en.wikipedia.org/w/index.php?oldid=1378888660
2. Wikibooks, "Vehicle Identification Numbers (VIN codes)/World Manufacturer Identifier (WMI)", revision 4672643 (2026-09-27): https://en.wikibooks.org/w/index.php?oldid=4672643
3. Wikibooks, "Vehicle Identification Numbers (VIN codes)/Toyota/VIN Codes", revision 4673541 (2026-10-01): https://en.wikibooks.org/w/index.php?oldid=4673541
   - Phase 0 read the unversioned page (https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Toyota/VIN_Codes) on 2026-10-10. Revision 4673541 was the current revision that day, and the table cites it.
4. Russian Wikipedia, "Идентификационный номер транспортного средства" (Vehicle identification number), revision 152804777 (2026-04-21): https://ru.wikipedia.org/w/index.php?oldid=152804777

All three sites publish their text under CC BY-SA 4.0. Each site's licence API confirmed this during Phase 0 research on 2026-10-10.

## Changes made

- **Selected rows.** We kept WMI rows for Uzbek plants, Toyota, BYD and other Chinese, Korean and Russian manufacturers. Each row was cut down to code, manufacturer, country and short plant notes, and the wording was shortened and paraphrased.
- **Merged non-wiki facts.** We added facts from two other sources:
  - **NHTSA vPIC `DecodeWMI` API:** https://vpic.nhtsa.dot.gov/api/vehicles/DecodeWMI/{wmi}?format=json. The vPIC FAQ (https://vpic.nhtsa.dot.gov/api/home/index/faq) calls the service free for use by the public. No licence is stated.
  - **carinfo.kiev.ua WMI dictionary:** https://carinfo.kiev.ua/cars/vin/wmi/dic?country=UZBEKISTAN. No licence is stated, so we took facts only (codes and manufacturers).
- **Added columns and notes.** We added the `source_url`, `source_licence`, `confidence` and `verified_on` columns, and `CONFLICT` notes where sources disagree.
- **Confidence is ours.** The `confidence` values are our own judgement of how well the sources agree; the sources do not provide them. See `docs/research/phase0_findings_vin.md`, section 4.

## Reusing this table

- **Attribution.** When you share the table or anything derived from it, keep this attribution, link the licence and say what you changed.
- **ShareAlike.** Derivatives must be shared under CC BY-SA 4.0 or a compatible licence.
- **No endorsement.** The original authors do not endorse this adaptation.
- **No warranty.** No manufacturer has checked the table. Confirm rows marked `low` or `CONFLICT` against a real VIN before relying on them.
