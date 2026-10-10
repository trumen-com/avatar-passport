<!-- Phase 0 desk research, collected 2026-10-10 by read-only research agents (web/API fetches). Verbatim summary; claims cite the URLs fetched. Third-party figures here are REPORTED, not measured by us. -->
<!-- Correction 2026-10-10: third-party VINs in this file are masked after position 11 (real-vehicle identifiers are not kept in the repo; the WMI/VDS prefix is what the evidence needs). -->

I've pulled together what both transcripts actually show, and filled the main gaps with live read-only checks: PyPI, the ISO chart PDF, the GB 16735 WTO draft, the RapidOCR source and the Wikimedia licence API. Neither earlier agent had written a conclusion, so everything below is built from their tool results.

## 1. OCR engine

Use **RapidOCR 3.10.0 on both Colab and CPU Docker**. Run the same ONNX CPU path on Colab so the two environments behave the same. EasyOCR is a Colab-only fallback if you really need a built-in character allowlist. PaddleOCR is too heavy for the time you have.

| | RapidOCR 3.10.0 (2026-10-08) | PaddleOCR 3.7.0 (2026-06-11) | EasyOCR 1.7.2 (2024-09, no release since) |
|---|---|---|---|
| Licence | Apache-2.0; onnxruntime is MIT | Apache-2.0 | Apache-2.0; its CRAFT detection weights are MIT (NAVER) |
| Python 3.12 | Pure-Python wheel (`>=3.8,<4`); onnxruntime 1.31.0 has a cp312 wheel and needs Python ≥3.11 | PyPI lists 3.8–3.13 (README badge says 3.8–3.12); paddlepaddle 3.3.1 has a cp312 wheel | Pure-Python wheel, but needs torch |
| Size | 27.3 MB wheel with models included, plus 23.8 MB onnxruntime | 194.8 MB paddlepaddle wheel plus paddlex | torch, plus 77.3 MB + 14.0 MB of models downloaded on first run |
| Allowlist | **None** (only a custom dictionary path), so filter afterwards | None found | `readtext(..., allowlist=..., blocklist=...)` |

- **RapidOCR defaults:** PP-OCRv6 "small" detection and recognition models (handle Chinese and Latin text).
- **onnxruntime:** it is the default engine but has not been installed automatically since 2.0.6, so install it yourself.
- **Colab:** the 2026.07 runtime is Python 3.12.13 with PyTorch 2.11.0.
- **No ready-made VIN model:** nothing VIN-specific turned up on Hugging Face.

```python
# pip install rapidocr==3.10.0 onnxruntime
from rapidocr import RapidOCR, EngineType
engine = RapidOCR()
res = engine("vin.jpg", use_det=True, use_cls=True, use_rec=True)  # path/URL/ndarray
for t, s in zip(res.txts or (), res.scores or ()): print(t, s)      # also res.boxes (N,4,2), res.elapse
res.vis("vis.jpg")
# Optional GPU on Colab (documented parameter keys):
# RapidOCR(params={"Det.engine_type": EngineType.TORCH, "Rec.engine_type": EngineType.TORCH, "EngineConfig.torch.use_cuda": True})
```

## 2. VIN validation

**Algorithm**
- **Allowed characters:** `[A-HJ-NPR-Z0-9]{17}`, i.e. no I, O or Q (49 CFR 565.13(g); the Uzbek regulation says the same).
- **Letter values:** A1 B2 C3 D4 E5 F6 G7 H8 J1 K2 L3 M4 N5 P7 R9 S2 T3 U4 V5 W6 X7 Y8 Z9. Digits keep their own value.
- **Weights by position:** 8 7 6 5 4 3 2 10 0 9 8 7 6 5 4 3 2.
- **Check digit:** take the weighted sum mod 11; a remainder of 10 becomes X.
- I ran this against three published sample VINs (US regulation, Chinese standard, Wikipedia) and all three came out correct.

```python
T={**{str(d):d for d in range(10)},**dict(zip("ABCDEFGH",range(1,9))),**dict(zip("JKLMN",range(1,6))),"P":7,"R":9,**dict(zip("STUVWXYZ",range(2,10)))}
W=[8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]
def cd(v): r=sum(T[c]*w for c,w in zip(v,W))%11; return "X" if r==10 else str(r)
```

**Where the check digit is mandatory**
- **US:** yes, 49 CFR 565.13(c).
- **Canada:** yes, Motor Vehicle Safety Regulations, Schedule IV s.115(3)(c), (7), (8).
- **China:** yes. In the GB 16735 approval draft filed with the WTO, clause 4.3.3 says position 9 is the check digit computed per Annex A, with the same letter values. The foreword says all technical content is mandatory. I did not read the final 2019 text, which is paywalled.
- **EU:** phasing in. Under Regulation (EU) 2021/535 Art. 12, new vehicle types need it from 2022-07-06, and registration is refused without it from 2026-07-07.
- **Uzbekistan:** no check-digit clause in the extracted part of Cabinet of Ministers Resolution 237 (2017). Clause 15 says vehicles assembled locally from kits keep the original manufacturer's VIN.
- **Real VINs fail it:** WP0ZZZ99ZTS******, KLATF08Y1VB******, VF1LM1B0H36****** and XTAGFK330JY****** are all real and all fail. So reject on a bad check digit only when the VIN starts with 1–5, L or H; everywhere else just warn.

**Position 10 (model year):** `ABCDEFGHJKLMNPRSTVWXY123456789` maps to 2010–2039, and also to 1980–2009.
- I, O, Q, U, Z and 0 are never year codes.
- For US light vehicles, a letter in position 7 means the 2010–2039 cycle.
- China requires a year code here. Outside North America it is often just 0.

**Position 11:** assembly-plant code, defined by each manufacturer (49 CFR 565.15(d)(2); GB 16735 clause 4.4.3).

**Safe OCR correction**
1. Uppercase, and strip `*`, spaces and `-`.
2. Replace I with 1, and O or Q with 0.
3. Positions 14–17 must be digits, so map S→5, B→8, G→6, Z→2, D→0 there.
4. Swap WMI characters (1↔L, 0↔D, 5↔S, 8↔B) only if the swap produces a known WMI.
5. Where the check digit is mandatory and fails, try single-character swaps and accept only if exactly one candidate passes.
6. Always show the result to the user to confirm, because a wrong VIN still passes the check 1 time in 11.

**Leading "0" for synthetic VINs:** yes. The ISO 3780 country chart (revised 4/13/2021) labels row "0" as "Spare Codes" with no country assigned: https://standards.iso.org/iso/3780/ed-4/en/Current%20WMI%20World%20Codes%20chart%204.13.21.pdf. "Spare" is not permanently reserved, so also tag synthetic VINs in the data.

## 3. NHTSA vPIC

**Endpoints** (base `https://vpic.nhtsa.dot.gov/api/vehicles/`)
- `DecodeVinValues/{VIN}?format=json&modelyear=YYYY` returns one flat record with 154 fields.
  - Useful fields: Make, Model, ModelYear, Manufacturer, ManufacturerId, VehicleType, BodyClass, PlantCountry, PlantCity, FuelTypePrimary, ElectrificationLevel, ErrorCode, ErrorText, SuggestedVIN, PossibleValues.
  - Partial VINs work with `*` in place of missing characters. NHTSA recommends always sending the model year.
- `DecodeVin/...` returns the same data as variable/value rows; `DecodeVinValuesExtended/...` adds NHTSA statistics fields.
- `DecodeWMI/{WMI}?format=json` returns Make, ManufacturerName, VehicleType, CommonName, CreatedOn, UpdatedOn.
- `GetWMIsForManufacturer/{name}?format=json` lists a manufacturer's WMIs.
- Batch: `POST DecodeVINValuesBatch/` with body `format=json&data=VIN,YEAR;VIN`, up to 50 VINs.

**ErrorCode values**

| Code | Meaning |
|---|---|
| 0 | Clean decode |
| 1 | Bad check digit |
| 2–4 | VIN corrected |
| 5 | Errors in several positions |
| 6 | Incomplete VIN |
| 7 | Manufacturer not registered with NHTSA |
| 8 | No detailed data |
| 11 | Bad model-year code |
| 14 | Some positions not decodable |
| 400 | Invalid characters |

**Usage limits are contradictory**
- The FAQ says it is free, needs no registration, has no daily limit, and handles 1000–2000 requests a minute (run big batches at night or weekends).
- The API page mentions "automated traffic rate control" and the downloads page mentions "quotas and rate limits".
- No rate-limit headers came back. Cache responses.
- There is an offline database dump, `vPICList_lite_2026_09` (PostgreSQL 17+, 67.8 MB), queried with `SELECT * FROM vpic.spVinDecode('...')`.

**What it returned for non-US codes**
- **DecodeWMI found:** LGX, LC0 and LPE (BYD; LC0 is listed as Bus), JTD/JTE/JTM/JTH/JTJ (Toyota/Lexus), NMT, KNA/KND, KMH/KM8, KL1/KL7/KL8/KLA, and L6T (listed as Zeekr's US company).
- **DecodeWMI returned nothing for:** XWB, NSA/NS2/NS4/NST, LVV/LVT/LVU/LNN, LGW/XZG, LB3, LS4–LS6, LW4/HLX, Z94, XTA, XW7, MR0, MHF, LFM, LVG, MZB.
- **Decoding a bare 3-character prefix:** XWB, Z94, XTA, LVV, LGW, NST, NS4 and NSA give error code "6,7,11" (manufacturer not registered with NHTSA). LGX, LC0, JTD and JTM give "6,8,11".
- **Bottom line:** vPIC tells you the make at most for cars sold in Tashkent, so you need your own WMI table (section 4).

## 4. WMI table

The source column uses short keys. Expand them to these URLs:
- W = https://en.wikipedia.org/w/index.php?oldid=1378888660
- B = https://en.wikibooks.org/w/index.php?oldid=4672643
- R = https://ru.wikipedia.org/w/index.php?oldid=152804777
- C = https://carinfo.kiev.ua/cars/vin/wmi/dic?country=UZBEKISTAN
- N = https://vpic.nhtsa.dot.gov/api/vehicles/DecodeWMI/{wmi}?format=json
- T = https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Toyota/VIN_Codes
- V = https://vin17.com/view/LW433B103M1******/jqbyragm (search snippet only, not fetched)

```csv
wmi,manufacturer,country,plant_or_notes,source_url,confidence
XWB,UzAuto Motors (ex GM Uzbekistan/UzDaewoo/Ravon),Uzbekistan,Asaka; used since 2004 per C; vPIC none,W|B|R|C,high
ULV,GM-Uzbekistan (Daewoo),Uzbekistan,pre-2004; CONFLICT: UL lies in Denmark range UH-UM (B),C,low
NSA,SamAvto/SAZ (ex SamKochAvto),Uzbekistan,Samarkand buses/trucks,B|C,medium
NS2,JV MAN Auto Uzbekistan,Uzbekistan,trucks,B,medium
NS4,Atlant Invest Group,Uzbekistan,Samarkand,R,low
NST,ADM-Jizzakh,Uzbekistan,Jizzakh Kia assembly; single source,R,low
UMS,UzOtoyol JV,Uzbekistan,CONFLICT: Denmark range (B),C,low
JTD,Toyota Motor Corp,Japan,car,N|W|T,high
JTE,Toyota Motor Corp,Japan,MPV/SUV (vPIC: Toyota/Lexus/Subaru),N|W|T,high
JTM,Toyota Motor Corp,Japan,SUV,N|W|T,high
JTN,Toyota Motor Corp,Japan,car,N|W|T,high
JTF,Toyota Motor Corp,Japan,van/truck; vPIC none,W|T,medium
JTH,Toyota (Lexus),Japan,Lexus car,N|W|T,high
JTJ,Toyota (Lexus),Japan,Lexus SUV,N|W|T,high
MR0,Toyota Motor Thailand,Thailand,Ban Pho/Samrong,W|T,medium
MHF,Toyota Indonesia,Indonesia,,W|T,medium
LFM,FAW Toyota,China,,W|T,medium
LVG,GAC Toyota,China,,W|T,medium
NMT,Toyota Motor Mfg Turkey,Turkey,,N|W|T,high
XW7,Toyota Motor Mfg Russia,Russia,,W|T,medium
LGX,BYD Auto,China,BYD/Fangchengbao,N|W|B,high
LC0,BYD Auto Industry,China,BYD/Denza; vPIC type Bus,N|W|B,high
LPE,BYD Auto,China,vPIC type Truck,N|W|B,high
MRX,BYD Auto (Thailand),Thailand,,W,low
LVV,Chery,China,Chery/Omoda/Jaecoo,W|B,medium
LVT,Chery,China,Chery/Exeed/Jetour,B,medium
LVU,Chery,China,Chery/Jetour/Jaecoo,B,medium
LNN,Chery,China,Chery/Omoda/Jaecoo,B,medium
LGW,Great Wall Motor,China,GWM/Haval/Tank/Wey/Ora,W|B,medium
XZG,Haval Motor Rus,Russia,,B,medium
LB3,Geely,China,Geely/Galaxy/Geometry,W|B,medium
L6T,Geely/Lynk&Co/Zeekr,China,vPIC: Zeekr US LLC,N|W|B,high
LS6,Changan/Deepal/Avatr,China,,B,medium
LW4,Li Auto,China,sample LW433B103M1******,B|V,medium
KNA,Kia,South Korea,car,N|W|B,high
KND,Kia,South Korea,SUV/MPV,N|W|B,high
KNC,Kia,South Korea,CONFLICT: car (W) vs truck (B),W|B,low
MZB,Kia India,India,,W|B,medium
KMH,Hyundai,South Korea,car,N|W|B,high
KM8,Hyundai,South Korea,SUV,N|W|B,high
MAL,Hyundai Motor India,India,,W|B,medium
Z94,Hyundai Motor Mfg Rus,Russia,2008-23; Solaris Auto 2023-,W|B,medium
MX3,Hyundai Trans Auto,Kazakhstan,,W|B,medium
KL1,GM Korea,South Korea,Chevrolet car,N|W|B,high
KL7,GM Korea,South Korea,Chevrolet SUV,N|W|B,high
KLA,Daewoo/GM Korea,South Korea,,N|W|B,high
LSG,SAIC-GM,China,Chevrolet (W/B); CONFLICT: vPIC make Cadillac,N|W|B,medium
XTA,AvtoVAZ (Lada),Russia,,W|B|R,medium
```

## 5. Licences to cite
- **Wikipedia (English and Russian) and Wikibooks:** CC BY-SA 4.0, confirmed via each site's licence API. Cite the revision ids above; a WMI table built from them must also be share-alike.
- **NHTSA vPIC:** the FAQ says it is "free for use by the public", with no licence or registration. data.gov lists it as "unknown-license". Calling it public domain as US-government work is my inference; I did not find it stated.
- **Law texts (49 CFR 565, Canadian regulations, EU 2021/535, Uzbek Resolution 237):** cite as sources.
- **ISO 3780 chart:** ISO copyright. Cite it, don't redistribute the PDF.
- **GB 16735 WTO draft:** cite only.
- **carinfo.kiev.ua:** no licence stated; use it for facts only.
- **Code:** RapidOCR, PaddleOCR and EasyOCR are Apache-2.0; onnxruntime is MIT.

## 6. Gaps
- **BYD Uzbekistan (Jizzakh):** no evidence for a WMI. Kit-assembled cars may simply keep the Chinese LGX/LC0 VIN.
- **Other UzAuto plants:** no WMI evidence for anything other than Asaka (XWB).
- **Fix:** ask a partner dealer for one real VIN per brand from a tech passport.
- **NST and NS4:** each has a single source.
- **Not read in full:** the final GB 16735-2019 text and the whole Uzbek regulation.
- **RapidOCR accuracy:** untested on stamped or etched VINs.

Evidence comes from the two transcripts:
- /home/akkomron/.claude/projects/-home-akkomron-Projects-avtopassport-skeleton/63e51715-48a9-4d2a-967e-a37c00625496/subagents/workflows/wf_4f8d5e33-171/agent-ae7fa7465567c07fa.jsonl
- /home/akkomron/.claude/projects/-home-akkomron-Projects-avtopassport-skeleton/63e51715-48a9-4d2a-967e-a37c00625496/subagents/workflows/wf_4f8d5e33-171/agent-a39c3e9a7ee54abee.jsonl
