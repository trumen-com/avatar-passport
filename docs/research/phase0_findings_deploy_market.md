<!-- Phase 0 desk research, collected 2026-10-10 by read-only research agents (web/API fetches). Verbatim summary; claims cite the URLs fetched. Third-party figures here are REPORTED, not measured by us. -->

I read both transcripts and pulled facts only from tool results that actually returned content. The fetches were run on 2026-10-10.

## 1. Hosting

**Hugging Face Spaces** ([hardware](https://huggingface.co/docs/hub/spaces-gpus), [overview](https://huggingface.co/docs/hub/spaces-overview))
- **CPU Basic:** 2 vCPU, 16 GB RAM, 50 GB disk that is not persistent. No hourly charge.
- **Paid CPU:** the only paid CPU tier is CPU Upgrade (8 vCPU, 32 GB) at $0.03/h. GPU tiers start at a T4 for $0.40/h.
- **Blocker:** since a docs change on 2026-07-21, creating a Docker or Gradio Space needs PRO ($9/mo) or Team ($20 per user per month). Free accounts that are verified and over 30 days old can only host 2 Gradio Spaces on ZeroGPU.
- **Sleep:** a CPU Basic Space sleeps after 48 h with no visitors and wakes on the next visit.
- **Docker:** `sdk: docker`, `app_port` defaults to 7860, and the container runs as user ID 1000.
- **Storage:** the old persistent storage was shut down (HF staff on the forum, 2026-03-27). The replacement is Storage Buckets mounted as volumes (changelog 2026-03-31). They cost $18/TB/mo private or $12/TB/mo public, with egress included. Free accounts get 100 GB of private storage.
- **GitHub Actions** ([docs](https://huggingface.co/docs/hub/spaces-github-actions)): use `huggingface/hub-sync@v0.3.0` with `HF_TOKEN`. Set `space_sdk: docker`, because it defaults to gradio. The alternative is `git push https://USER:$HF_TOKEN@huggingface.co/spaces/USER/SPACE main`.
- **File sizes:** with git push, files over 10 MB need git-xet or LFS. hub-sync uploads with `hf upload`, so it needs neither.

**Render free** ([docs](https://render.com/docs/free), [plans](https://render.com/docs/compute-plans))
- 0.1 CPU and 512 MB RAM. It spins down after 15 min idle and takes about 1 min to start again. 750 h/month.
- The filesystem is wiped on redeploy, restart or spin-down; the docs name local SQLite files explicitly. Free services cannot attach disks.
- Paid: 0.5 CPU/512 MB is $7/mo, 1 CPU/2 GB is $25/mo, and disks are $0.25/GB/mo.

**Recommendation:** a Docker Space on CPU Basic, which means paying for PRO. Render free is far too weak for YOLO-seg.

**SQLite persistence:** the cheapest option is to keep SQLite on local disk, restore it at boot, and snapshot it every few minutes to a private dataset or bucket with `huggingface_hub.CommitScheduler`. Do not run SQLite directly on a bucket mount: the hf-mount README warns of no cross-client locking, data up to 10 s stale, and slow random I/O. Fallback: Turso's free tier (5 GB, 100 databases).

## 2. Python 3.12 and GPU
- **ultralytics:** 8.4.175 (2026-10-09). YOLO26 support starts at v8.4.0 (2026-01-14).
- **onnxruntime:** 1.31.0 has cp312 wheels.
- **torch:** 2.14.1 on default PyPI is already a CUDA 13.0 build and needs NVIDIA driver 580 or newer.
  - For Blackwell (sm_120) use `--index-url https://download.pytorch.org/whl/cu130`, which has cp312.
  - cu128 stops at 2.11.0 and cu129 at 2.13.0. Blackwell support began in 2.7 with cu128.
  - I could not confirm directly that sm_120 is in the cu130 build list.
- **Paddle:** paddlepaddle 3.3.1 on PyPI and 3.4.0 on the Paddle index have wheels for Python 3.10–3.13. The wheel is 185 MB.
- **Colab:** the default is now Python 3.13.16 on Ubuntu 24.04 (changed 2026-08-19), with torch 2.11.0+cu130. To get Python 3.12.13, pin the "2026.07" runtime. The free GPU is a T4 (Google blog, Oct 2025), and runtimes are capped at 12 h.

## 3. Ultralytics AGPL-3.0 ([license](https://www.ultralytics.com/license), [request-license](https://www.ultralytics.com/request-license))
- Without an Enterprise licence you must publish the full source of the whole project under AGPL-3.0, including weights where applicable.
- Under §13, users of a modified version over a network must be offered its source. Ultralytics says SaaS or API use needs an Enterprise licence unless the project is open-sourced.
- Ultralytics says all YOLO trained models are AGPL by default, even if trained from scratch. ONNX exports are not addressed. There is no hackathon licence.

## 4. ONNX CPU latency at 640 px (Ultralytics docs; Xeon at 2.0 GHz, thread count not stated)

| Model | ms |
|---|---|
| YOLO26n-seg | 53.3 |
| YOLO26s-seg | 118.4 |
| YOLO26m-seg | 328.2 |
| YOLO26l-seg | 387.0 |
| YOLO26x-seg | 787.0 |
| YOLO11n-seg (search snippet only) | 65.9 |

No figure for a 2-vCPU machine exists, so this needs measuring.

## 5. PDF and timestamping
- **fpdf2** 2.8.9: LGPL-3.0, pure Python.
- **reportlab** 5.0.1: BSD; depends only on Pillow and charset-normalizer.
- **weasyprint** 70.0: BSD, but needs Pango and related system libraries (about 16 MB installed).
- **Pick:** reportlab or fpdf2, bundling a DejaVu font for Cyrillic.
- **freetsa.org:** POST to `https://freetsa.org/tsr`.
  - Supports SHA-1 to SHA-512. Certificates are at `files/tsa.crt` and `files/cacert.pem`.
  - A new certificate covers stamps from 2026-03-16 and is valid to 2040.
  - The site states no SLA, rate limit or warranty, and asks commercial users to get in touch.
- **Python clients:** `rfc3161ng` 2.1.3 (MIT, last released 2020), or `openssl ts` plus `requests`. An `rfc3161-client` issue lists freetsa as failing, so test it before relying on it.

## 6. Royal Motors
- **Toyota:** the distributor is Toyota Central Asia, not Royal Motors.
  - The [official dealer list](https://www.toyota-centralasia.com/ru/dealers) includes "Toyota Royal Motors" and "Royal Motors Qushbegi" in Tashkent.
  - Their channel ([t.me/toyota_rm](https://t.me/toyota_rm), phone 1853) refers to "Toyota RM Tashkent City" and "Toyota RM Qushbegi".
  - Line-up: Camry and Camry HEV, RAV4 HEV, Land Cruiser Prado (250), Land Cruiser 300 and 300 HEV.
  - On Yandex, the Tashkent City showroom lists polishing and film, but not body or paint repair.
- **BYD:** [byd-rm.uz](https://byd-rm.uz/about) says "BYD Чигатой … Входим в группу компаний Royal Motors".
  - Address: Обводная Нурафшон 7, phone +998 78 140 1111.
  - Their channel ([t.me/byd_chigatoy](https://t.me/byd_chigatoy)) lists Tashkent, Namangan, Kokand and Samarkand.
  - Models with [bydauto.uz](https://bydauto.uz) prices in UZS: Chazor 255.9M, Song Plus DM-i 374.1M, Seal 6 364.4M, Sealion 7 451.1M, Yuan Up 266.7M, e2 236.6M, Seagull 208.3M, Atto 8 529.8M, Ti7 513.5M.

## 7. Tashkent prices actually seen
Dollar prices are converted at the [Central Bank rate](https://cbu.uz/ru/arkhiv-kursov-valyut/json/) of 11,897.90 UZS per USD, dated 10.10.2026.
```
service,car_class,price_uzs,source_url,date_seen
painting (from),any,295000,https://auto.ustabor.uz/ru/price/body_work,2026-10-10
dent straightening (from),any,500000,https://auto.ustabor.uz/ru/price/body_work,2026-10-10
scratch removal (from),any,325000,https://auto.ustabor.uz/ru/price/body_work,2026-10-10
panel replacement (from),any,1200000,https://auto.ustabor.uz/ru/price/body_work,2026-10-10
polishing (from),any,1004167,https://auto.ustabor.uz/ru/price/body_work,2026-10-10
paint 1 panel (from $55),any,654000,https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-i-t-d-ID1YwdF.html,2026-10-10
dent repair without paint (from $88),any,1047000,https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-i-t-d-ID1YwdF.html,2026-10-10
paint 1 panel (from $77),any,916000,https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-shumo-tsiya-kostoprav-ID22hsA.html,2026-10-10
body polishing (flat),any,600000,https://www.olx.uz/d/obyavlenie/polirovka-avto-kuzova-za-lyuboy-mashina-600000-sum-600-tsyach-sum-ID4bVtP.html,2026-10-10
stone-chip dent repair (from),any,100000,https://www.olx.uz/d/obyavlenie/avto-vakuum-pachok-tugirlash-vakum-ID3goDG.html,2026-10-10
body/paint listing,Nexia/Lacetti/Matiz/foreign,200000,https://www.cenotavr.uz/tashkent/remont-kuzova-pokraska-polirovka-kuzova-90,2026-10-10
pre-purchase paint/body inspection (from),any,300000,https://www.olx.uz/d/obyavlenie/proverka-avto-po-kuzovu-i-probegu-po-nizkim-tsenam-vyezdnoy-osmotr-ID3Vjj4.html,2026-10-10
dealer polishing promo,any BYD,2000000,https://t.me/byd_chigatoy,2025-02-15
wheel alignment promo (normally 200000),BYD,100000,https://t.me/byd_chigatoy,2026-04-08
```
The inspection row is a direct competitor price for a condition check. Prices by car class from car-driver.uz (750k to 2.5M) appeared only in a search snippet, so I left them out.

## 8. OpenStreetMap
- **Tags:** `shop=car_repair` combined with `service:vehicle:body_repair=yes` or `service:vehicle:painting=yes`; also `craft=car_painter`.
- **Tashkent coverage (VK Overpass mirror, 2026-10-10):**

  | Query | Count |
  |---|---|
  | `shop=car_repair` | 328 |
  | …with `body_repair=yes` | 0 |
  | …with `painting=yes` | 0 |
  | …with any `car_repair=*` | 6 |
  | `craft=car_painter` | 0 |

  So OSM alone won't identify body shops; you need to match on names.
- **Endpoints:**
  - The main server `https://overpass-api.de/api/interpreter` allows under 10,000 queries and 1 GB per day; for regular use, divide those by 100. It returned 504 errors during testing.
  - The mirror `https://maps.mail.ru/osm/tools/overpass/api/interpreter` states no limits.
  - Send a unique User-Agent, and wait 30 s after a 429 error.
- **Example query:**
```
[out:json][timeout:90];
area["ISO3166-2"="UZ-TK"]["admin_level"="4"]->.a;
nwr["shop"="car_repair"](area.a)->.all;
(nwr.all["service:vehicle:body_repair"="yes"];
 nwr.all["name"~"кузов|kuzov|покрас|kraska|маляр|рихтов|polir|body|paint",i];);
out tags center;
```
- **Attribution:** show "© OpenStreetMap contributors" linked to openstreetmap.org/copyright, and say the data is under ODbL 1.0. In a PDF, print the URL.
- **Nominatim (geocoding):** at most 1 request per second, cache results, and no autocomplete.

## 9. Paint colours
These hex values are the swatch colours from the website, not factory paint codes.
```
name,model,hex_approx,source_url
White,BYD Chazor,#FFFFFF,https://bydauto.uz/ru/chazor
Gray,BYD Chazor,#566270,https://bydauto.uz/ru/chazor
Black,BYD Chazor,#1A232C,https://bydauto.uz/ru/chazor
Silver,BYD Song Plus DM-i,#505156,https://bydauto.uz/en/song-dmi-champion
Midnight Blue (name inferred from page text),BYD Yuan Up DM-i,#2A3678,https://bydauto.uz/ru/yuan-up-dmi
Atlantis Blue,BYD Ti7,#728FB3,https://bydauto.uz/en/ti7
Emerald Green,BYD Ti7,#028377,https://bydauto.uz/en/ti7
Star & Moon Silver,BYD Ti7,#C6C5CA,https://bydauto.uz/en/ti7
```

## 10. Gaps
- No Toyota colours: the Toyota Central Asia pages carry no colour data.
- Neither dealer publishes a body-shop price list. No shop gives per-panel prices by car class.
- The OSM count by name keywords timed out.
- No latency figure for 2 vCPUs.
- sm_120 in the cu130 build is unconfirmed.
- Not stated by the sources: whether free accounts are billed for buckets, whether Render free supports Docker, and freetsa's rate limits.
- toyotaroyalmotors.uz did not resolve in DNS.

Transcripts: `/home/akkomron/.claude/projects/-home-akkomron-Projects-avtopassport-skeleton/63e51715-48a9-4d2a-967e-a37c00625496/subagents/workflows/wf_4f8d5e33-171/`
