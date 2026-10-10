"""Build data/reference/repair_prices.csv from observed Tashkent repair prices.

Source rows
    docs/research/phase0_findings_deploy_market.md, section 7 ("Tashkent prices actually seen"),
    read on 2026-10-10. Each row in ROWS keeps exactly what was observed: price, currency, source
    and the row's own date. Nothing is estimated, so is_estimate is false on every row.
    Listing dates, the cenotavr.uz page text and the two BYD Telegram posts were checked against the
    Phase 0 tool output (OLX offers API, fetched page text, t.me/s/byd_chigatoy) on 2026-10-10.

Currency conversion (computed, not estimated)
    USD rows keep price_original in USD. price_uzs = price_original x fx_rate_uzs, using the Central
    Bank of Uzbekistan rate of 11897.90 UZS per USD dated 2026-10-10
    (https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/2026-10-10/). The result is rounded to the
    nearest 1,000 UZS, with exact halves rounded up (ROUND_HALF_UP). Example: 55 x 11897.90 =
    654,384.5 -> 654,000. Shops quote round sums, so the rounding avoids implying more precision
    than the quote has. UZS rows are copied unchanged (no rounding), and their fx_* columns stay empty.

Columns
    service            what the source offered, in plain English
    service_category   paint | dent | polish | scratch | panel_replacement | inspection | promo |
                       unspecified (the source does not say what the price buys)
    part               any (not specified by the source) | panel (one body panel) |
                       whole_body (the whole car body) | wheels
    car_class          as stated by the source ("any" = the source names no class)
    price_original     price as quoted, in `currency`
    currency           UZS | USD
    fx_rate_uzs, fx_date, fx_source
                       the conversion used (only for currency != UZS)
    price_uzs          integer UZS (computed as above)
    price_kind         from = the source says "from" (a lower bound) |
                       flat = one price stated without "from" |
                       promo = time-limited dealer promotion
    source_url         listing URL, or "phone: <shop name>, <YYYY-MM-DD>" for phone quotes
    source_type        web_listing | telegram_post | phone | dealer
    listing_date       date the source published or last refreshed the price, if known:
                       OLX rows carry the listing's last refresh date (OLX "last_refresh_time"; the
                       creation dates are in docs/research/market_prices.md section 2), Telegram rows
                       the post date, phone rows the call date. Empty when the source shows no date
                       (ustabor.uz) or no full date (cenotavr.uz shows "16 октября" without a year).
    retrieved_on       date we read the price
    is_estimate        true only for a price that is our own estimate (none so far)

Rows the cost estimator must exclude
    service_category "inspection" (a competitor's pre-purchase condition check, kept as a
    benchmark for our own pricing), "promo" (BYD dealer promotions: polishing until 1 March 2025,
    wheel alignment until 30 April 2026; both have ended) and "unspecified" (see below).
    They are context, not repair costs.

Known caveat
    The cenotavr.uz row is a single "Цена: 200 000 сум" on an advert that lists body repair,
    scratch removal, bumper repair, polishing and painting. The advert does not say which service
    the price is for, so the row is "unspecified". Its date has no year and cannot be later than
    2025-10-16, because 16 October 2026 had not yet come on 2026-10-10.

Adding prices collected by phone (template)
    Append a PriceRow to ROWS, then re-run this script. Use the shop's business name. If the shop is
    a private person, use a neutral label such as "private body shop, Chilonzor". Never record
    personal names or phone numbers.

        PriceRow(
            service="<service as the shop described it, e.g. paint one front door>",
            service_category="<paint|dent|polish|scratch|panel_replacement|inspection|promo|unspecified>",
            part="<any|panel|whole_body|wheels|...>",
            car_class="<as the shop stated it, e.g. any | Cobalt | BYD Song Plus>",
            price_original="<price exactly as quoted>",
            currency="<UZS|USD>",
            price_kind="<from|flat|promo>",
            source_url="phone: <shop name>, <YYYY-MM-DD>",
            source_type="phone",
            listing_date="<YYYY-MM-DD of the call>",
            retrieved_on="<YYYY-MM-DD of the call>",
        ),

Usage
    python scripts/build_repair_prices.py           # write data/reference/repair_prices.csv
    python scripts/build_repair_prices.py --check   # exit 1 if the CSV on disk is out of date
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "data" / "reference" / "repair_prices.csv"

RETRIEVED_ON = "2026-10-10"
ROUND_TO_UZS = Decimal(1000)

# currency -> (UZS per unit, rate date, source). CBU "Rate" 11897.90, "Date" 10.10.2026.
FX: dict[str, tuple[Decimal, str, str]] = {
    "USD": (Decimal("11897.90"), "2026-10-10", "https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/2026-10-10/"),
}

SERVICE_CATEGORIES = ("paint", "dent", "polish", "scratch", "panel_replacement", "inspection", "promo", "unspecified")
EXCLUDE_FROM_ESTIMATES = ("inspection", "promo", "unspecified")
PRICE_KINDS = ("from", "flat", "promo")
SOURCE_TYPES = ("web_listing", "telegram_post", "phone", "dealer")
CURRENCIES = ("UZS", *FX)

COLUMNS = (
    "service", "service_category", "part", "car_class", "price_original", "currency",
    "fx_rate_uzs", "fx_date", "fx_source", "price_uzs", "price_kind", "source_url",
    "source_type", "listing_date", "retrieved_on", "is_estimate",
)

USTABOR = "https://auto.ustabor.uz/ru/price/body_work"
OLX_1YWDF = "https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-i-t-d-ID1YwdF.html"
OLX_22HSA = "https://www.olx.uz/d/obyavlenie/pokraska-kosmetika-polirovka-keramika-shumo-tsiya-kostoprav-ID22hsA.html"
OLX_4BVTP = "https://www.olx.uz/d/obyavlenie/polirovka-avto-kuzova-za-lyuboy-mashina-600000-sum-600-tsyach-sum-ID4bVtP.html"
OLX_3GODG = "https://www.olx.uz/d/obyavlenie/avto-vakuum-pachok-tugirlash-vakum-ID3goDG.html"
OLX_3VJJ4 = "https://www.olx.uz/d/obyavlenie/proverka-avto-po-kuzovu-i-probegu-po-nizkim-tsenam-vyezdnoy-osmotr-ID3Vjj4.html"
CENOTAVR = "https://www.cenotavr.uz/tashkent/remont-kuzova-pokraska-polirovka-kuzova-90"
BYD_TG = "https://t.me/byd_chigatoy"


@dataclass(frozen=True)
class PriceRow:
    service: str
    service_category: str
    part: str
    car_class: str
    price_original: str          # as quoted; a decimal string, no thousands separators
    currency: str
    price_kind: str
    source_url: str
    source_type: str
    listing_date: str = ""       # YYYY-MM-DD, or "" if not recorded
    retrieved_on: str = RETRIEVED_ON
    is_estimate: bool = False


# Observed rows, in the order of phase0_findings_deploy_market.md section 7.
ROWS: tuple[PriceRow, ...] = (
    PriceRow("painting", "paint", "any", "any", "295000", "UZS", "from", USTABOR, "web_listing"),
    PriceRow("dent straightening", "dent", "any", "any", "500000", "UZS", "from", USTABOR, "web_listing"),
    PriceRow("scratch removal", "scratch", "any", "any", "325000", "UZS", "from", USTABOR, "web_listing"),
    PriceRow("body panel replacement", "panel_replacement", "panel", "any", "1200000", "UZS", "from", USTABOR, "web_listing"),
    PriceRow("polishing", "polish", "any", "any", "1004167", "UZS", "from", USTABOR, "web_listing"),
    PriceRow("paint one panel", "paint", "panel", "any", "55", "USD", "from", OLX_1YWDF, "web_listing",
             listing_date="2026-09-25"),
    PriceRow("dent repair without paint", "dent", "any", "any", "88", "USD", "from", OLX_1YWDF, "web_listing",
             listing_date="2026-09-25"),
    PriceRow("paint one panel", "paint", "panel", "any", "77", "USD", "from", OLX_22HSA, "web_listing",
             listing_date="2026-10-03"),
    PriceRow("body polishing, any car (advertised as a promo)", "polish", "whole_body", "any", "600000", "UZS",
             "flat", OLX_4BVTP, "web_listing", listing_date="2026-10-06"),
    PriceRow("stone-chip dent repair", "dent", "any", "any", "100000", "UZS", "from", OLX_3GODG, "web_listing",
             listing_date="2026-10-10"),
    PriceRow("body repair / paint / polish advert (service not stated)", "unspecified", "any",
             "Nexia/Lacetti/Matiz/foreign", "200000", "UZS", "flat", CENOTAVR, "web_listing"),
    PriceRow("pre-purchase paint/body inspection", "inspection", "whole_body", "any", "300000", "UZS", "from",
             OLX_3VJJ4, "web_listing", listing_date="2026-10-04"),
    PriceRow("body polishing (dealer promo)", "promo", "whole_body", "any BYD", "2000000", "UZS", "promo",
             BYD_TG, "telegram_post", listing_date="2025-02-15"),
    PriceRow("wheel alignment (dealer promo; normally 200000 UZS)", "promo", "wheels", "BYD", "100000", "UZS", "promo",
             BYD_TG, "telegram_post", listing_date="2026-04-08"),
)

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_row(row: PriceRow) -> None:
    """Raise ValueError if a row breaks the schema rules in the module docstring."""
    problems = []
    if not row.service.strip():
        problems.append("empty service")
    if row.service_category not in SERVICE_CATEGORIES:
        problems.append(f"service_category {row.service_category!r}")
    if not row.part.strip() or not row.car_class.strip():
        problems.append("empty part or car_class")
    if row.currency not in CURRENCIES:
        problems.append(f"currency {row.currency!r} has no FX rate")
    if row.price_kind not in PRICE_KINDS:
        problems.append(f"price_kind {row.price_kind!r}")
    if row.source_type not in SOURCE_TYPES:
        problems.append(f"source_type {row.source_type!r}")
    if row.source_type == "phone":
        if not re.match(r"^phone: .+, \d{4}-\d{2}-\d{2}$", row.source_url):
            problems.append("phone rows need source_url 'phone: <shop name>, <YYYY-MM-DD>'")
    elif not row.source_url.startswith("https://"):
        problems.append("source_url must be an https URL")
    if row.listing_date and not _DATE.match(row.listing_date):
        problems.append(f"listing_date {row.listing_date!r}")
    if not _DATE.match(row.retrieved_on):
        problems.append(f"retrieved_on {row.retrieved_on!r}")
    try:
        price = Decimal(row.price_original)
    except ArithmeticError:
        problems.append(f"price_original {row.price_original!r} is not a number")
    else:
        if not price.is_finite() or price <= 0:
            problems.append(f"price_original {row.price_original!r} must be positive")
        elif row.currency == "UZS" and price != price.to_integral_value():
            problems.append("UZS prices must be whole numbers")
    if problems:
        raise ValueError(f"{row.service!r} ({row.source_url}): " + "; ".join(problems))


def price_uzs(row: PriceRow) -> int:
    """UZS rows unchanged; other currencies converted and rounded to the nearest 1,000 UZS (half up)."""
    price = Decimal(row.price_original)
    if row.currency == "UZS":
        return int(price)
    rate = FX[row.currency][0]
    thousands = (price * rate / ROUND_TO_UZS).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return int(thousands * ROUND_TO_UZS)


def to_record(row: PriceRow) -> dict[str, str]:
    validate_row(row)
    rate, fx_date, fx_source = ("", "", "")
    if row.currency != "UZS":
        r, fx_date, fx_source = FX[row.currency]
        rate = f"{r:.2f}"
    return {
        "service": row.service,
        "service_category": row.service_category,
        "part": row.part,
        "car_class": row.car_class,
        "price_original": row.price_original,
        "currency": row.currency,
        "fx_rate_uzs": rate,
        "fx_date": fx_date,
        "fx_source": fx_source,
        "price_uzs": str(price_uzs(row)),
        "price_kind": row.price_kind,
        "source_url": row.source_url,
        "source_type": row.source_type,
        "listing_date": row.listing_date,
        "retrieved_on": row.retrieved_on,
        "is_estimate": "true" if row.is_estimate else "false",
    }


def render_csv(rows: tuple[PriceRow, ...] = ROWS) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(to_record(row))
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_CSV, help="output CSV (default: %(default)s)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if --out differs from the rebuilt CSV")
    args = ap.parse_args(argv)

    text = render_csv()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if current != text:
            print(f"{args.out} is out of date: run python scripts/build_repair_prices.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date ({len(ROWS)} rows)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="")
    n_conv = sum(1 for r in ROWS if r.currency != "UZS")
    n_excl = sum(1 for r in ROWS if r.service_category in EXCLUDE_FROM_ESTIMATES)
    print(f"wrote {args.out} ({len(ROWS)} rows; {n_conv} converted from USD; "
          f"{n_excl} {'/'.join(EXCLUDE_FROM_ESTIMATES)} rows for the estimator to exclude)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
