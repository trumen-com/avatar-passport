"""Passport report: frame hashes, JSON report and PDF (reportlab).

Frame hash (reproducible with sha256sum + sort): for each frame file, SHA-256 of its bytes; lines
"<file name>  <sha256 hex>" sorted by file name (then hash) and joined with "\\n" (no trailing newline);
frames_sha256 = SHA-256 of that UTF-8 text.

The PDF uses DejaVuSans (env AVTO_FONT overrides the path) so Cyrillic renders; if the font file is missing
it falls back to Helvetica and says so in the PDF. Callers pass blurred frames only - this module never
reads raw media on its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from ..schema import InspectionResult
from .store import json_default

log = logging.getLogger("avtopassport.passport.report")

SCHEMA_VERSION = "1"
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
TITLE = "AvtoPassport - vehicle condition passport"
FOOTER_SOURCE = "Source: AGPL-3.0, see repository"
FOOTER_PRICES = "Prices: public listings (see data/reference/repair_prices.csv); amounts are 'from' prices"
HASH_METHOD = (
    "SHA-256 of each frame file's bytes; lines '<file name>  <sha256 hex>' sorted by file name, joined with "
    "'\\n' (no trailing newline); frames_sha256 = SHA-256 of that UTF-8 text"
)
MAX_IMAGES = 4


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frames(paths: Sequence[Path]) -> tuple[str, list[dict[str, str]]]:
    """(combined SHA-256, per-file [{"name", "sha256"}] sorted by name) - see HASH_METHOD."""
    files = sorted(({"name": Path(p).name, "sha256": _sha256_file(Path(p))} for p in paths),
                   key=lambda d: (d["name"], d["sha256"]))
    text = "\n".join(f"{d['name']}  {d['sha256']}" for d in files)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), files


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=json_default, ensure_ascii=False))


def build_json(result: InspectionResult, *, diff: dict[str, Any] | None = None, decode: Any = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-safe passport report (decode may be a dict or a VinDecode dataclass)."""
    if is_dataclass(decode) and not isinstance(decode, type):
        decode = asdict(decode)
    return _jsonable({
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "result": result.to_dict(),
        "diff": diff,
        "decode": decode,
        "hash": {"algorithm": "SHA-256", "frames_sha256": result.frames_sha256, "method": HASH_METHOD},
        "prices_note": FOOTER_PRICES,
        "extra": extra or {},
    })


# ---------------------------------------------------------------------------------------------- PDF


def _register_fonts() -> tuple[str, str, str | None]:
    """(regular, bold, note) font names; DejaVu (or AVTO_FONT) when available, else Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    path = Path(os.environ.get("AVTO_FONT") or DEFAULT_FONT)
    if not path.is_file():
        note = f"Font {path} not found; using Helvetica (Cyrillic text may not render)."
        log.warning(note)
        return "Helvetica", "Helvetica-Bold", note
    try:
        regular = "AvtoSans"
        if regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(regular, str(path)))
        bold = regular
        bold_path = path.with_name(path.stem + "-Bold" + path.suffix)
        if bold_path.is_file():
            bold = "AvtoSans-Bold"
            if bold not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold, str(bold_path)))
        return regular, bold, None
    except Exception as exc:  # unreadable/invalid font file
        note = f"Font {path} could not be loaded ({exc}); using Helvetica (Cyrillic text may not render)."
        log.warning(note)
        return "Helvetica", "Helvetica-Bold", note


def _uzs(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _cost_text(line: dict[str, Any] | None) -> str:
    if not line or line.get("price_uzs") is None:
        return "no price"
    kind = line.get("price_kind") or ""
    if kind == "from":
        return f"from {_uzs(int(line['price_uzs']))} UZS"
    return f"{_uzs(int(line['price_uzs']))} UZS ({kind})" if kind else f"{_uzs(int(line['price_uzs']))} UZS"


def _decode_summary(decode: dict[str, Any] | None) -> str:
    if not decode:
        return "VIN not decoded"
    year = decode.get("model_year")
    if year is None and decode.get("model_year_candidates"):
        year = "/".join(str(y) for y in decode["model_year_candidates"])
    parts = [decode.get("make") or decode.get("manufacturer"), decode.get("model"), year]
    plant = ", ".join(str(p) for p in (decode.get("plant_city"), decode.get("plant_country") or decode.get("country"))
                      if p)
    text = " ".join(str(p) for p in parts if p) or "make/model not decoded"
    return f"{text}; plant/country: {plant}" if plant else text


def _diff_lines(diff: dict[str, Any] | None) -> list[str]:
    if diff is None:
        return ["No comparison (no previous inspection data supplied)."]
    if diff.get("first_inspection"):
        return ["First AvtoPassport inspection for this VIN - no earlier inspection to compare."]
    counts = ", ".join(f"{k}: {len(diff.get(k) or [])}" for k in ("new", "worse", "unchanged", "improved", "resolved"))
    lines = [f"Since the previous inspection - {counts}."]
    for key, label in (("new", "New"), ("worse", "Worse"), ("resolved", "Resolved")):
        for it in diff.get(key) or []:
            sev = it.get("severity_after") or it.get("severity_before") or "-"
            if key == "worse":
                sev = f"{it.get('severity_before')} -> {it.get('severity_after')}"
            lines.append(f"{label}: {it.get('part')} - {it.get('damage')} ({sev})")
    return lines


def _thumbnails(images: Sequence[Path], max_w: float, max_h: float) -> tuple[list[Any], list[str]]:
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image

    out: list[Any] = []
    skipped: list[str] = []
    for p in list(images)[:MAX_IMAGES]:
        try:
            w, h = ImageReader(str(p)).getSize()
            scale = min(max_w / w, max_h / h)
            out.append(Image(str(p), width=w * scale, height=h * scale))
        except Exception as exc:
            skipped.append(f"{Path(p).name}: {exc}")
    return out, skipped


def build_pdf(report: dict[str, Any], out_path: Path, *, images: Sequence[Path] = ()) -> Path:
    """Render a build_json() report to PDF. ``images`` must be blurred/processed frames (max 4 used)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold, font_note = _register_fonts()
    base = ParagraphStyle("base", fontName=regular, fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=base, fontSize=7.5, leading=9.5)
    head = ParagraphStyle("head", parent=base, fontName=bold, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3)
    title = ParagraphStyle("title", parent=base, fontName=bold, fontSize=16, leading=20, spaceAfter=6)
    cell_b = ParagraphStyle("cell_b", parent=small, fontName=bold)

    def para(text: Any, style: ParagraphStyle = base) -> Paragraph:
        return Paragraph(escape(str(text)), style)

    result: dict[str, Any] = report.get("result") or {}
    decode = report.get("decode")
    hash_info: dict[str, Any] = report.get("hash") or {}
    story: list[Any] = [para(TITLE, title)]
    if font_note:
        story.append(para(font_note, small))

    vin = result.get("vin") or f"not read ({result.get('vin_status', 'unknown')})"
    story += [
        para(f"VIN: {vin}  (status: {result.get('vin_status', '-')})", head),
        para(f"Vehicle: {_decode_summary(decode)}"),
        para(f"Inspection time (UTC): {result.get('created_utc') or '-'}"),
        para(f"Report generated (UTC): {report.get('generated_utc') or '-'}"),
        para("Frames SHA-256:", base),
        para(result.get("frames_sha256") or hash_info.get("frames_sha256") or "-",
             ParagraphStyle("mono", parent=small, fontName=regular)),
        para(f"Hash method: {hash_info.get('method') or HASH_METHOD}", small),
    ]

    # Findings table with costs matched by (part, damage).
    costs = {(c.get("part"), c.get("damage")): c for c in result.get("costs") or []}
    findings = result.get("findings") or []
    story.append(para("Damage findings", head))
    if findings:
        rows: list[list[Any]] = [[para(h, cell_b) for h in ("Part", "Damage", "Severity", "Action", "Cost")]]
        for f in findings:
            line = costs.get((f.get("part"), f.get("damage")))
            rows.append([para(f.get("part") or "-", small), para(f.get("damage") or "-", small),
                         para(f.get("severity") or "-", small), para(f.get("action") or "-", small),
                         para(_cost_text(line), small)])
        table = Table(rows, colWidths=[42 * mm, 32 * mm, 24 * mm, 22 * mm, 54 * mm], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    else:
        story.append(para("No damage detected."))
    total = result.get("total_from_uzs")
    story.append(Spacer(1, 4))
    story.append(para(f"Total: from {_uzs(int(total))} UZS (sum of priced lines)" if total is not None
                      else "Total: no price available", ParagraphStyle("tot", parent=base, fontName=bold)))
    bases = sorted({c.get("basis") for c in result.get("costs") or [] if c.get("basis")})
    for b in bases:
        story.append(para(f"Price basis: {b}", small))

    story.append(para("Changes since previous inspection", head))
    story += [para(t) for t in _diff_lines(report.get("diff"))]

    notes = result.get("notes") or []
    if notes:
        story.append(para("Notes", head))
        story += [para(f"- {n}", small) for n in notes]

    thumbs, skipped = _thumbnails(images, 41 * mm, 41 * mm)
    if thumbs or skipped:
        story.append(para("Frames (privacy-blurred)", head))
    if thumbs:
        story.append(Table([thumbs], colWidths=[43.5 * mm] * len(thumbs), hAlign="LEFT"))
    for s in skipped:
        story.append(para(f"Image skipped: {s}", small))

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 12 * mm, FOOTER_SOURCE)
        canvas.drawString(doc.leftMargin, 8.5 * mm, FOOTER_PRICES)
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, f"page {doc.page}")
        canvas.restoreState()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm, title=TITLE, author="AvtoPassport")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path
