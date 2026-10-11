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
LANGS = ("en", "ru", "uz")

# PDF text in English / Russian / Uzbek (Latin); same wording as web/i18n.js.
_T: dict[str, tuple[str, str, str]] = {
    "title": (TITLE, "AvtoPassport — паспорт состояния автомобиля", "AvtoPassport — avtomobil holati pasporti"),
    "status": ("status", "статус", "holat"),
    "vehicle": ("Vehicle", "Автомобиль", "Avtomobil"),
    "plant": ("plant/country", "завод/страна", "zavod/mamlakat"),
    "not_decoded": ("VIN not decoded", "VIN не расшифрован", "VIN aniqlanmadi"),
    "mm_not_decoded": ("make/model not decoded", "марка/модель не определены", "marka/model aniqlanmadi"),
    "not_read": ("not read", "не прочитан", "o‘qilmadi"),
    "time": ("Inspection time (UTC)", "Время осмотра (UTC)", "Ko‘rik vaqti (UTC)"),
    "generated": ("Report generated (UTC)", "Отчёт создан (UTC)", "Hisobot yaratildi (UTC)"),
    "frames_sha": ("Frames SHA-256:", "SHA-256 кадров:", "Kadrlar SHA-256:"),
    "hash_method": ("Hash method", "Метод хеша", "Xesh usuli"),
    "findings": ("Damage findings", "Найденные повреждения", "Topilgan shikastlar"),
    "cols": ("Part|Damage|Severity|Action|Cost", "Деталь|Повреждение|Степень|Действие|Цена",
             "Detal|Shikast|Darajasi|Harakat|Narx"),
    "no_damage": ("No damage detected.", "Повреждений не найдено.", "Shikast topilmadi."),
    "from": ("from {v} UZS", "от {v} сум", "{v} so‘mdan"),
    "no_price": ("no price", "нет цены", "narx yo‘q"),
    "total": ("Total: from {v} UZS (sum of priced lines)", "Итого: от {v} сум (сумма строк с ценой)",
              "Jami: {v} so‘mdan (narxli qatorlar yig‘indisi)"),
    "total_none": ("Total: no price available", "Итого: цены нет", "Jami: narx yo‘q"),
    "basis": ("Price basis", "Основа цены", "Narx asosi"),
    "changes": ("Changes since previous inspection", "Изменения с прошлого осмотра", "Oldingi ko‘rikdan beri o‘zgarishlar"),
    "no_cmp": ("No comparison (no previous inspection data supplied).", "Нет сравнения (нет данных прошлого осмотра).",
               "Solishtirish yo‘q (oldingi ko‘rik ma’lumoti yo‘q)."),
    "first": ("First AvtoPassport inspection for this VIN - no earlier inspection to compare.",
              "Первый осмотр этого VIN в AvtoPassport — сравнивать не с чем.",
              "Bu VIN uchun AvtoPassportdagi birinchi ko‘rik — solishtiradigan narsa yo‘q."),
    "since": ("Since the previous inspection", "С прошлого осмотра", "Oldingi ko‘rikdan beri"),
    "new": ("new", "новые", "yangi"), "worse": ("worse", "хуже", "yomonlashgan"),
    "unchanged": ("unchanged", "без изменений", "o‘zgarmagan"), "improved": ("improved", "лучше", "yaxshilangan"),
    "resolved": ("resolved", "устранено", "bartaraf etilgan"),
    "notes": ("Notes", "Примечания", "Izohlar"),
    "frames": ("Frames (privacy-blurred)", "Кадры (номера и лица размыты)", "Kadrlar (raqam va yuzlar xiralashtirilgan)"),
    "skipped": ("Image skipped", "Изображение пропущено", "Rasm o‘tkazib yuborildi"),
    "foot_src": (FOOTER_SOURCE, "Исходный код: AGPL-3.0, см. репозиторий", "Manba kodi: AGPL-3.0, repozitoriyga qarang"),
    "foot_prices": (FOOTER_PRICES, "Цены: открытые объявления (data/reference/repair_prices.csv); суммы — цены «от»",
                    "Narxlar: ochiq e’lonlar (data/reference/repair_prices.csv); summalar — boshlang‘ich narxlar"),
    "page": ("page {n}", "стр. {n}", "{n}-sahifa"),
}
_WORDS: dict[str, tuple[str, str]] = {  # English value -> (ru, uz)
    "dent": ("вмятина", "botiq"), "scratch": ("царапина", "tirnalish"), "crack": ("трещина", "yoriq"),
    "glass shatter": ("разбитое стекло", "singan oyna"), "lamp broken": ("разбитая фара", "singan chiroq"),
    "tire flat": ("спущенная шина", "teshilgan shina"),
    "minor": ("лёгкое", "yengil"), "moderate": ("среднее", "o‘rtacha"), "severe": ("сильное", "jiddiy"),
    "polish": ("полировка", "silliqlash"), "paint": ("покраска", "bo‘yash"), "repair": ("ремонт", "ta’mirlash"),
    "replace": ("замена", "almashtirish"),
    "read": ("прочитан", "o‘qildi"), "manual": ("введён вручную", "qo‘lda kiritildi"),
    "ambiguous": ("неточно", "noaniq"), "could_not_read": ("не прочитан", "o‘qilmadi"),
}
_PARTS: dict[str, tuple[str, str, str]] = {  # base part -> (ru, gender, uz)
    "front bumper": ("передний бампер", "m", "old bamper"), "back bumper": ("задний бампер", "m", "orqa bamper"),
    "grille": ("решётка радиатора", "f", "radiator panjarasi"), "hood": ("капот", "m", "kapot"),
    "roof": ("крыша", "f", "tom"), "trunk": ("крышка багажника", "f", "yukxona qopqog‘i"),
    "windshield": ("лобовое стекло", "n", "old oyna"), "back windshield": ("заднее стекло", "n", "orqa oyna"),
    "license plate": ("номерной знак", "m", "davlat raqami"), "back door": ("задняя дверь", "f", "orqa eshik"),
    "front door": ("передняя дверь", "f", "old eshik"), "back wheel": ("заднее колесо", "n", "orqa g‘ildirak"),
    "front wheel": ("переднее колесо", "n", "old g‘ildirak"),
    "back window": ("стекло задней двери", "n", "orqa eshik oynasi"),
    "front window": ("стекло передней двери", "n", "old eshik oynasi"), "fender": ("переднее крыло", "n", "old qanot"),
    "quarter panel": ("заднее крыло", "n", "orqa qanot"), "rocker panel": ("порог", "m", "ostona"),
    "headlight": ("фара", "f", "fara"), "tail light": ("задний фонарь", "m", "orqa chiroq"),
    "mirror": ("боковое зеркало", "n", "yon oyna"), "unknown": ("неизвестная деталь", "f", "noma’lum detal"),
}
_SIDE_RU = {"left": {"m": "левый", "f": "левая", "n": "левое"}, "right": {"m": "правый", "f": "правая", "n": "правое"}}
_SIDE_UZ = {"left": "chap", "right": "o‘ng"}


def _lang(lang: str | None) -> int:
    return LANGS.index(lang) if lang in LANGS else 0


def _t(key: str, lang: str | None, **kw: Any) -> str:
    return _T[key][_lang(lang)].format(**kw)


def _word(value: Any, lang: str | None) -> str:
    text = str(value) if value is not None else "-"
    i = _lang(lang)
    return _WORDS[text][i - 1] if i and text in _WORDS else text


def part_name(part: Any, lang: str | None) -> str:
    """'left_back-door' -> 'left back door' / 'Левая задняя дверь' / 'Chap orqa eshik'."""
    text = " ".join(str(part or "-").replace("_", " ").replace("-", " ").split())
    i = _lang(lang)
    side, _, rest = text.partition(" ")
    side, base = (side, rest) if side in ("left", "right") and rest else (None, text)
    p = _PARTS.get(base)
    if not i or p is None:
        return text
    name = (f"{_SIDE_RU[side][p[1]]} {p[0]}" if side else p[0]) if i == 1 else (f"{_SIDE_UZ[side]} {p[2]}" if side else p[2])
    return name[0].upper() + name[1:]


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


def _cost_text(line: dict[str, Any] | None, lang: str | None = "en") -> str:
    if not line or line.get("price_uzs") is None:
        return _t("no_price", lang)
    kind = line.get("price_kind") or ""
    if kind == "from":
        return _t("from", lang, v=_uzs(int(line["price_uzs"])))
    return f"{_uzs(int(line['price_uzs']))} UZS ({kind})" if kind else f"{_uzs(int(line['price_uzs']))} UZS"


def _decode_summary(decode: dict[str, Any] | None, lang: str | None = "en") -> str:
    if not decode:
        return _t("not_decoded", lang)
    year = decode.get("model_year")
    if year is None and decode.get("model_year_candidates"):
        year = "/".join(str(y) for y in decode["model_year_candidates"])
    parts = [decode.get("make") or decode.get("manufacturer"), decode.get("model"), year]
    plant = ", ".join(str(p) for p in (decode.get("plant_city"), decode.get("plant_country") or decode.get("country"))
                      if p)
    text = " ".join(str(p) for p in parts if p) or _t("mm_not_decoded", lang)
    return f"{text}; {_t('plant', lang)}: {plant}" if plant else text


def _diff_lines(diff: dict[str, Any] | None, lang: str | None = "en") -> list[str]:
    if diff is None:
        return [_t("no_cmp", lang)]
    if diff.get("first_inspection"):
        return [_t("first", lang)]
    keys = ("new", "worse", "unchanged", "improved", "resolved")
    counts = ", ".join(f"{_t(k, lang)}: {len(diff.get(k) or [])}" for k in keys)
    lines = [f"{_t('since', lang)} - {counts}."]
    for key in ("new", "worse", "resolved"):
        label = _t(key, lang).capitalize()
        for it in diff.get(key) or []:
            sev = _word(it.get("severity_after") or it.get("severity_before"), lang)
            if key == "worse":
                sev = f"{_word(it.get('severity_before'), lang)} -> {_word(it.get('severity_after'), lang)}"
            lines.append(f"{label}: {part_name(it.get('part'), lang)} - {_word(it.get('damage'), lang)} ({sev})")
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


def build_pdf(report: dict[str, Any], out_path: Path, *, images: Sequence[Path] = (), lang: str = "en") -> Path:
    """Render a build_json() report to PDF in lang (en / ru / uz). ``images`` must be blurred frames (max 4 used)."""
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
    story: list[Any] = [para(_t("title", lang), title)]
    if font_note:
        story.append(para(font_note, small))

    vin = result.get("vin") or f"{_t('not_read', lang)} ({_word(result.get('vin_status', 'unknown'), lang)})"
    story += [
        para(f"VIN: {vin}  ({_t('status', lang)}: {_word(result.get('vin_status', '-'), lang)})", head),
        para(f"{_t('vehicle', lang)}: {_decode_summary(decode, lang)}"),
        para(f"{_t('time', lang)}: {result.get('created_utc') or '-'}"),
        para(f"{_t('generated', lang)}: {report.get('generated_utc') or '-'}"),
        para(_t("frames_sha", lang), base),
        para(result.get("frames_sha256") or hash_info.get("frames_sha256") or "-",
             ParagraphStyle("mono", parent=small, fontName=regular)),
        para(f"{_t('hash_method', lang)}: {hash_info.get('method') or HASH_METHOD}", small),
    ]

    # Findings table with costs matched by (part, damage).
    costs = {(c.get("part"), c.get("damage")): c for c in result.get("costs") or []}
    findings = result.get("findings") or []
    story.append(para(_t("findings", lang), head))
    if findings:
        rows: list[list[Any]] = [[para(h, cell_b) for h in _t("cols", lang).split("|")]]
        for f in findings:
            line = costs.get((f.get("part"), f.get("damage")))
            rows.append([para(part_name(f.get("part"), lang), small), para(_word(f.get("damage"), lang), small),
                         para(_word(f.get("severity"), lang), small), para(_word(f.get("action"), lang), small),
                         para(_cost_text(line, lang), small)])
        table = Table(rows, colWidths=[42 * mm, 32 * mm, 24 * mm, 22 * mm, 54 * mm], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    else:
        story.append(para(_t("no_damage", lang)))
    total = result.get("total_from_uzs")
    story.append(Spacer(1, 4))
    story.append(para(_t("total", lang, v=_uzs(int(total))) if total is not None else _t("total_none", lang),
                      ParagraphStyle("tot", parent=base, fontName=bold)))
    bases = sorted({c.get("basis") for c in result.get("costs") or [] if c.get("basis")})
    for b in bases:
        story.append(para(f"{_t('basis', lang)}: {b}", small))

    story.append(para(_t("changes", lang), head))
    story += [para(t) for t in _diff_lines(report.get("diff"), lang)]

    notes = result.get("notes") or []
    if notes:
        story.append(para(_t("notes", lang), head))
        story += [para(f"- {n}", small) for n in notes]

    thumbs, skipped = _thumbnails(images, 41 * mm, 41 * mm)
    if thumbs or skipped:
        story.append(para(_t("frames", lang), head))
    if thumbs:
        story.append(Table([thumbs], colWidths=[43.5 * mm] * len(thumbs), hAlign="LEFT"))
    for s in skipped:
        story.append(para(f"{_t('skipped', lang)}: {s}", small))

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 12 * mm, _t("foot_src", lang))
        canvas.drawString(doc.leftMargin, 8.5 * mm, _t("foot_prices", lang))
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, _t("page", lang, n=doc.page))
        canvas.restoreState()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm, title=_t("title", lang), author="AvtoPassport")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path
