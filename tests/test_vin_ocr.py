"""VIN OCR: real RapidOCR (CPU, bundled models, offline) on cv2.putText renders, plus a scripted
stand-in engine for the deterministic window / ranking / status rules. Synthetic inputs only."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import cv2
import numpy as np
import pytest
import requests

from avtopassport.vin.checkdigit import compute_check_digit
from avtopassport.vin.ocr import DET_MAX_RATIO, REASON_PREFIX, VinOcrResult, _detect, make_engine, read_vin

VIN = "1HGCM82633A004352"   # public sample VIN, check digit verified (NA: mandatory)
# Published 11-character prefixes + synthetic serials (no real vehicle's full VIN): no check digit rule.
EU_A, EU_B = "WP0ZZZ99ZTS000000", "VF1LM1B0H36000000"
RAPIDOCR_MODELS = ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx", "ch_ppocr_mobile_v2.0_cls_mobile.onnx")


def render(lines: list[str], width: int = 1800, height: int = 300, scale: float = 3.0,
           thickness: int = 6) -> np.ndarray:
    """Black Hershey text, one centred line per entry, on a white BGR canvas."""
    img = np.full((height, width, 3), 255, np.uint8)
    step = height // (len(lines) + 1)
    for i, text in enumerate(lines, start=1):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(img, text, ((width - tw) // 2, step * i + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def rapidocr_skip_reason() -> str | None:
    try:
        import onnxruntime  # noqa: F401
        import rapidocr
    except ImportError as e:
        return f"rapidocr/onnxruntime not installed ({e})"
    folder = Path(rapidocr.__file__).parent / "models"
    missing = [name for name in RAPIDOCR_MODELS if not (folder / name).is_file()]
    return f"rapidocr models not bundled, a download would be needed: {missing}" if missing else None


@pytest.fixture(scope="module")
def engine() -> Iterator[Any]:
    reason = rapidocr_skip_reason()
    if reason:
        pytest.skip(reason)

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access in a unit test (rapidocr model download?)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(requests, "get", no_network)
        yield make_engine()


def reason_of(result: VinOcrResult) -> str:
    reasons = [text for text, _ in result.raw_lines if text.startswith(REASON_PREFIX)]
    assert reasons, result.raw_lines
    return reasons[-1]


# --- real RapidOCR ----------------------------------------------------------------------------

def test_reads_rendered_vin(engine: Any) -> None:
    flags = (engine.use_det, engine.use_cls, engine.use_rec)
    r = read_vin(render([VIN]), engine=engine)
    assert r.status == "read"
    assert r.best is not None
    assert r.best.validation.vin == VIN
    assert r.best.validation.status == "valid"
    assert r.best.source == "rot0/line1"
    assert r.candidates[0] is r.best
    assert r.timings_ms["det"] > 0 and r.timings_ms["rec"] > 0 and r.timings_ms["total"] > 0
    assert (engine.use_det, engine.use_cls, engine.use_rec) == flags   # engine state restored


def test_reads_label_and_known_wmi(engine: Any) -> None:
    r = read_vin(render(["VIN: XWB3L32EDKA123456"]), engine=engine)   # made-up VIN, real WMI
    assert r.status == "read"
    assert r.best is not None and r.best.validation.vin == "XWB3L32EDKA123456"
    assert r.best.validation.status == "valid_unverified"              # Uzbek VINs: no check digit rule


def test_reads_vertical_vin_via_rotation(engine: Any) -> None:
    r = read_vin(cv2.rotate(render([VIN]), cv2.ROTATE_90_CLOCKWISE), engine=engine)
    assert r.status == "read"
    assert r.best is not None and r.best.validation.vin == VIN


def test_blank_image_could_not_read(engine: Any) -> None:
    r = read_vin(np.full((300, 1800, 3), 255, np.uint8), engine=engine)
    assert r.status == "could_not_read"
    assert r.best is None and r.candidates == []
    assert "no text" in reason_of(r)


# --- input contract / failures (no model needed) ------------------------------------------------

class ExplodingEngine:
    use_det = use_cls = use_rec = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("engine exploded")


@pytest.mark.parametrize("image", [
    None,
    "not an image",
    np.zeros((100, 100, 3), np.float32),
    np.zeros((0, 0, 3), np.uint8),
    np.zeros((4, 4, 3), np.uint8),
    np.zeros((100, 100, 5), np.uint8),
    np.zeros(100, np.uint8),
])
def test_bad_images_never_raise(image: Any) -> None:
    r = read_vin(image, engine=ExplodingEngine())
    assert r.status == "could_not_read"
    assert r.best is None and r.candidates == []
    assert reason_of(r)


def test_engine_failure_never_raises() -> None:
    r = read_vin(np.full((200, 800, 3), 255, np.uint8), engine=ExplodingEngine())
    assert r.status == "could_not_read"
    assert "OCR failed" in reason_of(r)


class RecordingDetEngine:
    """Records each detection input's size and returns one vertical box in the middle of that input."""
    use_det = use_cls = use_rec = True
    max_side_len = 2000

    def __init__(self) -> None:
        self.shapes: list[tuple[int, int]] = []

    def __call__(self, img: np.ndarray, **flags: Any) -> Any:
        h, w = img.shape[:2]
        self.shapes.append((h, w))
        box = [[w / 2 - 10, 100], [w / 2 + 10, 100], [w / 2 + 10, 900], [w / 2 - 10, 900]]
        return SimpleNamespace(boxes=np.array([box], np.float32))


def test_tall_thin_detection_input_is_padded() -> None:
    # Regression: rapidocr pads only WIDE inputs before detection and its detector scales the short side up
    # to 736 px, so a VIN strip turned 90 degrees (2000x20 after downscaling) became ~70000x736 px (>6 GB).
    eng = RecordingDetEngine()
    quads = _detect(eng, np.full((4000, 40, 3), 255, np.uint8))     # downscaled to 2000x20, then padded
    (h, w), = eng.shapes
    assert h == 2000 and h <= DET_MAX_RATIO * w
    # the box is mapped back through the padding and the downscale to full-resolution pixels
    assert np.allclose(quads[0], [[0, 200], [40, 200], [40, 1800], [0, 1800]])
    eng = RecordingDetEngine()
    _detect(eng, np.full((300, 1800, 3), 255, np.uint8))             # ordinary inputs are left alone
    assert eng.shapes == [(300, 1800)]


class FlagStoringEngine:
    """Mimics rapidocr 3.10: a call stores its stage flags on the engine, works for a moment, then runs the
    stages the ENGINE's flags say. Counts calls that ran under another call's flags."""

    def __init__(self) -> None:
        self.use_det = self.use_cls = self.use_rec = True
        self.mixed = 0

    def __call__(self, img: np.ndarray, use_det: bool, use_cls: bool, use_rec: bool) -> Any:
        self.use_det, self.use_cls, self.use_rec = use_det, use_cls, use_rec
        time.sleep(0.002)
        if (self.use_det, self.use_cls, self.use_rec) != (use_det, use_cls, use_rec):
            self.mixed += 1
        if self.use_det:
            h, w = img.shape[:2]
            return SimpleNamespace(boxes=np.array([[[10, h / 2 - 12], [w - 10, h / 2 - 12], [w - 10, h / 2 + 12],
                                                    [10, h / 2 + 12]]], np.float32))
        return SimpleNamespace(txts=("XWB3L32EDKA123456",), scores=(0.9,))


def test_concurrent_reads_on_one_engine_keep_its_flags() -> None:
    # Regression: threads sharing an engine interleaved their flag save/restore; the engine was left with
    # use_cls = use_rec = False and calls could run each other's stages.
    eng = FlagStoringEngine()
    img = np.full((200, 800, 3), 255, np.uint8)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: read_vin(img, engine=eng, known_wmis=["xwb"]), range(8)))
    assert eng.mixed == 0
    assert (eng.use_det, eng.use_cls, eng.use_rec) == (True, True, True)
    assert all(r.status == "read" and r.best is not None and r.best.validation.vin == "XWB3L32EDKA123456"
               for r in results)


# --- window / ranking / status rules with a scripted engine ----------------------------------------

class ScriptedEngine:
    """Stands in for RapidOCR: the first detection pass returns one box per scripted line (top to
    bottom); later passes see nothing. read_vin recognises each crop upright, then turned 180 degrees:
    the upright call returns the next scripted line, the turned call returns nothing."""

    def __init__(self, lines: list[tuple[str, float]]) -> None:
        self.lines = lines
        self.det_calls = 0
        self.rec_calls = 0
        self.use_det = self.use_cls = self.use_rec = True

    def __call__(self, img: np.ndarray, use_det: bool | None = None, use_cls: bool | None = None,
                 use_rec: bool | None = None) -> Any:
        if use_det:
            self.det_calls += 1
            if self.det_calls > 1:
                return SimpleNamespace(boxes=None)
            h, w = img.shape[:2]
            step = h / (len(self.lines) + 1)
            boxes = [[[10, step * i - 12], [w - 10, step * i - 12], [w - 10, step * i + 12], [10, step * i + 12]]
                     for i in range(1, len(self.lines) + 1)]
            return SimpleNamespace(boxes=np.array(boxes, np.float32))
        call, self.rec_calls = self.rec_calls, self.rec_calls + 1
        if call % 2:
            return SimpleNamespace(txts=("",), scores=(0.0,))
        text, score = self.lines[call // 2]
        return SimpleNamespace(txts=(text,), scores=(score,))


def scripted(lines: list[tuple[str, float]], **kwargs: Any) -> VinOcrResult:
    eng = ScriptedEngine(lines)
    result = read_vin(np.full((400, 1200, 3), 255, np.uint8), engine=eng, **kwargs)
    assert (eng.use_det, eng.use_cls, eng.use_rec) == (True, True, True)
    return result


def valid_china_vin() -> str:
    body = "LGXCE4CB0N0123456"   # made-up, real BYD WMI; position 9 set to the computed check digit
    return body[:8] + compute_check_digit(body) + body[9:]


def test_single_plausible_window_is_read() -> None:
    r = scripted([(EU_A, 0.99)])
    assert r.status == "read"
    assert r.best is not None and r.best.validation.status == "valid_unverified"
    assert r.best.text == EU_A
    assert r.raw_lines == [(EU_A, 0.99)]


def test_whole_token_beats_windows_through_a_label() -> None:
    r = scripted([("VIN " + EU_A, 0.99)])
    assert r.status == "read"
    assert r.best is not None and r.best.validation.vin == EU_A
    assert len(r.candidates) > 1                       # the other windows are still listed


def test_overlong_token_is_not_trusted() -> None:
    r = scripted([(EU_A + "5", 0.99)])                 # 18 chars: which 17 is the VIN? Not guessed.
    assert r.status == "could_not_read"
    assert r.best is None
    assert {c.validation.vin for c in r.candidates} == {EU_A, EU_A[1:] + "5"}


def test_two_whole_tokens_are_ambiguous() -> None:
    r = scripted([(EU_A, 0.99), (EU_B, 0.98)])
    assert r.status == "ambiguous"
    assert r.best is not None and r.best.validation.vin == EU_A    # higher OCR score
    assert [c.validation.vin for c in r.candidates[:2]] == [EU_A, EU_B]


def test_window_cut_across_words_is_not_a_reading() -> None:
    # "...UZBEKISTAN" + "MAX WEIGHT 1690" stitches into the VIN-shaped "STANMAXWE1GHT1690".
    r = scripted([("MADE IN UZBEKISTAN", 0.97), ("MAX WEIGHT 1690 KG", 0.99)], known_wmis=frozenset())
    assert r.status == "could_not_read"
    assert r.best is None


def test_two_strong_vins_are_ambiguous() -> None:
    china = valid_china_vin()
    r = scripted([("XWB3L32EDKA123456", 0.95), (china, 0.97)], known_wmis=frozenset({"XWB", "LGX"}))
    assert r.status == "ambiguous"
    assert {c.validation.vin for c in r.candidates} >= {"XWB3L32EDKA123456", china}


def test_known_wmi_beats_junk_windows() -> None:
    r = scripted([("GM UZBEKISTAN", 0.99), ("VIN XWB3L32EDKA123456", 0.98), ("MAX WEIGHT 1690 KG 2021", 0.99)],
                 known_wmis=frozenset({"XWB"}))
    assert r.status == "read"
    assert r.best is not None and r.best.validation.vin == "XWB3L32EDKA123456"
    assert r.best.source == "rot0/line2"


def test_check_digit_needs_a_token_boundary() -> None:
    # A 17-char slice from the middle of a longer token passes a check digit 1 time in 11 by chance,
    # so it is not trusted on its own: listed as a candidate, but not read.
    china = valid_china_vin()
    r = scripted([("XX" + china + "XX", 0.99)], known_wmis=frozenset())
    assert r.status == "could_not_read"
    assert china in {c.validation.vin for c in r.candidates}
    r2 = scripted([(china, 0.99)], known_wmis=frozenset())
    assert r2.status == "read"
    assert r2.best is not None and r2.best.validation.vin == china and r2.best.validation.status == "valid"


def test_vin_split_over_two_lines_is_stitched() -> None:
    r = scripted([("XWB3L32ED", 0.99), ("KA123456", 0.97)], known_wmis=frozenset({"XWB"}))
    assert r.status == "read"
    assert r.best is not None and r.best.validation.vin == "XWB3L32EDKA123456"
    assert r.best.source == "rot0/line1+2"
    assert r.best.score == pytest.approx((9 * 0.99 + 8 * 0.97) / 17)


def test_text_without_vin_could_not_read() -> None:
    r = scripted([("GM UZBEKISTAN", 0.99), ("MADE IN UZBEKISTAN", 0.99)])
    assert r.status == "could_not_read"
    assert r.candidates == []                          # junk windows are not listed as near misses
    assert "no 17-character VIN" in reason_of(r)


def test_failing_mandatory_check_digit_is_reported_not_read() -> None:
    misread = "1HGCM85633A004352"   # two single swaps pass -> invalid, never guessed
    r = scripted([(misread, 0.99)], known_wmis=frozenset())
    assert r.status == "could_not_read"
    assert r.best is None
    assert [c.validation.vin for c in r.candidates] == [misread]
    assert r.candidates[0].validation.status == "invalid"
    assert set(r.candidates[0].validation.candidates) == {"1H6CM85633A004352", "1HGCM8S633A004352"}
    assert "closest windows" in reason_of(r)


def test_external_engine_lines_use_the_same_rules() -> None:
    from avtopassport.vin.ocr import _vin_from_lines

    r = _vin_from_lines([("GM UZBEKISTAN", 0.9), ("VIN XWB3L32EDKA123456", 0.8)],
                        known_wmis=frozenset({"XWB"}), prefix="easyocr", engine="easyocr")
    assert r.status == "read" and r.engine == "easyocr"
    assert r.best is not None and r.best.validation.vin == "XWB3L32EDKA123456"
    assert r.best.source == "easyocr/line2"
    empty = _vin_from_lines([], known_wmis=frozenset())
    assert empty.status == "could_not_read" and reason_of(empty)


def test_single_token_outranks_stitched_windows() -> None:
    from avtopassport.vin.ocr import _vin_from_lines

    known = frozenset({"XWB"})
    # The same VIN split over two lines (higher OCR score) and whole on a third: the whole token is kept.
    r = _vin_from_lines([("XWB3L32ED", 0.99), ("KA123456", 0.99), ("XWB3L32EDKA123456", 0.90)], known_wmis=known)
    assert r.status == "read" and r.best is not None and r.best.source == "ext/line3"
    # A different VIN stitched from two lines is a second reading (ambiguous), but ranks below the whole token
    # even with a higher OCR score (rank key 3 'single token' comes before key 4 'OCR score').
    r = _vin_from_lines([("XWB3L32EDKA123456", 0.80), ("XWB1L32ED", 0.99), ("KA765432", 0.99)], known_wmis=known)
    assert r.status == "ambiguous" and r.best is not None
    assert (r.best.validation.vin, r.best.source) == ("XWB3L32EDKA123456", "ext/line1")
    assert r.candidates[1].validation.vin == "XWB1L32EDKA765432"


@pytest.mark.parametrize("image", [np.full((400, 1200), 255, np.uint8), np.full((400, 1200, 1), 255, np.uint8),
                                   np.full((400, 1200, 4), 255, np.uint8)])
def test_gray_and_bgra_images_are_read(image: np.ndarray) -> None:
    r = read_vin(image, engine=ScriptedEngine([(EU_A, 0.99)]), known_wmis=frozenset())
    assert r.status == "read" and r.best is not None and r.best.validation.vin == EU_A
