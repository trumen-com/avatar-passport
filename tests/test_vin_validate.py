"""VIN normalisation / correction / validation rules (synthetic strings and public sample VINs only).

No real vehicle's full VIN is kept here (docs/research/vin_decoding.md: real VINs only with positions 12-17
masked): the non-mandatory examples are published 11-character prefixes completed with synthetic serials.
"""

from __future__ import annotations

import random
import re

import pytest

from avtopassport.vin.checkdigit import VIN_ALPHABET, check_digit_ok, compute_check_digit
from avtopassport.vin.validate import (
    YEAR_CODES,
    check_digit_mandatory,
    model_year_candidates,
    normalize,
    validate,
)

# Sample VINs with a correct check digit: the public sample VIN also used in test_notebook_build, the
# all-ones VIN and the hypothetical worked example of the Wikipedia VIN article (rev. 1378888660), whose
# check digit is 'X'. None is a scanned car.
PASSING = ["1HGCM82633A004352", "11111111111111111", "1M8GDM9AXKP042788"]
# Prefixes (positions 1-11) of real published VINs that fail the check digit (docs/research/vin_decoding.md
# 3.4) + synthetic serial "000000"; position 9 is as published, so these fail it too.
NO_CHECK_DIGIT = ["WP0ZZZ99ZTS000000", "KLATF08Y1VB000000", "VF1LM1B0H36000000", "XTAGFK330JY000000"]
CORRECTION_RE = re.compile(r"^pos (\d+): (.)->(.)$")
# 49 CFR 565 letter values as listed in docs/research/phase0_findings_vin.md, typed independently of checkdigit.py
LETTER_VALUES = "A1 B2 C3 D4 E5 F6 G7 H8 J1 K2 L3 M4 N5 P7 R9 S2 T3 U4 V5 W6 X7 Y8 Z9"
WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def reference_check_digit(vin: str) -> str:
    values = {pair[0]: int(pair[1:]) for pair in LETTER_VALUES.split()} | {str(d): d for d in range(10)}
    remainder = sum(values[c] * w for c, w in zip(vin, WEIGHTS)) % 11
    return "X" if remainder == 10 else str(remainder)


def test_check_digit_worked_example_by_hand() -> None:
    # 1M8GDM9A_KP042788: values 1 4 8 7 4 4 9 1 _ 2 7 0 4 2 7 8 8 times the weights above
    products = [8, 28, 48, 35, 16, 12, 18, 10, 0, 18, 56, 0, 24, 10, 28, 24, 16]
    assert sum(products) == 351 and 351 % 11 == 10
    assert compute_check_digit("1M8GDM9A0KP042788") == "X"


def test_check_digit_matches_independent_table() -> None:
    rng = random.Random(3779)
    for _ in range(2000):
        vin = "".join(rng.choice(VIN_ALPHABET) for _ in range(17))
        assert compute_check_digit(vin) == reference_check_digit(vin), vin


@pytest.mark.parametrize("vin", PASSING)
def test_public_samples_are_valid(vin: str) -> None:
    assert compute_check_digit(vin) == vin[8]
    r = validate(vin)
    assert r.status == "valid"
    assert r.vin == vin
    assert r.check_digit_ok is True
    assert r.check_digit_required is True
    assert r.corrections == ()


@pytest.mark.parametrize("vin", NO_CHECK_DIGIT)
def test_published_prefixes_without_check_digit_are_valid_unverified(vin: str) -> None:
    assert not check_digit_ok(vin)
    r = validate(vin)
    assert r.status == "valid_unverified"
    assert r.vin == vin
    assert r.check_digit_ok is False
    assert r.check_digit_required is False
    assert r.corrections == ()
    assert any("check digit" in reason for reason in r.reasons)


def test_check_digit_mandatory_regions() -> None:
    for first in "12345LH":
        assert check_digit_mandatory(first + "0" * 16)
    for first in "WKXJVSZ06":
        assert not check_digit_mandatory(first + "0" * 16)
    assert not check_digit_mandatory("")


def test_normalize_strips_separators_and_maps_ioq() -> None:
    assert normalize(" 1hg-cm8.2633:a0*04352 ") == "1HGCM82633A004352"
    assert normalize("IOQ") == "100"
    assert normalize("１ＨＧ") == "1HG"       # full-width OCR output
    assert normalize("XWB 123") == "XWB123"   # does not pad
    assert normalize("A" * 20) == "A" * 20    # does not truncate


@pytest.mark.parametrize("raw", [
    "1hgcm82633a004352",                       # lowercase
    "*1HGCM82633A004352*",                     # Code 39 barcode text keeps its start/stop asterisks
    " 1HG CM8\t2633\nA004352 ",                # spaces, tab, newline
    "1HGCM8 2633A004352",                 # no-break space
    "﻿1HGCM8​2633A004352­",     # BOM, zero-width space, soft hyphen (pasted text)
    "1HG–CM826−33A004352",           # en dash, minus sign
])
def test_separator_variants_still_validate(raw: str) -> None:
    r = validate(raw)
    assert (r.status, r.vin, r.corrections) == ("valid", "1HGCM82633A004352", ())


def test_ioq_mapping_is_recorded() -> None:
    r = validate("IHGCM82633AOQ4352")
    assert r.status == "valid"
    assert r.vin == "1HGCM82633A004352"
    assert r.corrections == ("pos 1: I->1", "pos 12: O->0", "pos 13: Q->0")


def test_positions_14_to_17_digit_fixes() -> None:
    r = validate("1HGCM82633A0043SZ")
    assert r.status == "valid"
    assert r.vin == "1HGCM82633A004352"
    assert r.corrections == ("pos 16: S->5", "pos 17: Z->2")

    # Z and S outside positions 14-17 are legal VIN letters and stay as read.
    r = validate("WP0ZZZ99ZTS00Z1Z4")
    assert r.vin == "WP0ZZZ99ZTS002124"
    assert r.corrections == ("pos 14: Z->2", "pos 16: Z->2")
    assert r.status == "valid_unverified"

    r = validate("WP0ZZZ99ZTS002A24")  # 'A' has no digit look-alike
    assert r.status == "invalid"
    assert any("positions 14-17" in reason for reason in r.reasons)


def test_unique_single_swap_is_accepted() -> None:
    good, misread = "1HGCM82633A004352", "1HGCMB2633A004352"   # pos 6: 8 read as B
    assert check_digit_ok(good) and not check_digit_ok(misread)
    r = validate(misread)
    assert r.status == "valid"
    assert r.vin == good
    assert r.check_digit_ok is True
    assert "pos 6: B->8" in r.corrections


def test_ambiguous_swaps_are_rejected() -> None:
    misread = "1HGCM85633A004352"
    assert not check_digit_ok(misread)
    alternatives = {"1H6CM85633A004352", "1HGCM8S633A004352"}   # pos 3 G->6, pos 7 5->S
    assert all(check_digit_ok(a) for a in alternatives)
    r = validate(misread)
    assert r.status == "invalid"
    assert r.vin == misread
    assert r.check_digit_ok is False
    assert set(r.candidates) == alternatives
    assert r.model_year_candidates == ()


def test_no_swap_when_check_digit_is_optional() -> None:
    r = validate("KLATF08Y1VB000000")
    assert r.status == "valid_unverified"
    assert r.vin == "KLATF08Y1VB000000"


@pytest.mark.parametrize("raw", ["", "   ", "1HGCM82633", "hello world", "1HGCM82633A00435#", "XWB3L32EDKA12345Ж"])
def test_short_or_garbage_is_unreadable(raw: str) -> None:
    r = validate(raw)
    assert r.status == "unreadable"
    assert r.vin is None
    assert r.check_digit_ok is None
    assert r.reasons


def test_too_long_is_invalid_not_truncated() -> None:
    r = validate("1HGCM82633A0043521")
    assert r.status == "invalid"
    assert r.vin is None


def test_wmi_swap_only_to_known_wmi() -> None:
    known = frozenset({"XWB", "LGX"})
    r = validate("XW8JF69VJBA123456", known_wmis=known)
    assert r.vin == "XWBJF69VJBA123456"
    assert "pos 3: 8->B" in r.corrections
    assert r.status == "valid_unverified"
    # Without WMI knowledge, or when the swap gives no known WMI, nothing changes.
    assert validate("XW8JF69VJBA123456").vin == "XW8JF69VJBA123456"
    assert validate("XW8JF69VJBA123456", known_wmis=frozenset({"JTD"})).vin == "XW8JF69VJBA123456"


def test_wmi_swap_never_overrides_a_passing_mandatory_check_digit() -> None:
    r = validate("1HGCM82633A004352", known_wmis=frozenset({"LHG"}))
    assert r.vin == "1HGCM82633A004352"
    assert r.status == "valid"


def test_wmi_swap_ambiguous_is_not_applied() -> None:
    r = validate("8BW3L32EDKA123456", known_wmis=frozenset({"BBW", "88W"}))
    assert r.vin == "8BW3L32EDKA123456"
    assert {c[:3] for c in r.candidates} == {"BBW", "88W"}


@pytest.mark.parametrize("known", [["XWB"], {"XWB"}, frozenset({"xwb"}), ("XWB ",), "XWB"])
def test_known_wmis_any_iterable_any_case(known: object) -> None:
    # A list used to raise TypeError ({...} & list); lowercase entries silently never matched.
    r = validate("XW8JF69VJBA123456", known_wmis=known)  # type: ignore[arg-type]
    assert (r.vin, r.corrections) == ("XWBJF69VJBA123456", ("pos 3: 8->B",))


def test_failed_wmi_swap_does_not_block_the_confusion_fix() -> None:
    # Regression: a WMI swap was applied although the swapped VIN failed its mandatory check digit, and the
    # confusion swaps then ran on that wrong VIN: knowing 'LHG' turned this correctable read into 'invalid'
    # with a fabricated "pos 1: 1->L" (and, for other VINs, into a wrong 'valid').
    misread = "1HGCMB2633A004352"     # public sample VIN with pos 6 '8' read as 'B'
    r = validate(misread, known_wmis=frozenset({"LHG"}))
    assert (r.status, r.vin, r.corrections) == ("valid", "1HGCM82633A004352", ("pos 6: B->8",))
    assert "LHGCMB2633A004352" in r.candidates
    assert any("not applied" in reason for reason in r.reasons)


def china_vin(body: str) -> str:
    """Made-up VIN body with a real Chinese WMI; position 9 set to the computed check digit."""
    return body[:8] + compute_check_digit(body) + body[9:]


@pytest.mark.parametrize(("truth", "misread", "fix"), [
    (china_vin("LGXCE4CB0N0123456"), "1" + china_vin("LGXCE4CB0N0123456")[1:], "pos 1: 1->L"),   # L read as 1
    (china_vin("LC0CE4CB0N0123456"), "LCD" + china_vin("LC0CE4CB0N0123456")[3:], "pos 3: D->0"),  # 0 read as D
])
def test_wmi_swap_verified_by_check_digit_is_applied(truth: str, misread: str, fix: str) -> None:
    r = validate(misread, known_wmis=frozenset({truth[:3]}))
    assert (r.status, r.vin, r.corrections, r.check_digit_ok) == ("valid", truth, (fix,), True)


def test_wmi_swap_plus_one_confusion_only_if_unique() -> None:
    # L read as 1 AND B read as 8: reachable only via the WMI swap + one confusion swap; unique -> accepted.
    r = validate("1GXC84CB1N0000000", known_wmis=frozenset({"LGX"}))
    assert (r.status, r.vin) == ("valid", "LGXCB4CB1N0000000")
    assert r.corrections == ("pos 1: 1->L", "pos 5: 8->B")
    # Here a confusion swap of the read AND a WMI swap + confusion swap both pass: not guessed.
    r = validate("1GXTJSRD9WC054061", known_wmis=frozenset({"LGX"}))
    assert r.status == "invalid" and r.vin == "1GXTJSRD9WC054061"
    assert {"1GXTJ5RD9WC054061", "LGXTJSRD9WC0S4061"} <= set(r.candidates)
    assert all(c.startswith("1GX") or c.startswith("LGX") for c in r.candidates)   # known WMI never swapped away


def test_wmi_swap_into_a_mandatory_region_must_pass_the_check_digit() -> None:
    known = frozenset({"5YJ"})
    r = validate("SYJ3E1EA0KF000000", known_wmis=known)   # '5YJ...' fails its check digit: not applied
    assert (r.status, r.vin, r.corrections) == ("valid_unverified", "SYJ3E1EA0KF000000", ())
    assert r.candidates == ("5YJ3E1EA0KF000000",)
    r = validate("SYJ3E1EA0KF000008", known_wmis=known)   # '5YJ...' passes: applied and verified
    assert (r.status, r.vin, r.corrections) == ("valid", "5YJ3E1EA0KF000008", ("pos 1: S->5",))


def test_every_correction_uses_the_documented_format() -> None:
    for raw in ["IHGCM82633AOQ4352", "1HGCM82633A0043SZ", "1HGCMB2633A004352"]:
        for c in validate(raw).corrections:
            m = CORRECTION_RE.match(c)
            assert m and 1 <= int(m.group(1)) <= 17 and m.group(2) != m.group(3)


def test_model_year_cycles() -> None:
    assert YEAR_CODES[0] == "A" and len(YEAR_CODES) == 30
    # position 10 'A' outside North America: both 30-year cycles
    assert model_year_candidates("WP0ZZZ99ZAS000000") == (1980, 2010)
    # North America with a letter at position 7: 2010-2039 cycle only
    assert model_year_candidates("1HGCM8L63AA004352") == (2010,)
    # North America with a digit at position 7: not decided, both cycles
    assert model_year_candidates("1HGCM82633A004352") == (2003, 2033)
    assert model_year_candidates("11111111111111111") == (2001, 2031)
    assert model_year_candidates("KLATF08Y1VB000000") == (1997, 2027)
    # '0', 'U', 'Z' are never year codes; wrong length gives nothing
    assert model_year_candidates("WP0ZZZ99Z0S000000") == ()
    assert model_year_candidates("WP0ZZZ99ZUS000000") == ()
    assert model_year_candidates("WP0ZZZ99ZZS000000") == ()
    assert model_year_candidates("1HGCM82633") == ()
    assert validate("1HGCM82633A004352").model_year_candidates == (2003, 2033)


def test_mandatory_region_needs_a_year_code_at_position_10() -> None:
    vin = "1HGCM826300004352"   # position 10 '0' is not a year code
    r = validate(vin[:8] + compute_check_digit(vin) + vin[9:])
    assert r.check_digit_ok is True
    assert r.status == "invalid"
