"""VIN normalisation, correction and validation (ISO 3779 / 49 CFR 565).

STUB written by main; the vin writer implements the bodies without changing signatures.
Rules (see docs/research/phase0_findings_vin.md):
- normalise: uppercase; strip spaces, '*', '-', '.', ':'; map I->1, O->0, Q->0 (never valid VIN letters).
- positions 14-17 (index 13..16) must be digits: map S->5, B->8, G->6, Z->2, D->0 there only.
- WMI swaps (1<->L, 0<->D, 5<->S, 8<->B) only if the result is a known WMI (``known_wmis``).
- check digit is MANDATORY only when the first char is 1-5 (North America), L or H (China);
  elsewhere a failing check digit is only a warning (real EU/Korean/Uzbek VINs fail it).
- where mandatory and failing: try single-character confusion swaps (0/O,1/I,8/B,5/S,2/Z,6/G);
  accept only if EXACTLY ONE candidate passes; else status 'invalid' with candidates listed.
- never guess: anything that is not 17 VIN-alphabet chars after normalisation is 'unreadable'/'invalid'.

Implementation notes:
- Normalisation also applies Unicode NFKC first (full-width OCR output such as 'Ｘ' or '：' becomes ASCII).
  Unicode dashes (en dash, minus sign, ...) are stripped like '-', invisible format characters (zero-width
  space, BOM, soft hyphen) like spaces.
- Statuses: 'unreadable' = characters outside the VIN alphabet or fewer than 17 usable characters;
  'invalid' = more than 17 characters, non-digits left in positions 14-17, or a mandatory check digit /
  model-year code that no unique swap repairs; 'valid_unverified' = check digit fails where it is not
  mandatory; 'valid' = check digit verified.
- Where the check digit is mandatory (North America, China) position 10 must also be a model-year code.
- A swap may not "break the alphabet": its result must be a VIN letter, a digit or 'X' at position 9,
  a digit in positions 14-17 and, where the check digit is mandatory, a year code at position 10.
- WMI swaps: a mandatory check digit has the last word.
  - Skipped when the VIN as read already passes a mandatory check digit (the read is verified; the WMI is
    simply missing from our table).
  - Not applied when several swaps give different known WMIs (those VINs are listed in ``candidates``).
  - Applied first (stub order) when the swapped VIN passes the check digit and year code its first
    character makes mandatory, or needs none.
  - Never applied on its own when the swapped VIN fails them: it is listed in ``candidates`` and, if the
    VIN as read fails a mandatory check digit too, the single confusion swaps are tried on the VIN as read
    AND on the WMI-swapped VIN, accepting only if exactly one candidate passes. (Applying the failed WMI
    swap first made the true VIN unreachable: a correctable '1..' misread became a wrong 'valid' VIN
    whenever our table knew its 'L..' twin, e.g. 1FM/LFM, 1B3/LB3.)
- ``known_wmis`` may be any iterable of WMIs; it is compared case-insensitively.
- ``vin`` is set whenever the input normalises to 17 VIN-alphabet characters, also for 'invalid':
  always check ``status`` before using it.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, Iterator, Literal

from .checkdigit import VIN_ALPHABET, VIN_RE, check_digit_ok, compute_check_digit  # noqa: F401

Status = Literal["valid", "valid_unverified", "invalid", "unreadable"]

# position 10 model-year codes, 30-year cycle (I, O, Q, U, Z, 0 never used)
YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"

_SEPARATORS = frozenset("*-.:\u2212")                # \u2212 minus sign; also whitespace, Pd and Cf characters
_SEPARATOR_CATEGORIES = frozenset({"Pd", "Cf"})       # dash punctuation; invisible format characters
_ALWAYS = {"I": "1", "O": "0", "Q": "0"}              # never valid VIN letters
_DIGIT_FIXES = {"S": "5", "B": "8", "G": "6", "Z": "2", "D": "0"}   # positions 14-17 only
_WMI_SWAPS = {"1": "L", "L": "1", "0": "D", "D": "0", "5": "S", "S": "5", "8": "B", "B": "8"}
_CONFUSIONS = (("0", "O"), ("1", "I"), ("8", "B"), ("5", "S"), ("2", "Z"), ("6", "G"))
_SWAPS = {a: b for pair in _CONFUSIONS for a, b in (pair, pair[::-1])}
_MANDATORY_FIRST = frozenset("12345LH")
_NORTH_AMERICA = frozenset("12345")
_DIGITS = frozenset("0123456789")
_CHECK_CHARS = _DIGITS | {"X"}


@dataclass(frozen=True)
class VinValidation:
    raw: str
    vin: str | None                      # normalised/corrected 17-char VIN, or None
    status: Status
    check_digit_ok: bool | None          # None if not computable
    check_digit_required: bool
    corrections: tuple[str, ...] = ()    # human-readable, e.g. "pos 3: O->0"
    candidates: tuple[str, ...] = ()     # alternative VINs considered
    model_year_candidates: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()


def _is_separator(c: str) -> bool:
    """True for a character normalize() strips (also a token boundary for the OCR windows)."""
    return c.isspace() or c in _SEPARATORS or unicodedata.category(c) in _SEPARATOR_CATEGORIES


def _wmi_set(known_wmis: Iterable[str] | None) -> frozenset[str]:
    """``known_wmis`` as an uppercase frozenset (a list, a set or lowercase entries also work)."""
    if not known_wmis:
        return frozenset()
    if isinstance(known_wmis, str):
        known_wmis = (known_wmis,)
    return _upper_wmis(frozenset(known_wmis))


@lru_cache(maxsize=16)
def _upper_wmis(wmis: frozenset[str]) -> frozenset[str]:
    return frozenset(str(w).strip().upper() for w in wmis)


def _normalize(raw: str) -> tuple[str, list[str]]:
    """normalize() plus the I/O/Q mappings as "pos N: A->B" (N = 1-based position after stripping)."""
    text = unicodedata.normalize("NFKC", raw).upper()
    kept = [c for c in text if not _is_separator(c)]
    out: list[str] = []
    corrections: list[str] = []
    for pos, c in enumerate(kept, start=1):
        mapped = _ALWAYS.get(c)
        if mapped:
            corrections.append(f"pos {pos}: {c}->{mapped}")
            c = mapped
        out.append(c)
    return "".join(out), corrections


def normalize(raw: str) -> str:
    """Uppercase, strip separators, map I/O/Q to 1/0/0. Does not truncate or pad."""
    return _normalize(raw or "")[0]


def check_digit_mandatory(vin: str) -> bool:
    """True if the first character is 1-5, L or H."""
    v = normalize(vin)
    return bool(v) and v[0] in _MANDATORY_FIRST


def model_year_candidates(vin: str) -> tuple[int, ...]:
    """Candidate model years from position 10 (both 30-year cycles within 1980-2039).
    For North-American VINs (first char 1-5) a LETTER at position 7 selects the 2010-2039 cycle."""
    v = normalize(vin)
    if len(v) != 17:
        return ()
    idx = YEAR_CODES.find(v[9])
    if idx < 0:
        return ()
    if v[0] in _NORTH_AMERICA and v[6].isalpha():    # 49 CFR 565.15: rule for light vehicles
        return (2010 + idx,)
    return (1980 + idx, 2010 + idx)


def _year_code_ok(v: str) -> bool:
    """Position 10 must be a model-year code where the check digit is mandatory (NA, China)."""
    return v[0] not in _MANDATORY_FIRST or v[9] in YEAR_CODES


def _acceptable(v: str) -> bool:
    return check_digit_ok(v) and _year_code_ok(v)


def _region_ok(v: str) -> bool:
    """Acceptable for its own region: passes the check digit and year code where those are mandatory."""
    return v[0] not in _MANDATORY_FIRST or _acceptable(v)


def _single_swaps(v: str, start: int = 0) -> Iterator[tuple[str, str]]:
    """Every single confusion swap of ``v`` (at index >= start) that keeps each position's alphabet."""
    for i, c in enumerate(v):
        alt = _SWAPS.get(c)
        if i < start:
            continue
        if alt is None or alt not in VIN_ALPHABET:
            continue
        if i == 8 and alt not in _CHECK_CHARS:
            continue
        if 13 <= i <= 16 and alt not in _DIGITS:
            continue
        cand = v[:i] + alt + v[i + 1:]
        if cand[0] in _MANDATORY_FIRST and cand[9] not in YEAR_CODES:
            continue
        yield cand, f"pos {i + 1}: {c}->{alt}"


def _wmi_swap(v: str, known_wmis: frozenset[str]) -> tuple[str | None, list[str], list[str], list[str]]:
    """``v`` with its WMI swapped to the ONE known WMI that 1<->L, 0<->D, 5<->S, 8<->B swaps reach, else None.

    Returns (swapped VIN or None, corrections, candidates, reasons); several reachable known WMIs are
    listed in candidates, not applied. Whether a swap is applied is decided by validate().
    """
    wmi = v[:3]
    if not known_wmis or wmi in known_wmis:
        return None, [], [], []
    options = [(c, _WMI_SWAPS[c]) if c in _WMI_SWAPS else (c,) for c in wmi]
    hits = sorted({"".join(p) for p in product(*options)} & known_wmis)
    if not hits:
        return None, [], [], []
    if len(hits) > 1:
        reason = f"WMI {wmi} is unknown and swaps give several known WMIs ({', '.join(hits)}); not corrected"
        return None, [], [h + v[3:] for h in hits], [reason]
    new = hits[0]
    corrections = [f"pos {i + 1}: {a}->{b}" for i, (a, b) in enumerate(zip(wmi, new)) if a != b]
    return new + v[3:], corrections, [], []


def validate(raw: str, *, known_wmis: frozenset[str] | None = None) -> VinValidation:
    """Normalise, apply the safe corrections above and classify. Pure function, no I/O."""
    raw = "" if raw is None else str(raw)
    v, corrections = _normalize(raw)
    known = _wmi_set(known_wmis)
    candidates: list[str] = []
    reasons: list[str] = []

    def result(status: Status, vin: str | None, ok: bool | None, *more: str) -> VinValidation:
        basis = vin or v
        years = model_year_candidates(vin) if vin and status in ("valid", "valid_unverified") else ()
        return VinValidation(raw, vin, status, ok, bool(basis) and basis[0] in _MANDATORY_FIRST,
                             tuple(corrections), tuple(candidates), years, tuple(reasons) + more)

    def classify(vin: str, *more: str) -> VinValidation:
        """'valid' / 'valid_unverified' for a VIN that is acceptable for its region."""
        if check_digit_ok(vin) and _year_code_ok(vin):
            note = ("check digit verified" if vin[0] in _MANDATORY_FIRST
                    else f"check digit matches (optional for VINs starting with {vin[0]!r})")
            return result("valid", vin, True, *more, note)
        return result("valid_unverified", vin, False, *more,
                      f"check digit mismatch (position 9 is {vin[8]!r}, computed {compute_check_digit(vin)!r}); "
                      f"not mandatory for VINs starting with {vin[0]!r}")

    junk = sorted({c for c in v if c not in VIN_ALPHABET})
    if junk:
        return result("unreadable", None, None, f"characters outside the VIN alphabet: {''.join(junk)!r}")
    if len(v) < 17:
        return result("unreadable", None, None, f"{len(v)} usable characters; a VIN has 17")
    if len(v) > 17:
        return result("invalid", None, None, f"{len(v)} characters; a VIN has exactly 17")

    chars = list(v)
    for i in range(13, 17):
        fix = _DIGIT_FIXES.get(chars[i])
        if fix:
            corrections.append(f"pos {i + 1}: {chars[i]}->{fix}")
            chars[i] = fix
    v = "".join(chars)

    bad_tail = [str(i + 1) for i in range(13, 17) if v[i] not in _DIGITS]
    if bad_tail:
        return result("invalid", v, check_digit_ok(v), f"positions 14-17 must be digits (pos {', '.join(bad_tail)})")

    required = v[0] in _MANDATORY_FIRST
    if required and _acceptable(v):
        return classify(v)                    # verified as read: no WMI swap

    alt, alt_fixes, wmi_alts, wmi_reasons = _wmi_swap(v, known)
    candidates += wmi_alts
    reasons += wmi_reasons
    wmi_note = f"WMI {v[:3]} corrected to known WMI {alt[:3]}" if alt is not None else ""
    if alt is not None and _region_ok(alt):             # stub order: the WMI swap first
        corrections.extend(alt_fixes)
        if alt[0] in _MANDATORY_FIRST:
            return result("valid", alt, True, f"{wmi_note}; check digit verified")
        return classify(alt, wmi_note)
    if alt is not None:                                 # the check digit rejects it: never applied on its own
        candidates.append(alt)
        reasons.append(f"WMI {v[:3]} -> known WMI {alt[:3]} not applied: that VIN fails the check digit or "
                       f"model-year code mandatory for VINs starting with {alt[0]!r}")
    if not required:
        return classify(v)

    # Mandatory and failing: single confusion swaps of the VIN as read and, when a WMI swap was rejected
    # above, of the WMI-swapped VIN too; accept only if exactly one candidate passes.
    problem = ("position 10 is not a model-year code" if check_digit_ok(v)
               else f"check digit mismatch (position 9 is {v[8]!r}, computed {compute_check_digit(v)!r})")
    bases = [(v, [], "", 0)] + ([(alt, alt_fixes, f"{wmi_note} and ", 3)] if alt is not None else [])
    passing: dict[str, tuple[list[str], str]] = {}
    for base, base_fixes, prefix, start in bases:     # the known WMI of a WMI-swapped base stays as it is
        for cand, fix in _single_swaps(base, start):
            if _acceptable(cand):
                passing.setdefault(cand, (base_fixes + [fix], f"{prefix}one confusion swap ({fix})"))
    if len(passing) == 1:
        (cand, (fixes, how)), = passing.items()
        corrections.extend(fixes)
        if cand[0] in _MANDATORY_FIRST:
            return result("valid", cand, True, f"{problem}; check digit verified after {how}")
        return classify(cand, f"{problem}; {how}")      # a swap at position 1 (5->S) left the region
    if passing:
        candidates += list(passing)
        return result("invalid", v, check_digit_ok(v), f"{problem}; mandatory for VINs starting with {v[0]!r}; "
                      f"{len(passing)} single swaps pass, not guessing")
    return result("invalid", v, check_digit_ok(v), f"{problem}; mandatory for VINs starting with {v[0]!r}; "
                  "no single confusion swap repairs it")
