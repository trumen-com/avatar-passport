"""ISO 3779 / 49 CFR 565 VIN check digit (position 9).

Letter values and weights per 49 CFR 565.15; verified against published sample VINs
(see docs/research/phase0_findings_vin.md).
"""

from __future__ import annotations

import re

VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # no I, O, Q
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

TRANSLITERATION: dict[str, int] = {
    **{str(d): d for d in range(10)},
    **dict(zip("ABCDEFGH", range(1, 9))),
    **dict(zip("JKLMN", range(1, 6))),
    "P": 7,
    "R": 9,
    **dict(zip("STUVWXYZ", range(2, 10))),
}
WEIGHTS: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def compute_check_digit(vin17: str) -> str:
    """Return the expected check digit ('0'-'9' or 'X') for a 17-character VIN.

    Raises ValueError if the string is not 17 characters of the VIN alphabet.
    """
    v = vin17.upper()
    if not VIN_RE.match(v):
        raise ValueError(f"not a 17-character VIN: {vin17!r}")
    r = sum(TRANSLITERATION[c] * w for c, w in zip(v, WEIGHTS)) % 11
    return "X" if r == 10 else str(r)


def check_digit_ok(vin17: str) -> bool:
    """True if position 9 matches the computed check digit."""
    try:
        return vin17.upper()[8] == compute_check_digit(vin17)
    except ValueError:
        return False
