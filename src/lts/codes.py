"""Industry-code concordances.

Every industry label (FIGARO/NACE, OECD ISIC4 activity codes, OECD TiMBC aggregates, BEA summary
codes) is represented as a set of 2-digit NACE Rev.2 / ISIC Rev.4 divisions ("01".."99").
Mapping between classifications is then done by set inclusion.
"""
from __future__ import annotations

import re

SECTIONS = {  # section letter -> divisions
    "A": range(1, 4), "B": range(5, 10), "C": range(10, 34), "D": [35], "E": range(36, 40),
    "F": range(41, 44), "G": range(45, 48), "H": range(49, 54), "I": range(55, 57), "J": range(58, 64),
    "K": range(64, 67), "L": [68], "M": range(69, 76), "N": range(77, 83), "O": [84], "P": [85],
    "Q": range(86, 89), "R": range(90, 94), "S": range(94, 97), "T": range(97, 99), "U": [99],
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTU"


def _sec(a: str) -> set[int]:
    return set(SECTIONS[a])


def divisions(code: str) -> frozenset[int] | None:
    """Return the set of 2-digit divisions covered by an activity code, or None if not parseable.

    Handles e.g. 'A01', 'C10-12', 'C10T12', 'C31_32', 'D35', 'D', 'BTE', 'GTI', 'M_N', 'B_D_E',
    'C10T18_31T33', 'C19T25', 'OTT', '_T' (total economy), 'L68A' -> None (imputed rent is part of 68).
    """
    c = code.strip()
    if c in ("_T", "TOTAL"):
        return frozenset(range(1, 100)) & frozenset(d for s in SECTIONS.values() for d in s)
    if "XL" in c or c in ("L68A", "ENERGYP") or c.startswith(("ICT", "HI", "MH", "LDI", "MLDI", "MHDI", "HDI", "INFO")):
        return None
    out: set[int] = set()
    parts = c.split("_")
    prefix = None
    for p in parts:
        if not p:
            return None
        m = re.fullmatch(r"([A-U])?(\d{2})(?:[T-](\d{2}))?", p)
        if m and (m.group(1) or prefix):
            prefix = m.group(1) or prefix
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            out |= set(range(lo, hi + 1))
            continue
        m = re.fullmatch(r"([A-U])T([A-U])", p)
        if m:
            a, b = LETTERS.index(m.group(1)), LETTERS.index(m.group(2))
            for L in LETTERS[a:b + 1]:
                out |= _sec(L)
            prefix = m.group(2)
            continue
        if re.fullmatch(r"[A-U]", p):
            out |= _sec(p)
            prefix = p
            continue
        return None
    return frozenset(out)


# BEA summary industries (2017 NAICS based) -> NACE divisions (approximate, documented in ASSUMPTIONS.md)
BEA_TO_NACE = {
    "111CA": [1], "113FF": [2, 3], "211": [6], "212": [5, 7, 8], "213": [9], "22": [35, 36],
    "23": [41, 42, 43], "321": [16], "327": [23], "331": [24], "332": [25], "333": [28], "334": [26],
    "335": [27], "3361MV": [29], "3364OT": [30], "337": [31], "339": [32], "311FT": [10, 11, 12],
    "313TT": [13], "315AL": [14, 15], "322": [17], "323": [18], "324": [19], "325": [20, 21], "326": [22],
    "42": [46], "441": [45], "445": [47], "452": [47], "4A0": [47], "481": [51], "482": [49], "483": [50],
    "484": [49], "485": [49], "486": [49], "487OS": [52], "493": [52], "511": [58], "512": [59],
    "513": [60, 61], "514": [63], "521CI": [64], "523": [66], "524": [65], "525": [64], "HS": [68],
    "ORE": [68], "532RL": [77], "5411": [69], "5415": [62], "5412OP": [70, 71, 72, 73, 74, 75],
    "55": [70], "561": [78, 79, 80, 81, 82], "562": [37, 38, 39], "61": [85], "621": [86], "622": [86],
    "623": [87], "624": [88], "711AS": [90, 91], "713": [92, 93], "721": [55], "722": [56], "81": [94, 95, 96, 97],
    "GFGD": [84], "GFGN": [84], "GFE": [84], "GSLG": [84, 85], "GSLE": [84, 35, 36, 49],
}


def figaro_to_oecd(code: str) -> str:
    """FIGARO/NACE label -> OECD ISIC4 activity code as used in OECD NA tables."""
    special = {"D35": "D", "O84": "O", "P85": "P", "L": "L", "T": "T", "U": "U", "B": "B", "F": "F", "I": "I"}
    if code in special:
        return special[code]
    return code.replace("-", "T")
