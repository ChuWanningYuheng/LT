"""Iteration 13a: the theory x outcome matrix as data (pre_registration_v13a.md, 'Матрица «теория × исход»',
edition after the pre-calculation review, journal item 3). Committed before any outcome is computed.

PRED[outcome][theory] = (prediction, basis): prediction one of '>', '<', '≈0', '>=0', '<=0', 'молчит';
basis 'h' (the hours measure) or '$' (the money analogue of the same measure, lambda_$ = 1, same valuation and
coverage). Outcome 113 is a ratio with theta0 = 1: '>' means R > 1.
Descriptive outcomes (112, 118, 119, 131) are not counted.
"""
from __future__ import annotations

THEORIES = ["М-ист", "М-воспр", "TSSI", "Ок/Ср", "Неокл.-К", "Неокл.-KN", "Кал/ПК", "Шейх", "Д–Л", "СП", "Мозли"]
S = ("молчит", "h")


def _row(**kw):
    out = {t: S for t in THEORIES}
    key = {"Mist": "М-ист", "Mvos": "М-воспр", "TSSI": "TSSI", "OkSr": "Ок/Ср", "NeoK": "Неокл.-К",
           "NeoKN": "Неокл.-KN", "Kal": "Кал/ПК", "Shaikh": "Шейх", "DL": "Д–Л", "SP": "СП", "Mos": "Мозли"}
    for k, v in kw.items():
        out[key[k]] = v if isinstance(v, tuple) else (v, "h")
    return out


PRED = {
    110: _row(Mist=">", NeoK=("≈0", "$"), Shaikh=">"),
    111: _row(Mist="<", NeoK=("≈0", "$")),
    113: _row(Mist=">", Mvos=">"),
    114: _row(Mvos="<", OkSr=">=0", SP=(">=0", "$")),
    115: _row(Mist="<", SP=(">=0", "$")),
    116: _row(NeoKN=(">", "$"), Kal=(">", "$"), Shaikh=">", DL=(">", "$")),
    117: _row(Mos="<"),
    120: _row(Mvos="<"),
    121: _row(Mvos=">", Shaikh=">"),
    122: _row(),
    123: _row(Mvos="<", NeoK=("≈0", "$")),
    124: _row(Mvos=">", NeoK=("≈0", "$"), Shaikh=">"),
    125: _row(Mist=">", NeoK=("≈0", "$"), Shaikh=">"),
    126: _row(NeoK=("≈0", "$"), Shaikh=">", DL=(">", "$")),
    127: _row(NeoK=("≈0", "$"), DL=("<", "$")),
    128: _row(NeoKN=(">", "$")),
    129: _row(TSSI="<"),
    130: _row(Mist="≈0", Mvos="≈0", Shaikh="≈0"),
}
# outcomes whose estimate is a common slope within phases (the equal-slopes condition applies to '≈0')
WITHIN = {110, 111, 114, 115, 117, 120, 121, 122, 123, 124, 125, 129, 130}
# conditional prediction recorded with the row (shown in the report line)
NOTES = {(116, "Кал/ПК"): "условно: при ослаблении профсоюзов (внешний факт)",
         (114, "Ок/Ср"): "перенос теоремы на агрегат — тест допущения",
         (129, "TSSI"): "эмпирический тезис Kliman (2012), из теории не следует"}
