"""Variables de estilo por respuesta (listas, código, encabezados, rechazos, etc.).

Se calculan sobre `resp_a_text` / `resp_b_text` y se exponen como una columna por
respuesta (`style_<nombre>_a`, `style_<nombre>_b`) más su diferencia A − B.
"""
from __future__ import annotations

import pandas as pd

STYLE_NAMES = [
    "n_bullets", "n_code_blocks", "n_headers", "n_newlines",
    "n_exclam", "n_question", "n_refusal", "upper_ratio",
]

_RE_BULLET = r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"
_RE_HEADER = r"(?m)^\s{0,3}#{1,6}\s"
_RE_REFUSAL = (
    r"(?i)(?:i'?m sorry|i am sorry|i apologi[sz]e|as an ai|as a language model|"
    r"i cannot|i can'?t (?:assist|help|provide|fulfill)|i'?m unable|i am unable|"
    r"i'?m not able|against my (?:programming|guidelines))"
)


def style_features(text: pd.Series) -> pd.DataFrame:
    """Devuelve un DataFrame con `STYLE_NAMES` para una serie de textos."""
    t = text.fillna("").astype(str)
    n_chars = t.str.len().clip(lower=1)
    upper = t.str.count(r"[A-Z]")
    out = pd.DataFrame({
        "n_bullets": t.str.count(_RE_BULLET),
        "n_code_blocks": t.str.count("```") // 2,
        "n_headers": t.str.count(_RE_HEADER),
        "n_newlines": t.str.count("\n"),
        "n_exclam": t.str.count("!"),
        "n_question": t.str.count(r"\?"),
        "n_refusal": t.str.count(_RE_REFUSAL),
        "upper_ratio": upper / n_chars,
    })
    return out.astype(float)


def build_style_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas `style_*_a`, `style_*_b` y `style_*_dif` (A − B) alineadas con `df`."""
    sa = style_features(df["resp_a_text"]).add_prefix("style_").add_suffix("_a")
    sb = style_features(df["resp_b_text"]).add_prefix("style_").add_suffix("_b")
    sa.index = sb.index = df.index
    dif = pd.DataFrame(
        {f"style_{n}_dif": sa[f"style_{n}_a"] - sb[f"style_{n}_b"] for n in STYLE_NAMES},
        index=df.index,
    )
    return pd.concat([sa, sb, dif], axis=1)
