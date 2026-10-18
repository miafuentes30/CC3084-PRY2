"""Variables comunes para los modelos (a partir de `train_clean.csv`).

Solo usa información de entrada: nunca `winner` ni `winner_*`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

COLS_BASE = [
    "prompt_len", "resp_a_len", "resp_b_len",
    "prompt_words", "resp_a_words", "resp_b_words", "n_turns",
]
COLS_NUM = COLS_BASE + [
    "dif_len", "dif_words",
    "log_prompt_len", "log_resp_a_len", "log_resp_b_len", "dif_log_len",
]
COLS_CAT = [cfg.COL_MODEL_A, cfg.COL_MODEL_B]


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables numéricas base + categóricas (model_a, model_b), sin etiquetas."""
    out = df[[cfg.COL_ID] + COLS_BASE + COLS_CAT].copy()
    out["dif_len"] = df["resp_a_len"] - df["resp_b_len"]
    out["dif_words"] = df["resp_a_words"] - df["resp_b_words"]
    for c in ["prompt_len", "resp_a_len", "resp_b_len"]:
        out[f"log_{c}"] = np.log1p(df[c])
    out["dif_log_len"] = out["log_resp_a_len"] - out["log_resp_b_len"]
    return out


def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    """Intercambia A↔B (modelos, textos, longitudes y etiqueta).

    Sirve para la prueba de sesgo de posición y para el aumento de datos:
    las probabilidades de A y B deben intercambiarse y la de empate mantenerse.
    """
    s = df.copy()
    pares = [
        (cfg.COL_MODEL_A, cfg.COL_MODEL_B),
        ("resp_a_text", "resp_b_text"),
        ("resp_a_len", "resp_b_len"),
        ("resp_a_words", "resp_b_words"),
    ]
    for a, b in pares:
        if a in s.columns and b in s.columns:
            s[[a, b]] = s[[b, a]].to_numpy()
    if "winner" in s.columns:
        s["winner"] = s["winner"].map({0: 1, 1: 0, 2: 2})
    if {cfg.COL_WIN_A, cfg.COL_WIN_B} <= set(s.columns):
        s[[cfg.COL_WIN_A, cfg.COL_WIN_B]] = s[[cfg.COL_WIN_B, cfg.COL_WIN_A]].to_numpy()
    return s
