"""Rutas y constantes del dataset LMSYS Chatbot Arena.

Competencia Kaggle: lmsys-chatbot-arena
Reto 11 – Negocios | CC3084 Data Science – Semestre II 2026
"""
from pathlib import Path

# ── Raíces del proyecto ─────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ── Archivos del dataset ────────────────────────────────────────────────────
F_TRAIN        = RAW / "train.csv"
F_TEST         = RAW / "test.csv"
F_SUBMISSION   = RAW / "sample_submission.csv"
F_TRAIN_CLEAN  = PROCESSED / "train_clean.csv"

# ── Nombres de columnas ─────────────────────────────────────────────────────
COL_ID       = "id"
COL_MODEL_A  = "model_a"
COL_MODEL_B  = "model_b"
COL_PROMPT   = "prompt"
COL_RESP_A   = "response_a"
COL_RESP_B   = "response_b"
COL_WIN_A    = "winner_model_a"
COL_WIN_B    = "winner_model_b"
COL_WIN_TIE  = "winner_tie"

# ── Agrupaciones de columnas ────────────────────────────────────────────────
TEXT_COLS   = [COL_PROMPT, COL_RESP_A, COL_RESP_B]
WINNER_COLS = [COL_WIN_A, COL_WIN_B, COL_WIN_TIE]

# ── Etiqueta unificada derivada de las tres columnas de ganador ─────────────
# 0 → model_a gana  |  1 → model_b gana  |  2 → empate
LABEL_MAP   = {COL_WIN_A: 0, COL_WIN_B: 1, COL_WIN_TIE: 2}
LABEL_NAMES = {0: "model_a gana", 1: "model_b gana", 2: "empate"}
