"""Carga de datos y resultados para la app (todo en caché)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import evaluation  # noqa: E402

MODEL_IDS = ["logreg", "xgboost", "embeddings", "deberta"]
RESULTS = ROOT / "results"
CLEAN_COLS = [cfg.COL_ID, cfg.COL_MODEL_A, cfg.COL_MODEL_B, "winner", "prompt_len", "resp_a_len",
              "resp_b_len", "prompt_words", "resp_a_words", "resp_b_words", "n_turns"]
CLASS_LABEL = {0: "Gana A", 1: "Gana B", 2: "Empate"}


@st.cache_data(show_spinner="Cargando el dataset limpio…")
def load_clean() -> pd.DataFrame | None:
    """Dataset limpio con partición y variables derivadas; None si aún no se generó."""
    if not cfg.F_TRAIN_CLEAN.exists() or not cfg.F_SPLITS.exists():
        return None
    df = pd.read_csv(cfg.F_TRAIN_CLEAN, usecols=CLEAN_COLS)
    df = df.merge(pd.read_csv(cfg.F_SPLITS), on=cfg.COL_ID, how="left")
    df["resultado"] = df["winner"].map(CLASS_LABEL)
    df["dif_len"] = df["resp_a_len"] - df["resp_b_len"]
    df["dif_words"] = df["resp_a_words"] - df["resp_b_words"]
    df["total_len"] = df["resp_a_len"] + df["resp_b_len"]
    df["log_dif_abs"] = np.log1p(df["dif_len"].abs())
    return df


@st.cache_data
def available_models() -> list[str]:
    available = []
    for model in MODEL_IDS:
        if not (RESULTS / "preds" / f"{model}_test.csv").exists():
            continue
        if model == "deberta":
            metric_file = RESULTS / "metrics" / "deberta.json"
            checkpoint = ROOT / "models" / "deberta_checkpoint" / "head.pt"
            try:
                metadata = json.loads(metric_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not checkpoint.exists() or "history" not in metadata or not metadata.get("full_run", False):
                continue
        available.append(model)
    return available


@st.cache_data
def load_meta(model: str) -> dict:
    p = RESULTS / "metrics" / f"{model}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_preds(model: str, split: str) -> pd.DataFrame | None:
    """Predicciones del modelo unidas con la etiqueta real (columna `winner`)."""
    clean = load_clean()
    p = RESULTS / "preds" / f"{model}_{split}.csv"
    if clean is None or not p.exists():
        return None
    pr = pd.read_csv(p)
    return pr.merge(clean[[cfg.COL_ID, "winner"]], on=cfg.COL_ID, how="inner", validate="one_to_one")


@st.cache_data
def metrics_from_preds(model: str, split: str) -> dict | None:
    """Métricas recalculadas con el mismo código para todos los modelos (comparación justa)."""
    df = load_preds(model, split)
    if df is None:
        return None
    return evaluation.calculate_metrics(df["winner"].to_numpy(), df[["p_a", "p_b", "p_tie"]].to_numpy())


def train_seconds(model: str) -> float | None:
    """Tiempo de entrenamiento (s) según el JSON de cada modelo (formatos distintos)."""
    m = load_meta(model)
    if model == "logreg":
        v = m.get("test_metrics", {}).get("variant_b", {})
        return v.get("training_time_total_seconds")
    t = m.get("times", {})
    base = t.get("training_sec")
    if base is None:
        return None
    # Embeddings: la codificación con el modelo congelado también es costo de entrenamiento.
    return base + (t.get("encoding_sec", 0) if model == "embeddings" else 0)


def swap_diff(model: str) -> float | None:
    m = load_meta(model)
    if model == "logreg":
        return m.get("swap_test", {}).get("aggregate_mae")
    return m.get("position_bias_swap_check", {}).get("mean_absolute_swap_diff")
