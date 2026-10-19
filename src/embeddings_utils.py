"""Utilidades de extracción y construcción de variables con Embeddings (Modelo 3).

Permite codificar textos en batches, guardar y cargar embeddings de disco (.npy),
y construir variables semánticas simétricas de alta calidad.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import features


def build_semantic_features(
    prompt_emb: np.ndarray,
    resp_a_emb: np.ndarray,
    resp_b_emb: np.ndarray,
) -> np.ndarray:
    """Construye variables semánticas simétricas a partir de los embeddings de prompt y respuestas.

    Variables generadas:
    - emb(A) - emb(B): Contraste directo entre respuestas (D dimensiones).
    - emb(A) * emb(B): Interacción y solapamiento semántico de Hadamard (D dimensiones).
    - cos(prompt, A): Similitud coseno entre la solicitud y respuesta A (1 dim).
    - cos(prompt, B): Similitud coseno entre la solicitud y respuesta B (1 dim).
    - cos(prompt, A) - cos(prompt, B): Ventaja semántica de relevancia (1 dim).

    Total de columnas: 2 * D + 3.
    """
    # Los embeddings ya vienen L2-normalizados, por lo que el producto punto es el coseno
    cos_p_a = np.sum(prompt_emb * resp_a_emb, axis=1, keepdims=True)
    cos_p_b = np.sum(prompt_emb * resp_b_emb, axis=1, keepdims=True)
    dif_cos = cos_p_a - cos_p_b

    diff_ab = resp_a_emb - resp_b_emb
    mult_ab = resp_a_emb * resp_b_emb

    features_sem = np.hstack([diff_ab, mult_ab, cos_p_a, cos_p_b, dif_cos])
    return features_sem.astype(np.float32)


def get_embedding_feature_names(embedding_dim: int) -> list[str]:
    """Nombres descriptivos para las columnas de características semánticas."""
    names = [f"emb_diff_{i}" for i in range(embedding_dim)]
    names += [f"emb_mult_{i}" for i in range(embedding_dim)]
    names += ["cos_prompt_a", "cos_prompt_b", "dif_cos_prompt"]
    return names


class FeaturePipelineEmbeddings:
    """Pipeline de preprocesamiento para combinar variables tabulares base y embeddings."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        semantic_features: np.ndarray,
    ) -> np.ndarray:
        """Ajusta y transforma sobre el conjunto de entrenamiento."""
        df_base = features.build_base_features(df)
        num_cols = [c for c in features.COLS_NUM if c in df_base.columns]
        cat_cols = features.COLS_CAT

        X_num = self.scaler.fit_transform(df_base[num_cols].fillna(0).to_numpy())
        X_cat = self.encoder.fit_transform(df_base[cat_cols])

        self.fitted = True
        return np.hstack([X_num, X_cat, semantic_features]).astype(np.float32)

    def transform(
        self,
        df: pd.DataFrame,
        semantic_features: np.ndarray,
    ) -> np.ndarray:
        """Transforma conjuntos de validación o prueba con los scalers ya ajustados."""
        if not self.fitted:
            raise RuntimeError("El pipeline no ha sido ajustado. Llame a fit_transform primero.")

        df_base = features.build_base_features(df)
        num_cols = [c for c in features.COLS_NUM if c in df_base.columns]
        cat_cols = features.COLS_CAT

        X_num = self.scaler.transform(df_base[num_cols].fillna(0).to_numpy())
        X_cat = self.encoder.transform(df_base[cat_cols])

        return np.hstack([X_num, X_cat, semantic_features]).astype(np.float32)
