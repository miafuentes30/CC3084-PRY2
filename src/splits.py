"""Particiones train/val/test compartidas por todos los modelos.

La partición se genera una sola vez (notebook 05) y se guarda en `data/splits.csv`.
Nadie debe crear su propia partición: todos cargan la misma con `load.load_splits()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as cfg

SEED = 42
N_FOLDS = 20            # cada fold ≈ 5 % de los datos
FOLDS_VAL = 3           # 3/20 = 15 %
FOLDS_TEST = 3          # 3/20 = 15 %  (los 14 restantes, 70 %, son train)


def make_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Devuelve un DataFrame `id, split` con split ∈ {train, val, test}.

    Estratifica por `winner` y agrupa por texto del prompt, de modo que un mismo
    prompt nunca queda repartido entre particiones (evita fuga de información).
    """
    grupos = pd.util.hash_pandas_object(df["prompt_text"], index=False).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["winner"], groups=grupos)):
        fold[idx] = k

    split = np.full(len(df), "train", dtype=object)
    split[fold < FOLDS_VAL] = "val"
    split[(fold >= FOLDS_VAL) & (fold < FOLDS_VAL + FOLDS_TEST)] = "test"

    return pd.DataFrame({cfg.COL_ID: df[cfg.COL_ID].to_numpy(), "split": split})


def check_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Verifica la partición y devuelve un resumen por split.

    Lanza AssertionError si hay ids sin partición o prompts repartidos entre splits.
    """
    m = df.merge(splits, on=cfg.COL_ID, how="left", validate="one_to_one")
    assert m["split"].notna().all(), "Hay filas sin partición asignada"
    assert len(splits) == len(df), "splits.csv no coincide con el dataset limpio"

    n_splits_por_prompt = m.groupby("prompt_text")["split"].nunique()
    assert (n_splits_por_prompt == 1).all(), "Hay prompts repartidos entre particiones"

    resumen = m.groupby("split").agg(n=("winner", "size"))
    resumen["% total"] = resumen["n"] / len(m) * 100
    props = m.groupby("split")["winner"].value_counts(normalize=True).unstack() * 100
    props.columns = [f"% {cfg.LABEL_NAMES[c]}" for c in props.columns]
    return resumen.join(props).round(2).loc[["train", "val", "test"]]
