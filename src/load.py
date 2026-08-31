"""Funciones de carga del dataset LMSYS Chatbot Arena."""
from __future__ import annotations

import pandas as pd

from . import config as cfg


def _check(path):
    """Verifica que el archivo exista antes de cargarlo."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path.name}' en {path.parent}.\n\n"
            "Para descargar el dataset ejecutá:\n"
            "  import kagglehub\n"
            "  path = kagglehub.competition_download('lmsys-chatbot-arena')\n\n"
            "O con la CLI de Kaggle:\n"
            "  kaggle competitions download -c lmsys-chatbot-arena -p data/raw/ --unzip\n\n"
            "Asegurate de tener tu API key en ~/.kaggle/kaggle.json"
        )
    return path


def load_train(nrows: int | None = None) -> pd.DataFrame:
    """Carga train.csv crudo. Usar `nrows` para explorar sin cargar todo."""
    return pd.read_csv(_check(cfg.F_TRAIN), nrows=nrows)


def load_test(nrows: int | None = None) -> pd.DataFrame:
    """Carga test.csv crudo."""
    return pd.read_csv(_check(cfg.F_TEST), nrows=nrows)


def load_submission() -> pd.DataFrame:
    """Carga sample_submission.csv."""
    return pd.read_csv(_check(cfg.F_SUBMISSION))


def load_train_clean(nrows: int | None = None) -> pd.DataFrame:
    """Carga el dataset limpio generado por el notebook 02.

    Requiere haber ejecutado notebooks/02_descripcion_y_limpieza.ipynb primero.
    """
    return pd.read_csv(_check(cfg.F_TRAIN_CLEAN), nrows=nrows)
