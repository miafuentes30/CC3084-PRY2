"""Carga de los CSV crudos con dtypes explicitos."""
from __future__ import annotations

import pandas as pd

from . import config as cfg


def _check(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path.name} en {path.parent}.\n"
            "Descarga data_format1.zip de "
            "https://tianchi.aliyun.com/competition/entrance/231576/information "
            "y descomprimelo en data/raw/."
        )
    return path


def load_user_info() -> pd.DataFrame:
    return pd.read_csv(_check(cfg.F_USER_INFO), dtype=cfg.DTYPES_USER_INFO)


def load_train() -> pd.DataFrame:
    return pd.read_csv(_check(cfg.F_TRAIN), dtype=cfg.DTYPES_TRAIN)


def load_log(nrows: int | None = None) -> pd.DataFrame:
    """Carga user_log. Usar `nrows` para explorar sin cargarlo entero."""
    return pd.read_csv(_check(cfg.F_USER_LOG), dtype=cfg.DTYPES_LOG, nrows=nrows)
