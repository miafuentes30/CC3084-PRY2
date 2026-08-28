"""Rutas y codificaciones del dataset IJCAI-15 Repeat Buyers (Tianchi 231576)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

F_USER_INFO = RAW / "user_info_format1.csv"
F_USER_LOG = RAW / "user_log_format1.csv"
F_TRAIN = RAW / "train_format1.csv"
F_TEST = RAW / "test_format1.csv"

ACTION_TYPE = {0: "clic", 1: "carrito", 2: "compra", 3: "favorito"}

AGE_RANGE = {
    0: "desconocido", 1: "<18", 2: "18-24", 3: "25-29",
    4: "30-34", 5: "35-39", 6: "40-49", 7: ">=50", 8: ">=50",
}

GENDER = {0: "femenino", 1: "masculino", 2: "desconocido"}

DTYPES_LOG = {
    "user_id": "int32",
    "item_id": "int32",
    "cat_id": "int16",
    "seller_id": "int32",
    "brand_id": "float32",  # tiene nulos -> no puede ser int
    "time_stamp": "int16",
    "action_type": "int8",
}
DTYPES_USER_INFO = {"user_id": "int32", "age_range": "float32", "gender": "float32"}
DTYPES_TRAIN = {"user_id": "int32", "merchant_id": "int32", "label": "int8"}
