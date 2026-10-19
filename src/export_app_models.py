"""Prepara los modelos que usa la app Streamlit para clasificar datos nuevos.

- Regresión logística (Variante B): se reajusta con el `C` ya elegido en
  `results/metrics/logreg.json`, sin tocar `results/` (solo escribe `models/logreg_pipeline.joblib`).
- XGBoost: reconstruye `models/xgboost_bundle.joblib` a partir de `models/xgboost_final.json`.

    python -m src.export_app_models
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load, logreg_model, xgb_model  # noqa: E402


def export_logreg() -> Path:
    meta = json.loads((ROOT / "results" / "metrics" / "logreg.json").read_text(encoding="utf-8"))
    c_value = meta["chosen_hyperparameters"]["variant_b"]["C"]
    model, seconds, _ = logreg_model.fit_model(load.load_split("train"), True, c_value)
    out = ROOT / "models" / "logreg_pipeline.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    print(f"Regresión logística (C={c_value}) ajustada en {seconds:.1f}s -> {out}")
    return out


def main() -> None:
    export_logreg()
    if (ROOT / "models" / "xgboost_final.json").exists():
        print(f"XGBoost -> {xgb_model.export_bundle()}")
    else:
        print("XGBoost: falta models/xgboost_final.json (corre antes `python -m src.xgb_model`).")


if __name__ == "__main__":
    main()
