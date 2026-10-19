"""Inferencia sobre datos nuevos: preprocesamiento invisible + modelos disponibles."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import logreg_model  # noqa: E402

UNKNOWN = "Desconocido / otro"
MODEL_DIR = ROOT / "models"

# Modelos cuyo artefacto pesado vive en Drive y no está en el repositorio.
UNAVAILABLE = {
    "embeddings": "No se versiona el clasificador entrenado ni los embeddings (se calculan con GPU).",
    "deberta": "Ejecuta `python -m src.run_deberta` para generar models/deberta_checkpoint.",
}


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models() -> dict:
    """Devuelve {id: objeto} para los modelos con artefacto local."""
    out = {}
    p = MODEL_DIR / "logreg_pipeline.joblib"
    if p.exists():
        out["logreg"] = joblib.load(p)
    p = MODEL_DIR / "xgboost_bundle.joblib"
    if p.exists():
        out["xgboost"] = joblib.load(p)
    checkpoint = MODEL_DIR / "deberta_checkpoint"
    if (checkpoint / "head.pt").exists():
        from transformers import AutoModel, AutoTokenizer
        import torch

        head = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
        encoder = AutoModel.from_pretrained(checkpoint).float()
        model = {"tokenizer": AutoTokenizer.from_pretrained(checkpoint), "encoder": encoder}
        model["reward"] = torch.nn.Linear(encoder.config.hidden_size, 1)
        model["reward"].load_state_dict(head["reward"])
        model["log_tau"] = head["log_tau"]
        model["encoder"].eval()
        model["reward"].eval()
        out["deberta"] = model
    return out


def predict_all(prompt: str, resp_a: str, resp_b: str, model_a: str, model_b: str) -> dict[str, np.ndarray]:
    """Preprocesa la entrada igual que el dataset limpio y devuelve [P(A), P(B), P(empate)] por modelo."""
    models = load_models()
    row = logreg_model.prepare_inference_row(prompt, resp_a, resp_b, model_a, model_b)
    out: dict[str, np.ndarray] = {}
    if "logreg" in models:
        out["logreg"] = logreg_model.predict_proba(models["logreg"], row)[0]
    if "xgboost" in models:
        b = models["xgboost"]
        x = b["builder"].transform(row)[b["columns"]]
        out["xgboost"] = b["clf"].predict_proba(x)[0]
    if "deberta" in models:
        import torch

        b = models["deberta"]
        tokenizer = b["tokenizer"]
        encoded = []
        for response in (resp_a, resp_b):
            encoded.append(tokenizer(prompt, response, max_length=384, truncation=True,
                                     padding="max_length", return_tensors="pt"))
        with torch.no_grad():
            scores = []
            for item in encoded:
                hidden = b["encoder"](**item).last_hidden_state[:, 0]
                scores.append(b["reward"](hidden).squeeze(-1))
            delta = scores[0] - scores[1]
            tau = torch.exp(b["log_tau"])
            logits = torch.stack((delta, -delta, tau - torch.abs(delta)), dim=1)
            out["deberta"] = torch.softmax(logits, dim=1)[0].numpy()
    return out


def style_summary(resp: str) -> dict:
    """Variables de estilo que ve XGBoost, para mostrarlas al usuario."""
    import pandas as pd
    from src.style_features import style_features
    return style_features(pd.Series([resp])).iloc[0].to_dict()
