"""Funciones comunes de evaluación, métricas y guardado de resultados para la Fase 2.

Asegura consistencia exacta en el cálculo de métricas entre todos los modelos:
- Log Loss (métrica principal)
- Accuracy y F1 Macro
- Reporte por clase
- Matriz de confusión
- Prueba de intercambio A/B (control de sesgo de posición)
- Exportación estándar a results/preds/ y results/metrics/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from . import config as cfg


def calculate_metrics(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> dict[str, Any]:
    """Calcula las métricas estándar del proyecto sobre las probabilidades predichas."""
    y_true_arr = np.asarray(y_true)
    y_pred = np.argmax(y_prob, axis=1)

    # Log loss principal (con clipping para evitar infinities)
    ll = float(log_loss(y_true_arr, y_prob, labels=[0, 1, 2]))
    acc = float(accuracy_score(y_true_arr, y_pred))
    f1_mac = float(f1_score(y_true_arr, y_pred, average="macro"))

    report = classification_report(
        y_true_arr,
        y_pred,
        labels=[0, 1, 2],
        target_names=["model_a", "model_b", "tie"],
        output_dict=True,
        zero_division=0,
    )

    # Métricas de línea base teórica trivial (predicción fija de proporciones de clase)
    clases, counts = np.unique(y_true_arr, return_counts=True)
    priors = counts / len(y_true_arr)
    baseline_probs = np.tile(priors, (len(y_true_arr), 1))
    base_ll = float(log_loss(y_true_arr, baseline_probs, labels=[0, 1, 2]))
    base_acc = float(np.max(priors))

    return {
        "log_loss": round(ll, 5),
        "accuracy": round(acc, 5),
        "f1_macro": round(f1_mac, 5),
        "baseline_comparison": {
            "trivial_log_loss": round(base_ll, 5),
            "trivial_accuracy": round(base_acc, 5),
            "log_loss_improvement": round(base_ll - ll, 5),
            "accuracy_improvement": round(acc - base_acc, 5),
        },
        "per_class": {
            "model_a": {
                "precision": round(report["model_a"]["precision"], 4),
                "recall": round(report["model_a"]["recall"], 4),
                "f1": round(report["model_a"]["f1-score"], 4),
            },
            "model_b": {
                "precision": round(report["model_b"]["precision"], 4),
                "recall": round(report["model_b"]["recall"], 4),
                "f1": round(report["model_b"]["f1-score"], 4),
            },
            "tie": {
                "precision": round(report["tie"]["precision"], 4),
                "recall": round(report["tie"]["recall"], 4),
                "f1": round(report["tie"]["f1-score"], 4),
            },
        },
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_title: str,
    output_path: Path | str,
) -> None:
    """Genera y guarda la matriz de confusión normalizada con estética profesional."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = ["Gana A", "Gana B", "Empate"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión – {model_title}", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Predicción del Modelo", fontsize=10)
    ax.set_ylabel("Etiqueta Real", fontsize=10)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def check_ab_swap_bias(
    val_probs: np.ndarray,
    swap_probs: np.ndarray,
) -> dict[str, float]:
    """Evalúa el sesgo de posición comparando predicciones de validación original vs invertida A/B.

    Si no hay sesgo:
    - P(A | swap) == P(B | orig)
    - P(B | swap) == P(A | orig)
    - P(tie | swap) == P(tie | orig)
    """
    diff_a = np.mean(np.abs(swap_probs[:, 0] - val_probs[:, 1]))
    diff_b = np.mean(np.abs(swap_probs[:, 1] - val_probs[:, 0]))
    diff_tie = np.mean(np.abs(swap_probs[:, 2] - val_probs[:, 2]))
    mean_diff = float((diff_a + diff_b + diff_tie) / 3.0)

    return {
        "mean_absolute_swap_diff": round(mean_diff, 5),
        "p_a_to_p_b_diff": round(float(diff_a), 5),
        "p_b_to_p_a_diff": round(float(diff_b), 5),
        "p_tie_invariance_diff": round(float(diff_tie), 5),
    }


def save_predictions_csv(ids: pd.Series | np.ndarray, probs: np.ndarray, output_path: Path | str) -> Path:
    """Guarda predicciones en el formato exacto requerido por el proyecto (id, p_a, p_b, p_tie)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        cfg.COL_ID: np.asarray(ids),
        "p_a": probs[:, 0].round(6),
        "p_b": probs[:, 1].round(6),
        "p_tie": probs[:, 2].round(6),
    })
    df_out.to_csv(out_file, index=False)
    return out_file


def save_metrics_json(metrics: dict[str, Any], output_path: Path | str) -> Path:
    """Guarda el archivo JSON de métricas estandarizado."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return out_file
