"""Regresión logística multinomial para preferencias de Chatbot Arena.

Este módulo concentra el entrenamiento, evaluación, diagnóstico A/B y la
inferencia reutilizable. El conjunto de prueba solo se evalúa después de fijar
la Variante B como modelo final, tal como establece el contrato del equipo.
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import features, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

SEED = 42
CLASS_ORDER = np.array([0, 1, 2], dtype=int)
CLASS_KEYS = ["a", "b", "tie"]
CLASS_LABELS_ES = ["Gana A", "Gana B", "Empate"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_CV_SPLITS = 5
NUMERIC_FEATURES = list(features.COLS_NUM)
CATEGORICAL_FEATURES = list(features.COLS_CAT)
FORBIDDEN_FEATURES = {"winner", cfg.COL_WIN_A, cfg.COL_WIN_B, cfg.COL_WIN_TIE}


def _json_ready(value: Any) -> Any:
    """Convierte objetos NumPy/Pandas a tipos serializables por JSON."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _logreg_kwargs(c_value: float, max_iter: int) -> dict[str, Any]:
    """Argumentos compatibles con sklearn antiguo y con sklearn >= 1.8.

    En sklearn 1.8 ``multi_class`` fue eliminado y ``penalty`` quedó
    deprecado. Al omitir ``penalty`` en esa versión se conserva la
    regularización L2 predeterminada; en versiones anteriores se explicita.
    """
    kwargs: dict[str, Any] = {
        "C": c_value,
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": SEED,
    }
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty is not None and penalty.default != "deprecated":
        kwargs["penalty"] = "l2"
    return kwargs


def build_pipeline(
    include_model_identity: bool,
    c_value: float = 1.0,
    max_iter: int = 1_000,
) -> Pipeline:
    """Construye el pipeline sin ajustarlo.

    Los valores numéricos se estandarizan y las identidades se codifican con
    categorías desconocidas ignoradas. Todo se ajusta dentro del pipeline.
    """
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", StandardScaler(), NUMERIC_FEATURES)
    ]
    if include_model_identity:
        transformers.append(
            (
                "models",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(**_logreg_kwargs(c_value, max_iter))
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Construye variables comunes y comprueba que no haya fuga de etiqueta."""
    if "winner" not in df:
        raise ValueError("El dataframe de entrenamiento no contiene 'winner'.")
    x = features.build_base_features(df)
    leaked = FORBIDDEN_FEATURES.intersection(x.columns)
    if leaked:
        raise ValueError(f"Fuga de etiqueta detectada: {sorted(leaked)}")
    y = df["winner"].to_numpy(dtype=int)
    if not set(np.unique(y)).issubset(set(CLASS_ORDER)):
        raise ValueError("Las etiquetas deben pertenecer a {0, 1, 2}.")
    return x, y


def _ordered_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Predice y ordena explícitamente las columnas como [A, B, empate]."""
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.named_steps["classifier"].classes_, dtype=int)
    if set(classes) != set(CLASS_ORDER):
        raise ValueError(f"Clases inesperadas en el modelo: {classes.tolist()}")
    positions = [int(np.flatnonzero(classes == label)[0]) for label in CLASS_ORDER]
    ordered = np.asarray(probabilities[:, positions], dtype=float)
    if not np.allclose(ordered.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Las probabilidades no suman uno.")
    return ordered


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predice probabilidades [A, B, empate] desde un dataframe limpio."""
    return _ordered_probabilities(model, features.build_base_features(df))


def make_cv_splits(
    train_df: pd.DataFrame,
    n_splits: int = N_CV_SPLITS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crea folds estratificados y agrupados exclusivamente sobre train."""
    if "prompt_text" not in train_df:
        raise ValueError("Se necesita 'prompt_text' para agrupar la validación cruzada.")
    y = train_df["winner"].to_numpy(dtype=int)
    groups = train_df["prompt_text"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(train_df, y, groups=groups))
    for fit_idx, valid_idx in folds:
        fit_groups = set(groups[fit_idx])
        valid_groups = set(groups[valid_idx])
        if fit_groups.intersection(valid_groups):
            raise AssertionError("Un prompt aparece en ambos lados de un fold.")
    return folds


def tune_regularization(
    train_df: pd.DataFrame,
    include_model_identity: bool,
    c_grid: Iterable[float] = C_GRID,
    n_splits: int = N_CV_SPLITS,
    max_iter: int = 1_000,
) -> dict[str, Any]:
    """Selecciona C por log loss medio con StratifiedGroupKFold."""
    x, y = prepare_xy(train_df)
    folds = make_cv_splits(train_df, n_splits=n_splits)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    convergence_messages: list[str] = []

    for c_value in c_grid:
        fold_losses: list[float] = []
        fold_times: list[float] = []
        warning_count = 0
        for fit_idx, valid_idx in folds:
            model = build_pipeline(include_model_identity, float(c_value), max_iter)
            fold_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x.iloc[fit_idx], y[fit_idx])
            fold_times.append(time.perf_counter() - fold_start)
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    warning_count += 1
                    convergence_messages.append(str(warning.message))
            probabilities = _ordered_probabilities(model, x.iloc[valid_idx])
            fold_losses.append(
                float(log_loss(y[valid_idx], probabilities, labels=CLASS_ORDER))
            )
        rows.append(
            {
                "C": float(c_value),
                "mean_log_loss": float(np.mean(fold_losses)),
                "std_log_loss": float(np.std(fold_losses, ddof=1)),
                "fold_log_loss": fold_losses,
                "mean_fit_time_seconds": float(np.mean(fold_times)),
                "convergence_warning_count": warning_count,
            }
        )

    best = min(rows, key=lambda row: (row["mean_log_loss"], row["C"]))
    return {
        "best_C": best["C"],
        "results": rows,
        "cv_folds": n_splits,
        "cv_time_seconds": time.perf_counter() - started,
        "convergence_warnings": sorted(set(convergence_messages)),
        "convergence_warning_count": int(
            sum(row["convergence_warning_count"] for row in rows)
        ),
    }


def fit_model(
    train_df: pd.DataFrame,
    include_model_identity: bool,
    c_value: float,
    max_iter: int = 1_000,
) -> tuple[Pipeline, float, list[str]]:
    """Ajusta una variante sobre entrenamiento completo y mide el tiempo."""
    x, y = prepare_xy(train_df)
    model = build_pipeline(include_model_identity, c_value, max_iter)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    elapsed = time.perf_counter() - started
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]
    return model, elapsed, sorted(set(messages))


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    inference_time_seconds: float,
) -> dict[str, Any]:
    """Calcula métricas globales, por clase y matriz de confusión."""
    predictions = CLASS_ORDER[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    per_class = {
        key: {
            "label": int(label),
            "name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, (key, label, name) in enumerate(
            zip(CLASS_KEYS, CLASS_ORDER, CLASS_LABELS_ES)
        )
    }
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASS_ORDER)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=CLASS_ORDER
        ).tolist(),
        "inference_time_seconds": float(inference_time_seconds),
        "n_observations": int(len(y_true)),
    }


def evaluate_model(
    model: Pipeline, df: pd.DataFrame
) -> tuple[dict[str, Any], np.ndarray]:
    """Predice una partición y devuelve métricas y probabilidades ordenadas."""
    x, y = prepare_xy(df)
    started = time.perf_counter()
    probabilities = _ordered_probabilities(model, x)
    elapsed = time.perf_counter() - started
    return evaluate_probabilities(y, probabilities, elapsed), probabilities


def evaluate_prior_baseline(
    train_y: np.ndarray, target_df: pd.DataFrame
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Baseline que predice siempre las proporciones observadas en train."""
    started_fit = time.perf_counter()
    counts = np.bincount(train_y, minlength=len(CLASS_ORDER)).astype(float)
    priors = counts / counts.sum()
    fit_time = time.perf_counter() - started_fit
    started_predict = time.perf_counter()
    probabilities = np.tile(priors, (len(target_df), 1))
    inference_time = time.perf_counter() - started_predict
    metrics = evaluate_probabilities(
        target_df["winner"].to_numpy(dtype=int), probabilities, inference_time
    )
    metrics["class_priors"] = priors.tolist()
    return metrics, probabilities, fit_time


def swap_consistency(model: Pipeline, validation_df: pd.DataFrame) -> dict[str, Any]:
    """Mide consistencia predictiva al intercambiar A y B en validación."""
    original = predict_proba(model, validation_df)
    swapped = predict_proba(model, features.swap_ab(validation_df))
    errors = {
        "p_a_original_vs_p_b_swapped": float(np.mean(np.abs(original[:, 0] - swapped[:, 1]))),
        "p_b_original_vs_p_a_swapped": float(np.mean(np.abs(original[:, 1] - swapped[:, 0]))),
        "p_tie_original_vs_p_tie_swapped": float(np.mean(np.abs(original[:, 2] - swapped[:, 2]))),
    }
    aggregate = float(np.mean(list(errors.values())))
    return {
        "mean_absolute_errors": errors,
        "aggregate_mae": aggregate,
        "consistency_score_1_minus_mae": float(1.0 - aggregate),
        "n_observations": int(len(validation_df)),
    }


def length_analysis(
    validation_df: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, Any]:
    """Resume resultados reales y predichos según extensión relativa."""
    max_len = validation_df[["resp_a_len", "resp_b_len"]].max(axis=1).clip(lower=1)
    relative_difference = (
        validation_df["resp_a_len"] - validation_df["resp_b_len"]
    ) / max_len
    groups = pd.Series("similares (±10%)", index=validation_df.index)
    groups.loc[relative_difference > 0.10] = "A más extensa (>10%)"
    groups.loc[relative_difference < -0.10] = "B más extensa (>10%)"
    group_order = [
        "A más extensa (>10%)",
        "similares (±10%)",
        "B más extensa (>10%)",
    ]
    result: dict[str, Any] = {"similarity_threshold_relative": 0.10, "groups": {}}
    y = validation_df["winner"].to_numpy(dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    for group in group_order:
        mask = (groups == group).to_numpy()
        if not mask.any():
            continue
        observed = np.bincount(y[mask], minlength=3) / mask.sum()
        result["groups"][group] = {
            "n": int(mask.sum()),
            "observed_proportions": dict(zip(CLASS_KEYS, observed.tolist())),
            "mean_predicted_probabilities": dict(
                zip(CLASS_KEYS, probabilities[mask].mean(axis=0).tolist())
            ),
            "accuracy": float(accuracy_score(y[mask], predicted[mask])),
            "log_loss": float(log_loss(y[mask], probabilities[mask], labels=CLASS_ORDER)),
        }

    decisive = y != 2
    unequal = validation_df["resp_a_len"].to_numpy() != validation_df["resp_b_len"].to_numpy()
    diff = validation_df["resp_a_len"].to_numpy() - validation_df["resp_b_len"].to_numpy()
    eligible = decisive & unequal
    longer_wins = ((diff > 0) & (y == 0)) | ((diff < 0) & (y == 1))
    result["eda_historical_longer_response_win_rate"] = 0.6163
    result["validation_longer_response_win_rate"] = float(longer_wins[eligible].mean())
    result["validation_decisive_unequal_n"] = int(eligible.sum())
    return result


def coefficient_analysis(model: Pipeline, train_df: pd.DataFrame) -> dict[str, Any]:
    """Extrae coeficientes estandarizados y pares numéricos correlacionados."""
    classifier = model.named_steps["classifier"]
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = []
    for class_position, class_value in enumerate(classifier.classes_):
        for feature_name, value in zip(transformed_names, classifier.coef_[class_position]):
            coefficients.append(
                {
                    "class": int(class_value),
                    "class_name": CLASS_LABELS_ES[int(class_value)],
                    "feature": str(feature_name),
                    "coefficient": float(value),
                }
            )
    x, _ = prepare_xy(train_df)
    correlations = x[NUMERIC_FEATURES].corr()
    pairs = []
    for i, first in enumerate(NUMERIC_FEATURES):
        for second in NUMERIC_FEATURES[i + 1 :]:
            value = float(correlations.loc[first, second])
            if abs(value) >= 0.90:
                pairs.append({"feature_1": first, "feature_2": second, "correlation": value})
    numeric_coefficients = [
        row for row in coefficients if row["feature"].startswith("numeric__")
    ]
    return {
        "standardized_numeric_coefficients": numeric_coefficients,
        "high_correlation_pairs_abs_ge_0_90": pairs,
        "interpretation_note": (
            "Los coeficientes numéricos corresponden a variables estandarizadas. "
            "La colinealidad y la regularización L2 reparten señal entre variables "
            "redundantes; no deben interpretarse como efectos causales aislados."
        ),
    }


def _prediction_frame(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": df[cfg.COL_ID].to_numpy(),
            "p_a": probabilities[:, 0],
            "p_b": probabilities[:, 1],
            "p_tie": probabilities[:, 2],
        }
    )
    if not np.allclose(output[["p_a", "p_b", "p_tie"]].sum(axis=1), 1.0):
        raise AssertionError("Las probabilidades del CSV no suman uno.")
    return output


def validate_prediction_frame(predictions: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Valida esquema, orden de IDs y probabilidades de una salida."""
    if predictions.columns.tolist() != ["id", "p_a", "p_b", "p_tie"]:
        raise AssertionError("El CSV no tiene las columnas requeridas y en el orden correcto.")
    if predictions["id"].tolist() != expected_ids.tolist():
        raise AssertionError("Los IDs no coinciden con la partición y su orden.")
    probabilities = predictions[["p_a", "p_b", "p_tie"]].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise AssertionError("Hay probabilidades inválidas.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("Las probabilidades no suman uno.")


def _plot_metric_comparison(validation: dict[str, Any], path: Path) -> None:
    models = ["Baseline", "Variante A", "Variante B"]
    keys = ["baseline", "variant_a", "variant_b"]
    colors = ["#9CA3AF", "#4C78A8", "#F58518"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(
        axes,
        ["log_loss", "f1_macro"],
        ["Log loss (menor es mejor)", "F1 macro (mayor es mejor)"],
    ):
        values = [validation[key][metric] for key in keys]
        bars = ax.bar(models, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
        ax.set_xlabel("")
    fig.suptitle("Validación: línea base y regresión logística multinomial", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(metrics: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS_ES,
        yticklabels=CLASS_LABELS_ES,
        ax=ax,
    )
    ax.set_title("Matriz de confusión del modelo final en prueba", fontweight="bold")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class(metrics: dict[str, Any], path: Path) -> None:
    rows = []
    for key in CLASS_KEYS:
        row = metrics["per_class"][key]
        for metric in ["precision", "recall", "f1"]:
            rows.append({"Clase": row["name"], "Métrica": metric.capitalize(), "Valor": row[metric]})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=plot_df, x="Clase", y="Valor", hue="Métrica", palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Rendimiento por clase del modelo final en prueba", fontweight="bold")
    ax.set_ylabel("Valor")
    ax.set_xlabel("")
    ax.legend(title="")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_search(searches: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, label, color in [
        ("variant_a", "Variante A", "#4C78A8"),
        ("variant_b", "Variante B", "#F58518"),
    ]:
        rows = searches[key]["results"]
        x = [row["C"] for row in rows]
        means = [row["mean_log_loss"] for row in rows]
        stds = [row["std_log_loss"] for row in rows]
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=4, label=label, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("C (escala logarítmica)")
    ax.set_ylabel("Log loss medio ± 1 DE")
    ax.set_title("Búsqueda de regularización con validación cruzada agrupada", fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_swap(swap: dict[str, Any], path: Path) -> None:
    errors = swap["mean_absolute_errors"]
    labels = ["P(A) vs P(B)\nintercambiada", "P(B) vs P(A)\nintercambiada", "P(empate)\nantes/después"]
    values = list(errors.values())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Error absoluto medio")
    ax.set_title("Consistencia predictiva bajo intercambio A/B", fontweight="bold")
    ax.axhline(swap["aggregate_mae"], color="#B91C1C", linestyle="--", label=f"MAE agregado = {swap['aggregate_mae']:.4f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_length(length: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [0.8, 1.7]})
    rates = [
        length["eda_historical_longer_response_win_rate"] * 100,
        length["validation_longer_response_win_rate"] * 100,
    ]
    bars = axes[0].bar(["EDA histórico", "Validación"], rates, color=["#9CA3AF", "#4C78A8"])
    axes[0].axhline(50, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel("La respuesta más larga gana (%)")
    axes[0].set_title("Batallas decisivas\ncon longitudes distintas")
    for bar, value in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}%", ha="center", va="bottom")

    groups = list(length["groups"].keys())
    x = np.arange(len(groups))
    for class_key, class_label, color in zip(CLASS_KEYS, CLASS_LABELS_ES, ["#4C78A8", "#F58518", "#54A24B"]):
        observed = [length["groups"][g]["observed_proportions"][class_key] for g in groups]
        predicted = [length["groups"][g]["mean_predicted_probabilities"][class_key] for g in groups]
        axes[1].plot(x, observed, marker="o", color=color, label=f"Real: {class_label}")
        axes[1].plot(x, predicted, marker="x", linestyle="--", color=color, label=f"Pred.: {class_label}")
    axes[1].set_xticks(x, groups, rotation=10, ha="right")
    axes[1].set_ylim(0, 0.65)
    axes[1].set_ylabel("Proporción / probabilidad media")
    axes[1].set_title("Resultado real y probabilidades según longitud relativa")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Del hallazgo de longitud del EDA al experimento predictivo", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(results: dict[str, Any], figures_dir: Path) -> list[str]:
    """Genera las seis figuras estáticas obligatorias."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": figures_dir / "logreg_comparacion_modelos.png",
        "confusion": figures_dir / "logreg_matriz_confusion.png",
        "classes": figures_dir / "logreg_rendimiento_clases.png",
        "cv": figures_dir / "logreg_busqueda_c.png",
        "swap": figures_dir / "logreg_consistencia_ab.png",
        "length": figures_dir / "logreg_longitud_validacion.png",
    }
    _plot_metric_comparison(results["validation_metrics"], paths["comparison"])
    _plot_confusion(results["test_metrics"]["variant_b"], paths["confusion"])
    _plot_per_class(results["test_metrics"]["variant_b"], paths["classes"])
    _plot_cv_search(results["hyperparameter_search"], paths["cv"])
    _plot_swap(results["swap_test"], paths["swap"])
    _plot_length(results["length_analysis_validation"], paths["length"])
    return [str(path) for path in paths.values()]


def _turns_to_text(value: str | list[Any] | tuple[Any, ...] | None) -> tuple[str, int]:
    """Aplica a una entrada nueva la misma unión de turnos del notebook 02."""
    if value is None:
        turns: list[Any] = []
    elif isinstance(value, (list, tuple)):
        turns = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped, strict=False)
                turns = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                turns = [value]
        else:
            turns = [value]
    else:
        turns = [value]
    text = "\n".join("" if turn is None else str(turn) for turn in turns).strip()
    return text, len(turns)


def prepare_inference_row(
    prompt: str | list[Any],
    response_a: str | list[Any],
    response_b: str | list[Any],
    model_a: str,
    model_b: str,
    row_id: str | int = "inference",
) -> pd.DataFrame:
    """Prepara una fila nueva con la misma lógica de limpieza y longitudes."""
    prompt_text, prompt_turns = _turns_to_text(prompt)
    response_a_text, _ = _turns_to_text(response_a)
    response_b_text, _ = _turns_to_text(response_b)
    return pd.DataFrame(
        [
            {
                cfg.COL_ID: row_id,
                cfg.COL_MODEL_A: model_a,
                cfg.COL_MODEL_B: model_b,
                "prompt_text": prompt_text,
                "resp_a_text": response_a_text,
                "resp_b_text": response_b_text,
                "prompt_len": len(prompt_text),
                "resp_a_len": len(response_a_text),
                "resp_b_len": len(response_b_text),
                "prompt_words": len(prompt_text.split()),
                "resp_a_words": len(response_a_text.split()),
                "resp_b_words": len(response_b_text.split()),
                "n_turns": prompt_turns,
            }
        ]
    )


def predict_single(
    model_or_path: Pipeline | str | Path,
    prompt: str | list[Any],
    response_a: str | list[Any],
    response_b: str | list[Any],
    model_a: str,
    model_b: str,
) -> dict[str, float]:
    """Función de inferencia lista para reutilizar desde Streamlit."""
    model = joblib.load(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    row = prepare_inference_row(prompt, response_a, response_b, model_a, model_b)
    probabilities = predict_proba(model, row)[0]
    return dict(zip(["p_a", "p_b", "p_tie"], map(float, probabilities)))


def run_full_experiment(
    root: Path = cfg.ROOT,
    c_grid: Iterable[float] = C_GRID,
    n_splits: int = N_CV_SPLITS,
    max_iter: int = 1_000,
) -> dict[str, Any]:
    """Ejecuta el experimento reproducible completo y escribe artefactos."""
    np.random.seed(SEED)
    train_df = load.load_split("train")
    validation_df = load.load_split("val")
    test_df = load.load_split("test")

    # Prueba inicial pequeña antes del entrenamiento completo.
    smoke_size = min(900, len(train_df))
    smoke_df = train_df.sample(smoke_size, random_state=SEED)
    smoke_model = build_pipeline(True, c_value=0.1, max_iter=max_iter)
    smoke_x, smoke_y = prepare_xy(smoke_df)
    smoke_model.fit(smoke_x, smoke_y)
    smoke_probabilities = _ordered_probabilities(smoke_model, smoke_x.head(20))
    smoke_test = {
        "n_rows": smoke_size,
        "passed": bool(np.allclose(smoke_probabilities.sum(axis=1), 1.0)),
    }

    searches = {
        "variant_a": tune_regularization(train_df, False, c_grid, n_splits, max_iter),
        "variant_b": tune_regularization(train_df, True, c_grid, n_splits, max_iter),
    }
    model_a, fit_time_a, final_warnings_a = fit_model(
        train_df, False, searches["variant_a"]["best_C"], max_iter
    )
    model_b, fit_time_b, final_warnings_b = fit_model(
        train_df, True, searches["variant_b"]["best_C"], max_iter
    )

    train_y = train_df["winner"].to_numpy(dtype=int)
    baseline_val, _, baseline_fit_time = evaluate_prior_baseline(train_y, validation_df)
    val_a, val_prob_a = evaluate_model(model_a, validation_df)
    val_b, val_prob_b = evaluate_model(model_b, validation_df)

    for key, metrics, fit_time in [
        ("variant_a", val_a, fit_time_a),
        ("variant_b", val_b, fit_time_b),
    ]:
        metrics["fit_time_seconds"] = fit_time
        metrics["cv_time_seconds"] = searches[key]["cv_time_seconds"]
        metrics["training_time_total_seconds"] = fit_time + searches[key]["cv_time_seconds"]
    baseline_val["fit_time_seconds"] = baseline_fit_time
    baseline_val["cv_time_seconds"] = 0.0
    baseline_val["training_time_total_seconds"] = baseline_fit_time

    # La selección se fija aquí, antes de cualquier acceso evaluativo a test.
    final_selection = {
        "model": "variant_b",
        "reason": (
            "Variante principal preespecificada: variables numéricas más identidad "
            "de model_a y model_b; test no intervino en la selección."
        ),
        "selected_before_test_evaluation": True,
    }

    baseline_test, _, _ = evaluate_prior_baseline(train_y, test_df)
    test_a, test_prob_a = evaluate_model(model_a, test_df)
    test_b, test_prob_b = evaluate_model(model_b, test_df)
    for key, metrics, fit_time in [
        ("variant_a", test_a, fit_time_a),
        ("variant_b", test_b, fit_time_b),
    ]:
        metrics["fit_time_seconds"] = fit_time
        metrics["cv_time_seconds"] = searches[key]["cv_time_seconds"]
        metrics["training_time_total_seconds"] = fit_time + searches[key]["cv_time_seconds"]
    baseline_test["fit_time_seconds"] = baseline_fit_time
    baseline_test["cv_time_seconds"] = 0.0
    baseline_test["training_time_total_seconds"] = baseline_fit_time

    model_dir = root / "models"
    prediction_dir = root / "results" / "preds"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "docs" / "figures"
    for directory in [model_dir, prediction_dir, metrics_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "logreg_pipeline.joblib"
    joblib.dump(model_b, model_path)
    reloaded = joblib.load(model_path)
    reloaded_probabilities = predict_proba(reloaded, validation_df)
    serialization_max_difference = float(np.max(np.abs(val_prob_b - reloaded_probabilities)))
    if not np.allclose(val_prob_b, reloaded_probabilities, atol=1e-12):
        raise AssertionError("El pipeline recuperado no reproduce las probabilidades.")

    val_predictions = _prediction_frame(validation_df, val_prob_b)
    test_predictions = _prediction_frame(test_df, test_prob_b)
    validate_prediction_frame(val_predictions, validation_df[cfg.COL_ID])
    validate_prediction_frame(test_predictions, test_df[cfg.COL_ID])
    val_path = prediction_dir / "logreg_val.csv"
    test_path = prediction_dir / "logreg_test.csv"
    val_predictions.to_csv(val_path, index=False)
    test_predictions.to_csv(test_path, index=False)

    results: dict[str, Any] = {
        "model_name": "Regresión logística multinomial (Variante B)",
        "seed": SEED,
        "label_order": {"0": "gana A", "1": "gana B", "2": "empate"},
        "probability_order": ["p_a", "p_b", "p_tie"],
        "splits": {
            "source": "data/splits.csv",
            "train_n": len(train_df),
            "validation_n": len(validation_df),
            "test_n": len(test_df),
            "grouping": "prompt_text",
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical_variant_b": CATEGORICAL_FEATURES,
            "forbidden_and_excluded": sorted(FORBIDDEN_FEATURES),
        },
        "estimator": {
            "solver": "lbfgs",
            "regularization": "L2",
            "max_iter": max_iter,
            "C_grid": list(map(float, c_grid)),
        },
        "smoke_test": smoke_test,
        "hyperparameter_search": searches,
        "chosen_hyperparameters": {
            "variant_a": {"C": searches["variant_a"]["best_C"]},
            "variant_b": {"C": searches["variant_b"]["best_C"]},
        },
        "selection": final_selection,
        "validation_metrics": {
            "baseline": baseline_val,
            "variant_a": val_a,
            "variant_b": val_b,
        },
        "test_metrics": {
            "baseline": baseline_test,
            "variant_a": test_a,
            "variant_b": test_b,
        },
        "validation_deltas_variant_b_minus_a": {
            "log_loss": val_b["log_loss"] - val_a["log_loss"],
            "f1_macro": val_b["f1_macro"] - val_a["f1_macro"],
            "accuracy": val_b["accuracy"] - val_a["accuracy"],
        },
        "swap_test": swap_consistency(model_b, validation_df),
        "length_analysis_validation": length_analysis(validation_df, val_prob_b),
        "coefficient_analysis": coefficient_analysis(model_b, train_df),
        "convergence": {
            "variant_a_cv": searches["variant_a"]["convergence_warnings"],
            "variant_b_cv": searches["variant_b"]["convergence_warnings"],
            "variant_a_final": final_warnings_a,
            "variant_b_final": final_warnings_b,
        },
        "artifacts": {
            "validation_predictions": str(val_path.relative_to(root)).replace("\\", "/"),
            "test_predictions": str(test_path.relative_to(root)).replace("\\", "/"),
            "pipeline": str(model_path.relative_to(root)).replace("\\", "/"),
            "serialization_max_probability_difference": serialization_max_difference,
        },
        "versions": {"scikit_learn": sklearn.__version__},
        "methodological_notes": {
            "test_usage": "Evaluado una sola vez después de fijar la Variante B como modelo final.",
            "causality": "Las asociaciones predictivas no demuestran causalidad ni que mayor longitud implique mayor calidad.",
        },
    }
    results["figures"] = generate_figures(results, figures_dir)
    metrics_path = metrics_dir / "logreg.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(results), file, ensure_ascii=False, indent=2)

    # Lectura final para detectar archivos JSON incompletos o inválidos.
    with metrics_path.open(encoding="utf-8") as file:
        loaded_results = json.load(file)
    if loaded_results["probability_order"] != ["p_a", "p_b", "p_tie"]:
        raise AssertionError("Orden de probabilidades incorrecto en el JSON.")
    return loaded_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Ejecuta el experimento completo.")
    parser.add_argument("--cv-folds", type=int, default=N_CV_SPLITS)
    parser.add_argument("--max-iter", type=int, default=1_000)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    results = run_full_experiment(n_splits=args.cv_folds, max_iter=args.max_iter)
    summary = {
        "best_C_A": results["chosen_hyperparameters"]["variant_a"]["C"],
        "best_C_B": results["chosen_hyperparameters"]["variant_b"]["C"],
        "validation_B": results["validation_metrics"]["variant_b"],
        "test_B": results["test_metrics"]["variant_b"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
