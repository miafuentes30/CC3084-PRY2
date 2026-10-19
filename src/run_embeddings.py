"""Script de ejecución del Modelo 3: Embeddings preentrenados + Clasificador.

Responsable: Roberto Barreda (23354)
Genera las predicciones, métricas y figuras estáticas requeridas por el informe.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import evaluation, features, load
from src.embeddings_utils import build_semantic_features
from src.fast_embed import FastEmbedder

# Fijar semilla global de reproducibilidad
SEED = 42
np.random.seed(SEED)

EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = ROOT / "results" / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = ROOT / "results" / "metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2 (ONNX DirectML GPU)"


def encode_or_load(df: pd.DataFrame, split_name: str, embedder: FastEmbedder):
    """Codifica los textos de prompt, response_a y response_b o los carga desde caché en disco."""
    p_path = EMB_DIR / f"{split_name}_prompt.npy"
    a_path = EMB_DIR / f"{split_name}_resp_a.npy"
    b_path = EMB_DIR / f"{split_name}_resp_b.npy"

    if p_path.exists() and a_path.exists() and b_path.exists():
        print(f"[{split_name.upper()}] Cargando embeddings cacheados desde {EMB_DIR}...")
        p_emb = np.load(p_path)
        a_emb = np.load(a_path)
        b_emb = np.load(b_path)
    else:
        print(f"\n[{split_name.upper()}] Codificando {len(df):,} filas con GPU DirectML...")
        t0 = time.time()
        print("  -> Codificando prompts...")
        p_emb = embedder.encode(df["prompt_text"].tolist())
        print("  -> Codificando response_a...")
        a_emb = embedder.encode(df["resp_a_text"].tolist())
        print("  -> Codificando response_b...")
        b_emb = embedder.encode(df["resp_b_text"].tolist())
        t_tot = time.time() - t0
        print(f"[{split_name.upper()}] Codificación completa en {t_tot:.1f}s ({len(df)*3/t_tot:.1f} textos/s).")

        np.save(p_path, p_emb)
        np.save(a_path, a_emb)
        np.save(b_path, b_emb)

    return p_emb, a_emb, b_emb


def main():
    print("=== INICIANDO PIPELINE MODELO 3: EMBEDDINGS (ROBERTO BARREDA) ===")
    t_start = time.time()

    # 1. Cargar particiones fijadas
    train = load.load_split("train")
    val = load.load_split("val")
    test = load.load_split("test")

    y_train = train["winner"].to_numpy()
    y_val = val["winner"].to_numpy()
    y_test = test["winner"].to_numpy()

    print(f"Particiones: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    # 2. Cargar modelo de embeddings acelerado y codificar
    embedder = FastEmbedder(batch_size=128, max_length=256)
    t_enc_0 = time.time()
    p_train, a_train, b_train = encode_or_load(train, "train", embedder)
    p_val, a_val, b_val = encode_or_load(val, "val", embedder)
    p_test, a_test, b_test = encode_or_load(test, "test", embedder)
    t_encoding = time.time() - t_enc_0

    # 3. Construir variables para cada variante
    print("\n--- Construyendo matrices de características para Variantes A, B y C ---")

    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    df_base_tr = features.build_base_features(train)
    df_base_va = features.build_base_features(val)
    df_base_te = features.build_base_features(test)

    num_cols = [c for c in features.COLS_NUM if c in df_base_tr.columns]
    X_base_tr_num = scaler.fit_transform(df_base_tr[num_cols].fillna(0).to_numpy())
    X_base_va_num = scaler.transform(df_base_va[num_cols].fillna(0).to_numpy())
    X_base_te_num = scaler.transform(df_base_te[num_cols].fillna(0).to_numpy())

    X_base_tr_cat = ohe.fit_transform(df_base_tr[features.COLS_CAT])
    X_base_va_cat = ohe.transform(df_base_va[features.COLS_CAT])
    X_base_te_cat = ohe.transform(df_base_te[features.COLS_CAT])

    X_var_a_tr = np.hstack([X_base_tr_num, X_base_tr_cat])
    X_var_a_va = np.hstack([X_base_va_num, X_base_va_cat])
    X_var_a_te = np.hstack([X_base_te_num, X_base_te_cat])

    # Matriz semántica de embeddings
    X_sem_tr = build_semantic_features(p_train, a_train, b_train)
    X_sem_va = build_semantic_features(p_val, a_val, b_val)
    X_sem_te = build_semantic_features(p_test, a_test, b_test)

    scaler_sem = StandardScaler()
    X_var_b_tr = scaler_sem.fit_transform(X_sem_tr)
    X_var_b_va = scaler_sem.transform(X_sem_va)
    X_var_b_te = scaler_sem.transform(X_sem_te)

    # Matriz combinada (Variante C)
    X_var_c_tr = np.hstack([X_var_a_tr, X_var_b_tr])
    X_var_c_va = np.hstack([X_var_a_va, X_var_b_va])
    X_var_c_te = np.hstack([X_var_a_te, X_var_b_te])

    print("Dimensiones de características:")
    print(f"  Variante (a) solo longitud/base: {X_var_a_tr.shape[1]} columnas")
    print(f"  Variante (b) solo embeddings:    {X_var_b_tr.shape[1]} columnas")
    print(f"  Variante (c) híbrido completo:   {X_var_c_tr.shape[1]} columnas")

    # 4. Ajuste y comparación de variantes en validación
    print("\n--- Entrenando y evaluando variantes en Validación ---")
    variantes_res = {}

    for name, X_tr, X_va in [
        ("Variante (a): Solo Longitud/Base", X_var_a_tr, X_var_a_va),
        ("Variante (b): Solo Embeddings", X_var_b_tr, X_var_b_va),
        ("Variante (c): Híbrido Longitud + Embeddings", X_var_c_tr, X_var_c_va),
    ]:
        t0 = time.time()
        clf = LogisticRegression(
            C=0.1,
            max_iter=500,
            solver="lbfgs",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0

        probs_va = clf.predict_proba(X_va)
        preds_va = np.argmax(probs_va, axis=1)

        ll = log_loss(y_val, probs_va, labels=[0, 1, 2])
        acc = accuracy_score(y_val, preds_va)
        f1 = f1_score(y_val, preds_va, average="macro")

        variantes_res[name] = {
            "log_loss": round(float(ll), 5),
            "accuracy": round(float(acc), 5),
            "f1_macro": round(float(f1), 5),
            "train_time_sec": round(fit_time, 2),
        }
        print(f"  -> {name}: Log Loss = {ll:.5f} | Acc = {acc*100:.2f}% | F1 Macro = {f1:.4f} (en {fit_time:.2f}s)")

    # 5. Ajuste fino de hiperparámetro C para la variante híbrida (c)
    print("\n--- Búsqueda de regularización C para Variante C en Validación ---")
    c_candidates = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_c = 0.1
    best_ll = float("inf")
    best_clf = None

    for c in c_candidates:
        clf = LogisticRegression(
            C=c,
            max_iter=600,
            solver="lbfgs",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_var_c_tr, y_train)
        probs_va = clf.predict_proba(X_var_c_va)
        ll = log_loss(y_val, probs_va, labels=[0, 1, 2])
        print(f"  C={c:<6} -> Val Log Loss = {ll:.5f}")
        if ll < best_ll:
            best_ll = ll
            best_c = c
            best_clf = clf

    print(f"Mejor C seleccionado: {best_c} con Log Loss = {best_ll:.5f}")

    # Medir tiempo de entrenamiento final e inferencia
    t_train_0 = time.time()
    best_clf.fit(X_var_c_tr, y_train)
    t_train_final = time.time() - t_train_0

    t_inf_0 = time.time()
    val_probs = best_clf.predict_proba(X_var_c_va)
    t_inf_val = time.time() - t_inf_0

    t_inf_0_te = time.time()
    test_probs = best_clf.predict_proba(X_var_c_te)
    t_inf_test = time.time() - t_inf_0_te

    # 6. Prueba de sesgo de posición (A/B swap) en validación
    print("\n--- Evaluando prueba de intercambio A/B (Sesgo de posición) ---")
    val_swapped = features.swap_ab(val)

    X_sem_val_swap = build_semantic_features(p_val, b_val, a_val)
    X_sem_val_swap_norm = scaler_sem.transform(X_sem_val_swap)

    df_base_va_sw = features.build_base_features(val_swapped)
    X_base_va_sw_num = scaler.transform(df_base_va_sw[num_cols].fillna(0).to_numpy())
    X_base_va_sw_cat = ohe.transform(df_base_va_sw[features.COLS_CAT])
    X_val_swap_c = np.hstack([X_base_va_sw_num, X_base_va_sw_cat, X_sem_val_swap_norm])

    val_swap_probs = best_clf.predict_proba(X_val_swap_c)
    swap_bias_res = evaluation.check_ab_swap_bias(val_probs, val_swap_probs)
    print(f"  Diferencia promedio de swap: {swap_bias_res['mean_absolute_swap_diff']:.5f}")
    print(f"  Detalle: |P(A)-P(B)|={swap_bias_res['p_a_to_p_b_diff']:.5f}, Empate={swap_bias_res['p_tie_invariance_diff']:.5f}")

    # 7. Cálculo de métricas completas en Validación y Test
    val_metrics = evaluation.calculate_metrics(y_val, val_probs)
    test_metrics = evaluation.calculate_metrics(y_test, test_probs)

    print("\n--- Resultados Finales en Prueba (Test Set) ---")
    print(f"  Log Loss: {test_metrics['log_loss']}")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Macro: {test_metrics['f1_macro']}")
    print(f"  Mejora sobre línea base trivial: Log Loss -{test_metrics['baseline_comparison']['log_loss_improvement']:.5f}")

    # 8. Generar y guardar figuras estáticas
    print("\n--- Generando figuras estáticas para el informe ---")

    # Figura 1: Matriz de confusión en test
    y_test_pred = np.argmax(test_probs, axis=1)
    fig_cm_path = FIG_DIR / "fig_confusion_embeddings.png"
    evaluation.plot_and_save_confusion_matrix(
        y_test,
        y_test_pred,
        model_title="Embeddings Preentrenados + Regresión Logística (Test)",
        output_path=fig_cm_path,
    )
    print(f"  Guardada: {fig_cm_path}")

    # Figura 2: Comparación obligatoria de variantes (a vs b vs c)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    var_names = list(variantes_res.keys())
    ll_values = [variantes_res[k]["log_loss"] for k in var_names]
    f1_values = [variantes_res[k]["f1_macro"] for k in var_names]
    labels_clean = ["(a) Solo Longitud", "(b) Solo Embeddings", "(c) Híbrido (Long + Emb)"]

    x = np.arange(len(labels_clean))
    width = 0.35

    rects1 = ax.bar(x - width/2, ll_values, width, label="Log Loss (menor es mejor)", color="#4C72B0")
    ax2 = ax.twinx()
    rects2 = ax2.bar(x + width/2, f1_values, width, label="F1 Macro (mayor es mejor)", color="#55A868")

    # Línea base trivial
    ax.axhline(1.0986, color="#C44E52", linestyle="--", alpha=0.8, label="Línea base trivial (1.098)")

    ax.set_ylabel("Log Loss", fontsize=11, color="#4C72B0")
    ax2.set_ylabel("F1 Macro", fontsize=11, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=10, fontweight="bold")
    ax.set_title("Comparación de Variantes – Modelo 3 (Embeddings)", fontsize=12, fontweight="bold", pad=12)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.95, 1.15)
    ax2.set_ylim(0.25, 0.45)
    ax2.grid(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig_var_path = FIG_DIR / "fig_embeddings_variantes.png"
    plt.savefig(fig_var_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {fig_var_path}")

    # Figura 3: Control de sesgo de posición A/B swap
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    indices = np.random.RandomState(SEED).choice(len(val_probs), size=100, replace=False)
    ax.scatter(val_probs[indices, 0], val_swap_probs[indices, 1], alpha=0.7, color="#4C72B0", edgecolors="none", s=25, label="P(A|orig) vs P(B|swap)")
    ax.scatter(val_probs[indices, 1], val_swap_probs[indices, 0], alpha=0.7, color="#DD8452", edgecolors="none", s=25, label="P(B|orig) vs P(A|swap)")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Simetría ideal (y = x)")
    ax.set_title("Control de Sesgo de Posición – Modelo 3 (Prueba A/B Swap)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Probabilidad en Configuración Original", fontsize=10)
    ax.set_ylabel("Probabilidad en Configuración Invertida", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig_swap_path = FIG_DIR / "fig_embeddings_swap.png"
    plt.savefig(fig_swap_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {fig_swap_path}")

    # 9. Guardar predicciones en formato requerido (results/preds/)
    f_val_preds = PRED_DIR / "embeddings_val.csv"
    f_test_preds = PRED_DIR / "embeddings_test.csv"
    evaluation.save_predictions_csv(val[cfg.COL_ID], val_probs, f_val_preds)
    evaluation.save_predictions_csv(test[cfg.COL_ID], test_probs, f_test_preds)
    print(f"  Guardadas predicciones: {f_val_preds} y {f_test_preds}")

    # 10. Guardar archivo de métricas JSON (results/metrics/embeddings.json)
    full_metrics = {
        "model_id": "embeddings",
        "model_name": "Embeddings Preentrenados (all-MiniLM-L6-v2) + Regresion Logistica",
        "responsible": "Roberto Barreda (23354)",
        "hardware": "NVIDIA GeForce RTX 5050 (DirectML GPU) + AMD Ryzen 5 8400F",
        "hyperparameters": {
            "embedding_model": MODEL_NAME,
            "embedding_dim": 384,
            "classifier": "LogisticRegression(multi_class='multinomial')",
            "C": best_c,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 600,
            "random_state": SEED,
        },
        "times": {
            "total_elapsed_sec": round(time.time() - t_start, 2),
            "encoding_sec": round(t_encoding, 2),
            "training_sec": round(t_train_final, 2),
            "val_inference_sec": round(t_inf_val, 4),
            "test_inference_sec": round(t_inf_test, 4),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "variants_comparison": variantes_res,
        "position_bias_swap_check": swap_bias_res,
        "model_drive_link": "https://drive.google.com/drive/folders/cc3084_embeddings_roberto_checkpoints",
    }

    f_metrics = METRIC_DIR / "embeddings.json"
    evaluation.save_metrics_json(full_metrics, f_metrics)
    print(f"  Guardadas métricas: {f_metrics}")

    print("\n=== PIPELINE DE MODELO 3 COMPLETADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
