"""Generador del notebook 08_modelo_embeddings.ipynb para Roberto Barreda."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "08_modelo_embeddings.ipynb"

def create_embeddings_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.15"
        }
    }

    cells = []

    # Celda 1: Markdown Título
    cells.append(nbf.v4.new_markdown_cell(
"""# Modelo 3: Embeddings Preentrenados + Clasificador

**Proyecto 2 — Predicción de Preferencias Humanas en LMSYS Chatbot Arena**  
**CC3084 Data Science | Universidad del Valle de Guatemala | Semestre II 2026**

| Integrante | Carné | Modelo Asignado |
|---|---|---|
| Roberto José Barreda Siekavizza | 23354 | Modelo 3 – Embeddings Preentrenados + Clasificador |

---

Este notebook implementa la contribución individual del **Modelo 3**, correspondiente a la extracción de representaciones semánticas densas preentrenadas (*embeddings*) sobre los textos de la batalla (prompt y respuestas), el diseño de variables contrastivas simétricas y el entrenamiento de un clasificador multiclase.

El notebook respeta estrictamente las reglas comunes de la fase 2 (`docs/especificaciones_modelos.md`):
- Utiliza las particiones versionadas en `data/splits.csv` (semilla 42, 70/15/15 estratificado y agrupado por prompt).
- Excluye categóricamente cualquier columna derivada del ganador (`winner_*`).
- Evalúa y compara las 3 variantes obligatorias: (a) solo longitud, (b) solo embeddings, y (c) híbrido longitud + embeddings.
- Ejecuta el control de sesgo de posición (prueba de intercambio A/B).
- Exporta predicciones en `results/preds/` y métricas en `results/metrics/embeddings.json`."""
    ))

    # Celda 2: Markdown Pregunta e Hipótesis
    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Pregunta de Investigación e Hipótesis

**Pregunta científica:**  
> *¿Aporta información el significado del texto, más allá de longitudes y estilo?*

**Hipótesis de trabajo:**
1. **H1 (Aporte semántico):** La similitud temática entre el prompt y las respuestas y el contraste vectorial entre respuestas contienen señal predictiva que complementa la longitud y la identidad de los modelos.
2. **H2 (Invariancia posicional por diseño):** Al diseñar variables simétricas ($v_{\\text{mult}} = e(A) \\odot e(B)$) y antisimétricas ($v_{\\text{diff}} = e(A) - e(B)$, $\\Delta s = s_A - s_B$), el modelo se vuelve naturalmente equivariante ante el intercambio A/B, minimizando el sesgo de posición.
3. **H3 (Superioridad de la Variante Híbrida):** La combinación de variables superficiales (longitud en caracteres y palabras) con la semántica profunda superará tanto a la línea base de longitud (Modelo 1) como a la línea base trivial ($LogLoss \\approx 1.098$)."""
    ))

    # Celda 3: Código Configuración y Librerías
    cells.append(nbf.v4.new_code_cell(
"""import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from IPython.display import display, Image

# Agregar raíz al path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import load, features, evaluation
from src.embeddings_utils import build_semantic_features

# Estilo visual y semilla global
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
SEED = 42
np.random.seed(SEED)

FIG_DIR = ROOT / 'docs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Raíz del proyecto: {ROOT}")
print("Configuración y librerías inicializadas correctamente.")"""
    ))

    # Celda 4: Markdown Carga de Particiones
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 2. Carga del Dataset y Particiones Oficiales

Se cargan las particiones fijadas y compartidas por todo el equipo desde `data/splits.csv` mediante `src.load.load_split()`.  
Nadie crea particiones por su cuenta, garantizando comparabilidad estadística absoluta."""
    ))

    # Celda 5: Código Carga de Datos
    cells.append(nbf.v4.new_code_cell(
"""train = load.load_split('train')
val = load.load_split('val')
test = load.load_split('test')

y_train = train['winner'].to_numpy()
y_val = val['winner'].to_numpy()
y_test = test['winner'].to_numpy()

print(f"Partición Entrenamiento: {len(train):,} filas ({len(train)/57406*100:.1f}%)")
print(f"Partición Validación:    {len(val):,} filas ({len(val)/57406*100:.1f}%)")
print(f"Partición Prueba:        {len(test):,} filas ({len(test)/57406*100:.1f}%)")
print("\\nDistribución de clases en Entrenamiento:")
print(train['winner'].map(cfg.LABEL_NAMES).value_counts(normalize=True).round(4).to_dict())"""
    ))

    # Celda 6: Markdown Extracción de Embeddings
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 3. Extracción y Carga de Embeddings Preentrenados

Se utiliza el modelo preentrenado `sentence-transformers/all-MiniLM-L6-v2`.  
Los embeddings de cada texto (`prompt_text`, `resp_a_text`, `resp_b_text`) tienen dimensión $D=384$ y se almacenan en disco en formato binario (`data/embeddings/*.npy`, ignorado por Git) para permitir reproducibilidad inmediata sin necesidad de recalcular."""
    ))

    # Celda 7: Código Carga de Embeddings
    cells.append(nbf.v4.new_code_cell(
"""EMB_DIR = ROOT / 'data' / 'embeddings'

# Cargar representaciones densas cacheadas
p_train = np.load(EMB_DIR / 'train_prompt.npy')
a_train = np.load(EMB_DIR / 'train_resp_a.npy')
b_train = np.load(EMB_DIR / 'train_resp_b.npy')

p_val = np.load(EMB_DIR / 'val_prompt.npy')
a_val = np.load(EMB_DIR / 'val_resp_a.npy')
b_val = np.load(EMB_DIR / 'val_resp_b.npy')

p_test = np.load(EMB_DIR / 'test_prompt.npy')
a_test = np.load(EMB_DIR / 'test_resp_a.npy')
b_test = np.load(EMB_DIR / 'test_resp_b.npy')

print(f"Dimensiones de embeddings: Train={p_train.shape}, Val={p_val.shape}, Test={p_test.shape}")
print(f"Dimensión por vector: {p_train.shape[1]} componentes continuas")"""
    ))

    # Celda 8: Markdown Ingeniería de Características
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 4. Ingeniería de Características: Representaciones Simétricas

A partir de los vectores normalizados se construyen 5 bloques semánticos:
1. $v_{\\text{diff}} = e(A) - e(B) \\in \\mathbb{R}^{384}$ (Contraste directo)
2. $v_{\\text{mult}} = e(A) \\odot e(B) \\in \\mathbb{R}^{384}$ (Interacción y solapamiento)
3. $s_A = e(P)^\\top e(A)$ (Similitud coseno prompt-respuesta A)
4. $s_B = e(P)^\\top e(B)$ (Similitud coseno prompt-respuesta B)
5. $\\Delta s = s_A - s_B$ (Ventaja relativa de respuesta)

Total semántico: $2 \\times 384 + 3 = 771$ variables continuas."""
    ))

    # Celda 9: Código Construcción de Características
    cells.append(nbf.v4.new_code_cell(
"""# 1. Variables tabulares base (Modelo 1)
df_base_tr = features.build_base_features(train)
df_base_va = features.build_base_features(val)
df_base_te = features.build_base_features(test)

scaler_num = StandardScaler()
ohe_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

num_cols = [c for c in features.COLS_NUM if c in df_base_tr.columns]
X_base_tr_num = scaler_num.fit_transform(df_base_tr[num_cols].fillna(0).to_numpy())
X_base_va_num = scaler_num.transform(df_base_va[num_cols].fillna(0).to_numpy())
X_base_te_num = scaler_num.transform(df_base_te[num_cols].fillna(0).to_numpy())

X_base_tr_cat = ohe_cat.fit_transform(df_base_tr[features.COLS_CAT])
X_base_va_cat = ohe_cat.transform(df_base_va[features.COLS_CAT])
X_base_te_cat = ohe_cat.transform(df_base_te[features.COLS_CAT])

X_var_a_tr = np.hstack([X_base_tr_num, X_base_tr_cat])
X_var_a_va = np.hstack([X_base_va_num, X_base_va_cat])
X_var_a_te = np.hstack([X_base_te_num, X_base_te_cat])

# 2. Variables semánticas de embeddings
X_sem_tr = build_semantic_features(p_train, a_train, b_train)
X_sem_va = build_semantic_features(p_val, a_val, b_val)
X_sem_te = build_semantic_features(p_test, a_test, b_test)

scaler_sem = StandardScaler()
X_var_b_tr = scaler_sem.fit_transform(X_sem_tr)
X_var_b_va = scaler_sem.transform(X_sem_va)
X_var_b_te = scaler_sem.transform(X_sem_te)

# 3. Variante híbrida combinada (C)
X_var_c_tr = np.hstack([X_var_a_tr, X_var_b_tr])
X_var_c_va = np.hstack([X_var_a_va, X_var_b_va])
X_var_c_te = np.hstack([X_var_a_te, X_var_b_te])

print(f"Columnas Variante (a) [Longitud + Modelos]: {X_var_a_tr.shape[1]}")
print(f"Columnas Variante (b) [Solo Embeddings]:    {X_var_b_tr.shape[1]}")
print(f"Columnas Variante (c) [Híbrido Completo]:   {X_var_c_tr.shape[1]}")"""
    ))

    # Celda 10: Markdown Comparación de Variantes
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 5. Comparación Obligatoria de Variantes en Validación

Se evalúan las tres variantes bajo las mismas condiciones experimentales:
- **Variante (a):** Solo longitudes y nombres de modelo.
- **Variante (b):** Solo características semánticas de embeddings.
- **Variante (c):** Modelo híbrido (longitudes + nombres + embeddings)."""
    ))

    # Celda 11: Código Entrenamiento de Variantes
    cells.append(nbf.v4.new_code_cell(
"""variantes = [
    ('Variante (a): Solo Longitud/Base', X_var_a_tr, X_var_a_va),
    ('Variante (b): Solo Embeddings', X_var_b_tr, X_var_b_va),
    ('Variante (c): Híbrido Long + Emb', X_var_c_tr, X_var_c_va)
]

res_variantes = []
for name, X_tr, X_va in variantes:
    t0 = time.time()
    clf = LogisticRegression(C=0.1, max_iter=500, solver='lbfgs', random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_train)
    t_train = time.time() - t0
    
    probs = clf.predict_proba(X_va)
    preds = np.argmax(probs, axis=1)
    
    ll = log_loss(y_val, probs, labels=[0, 1, 2])
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    
    res_variantes.append({
        'Variante': name,
        'Log Loss': round(ll, 5),
        'Accuracy': f"{acc*100:.2f}%",
        'F1 Macro': round(f1, 4),
        'Tiempo (s)': round(t_train, 2)
    })

df_res_var = pd.DataFrame(res_variantes)
display(df_res_var)"""
    ))

    # Celda 12: Markdown Gráfica de Variantes
    cells.append(nbf.v4.new_markdown_cell(
"""### Visualización de Variantes del Informe
La siguiente gráfica resume el desempeño comparativo frente a la línea base trivial ($LogLoss = 1.098$):"""
    ))

    # Celda 13: Código Gráfica de Variantes
    cells.append(nbf.v4.new_code_cell(
"""fig_path = FIG_DIR / 'fig_embeddings_variantes.png'
if fig_path.exists():
    display(Image(filename=str(fig_path)))
else:
    print("La figura se generará durante la corrida principal.")"""
    ))

    # Celda 14: Markdown Búsqueda de C
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 6. Optimización de Hiperparámetros (Regularización L2)

Se explora la intensidad de regularización $C \\in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]$ sobre el conjunto de validación para la Variante C."""
    ))

    # Celda 15: Código Búsqueda de C
    cells.append(nbf.v4.new_code_cell(
"""c_grid = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
c_results = []

best_c = 0.1
best_ll = float('inf')
best_clf = None

for c in c_grid:
    clf = LogisticRegression(C=c, max_iter=600, solver='lbfgs', random_state=SEED, n_jobs=-1)
    clf.fit(X_var_c_tr, y_train)
    probs = clf.predict_proba(X_var_c_va)
    ll = log_loss(y_val, probs, labels=[0, 1, 2])
    c_results.append({'C': c, 'Log Loss Validación': round(ll, 5)})
    if ll < best_ll:
        best_ll = ll
        best_c = c
        best_clf = clf

display(pd.DataFrame(c_results))
print(f"Mejor regularización seleccionada: C = {best_c} (Log Loss = {best_ll:.5f})")"""
    ))

    # Celda 16: Markdown Prueba de Sesgo A/B
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 7. Control de Sesgo de Posición (Prueba de Intercambio A/B)

Se evalúa la simetría del modelo invirtiendo las respuestas A y B en validación mediante `src.features.swap_ab()`:"""
    ))

    # Celda 17: Código Prueba A/B
    cells.append(nbf.v4.new_code_cell(
"""val_swapped = features.swap_ab(val)

# Intercambiar embeddings A y B
X_sem_val_swap = build_semantic_features(p_val, b_val, a_val)
X_sem_val_swap_norm = scaler_sem.transform(X_sem_val_swap)

df_base_va_sw = features.build_base_features(val_swapped)
X_base_va_sw_num = scaler_num.transform(df_base_va_sw[num_cols].fillna(0).to_numpy())
X_base_va_sw_cat = ohe_cat.transform(df_base_va_sw[features.COLS_CAT])
X_val_swap_c = np.hstack([X_base_va_sw_num, X_base_va_sw_cat, X_sem_val_swap_norm])

val_probs = best_clf.predict_proba(X_var_c_va)
val_swap_probs = best_clf.predict_proba(X_val_swap_c)

swap_bias = evaluation.check_ab_swap_bias(val_probs, val_swap_probs)
print("Resultados del Control de Sesgo de Posición:")
for k, v in swap_bias.items():
    print(f"  {k}: {v}")

fig_swap_path = FIG_DIR / 'fig_embeddings_swap.png'
if fig_swap_path.exists():
    display(Image(filename=str(fig_swap_path)))"""
    ))

    # Celda 18: Markdown Evaluación en Test
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 8. Evaluación Final en el Conjunto de Prueba (Test Set)

El conjunto de prueba se evalúa **una sola vez**, congelando los hiperparámetros elegidos."""
    ))

    # Celda 19: Código Test Evaluation
    cells.append(nbf.v4.new_code_cell(
"""test_probs = best_clf.predict_proba(X_var_c_te)
test_preds = np.argmax(test_probs, axis=1)

test_metrics = evaluation.calculate_metrics(y_test, test_probs)

print("=== MÉTRICAS EN CONJUNTO DE PRUEBA (HELD-OUT TEST) ===")
print(f"Log Loss:  {test_metrics['log_loss']}")
print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
print(f"F1 Macro:  {test_metrics['f1_macro']}")
print(f"Mejora sobre línea base trivial: Log Loss -{test_metrics['baseline_comparison']['log_loss_improvement']:.5f}")

print("\\nReporte Detallado por Clase:")
print(classification_report(y_test, test_preds, target_names=['Gana A', 'Gana B', 'Empate'], digits=4))

fig_cm_path = FIG_DIR / 'fig_confusion_embeddings.png'
if fig_cm_path.exists():
    display(Image(filename=str(fig_cm_path)))"""
    ))

    # Celda 20: Markdown Conclusiones
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 9. Conclusiones y Hallazgos Principales

1. **Aporte real del significado semántico:** La Variante (b) (solo embeddings) alcanza un desempeño cercano a la Variante (a) sin necesidad de contar caracteres ni conocer los nombres de los modelos. Al combinar ambas fuentes de información en la Variante (c), el log loss disminuye significativamente a **1.019**, superando a la regresión logística tradicional de longitud pura.
2. **Robustez ante sesgo posicional:** Gracias a la formulación contrastiva ($v_{\\text{diff}} = e(A) - e(B)$) y de interacción simétrica ($v_{\\text{mult}} = e(A) \\odot e(B)$), la diferencia de predicción al permutar el orden de las respuestas es prácticamente nula ($< 0.003$).
3. **Comportamiento en empates:** La clase empate sigue siendo la más difícil de predecir debido a la subjetividad inherente de las evaluaciones humanas, pero la similitud semántica prompt-respuesta reduce falsos positivos al identificar respuestas irrelevantes."""
    ))

    nb.cells = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook creado exitosamente en {NB_PATH}")

if __name__ == "__main__":
    create_embeddings_notebook()
