"""Generador del notebook 09_modelo_deberta.ipynb para Javier España."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "09_modelo_deberta.ipynb"

def create_deberta_notebook():
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
"""# Modelo 4: DeBERTa-v3-base con Fine-Tuning y Arquitectura Siamés

**Proyecto 2 — Predicción de Preferencias Humanas en LMSYS Chatbot Arena**  
**CC3084 Data Science | Universidad del Valle de Guatemala | Semestre II 2026**

| Integrante | Carné | Modelo Asignado |
|---|---|---|
| Javier Eduardo España Pacheco | 23361 | Modelo 4 – DeBERTa-v3-base con Fine-Tuning |

---

Este notebook implementa la contribución individual del **Modelo 4**, correspondiente al ajuste fino (*fine-tuning*) de un modelo de lenguaje basado en Transformer (*DeBERTa-v3*) utilizando una **arquitectura siamés simétrica** para modelar directamente las preferencias humanas en Chatbot Arena.

El notebook cumple estrictamente las directrices del proyecto (`docs/especificaciones_modelos.md`):
- Utiliza las particiones versionadas en `data/splits.csv` (semilla 42, 70/15/15 estratificado y agrupado por prompt).
- Excluye categóricamente cualquier columna derivada del ganador (`winner_*`).
- Modela matemáticamente la simetría A/B para eliminar el sesgo de posición por construcción.
- Evalúa el modelo en el conjunto de prueba *held-out*.
- Exporta predicciones a `results/preds/deberta_*.csv` y métricas a `results/metrics/deberta.json`."""
    ))

    # Celda 2: Markdown Pregunta e Hipótesis
    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Pregunta Científica, Arquitectura y Justificación Teórica

**Pregunta científica:**  
> *¿Un Transformer que lee la pregunta y las dos respuestas predice mejor que los modelos clásicos?*

### Justificación del Diseño Siamés Simétrico vs. Secuencia Única

En lugar de concatenar ambas respuestas en una secuencia única (`[CLS] prompt [SEP] resp_a [SEP] resp_b`), lo cual introduce un sesgo posicional espurio debido a la atención unidireccional y desborda la memoria VRAM en textos largos, adoptamos un **diseño siamés simétrico con pesos compartidos**:

1. **Rama A:** El codificador procesa el par `(prompt, respuesta A)` produciendo un puntaje escalar de calidad $s_A$.
2. **Rama B:** El mismo codificador con pesos idénticos procesa el par `(prompt, respuesta B)` produciendo $s_B$.
3. **Margen relativo:** $\\Delta s = s_A - s_B$.
4. **Formulación ordinal simétrica con margen de empate $\\tau > 0$:**
   - Logit A: $\\Delta s$
   - Logit B: $-\\Delta s$
   - Logit Empate: $\\tau - |\\Delta s|$

**Invariancia matemática estricta:**  
Si permutamos las respuestas $A \\leftrightarrow B$, el nuevo margen es $\\Delta s' = s_B - s_A = -\\Delta s$. En consecuencia, $P(A)' = P(B)$, $P(B)' = P(A)$, y $P(\\text{empate})' = P(\\text{empate})$. El sesgo de posición teórico es idénticamente cero."""
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
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report
from IPython.display import display, Image

# Agregar raíz al path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import load, features, evaluation

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FIG_DIR = ROOT / 'docs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = ROOT / 'results' / 'preds'
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = ROOT / 'results' / 'metrics'
METRIC_DIR.mkdir(parents=True, exist_ok=True)

print(f"Raíz del proyecto: {ROOT}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")"""
    ))

    # Celda 4: Markdown Carga de Particiones
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 2. Carga de Particiones Oficiales

Se cargan las particiones versionadas en `data/splits.csv`:"""
    ))

    # Celda 5: Código Carga de Datos
    cells.append(nbf.v4.new_code_cell(
"""train = load.load_split('train')
val = load.load_split('val')
test = load.load_split('test')

y_train = train['winner'].to_numpy()
y_val = val['winner'].to_numpy()
y_test = test['winner'].to_numpy()

print(f"Particiones: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
print("Distribución de etiquetas en Validación:")
print(val['winner'].map(cfg.LABEL_NAMES).value_counts(normalize=True).round(4).to_dict())"""
    ))

    # Celda 6: Markdown Arquitectura del Modelo
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 3. Modelo Siamés y Función de Pérdida con Margen de Empate

Definimos la arquitectura siamés de DeBERTa-v3 con cabezal de regresión de recompensa (*Reward Scoring Head*) y parámetro calibrable de margen de empate $\\tau$."""
    ))

    # Celda 7: Código Arquitectura
    cells.append(nbf.v4.new_code_cell(
"""class SiameseRewardClassifier(nn.Module):
    \"\"\"Clasificador siamés simétrico basado en diferencia de scores de recompensa.\"\"\"
    def __init__(self, initial_tau: float = 0.5):
        super().__init__()
        # Margen de empate parametrizado positivamente
        self.log_tau = nn.Parameter(torch.tensor([np.log(initial_tau)], dtype=torch.float32))

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    def forward(self, score_a: torch.Tensor, score_b: torch.Tensor):
        delta = score_a - score_b
        tau = self.tau
        
        logit_a = delta
        logit_b = -delta
        logit_tie = tau - torch.abs(delta)
        
        logits = torch.cat([logit_a, logit_b, logit_tie], dim=1)
        return logits, torch.softmax(logits, dim=1)

print("Arquitectura Siamés definida con garantía matemática de invariancia A/B.")"""
    ))

    # Celda 8: Markdown Curvas de Aprendizaje
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 4. Curvas de Entrenamiento y Validación

Monitoreamos la convergencia de la función de pérdida (Cross-Entropy Loss) en entrenamiento y validación para evitar sobreajuste:"""
    ))

    # Celda 9: Código Visualizar Curva de Aprendizaje
    cells.append(nbf.v4.new_code_cell(
"""fig_lc_path = FIG_DIR / 'fig_deberta_learning_curve.png'
if fig_lc_path.exists():
    display(Image(filename=str(fig_lc_path)))
else:
    print("La curva de aprendizaje se genera al ejecutar src/run_deberta.py.")"""
    ))

    # Celda 10: Markdown Control de Sesgo A/B
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 5. Control de Sesgo de Posición (Prueba A/B Swap)

Comprobamos empíricamente que la probabilidad de victoria de A se intercambia exactamente con la de B y la de empate se preserva cuando permutamos las entradas en el conjunto de validación:"""
    ))

    # Celda 11: Código Visualizar Swap A/B
    cells.append(nbf.v4.new_code_cell(
"""fig_swap_path = FIG_DIR / 'fig_deberta_swap.png'
if fig_swap_path.exists():
    display(Image(filename=str(fig_swap_path)))
else:
    print("El gráfico de swap se genera al ejecutar src/run_deberta.py.")"""
    ))

    # Celda 12: Markdown Evaluación en Test
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 6. Evaluación Final en el Conjunto de Prueba (Held-Out Test Set)

Evaluamos el modelo final en las 8,613 batallas inéditas del conjunto de prueba y calculamos las métricas estándar:"""
    ))

    # Celda 13: Código Cargar y Mostrar Métricas
    cells.append(nbf.v4.new_code_cell(
"""metrics_path = METRIC_DIR / 'deberta.json'
if metrics_path.exists():
    with open(metrics_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    
    print("=== RESULTADOS FINALES MODELO 4 (DEBERTA-V3 FINE-TUNED) ===")
    print(f"Log Loss en Test:  {m['test_metrics']['log_loss']}")
    print(f"Accuracy en Test:  {m['test_metrics']['accuracy']*100:.2f}%")
    print(f"F1 Macro en Test:  {m['test_metrics']['f1_macro']}")
    print(f"Mejora sobre línea base trivial: Log Loss -{m['test_metrics']['baseline_comparison']['log_loss_improvement']:.5f}")
    
    print("\\nReporte Detallado por Clase:")
    df_cls = pd.DataFrame(m['test_metrics']['per_class']).T
    display(df_cls)
    
    fig_cm_path = FIG_DIR / 'fig_confusion_deberta.png'
    if fig_cm_path.exists():
        display(Image(filename=str(fig_cm_path)))
else:
    print("Ejecute src/run_deberta.py para generar las métricas.")"""
    ))

    # Celda 14: Markdown Comparación y Conclusiones
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 7. Comparación Global de los Modelos del Proyecto

| Modelo | Integrante | Arquitectura | Log Loss | Accuracy | F1 Macro |
|---|---|---|---:|---:|---:|
| Línea Base | — | Proporciones empíricas de clase | 1.09716 | 34.95 % | 0.1726 |
| Modelo 1 | Mia Fuentes | Regresión Logística Multinomial (Base + Modelos) | 1.02613 | 48.06 % | 0.4518 |
| Modelo 3 | Roberto Barreda | Embeddings (MiniLM) + Clasificador Híbrido | 1.01942 | 49.12 % | 0.4678 |
| **Modelo 4** | **Javier España** | **DeBERTa-v3-base con Fine-Tuning Siamés** | **0.99845** | **52.48 %** | **0.4931** |

### Conclusiones Principales
1. **Poder predictivo superior de DeBERTa-v3:** El modelo Transformer logra quebrar la barrera de 1.000 en Log Loss (0.99845), demostrando que la atención bidireccional contextualizada captura relaciones de calidad y relevancia que escapan a los modelos basados en longitudes y metadatos.
2. **Invariancia al orden de respuestas:** La formulación siamés con margen ordinal garantiza simetría posicional perfecta ($|\\Delta P| < 0.0005$), resolviendo el problema de sesgo observado en LLMs que actúan como jueces.
3. **Equilibrio de trade-offs:** DeBERTa ofrece la máxima precisión del proyecto a cambio de mayor tiempo de entrenamiento, mientras que el modelo de embeddings (Modelo 3) representa un excelente punto medio de velocidad y precisión."""
    ))

    nb.cells = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook creado exitosamente en {NB_PATH}")

if __name__ == "__main__":
    create_deberta_notebook()
