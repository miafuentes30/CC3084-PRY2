"""Generador del notebook 07_modelo_xgboost.ipynb (Modelo 2)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "07_modelo_xgboost.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        MD("""# Modelo 2: XGBoost + variables de estilo + Bradley–Terry–Davidson

**Proyecto 2 — LMSYS Chatbot Arena | CC3084 Data Science**
Responsable: Angel Esteban Esquit Hernández (23221)

**Pregunta:** ¿mejoran las relaciones no lineales y el contenido de estilo la línea base?
¿Aporta la fuerza relativa de cada modelo (Bradley–Terry–Davidson)?

Reglas comunes (`docs/especificaciones_modelos.md`): particiones de `data/splits.csv`, semilla 42,
log loss como métrica principal, prueba usada una sola vez y prueba de intercambio A/B.
Documentación completa en `docs/modelo_xgboost.md`."""),
        CODE("""import sys, json
from pathlib import Path
ROOT = Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from IPython.display import Image, display

from src import load, evaluation
from src.btd import BTD
from src.xgb_model import attach_rewards, run_pipeline
SEED = 42"""),
        MD("## 1. Datos y particiones"),
        CODE("""train, val, test = (attach_rewards(load.load_split(s)) for s in ("train", "val", "test"))
print({k: len(v) for k, v in zip(("train", "val", "test"), (train, val, test))})
print("Reward model disponible:", {"reward_a", "reward_b"} <= set(train.columns))"""),
        MD("""## 2. Bradley–Terry–Davidson
Implementación propia (NumPy/SciPy): `P(A)=e^{d/2}/Z`, `P(B)=e^{-d/2}/Z`, `P(empate)=ν/Z`,
con `d = θ_A − θ_B` y `Z = 2cosh(d/2)+ν`. Se ajusta **solo con entrenamiento**; para las filas
de entrenamiento se usan puntajes fuera de muestra (5 folds agrupados por prompt) para que
XGBoost no vea puntajes calculados con la propia fila."""),
        CODE("""btd = BTD(l2=1e-2).fit(train["model_a"], train["model_b"], train["winner"])
print(f"nu (propensión a empate) = {btd.nu_:.3f}")
pd.DataFrame(btd.ranking(), columns=["modelo", "theta"]).head(15)"""),
        MD("""## 3. Entrenamiento de las variantes (a) sin BTD, (b) con BTD, (c) BTD + reward model
`n_trials` = intentos de Optuna por variante (30–50 para la corrida final; usar 3 para una prueba)."""),
        CODE("""metrics = run_pipeline(train, val, test, n_trials=30, device="cpu")"""),
        MD("## 4. Resultados"),
        CODE("""pd.DataFrame(metrics["variants_comparison"]).T[["n_features", "log_loss", "accuracy", "f1_macro", "train_time_sec"]]"""),
        CODE("""print("Variante final:", metrics["final_variant"])
for k in ("validation_metrics", "test_metrics"):
    m = metrics[k]
    print(k, "| log loss", m["log_loss"], "| acc", m["accuracy"], "| F1 macro", m["f1_macro"],
          "| mejora vs trivial", m["baseline_comparison"]["log_loss_improvement"])
print("Swap A/B:", metrics["position_bias_swap_check"])
print("Tiempos:", metrics["times"])"""),
        CODE("""FIG = ROOT / "docs" / "figures"
for f in ("fig_xgboost_variantes", "fig_xgboost_importancia", "fig_confusion_xgboost", "fig_xgboost_swap"):
    display(Image(filename=str(FIG / f"{f}.png")))"""),
        MD("""## 5. Conclusiones
Resultados de la corrida final (`results/metrics/xgboost.json`, 40 intentos de Optuna por variante):

- Ganó la variante **(b) con BTD** (log loss de validación 0.9931 frente a 0.9938 sin BTD). En **prueba**: log loss **0.9909**,
  accuracy 50.7 %, F1 macro 0.496, frente a 1.0986 y ~35 % de la línea base trivial.
- El aporte del BTD es pequeño: longitud y estilo ya explican casi toda la señal. Aun así, las variables derivadas del BTD
  (`btd_p_a`, `btd_p_b`, `dif_btd`) son las de mayor importancia junto con `dif_len`.
- Los empates siguen siendo lo más difícil (recall 33 %), aunque mejor que la regresión logística (20 %).
- Prueba de intercambio A/B: diferencia media 0.012; el aumento de datos con A↔B intercambiados fue elegido por la búsqueda.
- La variante (c) con reward model no se ejecutó (requiere `data/reward_scores.csv`)."""),
    ]
    nbf.write(nb, NB_PATH)
    print(f"Notebook generado: {NB_PATH}")


if __name__ == "__main__":
    create_notebook()
