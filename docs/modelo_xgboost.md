# Modelo 2: XGBoost + variables de estilo + Bradley–Terry–Davidson

Responsable: Angel Esteban Esquit Hernández (23221). Especificación: `docs/especificaciones_modelos.md` §2.

## Archivos
| Archivo | Contenido |
|---|---|
| `src/xgb_model.py` | Variables, entrenamiento, búsqueda Optuna, figuras y guardado de resultados |
| `src/btd.py` | Bradley–Terry–Davidson propio (NumPy/SciPy, L-BFGS con gradiente analítico) |
| `src/style_features.py` | Variables de estilo por respuesta |
| `src/compute_reward_scores.py` | (Opcional) puntajes del reward model → `data/reward_scores.csv` |
| `src/create_nb_xgboost.py` | Genera `notebooks/07_modelo_xgboost.ipynb` |
| `src/export_app_models.py` | Prepara `models/` para la app Streamlit (`app/`) |
| `tests/test_xgboost.py` | BTD vs. tabla de victorias, estilo, simetría A/B y formato de salida |

## Diseño
- **Entrada:** variables del modelo 1 sin escalar (`features.COLS_NUM`, `model_a`/`model_b` con one-hot), 8 variables de estilo por respuesta (+ diferencia A − B): viñetas, bloques de código, encabezados markdown, saltos de línea, `!`, `?`, frases de rechazo y proporción de mayúsculas.
- **BTD:** `P(A)=e^{d/2}/Z`, `P(B)=e^{-d/2}/Z`, `P(empate)=ν/Z`, `d=θ_A−θ_B`, `Z=2cosh(d/2)+ν`. Se ajusta solo con entrenamiento (L2 suave 1e-2). Variables: `btd_a`, `btd_b`, `dif_btd` y las tres probabilidades del BTD.
- **Sin fuga en entrenamiento:** los puntajes BTD de las filas de entrenamiento son fuera de muestra (5 folds agrupados por prompt); val/test usan el BTD ajustado con todo el entrenamiento.
- **Reward model (opcional):** si existe `data/reward_scores.csv` (`id, reward_a, reward_b`) se activa la variante (c).
- **Hiperparámetros:** Optuna TPE (semilla 42), `n_trials` por variante, minimizando log loss de validación con `early_stopping_rounds=50`. Espacio: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda` y `augment_swap` (duplicar el train con A↔B intercambiados, para reducir el sesgo de posición).
- **Variante final:** la de menor log loss en validación. La prueba se evalúa una sola vez, con esa variante.

## Cómo correr
```bash
pip install -r requirements.txt
python -m src.xgb_model --nrows 2000 --trials 3      # prueba rápida, escribe en cache/ (no toca results/)
python -m src.xgb_model --trials 40                   # corrida final (CPU); --device cuda para GPU
python -m src.create_nb_xgboost                       # regenera el notebook 07
python -m pytest tests/test_xgboost.py
```
Para la variante (c), quien tenga GPU: `python -m src.compute_reward_scores --device cuda` y compartir `data/reward_scores.csv` (no se versiona).

## Salidas
`results/preds/xgboost_{val,test}.csv`, `results/metrics/xgboost.json`, `models/xgboost_final.json`,
`docs/figures/fig_xgboost_variantes.png`, `fig_xgboost_importancia.png`, `fig_confusion_xgboost.png`, `fig_xgboost_swap.png`.

## Resultados (corrida final: 40 intentos de Optuna por variante, CPU, semilla 42)

| Variante | Log loss (val) | F1 macro (val) | Accuracy (val) |
|---|---|---|---|
| (a) Sin BTD | 0.99379 | 0.4944 | 50.75 % |
| **(b) Con BTD** (final) | **0.99311** | 0.4952 | 50.83 % |
| (c) BTD + reward model | no ejecutada (falta `data/reward_scores.csv`) | | |

Variante final (b), evaluada una sola vez en **prueba**: log loss **0.9909**, accuracy 50.67 %, F1 macro 0.4958
(línea base trivial: 1.0986 y ~35 %). Recall de empates: 33.4 %. Prueba de intercambio A/B: diferencia media 0.0121.
El BTD aporta una mejora pequeña (−0.0007 en validación); la longitud y las variables de estilo ya capturan casi toda la señal.

Comparación en prueba con los otros modelos (recalculada con `src/evaluation.py` desde `results/preds/`):

| Modelo | Log loss | Accuracy | F1 macro |
|---|---|---|---|
| **XGBoost (este)** | **0.9909** | 50.7 % | 0.496 |
| Embeddings + LR | 1.0143 | 48.6 % | 0.473 |
| Regresión logística | 1.0253 | 47.8 % | 0.451 |
| DeBERTa-v3 | Pendiente de corrida completa | Pendiente | Pendiente |

Variables más importantes (ganancia): `btd_p_a`, `btd_p_b`, `dif_btd`, `dif_len`, `dif_words`, `style_n_newlines_dif`.
Hiperparámetros finales: `max_depth=4`, `learning_rate≈0.051`, `subsample≈0.71`, `colsample_bytree≈0.37`,
`min_child_weight=10`, `reg_lambda≈4.7`, 719 árboles, con aumento A↔B (`augment_swap=True`).
Tiempos: variables 54 s, búsqueda 519 s (variante final), ajuste final 15 s.
