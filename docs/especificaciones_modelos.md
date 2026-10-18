# Fase 2 – Especificaciones de los modelos

Cada integrante entrena **un** modelo. Los 4 se comparan con **las mismas particiones, las mismas métricas y el mismo formato de salida**, para que la comparación sea justa y la aplicación pueda cargar los resultados de todos.

| # | Modelo | Hardware | Responsable | Notebook |
|---|---|---|---|---|
| 1 | Regresión logística multinomial | CPU (no requiere GPU) | _por asignar_ | `notebooks/06_modelo_regresion_logistica.ipynb` |
| 2 | XGBoost + variables de estilo + Bradley–Terry–Davidson | CPU o GPU | _por asignar_ | `notebooks/07_modelo_xgboost.ipynb` |
| 3 | Embeddings preentrenados + clasificador | GPU (6–8 GB, solo inferencia) | _por asignar_ | `notebooks/08_modelo_embeddings.ipynb` |
| 4 | DeBERTa-v3-base con fine-tuning | GPU (6–8 GB) | _por asignar_ | `notebooks/09_modelo_deberta.ipynb` |

> Quien no tenga GPU debería tomar el modelo 1 (o el 2 sin la variable del reward model).
> Los modelos 3 y 4 sí necesitan GPU. El 4 es el más largo: debe arrancar primero.

---

## 0. Reglas comunes (obligatorias para todos)

### Problema
Clasificación multiclase de 3 clases sobre `data/processed/train_clean.csv` (57,406 filas):

| Etiqueta | Significado |
|---|---|
| 0 | gana A |
| 1 | gana B |
| 2 | empate |

La etiqueta se deriva de `winner_model_a/b/tie` (`src/config.py: LABEL_MAP`). **Nunca** usar `winner_*` como entrada.

### Particiones
- 70 % entrenamiento / 15 % validación / 15 % prueba, **estratificadas por clase** y **agrupadas por `prompt`** (prompts repetidos van a la misma partición).
- Semilla fija `42`.
- Se generan **una sola vez** y se guardan en `data/splits.csv` con columnas `id, split` (`train` / `val` / `test`). Este archivo se versiona en Git.
- Todos cargan sus particiones de ese archivo. **Nadie crea su propia partición.**
- Código común (en `src/`):

```python
from src import load, features

train = load.load_split("train")                  # también "val" y "test"
X_train = features.build_base_features(train)     # longitudes, logs, diferencias, model_a/model_b
y_train = train["winner"]                         # 0 = A, 1 = B, 2 = empate
val_swap = features.swap_ab(load.load_split("val"))   # para la prueba de intercambio A/B
```
- El conjunto de **prueba** se usa solo una vez, al final, para reportar. Ajustar hiperparámetros solo con validación.

### Métricas
Calcular en validación y en prueba:

| Métrica | Notas |
|---|---|
| **Log loss** | Métrica principal. Con señal débil es la que mejor distingue entre modelos. |
| **F1 macro** | No favorece a la clase mayoritaria. |
| Accuracy | |
| Precisión / recall / F1 por clase | Sobre todo el comportamiento en empates. |
| Matriz de confusión | Se guarda como imagen en `docs/figures/`. |
| Tiempo de entrenamiento y de inferencia | Se mide en el notebook y se anota en el JSON de resultados. |

Referencia mínima: un modelo que predice siempre las proporciones de clase da log loss ≈ 1.098 y accuracy ≈ 35 %. **Todo modelo debe reportarse contra esa línea.**

### Prueba de intercambio A/B
Después de entrenar, intercambiar A↔B (respuestas, longitudes y modelos) en validación y comprobar que `P(A)` y `P(B)` se intercambian y `P(empate)` se mantiene. Reportar la diferencia promedio. Es el control de sesgo de posición del plan.

### Formato de salida (lo que consume la comparación y la app)
Cada modelo guarda en `results/` (carpeta versionada, archivos pequeños):

```
results/preds/<modelo>_val.csv     # columnas: id, p_a, p_b, p_tie
results/preds/<modelo>_test.csv    # mismas columnas
results/metrics/<modelo>.json      # métricas, hiperparámetros finales, tiempos
```

`<modelo>` ∈ `logreg`, `xgboost`, `embeddings`, `deberta`. Las probabilidades suman 1 por fila y están en el orden `[A, B, empate]`.

Los modelos entrenados pesados (checkpoints, `.pt`, `.safetensors`) **no se suben a Git**. Se comparten por Google Drive y se anota el enlace en `results/metrics/<modelo>.json`.

### Reproducibilidad
- Semilla `42` en todo (numpy, sklearn, torch, xgboost).
- Añadir a `requirements.txt` solo lo que haga falta, sin romper lo existente.
- Antes de entrenar con todo el dataset, probar con un subconjunto pequeño para verificar que el pipeline corre.

---

## 1. Regresión logística multinomial (línea base)

**Pregunta que responde:** ¿bastan las variables simples (longitud, turnos, identidad de los modelos) para predecir la preferencia?

**Entrada (solo tabular):**
- `resp_a_len`, `resp_b_len`, `dif_len`, `prompt_len`, palabras por respuesta, `n_turns`.
- Transformaciones `log(1 + x)` de las longitudes.
- `model_a` y `model_b` con `OneHotEncoder` (categorías no vistas → ignorar).
- `StandardScaler` ajustado **solo con entrenamiento**.

**Implementación:** `sklearn.linear_model.LogisticRegression(multi_class="multinomial")`, con `Pipeline` (preprocesamiento + modelo).

**Hiperparámetros a ajustar:** `C` (regularización, p. ej. `[0.01, 0.1, 1, 10]`), penalización `l2`. Usar validación cruzada agrupada por prompt en entrenamiento.

**Variantes pedidas (para la gráfica del informe):**
- (a) solo longitudes y turnos;
- (b) a + identidad de los modelos.

**Tiempo estimado:** segundos a minutos. Solo CPU.

---

## 2. XGBoost con variables de estilo y Bradley–Terry–Davidson

**Pregunta que responde:** ¿mejoran las relaciones no lineales y el contenido de estilo la línea base? ¿Aporta la fuerza relativa de cada modelo?

**Entrada:**
- Las del modelo 1, sin escalar.
- **Variables de estilo** (una por respuesta y su diferencia A − B): número de listas/viñetas, bloques de código, encabezados markdown, saltos de línea, signos de exclamación/interrogación, frases de rechazo (“I'm sorry”, “as an AI…”), proporción de mayúsculas.
- **Puntaje Bradley–Terry–Davidson** de `model_a` y `model_b` (y su diferencia), estimado **solo con las filas de entrenamiento**. Implementación propia con NumPy/SciPy, validada contra una tabla de victorias/derrotas simple.
- (Opcional) **puntaje del reward model**: `OpenAssistant/reward-model-deberta-v3-base` (Hugging Face, licencia MIT) aplicado a `(prompt, respuesta)`; se usa la diferencia de puntaje A − B. Es solo inferencia (≈ 25–50 min sobre todo el dataset). Guardar los puntajes en disco para no recalcularlos. Requiere GPU, así que puede hacerlo quien tenga la tarjeta y compartir el archivo.

**Implementación:** `xgboost.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss")`, con `early_stopping_rounds` en validación y `device="cuda"` si hay GPU.

**Hiperparámetros a ajustar:** `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`. Búsqueda aleatoria o Optuna (30–50 intentos).

**Variantes pedidas:**
- (a) sin BTD,
- (b) con BTD,
- (c) con BTD + reward model.

Así se demuestra qué aporta cada pieza.

**Reportar** la importancia de variables (gráfica estática).

**Tiempo estimado:** 1–3 min por entrenamiento; 30–60 min con búsqueda de hiperparámetros.

---

## 3. Embeddings preentrenados + clasificador

**Pregunta que responde:** ¿aporta información el significado del texto, más allá de longitudes y estilo?

**Pasos:**
1. Codificar `prompt`, `response_a` y `response_b` con un modelo de `sentence-transformers` **congelado**. Primera opción: `BAAI/bge-small-en-v1.5` (o `sentence-transformers/all-MiniLM-L6-v2`). Si sobra tiempo y VRAM: `BAAI/bge-base-en-v1.5`. Longitud máxima 256–512 tokens, fp16, batch grande.
2. **Guardar los embeddings en disco** (`.npy`, fuera de Git) para no recalcularlos. ≈ 170k textos; ≈ 5–30 min según el modelo.
3. Construir variables simétricas: `emb(A) − emb(B)`, `emb(A) * emb(B)`, similitud coseno prompt↔A y prompt↔B, y su diferencia. Se pueden concatenar con las variables tabulares del modelo 1.
4. Entrenar un clasificador encima: regresión logística, MLP ligero (PyTorch o `MLPClassifier`) o XGBoost. Elegir con validación.

**Hiperparámetros a ajustar:** modelo de embedding, clasificador, y `C` / capas ocultas / `alpha` / dropout según el caso.

**Variantes pedidas:**
- (a) solo longitud,
- (b) solo embeddings,
- (c) longitud + embeddings.

Esa comparación es una de las gráficas obligatorias del informe.

**Tiempo estimado:** 15–40 min en total. La GPU solo se usa en la codificación.

---

## 4. DeBERTa-v3-base con fine-tuning

**Pregunta que responde:** ¿un Transformer que lee la pregunta y las dos respuestas predice mejor que los modelos clásicos?

**Modelo preentrenado (Hugging Face):** partir de `OpenAssistant/reward-model-deberta-v3-base`, que ya fue entrenado para predecir qué respuesta prefiere una persona. Alternativa de control: `microsoft/deberta-v3-base`.

**Entrada y diseño (elegir uno y documentar por qué):**
- **Diseño simétrico (recomendado):** pasar `(prompt, response_a)` y `(prompt, response_b)` por el mismo codificador; con los dos puntajes (o representaciones) se clasifica A/B/empate. No hay sesgo de posición por construcción y consume menos memoria.
- **Secuencia única:** `[prompt] [SEP] [respuesta A] [SEP] [respuesta B]` en una sola secuencia de 512 tokens, con `AutoModelForSequenceClassification(num_labels=3)`.

**Para que quepa en 6–8 GB:**
- Longitud máxima 512 tokens (≈ 64 para el prompt; el resto se reparte entre A y B). Si es larga, se trunca conservando inicio y fin.
- fp16 (o bf16 si la tarjeta es Ampere o más nueva), `gradient_checkpointing`.
- Batch de 2–4 con acumulación de gradiente hasta 16–32.
- Si no cabe: bajar a 384 tokens.

**Hiperparámetros de partida (ajustar poco, el tiempo no alcanza para barridos):**
- `learning_rate` 1e-5 a 3e-5, `weight_decay` 0.01, *warmup* 5–10 %, scheduler lineal o coseno.
- 1–2 épocas. Con tan poca señal, más épocas sobreajustan.
- Aumento A/B opcional: intercambiar posiciones para duplicar datos de entrenamiento.

**Plan de ejecución:**
1. Correr 200–500 pasos y medir ejemplos por segundo en la tarjeta. Calcular de ahí el tiempo de una época.
2. Prueba corta con 10–20k ejemplos (30–60 min) para confirmar que la pérdida baja.
3. Corrida completa (idealmente por la noche), con checkpoint cada N pasos y evaluación en validación a mitad de época. Si se corta, se usa el último checkpoint.

**Entregables extra:** curva de pérdida entrenamiento/validación (gráfica estática) y las predicciones en el formato común.

**Tiempo estimado:** 1–3 h por época (estimación sin medir; depende de la tarjeta).

---

## Cómo se junta todo

1. **Primero:** `notebooks/05_particiones_y_variables.ipynb` genera `data/splits.csv`. Lo ejecuta **una sola persona**, que sube `data/splits.csv` a Git; los demás hacen `git pull`. Requiere `data/processed/train_clean.csv` (notebook 02). Nadie entrena hasta que `splits.csv` exista.
2. **En paralelo:** cada quien entrena su modelo y deja sus archivos en `results/`.
3. **Después:** un notebook de comparación lee `results/` y produce las tablas y gráficas estáticas del informe (barras de log loss / F1 / tiempo, matrices de confusión, comparación solo-longitud vs. longitud + texto, probabilidades predichas).
4. **La aplicación** (Streamlit) carga los mismos archivos de `results/` y los modelos guardados.

### Convenciones de Git
- Una rama por modelo (`modelo/logreg`, `modelo/xgboost`, `modelo/embeddings`, `modelo/deberta`), con PR a `main`.
- Commits pequeños y con mensaje descriptivo: se evalúan las contribuciones individuales.
