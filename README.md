# CC3084 – Proyecto 2: Análisis Exploratorio

**Reto 11 – Negocios:** [LMSYS Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

| Integrante | Carné |
|---|---|
| Mia Alejandra Fuentes Mérida | 23775 |
| Roberto José Barreda Siekavizza | 23354 |
| Javier Eduardo España Pacheco | 23361 |
| Angel Esteban Esquit Hernández | 23221 |

---

## Descripción del dataset

El dataset proviene de **Chatbot Arena** (LMSYS), una plataforma donde usuarios reales comparan dos LLMs (modelos de lenguaje grande) de forma anónima. Cada fila del `train.csv` representa una "batalla" entre dos modelos: el usuario emite un prompt, recibe ambas respuestas y elige su preferida (o declara empate).

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | str | Identificador único de la batalla |
| `model_a` | str | Nombre del primer modelo |
| `model_b` | str | Nombre del segundo modelo |
| `prompt` | str (JSON list) | Turnos de conversación del usuario |
| `response_a` | str (JSON list) | Respuestas del modelo A |
| `response_b` | str (JSON list) | Respuestas del modelo B |
| `winner_model_a` | int (0/1) | 1 si el usuario prefirió el modelo A |
| `winner_model_b` | int (0/1) | 1 si el usuario prefirió el modelo B |
| `winner_tie` | int (0/1) | 1 si el usuario declaró empate |

---

## Setup

```bash
pip install -r requirements.txt
```

Para descargar el dataset (requiere cuenta Kaggle autenticada):

```python
import kagglehub
path = kagglehub.competition_download('lmsys-chatbot-arena')
print("Path:", path)
```

O configurar `~/.kaggle/kaggle.json` con tu API token y ejecutar:

```bash
kaggle competitions download -c lmsys-chatbot-arena -p data/raw/ --unzip
```

Los archivos deben quedar en `data/raw/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

## Uso de los módulos `src`

```python
from src import load, config

# Cargar train completo
train = load.load_train()

# Cargar solo N filas (para exploración rápida)
train = load.load_train(nrows=5_000)

# Dataset limpio (generado por notebook 02)
clean = load.load_train_clean()
```

## Regresión logística multinomial (Fase 2)

La contribución de regresión logística compara dos variantes reproducibles:

- Variante A: longitudes, palabras, diferencias, transformaciones logarítmicas y turnos.
- Variante B: las variables anteriores más `model_a` y `model_b` codificados con *one-hot*.

Primero debe existir `data/processed/train_clean.csv`, generado por el notebook 02. Las
particiones de `data/splits.csv` se cargan sin regenerarlas. Para reproducir entrenamiento,
evaluación, predicciones y figuras:

```bash
python -m src.logreg_model --run --cv-folds 5 --max-iter 1000
python -m pytest tests/test_logreg.py -q
```

El análisis ejecutado se encuentra en
`notebooks/06_modelo_regresion_logistica.ipynb` y la redacción para el informe en
`docs/modelo_regresion_logistica.md`. Los CSV y métricas se guardan bajo `results/`.
El pipeline `models/logreg_pipeline.joblib` se genera localmente y está excluido de Git.

## XGBoost + estilo + Bradley–Terry–Davidson (Modelo 2)

Compara tres variantes: (a) sin BTD, (b) con BTD y (c) BTD + reward model (opcional).
Detalles en `docs/modelo_xgboost.md`.

```bash
python -m src.xgb_model --nrows 2000 --trials 3   # prueba rápida
python -m src.xgb_model --trials 40               # corrida final
```

## Aplicación Streamlit

```bash
pip install -r requirements.txt
python -m src.export_app_models      # prepara los modelos para clasificar casos nuevos (logreg y XGBoost)
streamlit run app/streamlit_app.py
```

Pestañas: resumen, exploración de los datos, resultados de los 4 modelos (se pueden ocultar) y clasificación de un caso nuevo.
Requiere `data/processed/train_clean.csv` (notebook 02) y `data/splits.csv`. La clasificación en vivo está disponible para
regresión logística y XGBoost; embeddings y DeBERTa muestran sus resultados, pero sus artefactos viven en Drive.
