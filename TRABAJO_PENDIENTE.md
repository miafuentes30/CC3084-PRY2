# Trabajo Pendiente – CC3084 Proyecto 2

> **Cómo usar este archivo:**
> - Elegí una parte que esté disponible (sin tachar).
> - Una vez que termines tu parte **eliminá el bloque completo** de esa parte de este archivo.
> - Así los demás sabrán qué queda libre y no habrá confusión.
>
> ADVERTENCIA: Las partes son **secuenciales**: la Parte 3 y la Parte 4 requieren haber corrido la Parte 2 primero (genera `data/processed/train_clean.csv`).

---

## [COMPLETADA] Parte 1 – Introducción y Carga *(COMPLETADA)*

Notebook: `notebooks/01_introduccion_y_carga.ipynb`

Esta parte ya fue realizada. Incluye:
- Situación problemática, problema científico y objetivos
- Configuración del ambiente y módulos `src/`
- Descarga del dataset con `kagglehub`
- Primera exploración: shape, dtypes, `.head()`, `.info()`

---

## [COMPLETADA] Parte 2 – Descripción y Limpieza de Datos *(COMPLETADA — pendiente de ejecución)*

Notebook: `notebooks/02_descripcion_y_limpieza.ipynb`

El código de esta parte ya está escrito y cubre las 7 tareas del enunciado original (descripción de
variables, análisis de nulos, duplicados por `id`, limpieza de texto JSON, columna `winner` derivada,
features de longitud y guardado de `data/processed/train_clean.csv`), y su lógica fue validada con datos
sintéticos.

**IMPORTANTE — falta un paso manual:** el notebook no se ejecutó contra el dataset real porque este
entorno no tenía credenciales de Kaggle ni `data/raw/train.csv` disponibles. La primera persona del
equipo que lo abra con el dataset ya descargado debe:
1. Ejecutar todas las celdas de principio a fin (`Run All`).
2. Revisar que `data/processed/train_clean.csv` se generó correctamente.
3. Confirmar que la cifra de duplicados/nulos en las celdas coincide con lo esperado y hacer commit del
   notebook con sus outputs.

---

##  Parte 3 – EDA: Variables Numéricas y Distribuciones

**Notebook a crear:** `notebooks/03_eda_numericas_texto.ipynb`

**Prerequisito:** Haber completado la Parte 2 (necesitás `data/processed/train_clean.csv`).

**Tareas a realizar:**

1. **Estadística descriptiva**
   - `.describe()` sobre las columnas numéricas derivadas (longitudes, número de turnos)
   - Tabla de estadísticos: media, mediana, std, mín, máx, cuartiles

2. **Distribución de la variable objetivo**
   - Gráfico de barras / pie chart: ¿cuántas victorias tiene model_a vs model_b vs empate?
   - Tabla de frecuencias y proporciones

3. **Histogramas de longitudes**
   - Histograma de `prompt_len`, `resp_a_len`, `resp_b_len`
   - Escala logarítmica si hay skew muy fuerte
   - Descripción de lo que se observa

4. **Boxplots por resultado**
   - Boxplot de `resp_a_len` y `resp_b_len` agrupados por `winner`
   - ¿Las respuestas más largas tienden a ganar?
   - Identificar outliers

5. **Análisis de outliers**
   - Identificar filas con longitudes extremas (percentil 99+)
   - ¿Son errores o casos válidos?
   - Decisión documentada

6. **Distribución de modelos participantes**
   - ¿Cuántas veces aparece cada modelo (como model_a o model_b)?
   - Gráfico de barras horizontal ordenado

**Una vez terminado, eliminá este bloque del archivo.**

---

##  Parte 4 – EDA: Variables Categóricas + Hallazgos y Conclusiones

**Notebook a crear:** `notebooks/04_eda_categoricas_conclusiones.ipynb`

**Prerequisito:** Haber completado la Parte 2 (necesitás `data/processed/train_clean.csv`).

**Tareas a realizar:**

1. **Tablas de frecuencia de modelos**
   - Frecuencia y proporción de aparición de cada modelo
   - Tabla ordenada de mayor a menor

2. **Win rate por modelo**
   - Para cada modelo: `victorias / (veces que aparece)`
   - Gráfico de barras con el win rate de cada modelo
   - Separar entre cuando el modelo aparece como A vs como B (sesgo de posición)

3. **Análisis de empates**
   - ¿Qué modelos tienen mayor tasa de empate?
   - Gráfico comparativo

4. **Correlaciones**
   - Heatmap de correlación entre variables numéricas (longitudes, winner)
   - ¿Longitud de respuesta correlaciona con ganar?
   - Scatter plots: `resp_a_len` vs `resp_b_len`, coloreado por `winner`

5. **Análisis de sesgo de posición**
   - ¿El modelo en posición A (model_a) tiene ventaja sobre el modelo en posición B?
   - Proporción de victorias de model_a vs model_b en total

6. **Hallazgos**
   - Celda markdown con resumen de todos los hallazgos encontrados
   - Mínimo 5 hallazgos concretos con su justificación

7. **Conclusiones**
   - ¿Qué factores parecen influir en la preferencia humana?
   - ¿Qué pasos siguientes se recomiendan para el modelado?
   - Implicaciones para negocios (¿qué modelo usar en producción?)

**Una vez terminado, eliminá este bloque del archivo.**
