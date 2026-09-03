# Trabajo Pendiente. CC3084 Proyecto 2

> **Cómo usar este archivo:**
> - Elegí una parte que esté disponible (sin marcar como completada).
> - Una vez que termines tu parte, reemplazá su bloque por un resumen corto de lo que quedó cubierto,
>   para que quien siga no repita trabajo.
>
> ADVERTENCIA: las partes son **secuenciales**. La Parte 3 y la Parte 4 requieren haber corrido la
> Parte 2 primero, que es la que genera `data/processed/train_clean.csv`.

---

## [COMPLETADA] Parte 1. Introducción y Carga

Notebook: `notebooks/01_introduccion_y_carga.ipynb` (ejecutado, con outputs guardados)

Cubre: situación problemática, problema científico, objetivos, configuración del ambiente y módulos
`src/`, descarga con `kagglehub` y primera exploración del archivo crudo (shape, dtypes, nulos a nivel
de celda, distribución del resultado y listado de los 64 modelos).

Cifras establecidas ahí y que el resto del informe da por conocidas: **57,477 batallas x 9 columnas**,
**64 modelos distintos**, **cero nulos a nivel de celda** y un reparto del resultado de
34.91% / 34.19% / 30.90% entre victoria de A, victoria de B y empate.

---

## [COMPLETADA] Parte 2. Descripción y Limpieza de Datos

Notebook: `notebooks/02_descripcion_y_limpieza.ipynb`

Cubre: descripción de las nueve variables, duplicados por `id` y por contenido, parseo de las columnas
JSON con conteo de fallos, análisis de nulos a nivel de celda y de turnos sin respuesta, variable
objetivo unificada `winner`, variables de longitud y guardado de `data/processed/train_clean.csv`.

**PENDIENTE: ejecutarlo contra el dataset real.** El notebook nunca se corrió con los datos de Kaggle,
solo se validó su lógica. Quien tenga `data/raw/train.csv` descargado debe hacer `Run All`, revisar que
las cifras de duplicados, fallos de parseo y turnos sin respuesta sean razonables, y comitear el
notebook con sus outputs. Sin ese paso no existe `train_clean.csv` y las Partes 3 y 4 no corren.

---

## [COMPLETADA] Parte 3. Análisis Exploratorio: Variables Cuantitativas

Notebook: `notebooks/03_eda_numericas_texto.ipynb`

Cubre: estadística descriptiva con percentiles extremos y medidas de forma, verificación de la
variable objetivo sobre el dataset limpio, histogramas de longitudes en escala lineal y logarítmica,
distribución del número de turnos con su tasa de empate, diagramas de caja de longitud contra
resultado, cuantificación del sesgo por verbosidad, tratamiento documentado de valores atípicos y
participación de cada modelo con su reparto entre las posiciones A y B.

Figuras generadas: `fig_hist_longitudes.png`, `fig_turnos.png`,
`fig_boxplot_longitud_resultado.png`, `fig_participacion_modelos.png`.

**Igual que la Parte 2, falta ejecutarlo contra el dataset real.**

---

## Parte 4. EDA de Variables Categóricas, Hallazgos y Conclusiones

**Notebook a crear:** `notebooks/04_eda_categoricas_conclusiones.ipynb`

**Prerequisito:** haber corrido la Parte 2 para tener `data/processed/train_clean.csv`.

### Qué NO repetir

Para que el informe final no quede con secciones duplicadas, tres puntos del plan original ya están
resueltos en la Parte 3 y conviene retomarlos en lugar de rehacerlos:

- **Tabla de frecuencia y proporción de aparición de cada modelo:** hecha en la sección 8, junto con
  el gráfico de barras horizontal y la concentración del top 10. Partí de ahí directo al win rate.
- **Reparto de cada modelo entre las posiciones A y B:** ya medido en la misma sección. Sirve como
  control de que la asignación es aleatoria; el análisis de sesgo de posición sobre las *victorias*
  sigue pendiente y es tuyo.
- **Relación entre longitud y victoria:** la sección 6 ya cuantifica con qué frecuencia gana la
  respuesta más larga. En el heatmap de correlaciones no vuelvas a plantear la pregunta, citá el
  resultado y enfocá el heatmap en el conjunto de variables.

### Tareas

1. **Win rate por modelo**
   - Para cada modelo: `victorias / apariciones`, reportando siempre el número de batallas que
     respalda cada porcentaje (la participación es muy desigual, hay modelos con muy pocos casos).
   - Gráfico de barras con el win rate, filtrando o marcando los modelos con muestra insuficiente.
   - Win rate por separado según aparezca como A o como B.

2. **Análisis de empates por modelo**
   - Qué modelos empatan más seguido y contra qué tipo de rival.
   - Gráfico comparativo.

3. **Correlaciones**
   - Heatmap entre las variables numéricas y el resultado.
   - Scatter de `resp_a_len` contra `resp_b_len` coloreado por `winner`.

4. **Sesgo de posición**
   - Proporción de victorias de model_a contra model_b en el total.
   - Contrastarlo con el reparto A/B por modelo de la Parte 3 para separar el sesgo del evaluador
     humano de un posible desbalance en la asignación.

5. **Hallazgos**
   - Mínimo 5 hallazgos concretos con su justificación numérica.
   - La Parte 3 cierra con sus propios hallazgos sobre las variables cuantitativas: complementalos,
     no los repitas.

6. **Conclusiones del informe**
   - Qué factores parecen influir en la preferencia humana.
   - Qué pasos se recomiendan para el modelado.
   - Implicaciones de negocio: qué modelo conviene usar en producción y bajo qué criterio.
