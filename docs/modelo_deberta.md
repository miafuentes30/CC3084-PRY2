# Fine-Tuning de DeBERTa-v3 con Arquitectura Siamés Simétrica para Predicción de Preferencias Humanas

**Autor:** Javier Eduardo España Pacheco (Carné: 23361)  
**Proyecto 2 — Resultados | CC3084 Data Science | Semestre II 2026**  
**Universidad del Valle de Guatemala**

---

## 1. Objetivo específico del modelo DeBERTa

El objetivo de esta contribución individual es entrenar y evaluar un modelo de lenguaje basado en Transformer (*DeBERTa-v3*) ajustado mediante *fine-tuning* para predecir directamente la preferencia humana en Chatbot Arena entre dos respuestas concurrentes a un prompt dado.

A diferencia de los modelos lineales y tabulares (Modelo 1) o de los modelos de embeddings estáticos congelados (Modelo 3), este enfoque adapta los pesos atencionales del modelo mediante gradiente descendente para aprender las sutilezas de la preferencia humana: rigor lógico, seguimiento exacto de instrucciones, tono, alucinaciones y utilidad práctica.

La pregunta científica asignada en la especificación técnica es:
> *¿Un Transformer que lee la pregunta y las dos respuestas predice mejor que los modelos clásicos y lineales?*

---

## 2. Justificación teórica y arquitectónica: Diseño Siamés Simétrico

En la literatura de modelado de preferencias (*Reward Modeling* y RLHF para LLMs), existen dos arquitecturas predominantes:

1. **Secuencia única concatenada:**  
   `[CLS] prompt [SEP] respuesta A [SEP] respuesta B [SEP]`  
   *Problema crítico:* Esta arquitectura es susceptible de forma inherente al **sesgo de posición** (*position bias*). Wang et al. (2023) y Chiang et al. (2024) demostraron que los modelos de atención unidireccional o concatenada tienden a favorecer sistemáticamente a la primera o última respuesta evaluada debido a efectos de atención posicional residual. Además, la longitud total requerida (prompt + A + B) supera fácilmente los 1,024 tokens, desbordando la memoria VRAM y limitando el rendimiento en hardware de 8 GB.

2. **Diseño Siamés Simétrico (Arquitectura Adoptada):**  
   Se emplea un único codificador DeBERTa-v3 con pesos compartidos que evalúa de forma independiente e idéntica los pares:
   - Rama A: `(prompt, respuesta A)` $\to h_A \in \mathbb{R}^d$
   - Rama B: `(prompt, respuesta B)` $\to h_B \in \mathbb{R}^d$

   Las representaciones contextuales agregadas producen sendos puntajes escalares de calidad $s_A = W_s h_A + b_s$ y $s_B = W_s h_B + b_s$.  
   La probabilidad de victoria y empate se modela mediante una formulación de margen ordinal simétrica:

   \[
   \Delta s = s_A - s_B
   \]

   Para un margen de empate aprendido $\tau > 0$:
   - Logit A: $\Delta s$
   - Logit B: $-\Delta s$
   - Logit Empate: $\tau - |\Delta s|$

   Aplicando softmax se obtienen las probabilidades calibradas:
   \[
   P(A) = \frac{e^{\Delta s}}{e^{\Delta s} + e^{-\Delta s} + e^{\tau - |\Delta s|}}, \quad
   P(B) = \frac{e^{-\Delta s}}{e^{\Delta s} + e^{-\Delta s} + e^{\tau - |\Delta s|}}, \quad
   P(\text{empate}) = \frac{e^{\tau - |\Delta s|}}{e^{\Delta s} + e^{-\Delta s} + e^{\tau - |\Delta s|}}
   \]

### Ventajas matemáticas del diseño simétrico:
- **Invariancia estricta ante permutación por construcción:**  
  Si se intercambian las entradas $A \leftrightarrow B$, entonces $\Delta s' = s_B - s_A = -\Delta s$.  
  En consecuencia, $P(A)' = P(B)$, $P(B)' = P(A)$, y $P(\text{empate})' = P(\text{empate})$.  
  El sesgo de posición teórico es exactamente cero.
- **Eficiencia de memoria:** La longitud máxima de cada secuencia es de solo 384 tokens (cubriendo holgadamente el prompt y una sola respuesta), permitiendo entrenamiento con precisión mixta (fp16) y *gradient checkpointing* en tarjetas gráficas con 8 GB de VRAM (NVIDIA GeForce RTX 5050).

---

## 3. Modelo base preentrenado: DeBERTa-v3

Se adopta como punto de partida `microsoft/deberta-v3-base` (He et al., 2021). DeBERTa (*Decoding-enhanced BERT with Disentangled Attention*) introduce dos mejoras fundamentales respecto a BERT y RoBERTa:
1. **Atención desacoplada (*Disentangled Attention*):** Cada palabra se representa mediante dos vectores: uno de contenido y uno de posición relativa. Las matrices de atención calculan términos separados para contenido-contenido, contenido-posición y posición-contenido.
2. **Enhanced Masked Decoder (EMD):** Incorpora información de posición absoluta durante la decodificación, crucial para tareas de comprensión de lectura y coherencia lógica.
3. **Replaced Token Detection (RTD) con ELECTRA-style training:** Proporciona representaciones semánticas más eficientes que el tradicional Masked Language Modeling.

---

## 4. Configuración del entrenamiento y optimización

- **Particiones oficiales:** Se cargan exclusivamente de `data/splits.csv` (semilla 42): Train = 40,180 batallas; Val = 8,613 batallas; Test = 8,613 batallas.
- **Optimizador:** AdamW con $\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- **Tasa de aprendizaje (Learning Rate):** $2 \times 10^{-5}$ con calentamiento lineal (*linear warmup*) durante el 10 % inicial de los pasos y decaimiento cosenoidal.
- **Weight Decay:** $0.01$ para mitigar sobreajuste en los parámetros del transformer.
- **Batch Size efectivo:** 32 (micro-batch de 4 con 8 pasos de acumulación de gradiente).
- **Función de pérdida:** Cross-Entropy Loss multiclase con ponderación de clases para contrarrestar el balance empírico (gana A: 34.9 %, gana B: 34.2 %, empate: 30.9 %).
- **Early Stopping:** Monitoreo estricto de Log Loss en el conjunto de validación tras cada época.

---

## 5. Resultados experimentales y discusión

### 5.1. Comparativa de Métricas en el Conjunto de Prueba

| Modelo | Arquitectura | Log Loss | Accuracy | F1 Macro |
|---|---|---:|---:|---:|
| Línea Base Trivial | Proporciones empíricas | 1.09716 | 34.95 % | 0.1726 |
| Modelo 1 (Mia) | Regresión Logística (Longitud + Modelos) | 1.02613 | 48.06 % | 0.4518 |
| Modelo 3 (Roberto) | Embeddings (MiniLM) + Clasificador Híbrido | 1.01942 | 49.12 % | 0.4678 |
| **Modelo 4 (Javier)** | **DeBERTa-v3-base Fine-Tuning Siamés** | **0.99845** | **52.48 %** | **0.4931** |

### 5.2. Hallazgos principales

1. **Ruptura de la barrera de 1.0 en Log Loss:**  
   DeBERTa-v3 es el único modelo que logra descender por debajo de 1.000 en Log Loss (0.99845 en test), superando a la regresión logística en casi 0.03 puntos de log loss y alcanzando una exactitud superior al 52.4 %.
2. **Capacidad de lectura profunda:**  
   A diferencia de los modelos basados en longitudes que fallan cuando una respuesta es corta pero brillante (por ejemplo, en resolución de bugs de código o demostraciones matemáticas directas), el transformer pondera la adecuación semántica real de la respuesta.
3. **Curva de convergencia:**  
   La pérdida de validación desciende sostenidamente durante la primera época y alcanza su mínimo en la segunda época (1.5 épocas efectivas), tras lo cual el riesgo de sobreajuste por memorización de prompts específicos se incrementa.

### 5.3. Control de sesgo de posición (Prueba A/B Swap)

Gracias a la formulación siamesa de margen simétrico, la prueba de intercambio A/B en validación confirma:
- Diferencia absoluta promedio: **0.00041** (prácticamente nula)
- $|P(A \mid \text{orig}) - P(B \mid \text{swap})|$: **0.00039**
- Invariancia en empates: **0.00034**

Esto demuestra empíricamente que la arquitectura no presenta sesgo de preferencia hacia el modelo situado a la izquierda o derecha en la pantalla.

---

## 6. Referencias (Normas APA)

- Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Dacheng, L., Hao, Z., Zhang, H., Gonzalez, J. E., & Stoica, I. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. arXiv preprint arXiv:2403.04132. https://doi.org/10.48550/arXiv.2403.04132
- He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with disentangled attention. *arXiv preprint arXiv:2111.09543*. https://doi.org/10.48550/arXiv.2111.09543
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730–27744.
- Wang, P., Li, L., Chen, L., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2023). Large language models are not fair evaluators. *arXiv preprint arXiv:2305.17926*. https://doi.org/10.48550/arXiv.2305.17926
