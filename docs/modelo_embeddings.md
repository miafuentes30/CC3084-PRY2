# Representaciones semánticas densas (Embeddings) para preferencias humanas en Chatbot Arena

**Autor:** Roberto José Barreda Siekavizza (Carné: 23354)  
**Proyecto 2 — Resultados | CC3084 Data Science | Semestre II 2026**  
**Universidad del Valle de Guatemala**

---

## 1. Objetivo específico del modelo de embeddings

El objetivo central de este modelo es determinar si el significado semántico profundo del prompt y de las respuestas contiene señal predictiva sobre la preferencia humana en una evaluación ciega por pares, más allá de lo que pueden capturar variables superficiales de longitud, formato o la mera identidad de los modelos.

Para ello, se implementa una arquitectura basada en un codificador denso preentrenado (*sentence transformer* congelado: `sentence-transformers/all-MiniLM-L6-v2`) sobre el cual se construyen representaciones contrastivas simétricas que alimentan a un clasificador multiclase multinomial optimizado con regularización L2.

Esta investigación responde directamente a la pregunta científica formulada en la planificación común:
> *¿Aporta información el significado del texto, más allá de longitudes y estilo?*

---

## 2. Justificación a partir del EDA y revisión bibliográfica

El análisis exploratorio preliminar (sobre 57,406 batallas limpias) reveló dos patrones fundamentales:
1. **Sesgo de longitud (Verbosity Bias):** La respuesta más extensa gana en el 61.63 % de las batallas decisivas con longitudes dispares, y la diferencia de longitud (`dif_len`) correlaciona moderadamente con las victorias (+0.177 con A, −0.179 con B).
2. **Límite de las variables tabulares:** La regresión logística de línea base (Modelo 1) alcanza un log loss de 1.027 al incluir identidades de modelo, pero no tiene capacidad de evaluar la *calidad temática*, la *relevancia* respecto al prompt ni la *precisión argumentativa* del texto.

En la literatura de procesamiento de lenguaje natural y modelado de preferencias humanas:
- **Reimers & Gurevych (2019)** introdujeron *Sentence-BERT*, demostrando que el agrupamiento siamés de representaciones densas con pooling medio y distancia coseno supera sustancialmente a las representaciones tradicionales de conteo (TF-IDF) y permite comparaciones semánticas a gran escala con bajo costo computacional.
- **Wang et al. (2020)** demostraron con *MiniLM* que la destilación de conocimiento de modelos Transformer profundos preserva más del 99 % de la fidelidad semántica reduciendo el número de parámetros a 22 millones y generando vectores de 384 dimensiones altamente discriminativos.
- **Chiang et al. (2024)** destacan en la plataforma Chatbot Arena que, aunque los humanos premian la exhaustividad, en tareas lógicas, de código y de resolución de problemas específicos, la concordancia semántica directa entre la pregunta del usuario y la respuesta es el criterio determinante de satisfacción.

---

## 3. Fundamento matemático del modelo

### 3.1. Extracción de embeddings densos y normalización

Dado un texto $t \in \{p, r_a, r_b\}$ correspondiente al prompt, respuesta A y respuesta B respectivamente, el codificador produce una secuencia de vectores contextuales $h_1, h_2, \dots, h_L \in \mathbb{R}^D$ ($D = 384$). Se aplica *mean pooling* ponderado por la máscara de atención:

\[
e_{\text{pool}}(t) = \frac{\sum_{i=1}^L m_i h_i}{\sum_{i=1}^L m_i}, \quad m_i \in \{0, 1\}
\]

Seguidamente, cada vector se proyecta a la hiperesfera unitaria mediante normalización $L_2$:

\[
e(t) = \frac{e_{\text{pool}}(t)}{\|e_{\text{pool}}(t)\|_2}
\]

De esta forma, el producto punto entre dos embeddings equivale exactamente a su similitud coseno:

\[
\cos(u, v) = u^\top v \quad \forall u, v \in \mathbb{S}^{D-1}
\]

### 3.2. Construcción de representaciones simétricas e invariancia posicional

Para comparar de forma no sesgada las respuestas A y B frente al prompt $p$, se formulan cinco bloques de características semánticas:

1. **Diferencia vectorial de contraste:**
   \[
   v_{\text{diff}} = e(r_a) - e(r_b) \in \mathbb{R}^D
   \]
   Captura la dirección en el espacio latente hacia la cual la respuesta A se distancia de la respuesta B. Al permutar $A \leftrightarrow B$, $v_{\text{diff}}' = -v_{\text{diff}}$ (antisimétrico).

2. **Producto de Hadamard (interacción semántica):**
   \[
   v_{\text{mult}} = e(r_a) \odot e(r_b) \in \mathbb{R}^D
   \]
   Mide la coincidencia coordenada a coordenada del contenido tratado por ambos modelos. Es estrictamente simétrico ante permutaciones ($v_{\text{mult}}' = v_{\text{mult}}$).

3. **Relevancia temático-contextual con el prompt:**
   \[
   s_a = e(p)^\top e(r_a) \in [-1, 1], \quad s_b = e(p)^\top e(r_b) \in [-1, 1]
   \]
   Mide qué tan alineada está cada respuesta respecto a la consulta original del usuario.

4. **Ventaja relativa de relevancia:**
   \[
   \Delta s = s_a - s_b \in [-2, 2]
   \]
   Indica cuál de las dos alternativas responde con mayor cercanía al requerimiento formulado.

La concatenación semántica total tiene dimensión $2D + 3 = 771$ variables continuas.

### 3.3. Clasificador multinomial

Las características se normalizan mediante `StandardScaler` ajustado únicamente en el conjunto de entrenamiento. Para las tres clases $k \in \{0: \text{gana A}, 1: \text{gana B}, 2: \text{empate}\}$, la probabilidad predicha se obtiene mediante la función *softmax*:

\[
P(Y_i = k \mid x_i) = \frac{\exp(\beta_k^\top x_i + b_k)}{\sum_{j=0}^2 \exp(\beta_j^\top x_j + b_j)}
\]

Optimizada con el algoritmo L-BFGS y regularización $L_2$ controlada por el hiperparámetro $C$:

\[
\min_{\beta, b} -\sum_{i=1}^N \sum_{k=0}^2 y_{ik} \log P(Y_i = k \mid x_i) + \frac{1}{2C} \sum_{k=0}^2 \|\beta_k\|_2^2
\]

---

## 4. Variantes evaluadas

Para responder rigurosamente a los requisitos de la fase 2 y demostrar el aporte de cada componente, se comparan tres variantes preespecificadas:

- **Variante (a) Solo Longitud y Base:** Emplea las 13 variables numéricas continuas de longitud, turnos y diferencias más las variables indicadoras *One-Hot* de los modelos rivales (idéntica a la Variante B del Modelo 1).
- **Variante (b) Solo Embeddings:** Utiliza exclusivamente los 771 predictores derivados de las representaciones vectoriales densas ($v_{\text{diff}}, v_{\text{mult}}, s_a, s_b, \Delta s$).
- **Variante (c) Híbrido Completo (Longitud + Modelos + Embeddings):** Integra las variables tabulares estandarizadas con los 771 predictores semánticos.

---

## 5. Particiones y reproducibilidad

Se utilizan de forma estricta las particiones fijadas y versionadas en `data/splits.csv` (semilla 42, estratificación por clase y agrupación por `prompt_text`):
- **Entrenamiento:** 40,180 observaciones (70 %).
- **Validación:** 8,613 observaciones (15 %).
- **Prueba (Held-Out Test):** 8,613 observaciones (15 %).

El conjunto de prueba se evaluó una sola vez tras haber congelado el modelo híbrido final.

---

## 6. Resultados y Discusión

### 6.1. Comparación de Variantes en Validación

| Variante | Log Loss | Accuracy | F1 Macro | Tiempo Entrenamiento |
|---|---:|---:|---:|---:|
| Baseline trivial (proporciones) | 1.09716 | 34.95 % | 0.1726 | < 0.001 s |
| (a) Solo Longitud y Base | 1.02742 | 48.06 % | 0.4518 | 3.8 s |
| (b) Solo Embeddings | 1.03185 | 47.45 % | 0.4432 | 18.2 s |
| **(c) Híbrido (Longitud + Embeddings)** | **1.01894** | **49.21 %** | **0.4684** | **24.5 s** |

**Hallazgo crucial:**
1. Los embeddings por sí solos logran una precisión del 47.45 % sin tener acceso a métricas explícitas de longitud en caracteres ni a la identidad de los modelos.
2. Al combinar la señal de longitud con la semántica densa en la **Variante (c)**, el log loss desciende por debajo de 1.020 (1.01894), superando a la regresión logística tradicional de longitud por casi una centésima completa en log loss y +1.15 % en accuracy. Esto confirma estadísticamente que **el significado del texto sí aporta información complementaria a la forma y longitud**.

### 6.2. Evaluación Final en el Conjunto de Prueba (Test Set)

Sobre las 8,613 batallas inéditas del conjunto de prueba:
- **Log Loss:** **1.01942**
- **Accuracy:** **49.12 %**
- **F1 Macro:** **0.4678**
- **Mejora sobre línea base trivial:** Log Loss reducido en **0.0777** puntos y Accuracy incrementada en **+14.17 %**.

### 6.3. Desempeño por Clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|---|---:|---:|---:|---:|
| Gana Model A (0) | 0.4952 | 0.5841 | 0.5360 | 3,010 |
| Gana Model B (1) | 0.4918 | 0.5786 | 0.5317 | 2,947 |
| Empate (2) | 0.4721 | 0.2894 | 0.3587 | 2,656 |

Al igual que en otros modelos, la clase **empate** presenta la mayor dificultad de clasificación debido a que muchas situaciones de empate corresponden a desacuerdos subjetivos o empates técnicos donde ambas respuestas son excelentes. No obstante, las variables semánticas elevan la precisión en empates a 47.21 %, reduciendo falsos positivos.

---

## 7. Control de Sesgo de Posición (Prueba A/B Swap)

Al intercambiar las respuestas A y B en el conjunto de validación (`src.features.swap_ab`):
- Diferencia absoluta promedio: **0.00312**
- $|P(A \mid \text{orig}) - P(B \mid \text{swap})|$: **0.00321**
- $|P(B \mid \text{orig}) - P(A \mid \text{swap})|$: **0.00318**
- Invariancia de empate $|P(\text{tie} \mid \text{orig}) - P(\text{tie} \mid \text{swap})|$: **0.00297**

La divergencia cercana a cero confirma que la formulación antisimétrica de $v_{\text{diff}}$ y simétrica de $v_{\text{mult}}$ neutraliza eficazmente el sesgo posicional.

---

## 8. Referencias (Normas APA)

- Böhning, D. (1992). Multinomial logistic regression algorithm. *Annals of the Institute of Statistical Mathematics*, 44(1), 197–200. https://doi.org/10.1007/BF00068689
- Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Dacheng, L., Hao, Z., Zhang, H., Gonzalez, J. E., & Stoica, I. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. arXiv preprint arXiv:2403.04132. https://doi.org/10.48550/arXiv.2403.04132
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained Transformers. *Advances in Neural Information Processing Systems*, 33, 5776–5788.
