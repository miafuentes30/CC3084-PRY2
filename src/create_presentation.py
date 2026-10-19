"""Generador de la presentación ejecutiva PowerPoint (.pptx) para el Proyecto 2 (LMSYS Arena).

Diseño profesional 16:9, paleta moderna, métricas reales, visualizaciones y división en 4 partes:
1. Mia (Intro, EDA, Limpieza, Baseline, Modelo 1 Regresión Logística)
2. Angel (Feature Engineering de Estilo, BTD Elo, Modelo 2 XGBoost)
3. Roberto (Embeddings Semánticos all-MiniLM-L6-v2, Clasificador Híbrido, Modelo 3)
4. Pacheco/Javier (Transformer DeBERTa-v3 Siamés, Lecciones de Fine-Tuning, App Streamlit E2E, Conclusiones)
"""
from __future__ import annotations

import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]

# Colores del tema
NAVY = RGBColor(15, 23, 42)        # Slate 900
SLATE_DARK = RGBColor(30, 41, 59)   # Slate 800
BLUE_ACCENT = RGBColor(37, 99, 235) # Blue 600
CYAN_ACCENT = RGBColor(6, 182, 212) # Cyan 500
GREEN_ACCENT = RGBColor(16, 185, 129)# Emerald 500
AMBER_ACCENT = RGBColor(245, 158, 11)# Amber 500
PURPLE_ACCENT = RGBColor(147, 51, 234)# Purple 600
LIGHT_BG = RGBColor(248, 250, 252)  # Slate 50
CARD_BG = RGBColor(255, 255, 255)   # White
BORDER_COLOR = RGBColor(226, 232, 240) # Slate 200
TEXT_MAIN = RGBColor(15, 23, 42)
TEXT_MUTED = RGBColor(100, 116, 139)
WHITE = RGBColor(255, 255, 255)

SPEAKER_COLORS = {
    "Mia": RGBColor(225, 29, 72),      # Rose 600
    "Angel": RGBColor(217, 119, 6),    # Amber 600
    "Roberto": RGBColor(13, 148, 136), # Teal 600
    "Pacheco": RGBColor(79, 70, 229),  # Indigo 600
    "Equipo": RGBColor(37, 99, 235),
}

def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text, speaker_name, slide_num):
        # Header background banner
        header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        header_box.fill.solid()
        header_box.fill.fore_color.rgb = SLATE_DARK
        header_box.line.color.rgb = SLATE_DARK

        # Category / Breadcrumb
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(8), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CYAN_ACCENT

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(8.5), Inches(0.65))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(20)
        p_t.font.bold = True
        p_t.font.color.rgb = WHITE

        # Speaker Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.2), Inches(0.28), Inches(2.2), Inches(0.55))
        badge.fill.solid()
        badge.fill.fore_color.rgb = SPEAKER_COLORS.get(speaker_name, BLUE_ACCENT)
        badge.line.color.rgb = WHITE
        tf_b = badge.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = f"Expone: {speaker_name}"
        p_b.font.size = Pt(12)
        p_b.font.bold = True
        p_b.font.color.rgb = WHITE
        p_b.alignment = PP_ALIGN.CENTER

        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(12.2), Inches(7.05), Inches(0.9), Inches(0.35))
        tf_num = num_box.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num} / 16"
        p_num.font.size = Pt(10)
        p_num.font.color.rgb = TEXT_MUTED
        p_num.alignment = PP_ALIGN.RIGHT

    def add_card(slide, left, top, width, height, title="", border_color=BORDER_COLOR, bg_color=CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        if title:
            tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(width - 0.4), Inches(0.4))
            tf = tb.text_frame
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = BLUE_ACCENT
        return card

    # =========================================================================
    # SLIDE 1: PORTADA (EQUIPO)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.color.rgb = NAVY

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.333), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Predicción de Preferencias Humanas en Chatbot Arena"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Modelado Predictivo E2E: De Regresión Logística y Modelos de Estilo a Embeddings GPU y DeBERTa-v3"
    p1_sub.font.size = Pt(16)
    p1_sub.font.color.rgb = CYAN_ACCENT
    p1_sub.space_before = Pt(12)

    p1_ctx = tf1.add_paragraph()
    p1_ctx.text = "CC3084 Data Science · Proyecto 2: Resultados · Universidad del Valle de Guatemala · Semestre II 2026"
    p1_ctx.font.size = Pt(12)
    p1_ctx.font.color.rgb = RGBColor(148, 163, 184)
    p1_ctx.space_before = Pt(8)

    # 4 Cards de Integrantes y Modelos
    members = [
        ("1. Mia Fuentes", "23775", "Modelo 1: Regresión Logística", "Longitud + Identidad One-Hot", SPEAKER_COLORS["Mia"]),
        ("2. Angel Esquit", "23221", "Modelo 2: XGBoost + BTD", "Features de Estilo + Rating Elo BTD", SPEAKER_COLORS["Angel"]),
        ("3. Roberto Barreda", "23354", "Modelo 3: Embeddings MiniLM", "Semántica D=384 + GPU DirectML", SPEAKER_COLORS["Roberto"]),
        ("4. Javier Pacheco", "23361", "Modelo 4: Transformer DeBERTa", "Arquitectura Siamés + App Streamlit", SPEAKER_COLORS["Pacheco"]),
    ]
    for i, (name, carnet, model, tech, color) in enumerate(members):
        left = 1.0 + i * 2.9
        c = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(3.9), Inches(2.7), Inches(2.8))
        c.fill.solid()
        c.fill.fore_color.rgb = SLATE_DARK
        c.line.color.rgb = color
        c.line.width = Pt(2)
        tb = s1.shapes.add_textbox(Inches(left + 0.15), Inches(4.05), Inches(2.4), Inches(2.5))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        
        p_c = tf.add_paragraph()
        p_c.text = f"Carné: {carnet}"
        p_c.font.size = Pt(10)
        p_c.font.color.rgb = RGBColor(148, 163, 184)
        
        p_m = tf.add_paragraph()
        p_m.text = f"\n{model}"
        p_m.font.size = Pt(11)
        p_m.font.bold = True
        p_m.font.color.rgb = color
        
        p_t = tf.add_paragraph()
        p_t.text = f"\n{tech}"
        p_t.font.size = Pt(10)
        p_t.font.color.rgb = RGBColor(203, 213, 225)

    # =========================================================================
    # PARTE 1: MIA FUENTES (SLIDES 2, 3, 4, 5)
    # =========================================================================
    # SLIDE 2: Problema y Partición Oficial
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "El Desafío LMSYS Chatbot Arena y Protocolo Anti-Leakage", "Fase 1: Planteamiento y Datos", "Mia", 2)
    
    add_card(s2, 0.8, 1.4, 5.6, 5.4, "El Problema de Negocio y Ciencia de Datos")
    tb2_l = s2.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf2_l = tb2_l.text_frame
    tf2_l.word_wrap = True
    p = tf2_l.paragraphs[0]
    p.text = "• Batallas ciegas cara a cara: Un usuario hace una pregunta; dos LLMs anónimos (A y B) responden simultáneamente. El usuario juzga cuál respuesta fue mejor o si hubo empate."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MAIN
    
    p = tf2_l.add_paragraph()
    p.text = "\n• Desafío analítico: Predecir la decisión humana (Gana A, Gana B, Empate) usando únicamente el texto del prompt, las respuestas generadas y los identificadores de modelo."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MAIN

    p = tf2_l.add_paragraph()
    p.text = "\n• Distribución empírica de clases en 57,406 batallas limpias:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf2_l.add_paragraph()
    p.text = "  - Gana A: 34.93%\n  - Gana B: 34.22%\n  - Empate (Tie): 30.85%\n• Línea base trivial (clasificador proporcional): Log Loss = 1.0972, Exactitud = 34.95%."
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MUTED

    add_card(s2, 6.8, 1.4, 5.7, 5.4, "Protocolo de Partición y Prevención de Fugas")
    tb2_r = s2.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf2_r = tb2_r.text_frame
    tf2_r.word_wrap = True
    p = tf2_r.paragraphs[0]
    p.text = "• Partición versionada estricta (data/splits.csv):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    
    p = tf2_r.add_paragraph()
    p.text = "  - Train: 40,180 batallas (70%)\n  - Validación: 8,613 batallas (15%)\n  - Prueba (Held-Out Test): 8,613 batallas (15%)"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MAIN

    p = tf2_r.add_paragraph()
    p.text = "\n• Regla de Oro Metodológica:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Mia"]

    p = tf2_r.add_paragraph()
    p.text = "  1. Agrupamiento por prompt: Ningún prompt idéntico se comparte entre train, val y test, evitando memorización espuria de preguntas repetitivas.\n  2. Cero Data Leakage: Las columnas winner_model_a, winner_model_b y winner_tie se excluyen terminantemente antes de ingresar a cualquier modelo.\n  3. Test evaluado exactamente UNA vez al final de la selección de hiperparámetros."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 3: EDA Hallazgos Clave
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Hallazgos del Análisis Exploratorio: El Sesgo de Verborrea", "Fase 1: EDA y Características", "Mia", 3)
    
    # Imagen de scatter / boxplot si existe
    img_scatter = ROOT / "docs" / "figures" / "fig_scatter_longitudes_resultado.png"
    if img_scatter.exists():
        s3.shapes.add_picture(str(img_scatter), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s3, 0.8, 1.5, 5.8, 5.2, "Visualización de Longitudes")

    add_card(s3, 6.9, 1.5, 5.6, 5.2, "3 Descubrimientos Cruciales en los Datos")
    tb3 = s3.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "1. Sesgo de Verborrea (Verbosity Bias):"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf3.add_paragraph()
    p.text = "Los usuarios humanos muestran una fuerte inclinación a preferir la respuesta más larga, detallada o estructurada con viñetas, incluso cuando el contenido central es equivalente."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf3.add_paragraph()
    p.text = "\n2. Diferencia Logarítmica de Longitud (dif_log_len):"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf3.add_paragraph()
    p.text = "La variable log(len_A) - log(len_B) es el predictor cuantitativo individual más potente de todo el conjunto de datos, logrando separar victorias sin necesidad de NLP complejo."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf3.add_paragraph()
    p.text = "\n3. Jerarquía y Reputación de Modelos:"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf3.add_paragraph()
    p.text = "Modelos líderes como GPT-4 y Claude-3 superan consistentemente a modelos open-source de 7B parámetros. Conocer quién respondió aporta una probabilidad a priori muy marcada."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 4: Modelo 1 Regresión Logística
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Modelo 1: Regresión Logística Multinomial y Búsqueda de C", "Fase 2: Modelos Clásicos", "Mia", 4)

    img_c = ROOT / "docs" / "figures" / "logreg_busqueda_c.png"
    if img_c.exists():
        s4.shapes.add_picture(str(img_c), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s4, 0.8, 1.5, 5.8, 5.2, "Búsqueda de Hiperparámetro C")

    add_card(s4, 6.9, 1.5, 5.6, 5.2, "Metodología y Resultados de Variante B")
    tb4 = s4.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    
    p = tf4.paragraphs[0]
    p.text = "• Comparativa de Variantes Evaluadas:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf4.add_paragraph()
    p.text = "  - Variante A (Solo Longitudes):\n    Log Loss = 1.0654 | Acc = 43.97%\n  - Variante B (Longitudes + Identidad One-Hot 64 LLMs):\n    Log Loss = 1.0253 | Acc = 47.77% | F1 = 0.4505"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf4.add_paragraph()
    p.text = "\n• Optimización de Hiperparámetros:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf4.add_paragraph()
    p.text = "Se exploraron valores de C en escala logarítmica [0.001, 10.0] mediante Validación Cruzada estratificada de 5 pliegues. El valor óptimo fue C = 1.0 con regularización L2 y solver lbfgs."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf4.add_paragraph()
    p.text = "\n• Conclusión de Modelo 1:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf4.add_paragraph()
    p.text = "Demuestra que la identidad del modelo y la longitud reducen significativamente el log loss respecto al baseline trivial (de 1.0972 a 1.0253) en apenas 1.2 segundos de cómputo."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 5: Diagnóstico y Swap Test de Modelo 1
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Control de Sesgo Posicional y Matriz de Confusión en LogReg", "Fase 2: Diagnóstico Modelo 1", "Mia", 5)

    img_cm_lr = ROOT / "docs" / "figures" / "logreg_matriz_confusion.png"
    if img_cm_lr.exists():
        s5.shapes.add_picture(str(img_cm_lr), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s5, 0.8, 1.5, 5.8, 5.2, "Matriz de Confusión")

    add_card(s5, 6.9, 1.5, 5.6, 5.2, "Hallazgos de la Prueba de Consistencia A/B")
    tb5 = s5.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "• Prueba de Intercambio A/B (Swap Invariance):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p = tf5.add_paragraph()
    p.text = "Si invertimos las respuestas (A pasa a ser B y viceversa), un modelo perfecto debería predecir exactamente P(B) donde antes predecía P(A)."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf5.add_paragraph()
    p.text = "\n• Resultado en Regresión Logística:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT
    p = tf5.add_paragraph()
    p.text = "La discrepancia media absoluta fue de 0.00049. Es sumamente consistente gracias a que las variables dif_len y dif_words son antisimétricas por diseño."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf5.add_paragraph()
    p.text = "\n• El Gran Talón de Aquiles: La Clase Empate (Tie):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Mia"]
    p = tf5.add_paragraph()
    p.text = "La precisión en empates es de 40.8% pero el Recall es de solo 20.5% (F1 = 0.27). La regresión lineal tiende a forzar un ganador binario cuando las respuestas difieren en longitud."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # PARTE 2: ANGEL ESQUIT (SLIDES 6, 7, 8)
    # =========================================================================
    # SLIDE 6: Feature Engineering de Estilo y Bradley-Terry-Davidson
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Modelo 2: Ingeniería de Estilo y Modelado Bradley–Terry–Davidson", "Fase 2: Gradient Boosting", "Angel", 6)

    add_card(s6, 0.8, 1.4, 5.6, 5.4, "Más Allá de la Longitud: Variables de Estilo")
    tb6_l = s6.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf6_l = tb6_l.text_frame
    tf6_l.word_wrap = True
    p = tf6_l.paragraphs[0]
    p.text = "¿Qué hace que una respuesta se vea profesional y convincente a ojos humanos?"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    
    p = tf6_l.add_paragraph()
    p.text = "\nSe extrajeron 8 variables sintácticas y de formato independientes:"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN
    p = tf6_l.add_paragraph()
    p.text = "1. Bloques de código (```): Presencia de snippets y scripts.\n2. Listas con viñetas (- / *): Estructuración del pensamiento.\n3. Listas numeradas (1. / 2.): Instrucciones paso a paso.\n4. Saltos de párrafo (\\n\\n): Legibilidad y aire visual.\n5. Enlaces Markdown ([texto](url)): Evidencia externa.\n6. Formato negrita (**texto**): Énfasis en conceptos clave.\n7. Encabezados (# / ##): Jerarquía visual del contenido.\n8. Longitud promedio de palabra: Nivel léxico del texto."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    add_card(s6, 6.8, 1.4, 5.7, 5.4, "Rating Elo BTD: Bradley–Terry–Davidson")
    tb6_r = s6.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf6_r = tb6_r.text_frame
    tf6_r.word_wrap = True
    p = tf6_r.paragraphs[0]
    p.text = "Superando el One-Hot con Fuerza Latente Elo:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Angel"]

    p = tf6_r.add_paragraph()
    p.text = "En lugar de 128 columnas dispersas de One-Hot que dispersan la varianza, ajustamos un modelo BTD puro sobre el histórico de batallas de entrenamiento."
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf6_r.add_paragraph()
    p.text = "\n• Formulación BTD con Empates:\nCada modelo i tiene una habilidad latente theta_i y existe un parámetro de empate nu. La probabilidad de victoria viene dada por:\n  P(i vence a j) = exp(theta_i) / [exp(theta_i) + exp(theta_j) + exp(nu + (theta_i+theta_j)/2)]"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf6_r.add_paragraph()
    p.text = "\n• Ventaja: Reduce 128 variables categóricas a 3 probabilidades compactas y continuas: p_btd_a, p_btd_b, p_btd_tie."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    # SLIDE 7: XGBoost Optimización y Variantes
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "XGBoost: Búsqueda Bayesiana Optuna y Rompiendo el 1.0", "Fase 2: Gradient Boosting", "Angel", 7)

    img_xgb_var = ROOT / "docs" / "figures" / "fig_xgboost_variantes.png"
    if img_xgb_var.exists():
        s7.shapes.add_picture(str(img_xgb_var), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s7, 0.8, 1.5, 5.8, 5.2, "Comparación de Variantes XGBoost")

    add_card(s7, 6.9, 1.5, 5.6, 5.2, "El Mejor Modelo del Proyecto en Test")
    tb7 = s7.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "• Comparativa de Variantes de XGBoost:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf7.add_paragraph()
    p.text = "  - Base (Longitudes + One-Hot): Log Loss = 1.0118\n  - Con Estilo Sintáctico: Log Loss = 1.0065\n  - Con BTD Elo + Data Augmentation Swap:\n    Log Loss = 0.9909 | Acc = 50.67% | F1 = 0.4958"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf7.add_paragraph()
    p.text = "\n• Optimización de Hiperparámetros (Optuna):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf7.add_paragraph()
    p.text = "Ajustado con max_depth=4, learning_rate=0.051, subsample=0.71, colsample=0.37, reg_lambda=4.74 y 719 árboles con Early Stopping."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf7.add_paragraph()
    p.text = "\n• Hito del Proyecto:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf7.add_paragraph()
    p.text = "¡Es el campeón absoluto de rendimiento general! Log Loss de 0.9909 en test, superando a la regresión logística en más de 0.034 puntos."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 8: Importancia de Features XGBoost
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Importancia de Características e Interpretabilidad", "Fase 2: Diagnóstico Modelo 2", "Angel", 8)

    img_xgb_imp = ROOT / "docs" / "figures" / "fig_xgboost_importancia.png"
    if img_xgb_imp.exists():
        s8.shapes.add_picture(str(img_xgb_imp), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s8, 0.8, 1.5, 5.8, 5.2, "Importancia de Variables")

    add_card(s8, 6.9, 1.5, 5.6, 5.2, "Lecciones Extraídas del Árbol de Decisión")
    tb8 = s8.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "• Top 3 de Variables con Mayor Ganancia (Gain):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf8.add_paragraph()
    p.text = "  1. p_btd_a y p_btd_b: La probabilidad previa calculada por Bradley-Terry domina la raíz de los árboles.\n  2. dif_log_len y dif_len: La diferencia de longitud modula fuertemente las predicciones.\n  3. Viñetas y código: La presencia de listas con viñetas y bloques de código inclina las preferencias en empates técnicos."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf8.add_paragraph()
    p.text = "\n• Simetría Asegurada por Data Augmentation:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Angel"]
    p = tf8.add_paragraph()
    p.text = "Al entrenar con cada batalla duplicada e invertida (A<->B), el sesgo de posición en test se redujo a solo 0.0121, garantizando justicia posicional."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # PARTE 3: ROBERTO BARREDA (SLIDES 9, 10, 11)
    # =========================================================================
    # SLIDE 9: Embeddings Semánticos y Arquitectura
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Modelo 3: Embeddings Preentrenados y Similitud Coseno", "Fase 2: Representación Semántica", "Roberto", 9)

    add_card(s9, 0.8, 1.4, 5.6, 5.4, "¿Cómo Capturar el Significado del Texto?")
    tb9_l = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf9_l = tb9_l.text_frame
    tf9_l.word_wrap = True
    p = tf9_l.paragraphs[0]
    p.text = "Limitación de los modelos anteriores:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf9_l.add_paragraph()
    p.text = "La regresión logística y XGBoost no leen las palabras. No saben si una respuesta responde a la pregunta o si es texto incoherente pero largo."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf9_l.add_paragraph()
    p.text = "\n• Modelo Seleccionado: all-MiniLM-L6-v2:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Roberto"]
    p = tf9_l.add_paragraph()
    p.text = "  - Dimensión de embedding: D = 384.\n  - Salida normalizada en la esfera unitaria L2 (el producto punto equivale al coseno).\n  - Genera vectores densos de significado para Prompt (P), Respuesta A (A) y Respuesta B (B)."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf9_l.add_paragraph()
    p.text = "\n• Inferencia Acelerada con DirectML en RTX 5050:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf9_l.add_paragraph()
    p.text = "Superamos la incompatibilidad de CUDA en Blackwell utilizando ONNX Runtime con DirectML, alcanzando más de 310 textos/segundo en GPU."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    add_card(s9, 6.8, 1.4, 5.7, 5.4, "Ingeniería de Características Semánticas")
    tb9_r = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf9_r = tb9_r.text_frame
    tf9_r.word_wrap = True

    p = tf9_r.paragraphs[0]
    p.text = "Espacio de Variables Construido (771 dimensiones):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf9_r.add_paragraph()
    p.text = "1. Diferencia vectorial: (u_A - u_B) [384 dims]\n   Captura el vector de desplazamiento semántico de A hacia B."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    p = tf9_r.add_paragraph()
    p.text = "\n2. Producto elemento a elemento: (u_A * u_B) [384 dims]\n   Mide el solapamiento temático directo entre ambas respuestas."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    p = tf9_r.add_paragraph()
    p.text = "\n3. Similitudes Coseno Escalares [3 dims]:\n   - cos(Prompt, A): ¿Qué tanto responde A a la duda?\n   - cos(Prompt, B): ¿Qué tanto responde B a la duda?\n   - cos(A, B): ¿Qué tan similares son entre sí?"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    # SLIDE 10: Rendimiento del Modelo Híbrido
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Modelo 3: Variante Híbrida y Resultados en Test", "Fase 2: Semántica + Estilo", "Roberto", 10)

    img_emb_var = ROOT / "docs" / "figures" / "fig_embeddings_variantes.png"
    if img_emb_var.exists():
        s10.shapes.add_picture(str(img_emb_var), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s10, 0.8, 1.5, 5.8, 5.2, "Variantes Embeddings")

    add_card(s10, 6.9, 1.5, 5.6, 5.2, "Hallazgo Clave: La Semántica Requiere Longitud")
    tb10 = s10.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "• Comparación de Variantes en Validación:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf10.add_paragraph()
    p.text = "  - Variante (b) [Solo Embeddings Semánticos 771D]:\n    Log Loss = 1.0754 | Acc = 43.30% (Débil)\n  - Variante (a) [Longitud + Identidades]:\n    Log Loss = 1.0268 | Acc = 48.10%\n  - Variante (c) [Híbrido: Semántica + Longitud + Modelos]:\n    Log Loss = 1.0274 | Acc = 48.67% | F1 = 0.4747"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf10.add_paragraph()
    p.text = "\n• Evaluación Final en Prueba (Held-Out Test Set):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf10.add_paragraph()
    p.text = "Con C = 0.1 y solver L-BFGS, el modelo híbrido alcanzó en Test:\n  - Log Loss: 1.0143\n  - Accuracy: 48.57%\n  - F1 Macro: 0.4732"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf10.add_paragraph()
    p.text = "\n• Conclusión Científica:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf10.add_paragraph()
    p.text = "Los embeddings por sí solos capturan el tópico pero ignoran la exhaustividad. Al combinarse con longitud e identidades, superan a la regresión logística tradicional."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 11: Diagnóstico y Swap de Modelo 3
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Matriz de Confusión y Análisis de Simetría en Embeddings", "Fase 2: Diagnóstico Modelo 3", "Roberto", 11)

    img_cm_emb = ROOT / "docs" / "figures" / "fig_confusion_embeddings.png"
    if img_cm_emb.exists():
        s11.shapes.add_picture(str(img_cm_emb), Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    else:
        add_card(s11, 0.8, 1.5, 5.8, 5.2, "Matriz de Confusión Embeddings")

    add_card(s11, 6.9, 1.5, 5.6, 5.2, "Análisis de Fortalezas y Debilidades")
    tb11 = s11.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.4))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "• Desglose por Clase en Test:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf11.add_paragraph()
    p.text = "  - Gana A: Prec = 50.7%, Rec = 56.1%, F1 = 0.533\n  - Gana B: Prec = 50.2%, Rec = 57.2%, F1 = 0.535\n  - Empate: Prec = 42.4%, Rec = 30.7%, F1 = 0.356"
    p.font.size = Pt(11.5)
    p.font.color.rgb = TEXT_MAIN

    p = tf11.add_paragraph()
    p.text = "\n• Mejora en la Detección de Empates:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf11.add_paragraph()
    p.text = "El F1 de empates subió de 0.27 (en Modelo 1) a 0.356 gracias a que el producto elemento a elemento (u_A * u_B) y cos(A, B) le informan al clasificador cuándo ambas respuestas son idénticas en significado."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf11.add_paragraph()
    p.text = "\n• Sesgo de Posición (A/B Swap):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT
    p = tf11.add_paragraph()
    p.text = "Diferencia de swap observada: 0.075. Ocurre porque la regresión multiclase no fuerza una restricción de antisimetría exacta en las 771 dimensiones."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # PARTE 4: JAVIER PACHECO (SLIDES 12, 13, 14, 15, 16)
    # =========================================================================
    # SLIDE 12: Modelo 4 DeBERTa Siamés y Teoría de Reward Modeling
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Modelo 4: DeBERTa-v3 Siamés y Teoría de Reward Modeling", "Fase 2: Deep Learning E2E", "Pacheco", 12)

    add_card(s12, 0.8, 1.4, 5.6, 5.4, "Arquitectura Siamés Simétrica vs. Secuencia Única")
    tb12_l = s12.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf12_l = tb12_l.text_frame
    tf12_l.word_wrap = True

    p = tf12_l.paragraphs[0]
    p.text = "La Trampa del Cross-Encoder Tradicional:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SPEAKER_COLORS["Pacheco"]

    p = tf12_l.add_paragraph()
    p.text = "Concatenar [CLS] Prompt [SEP] A [SEP] B sufre de sesgo de posición inducido por el orden de los tokens y desborda los 1,024 tokens de contexto, consumiendo enorme VRAM."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf12_l.add_paragraph()
    p.text = "\n• Diseño Siamés Simétrico con Pesos Compartidos:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf12_l.add_paragraph()
    p.text = "1. Rama A evalúa (Prompt, A) -> Puntaje de recompensa s_A\n2. Rama B evalúa (Prompt, B) -> Puntaje de recompensa s_B\n3. Margen relativo ordinal: Delta = s_A - s_B"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf12_l.add_paragraph()
    p.text = "\n• Margen de Empate Calibrado (tau):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf12_l.add_paragraph()
    p.text = "Logit A = Delta · gamma\nLogit B = -Delta · gamma\nLogit Empate = tau - |Delta| · gamma\n¡Garantiza P(A<->B) con diferencia de swap exactamente cero!"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    add_card(s12, 6.8, 1.4, 5.7, 5.4, "Análisis de la Corrida y Lecciones de Cómputo")
    tb12_r = s12.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf12_r = tb12_r.text_frame
    tf12_r.word_wrap = True

    p = tf12_r.paragraphs[0]
    p.text = "Honestidad Científica: Resultados Reales vs. Meta Teórica:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT

    p = tf12_r.add_paragraph()
    p.text = "• Meta teórica en literatura (Kaggle LMSYS top 10%): DeBERTa-v3 fine-tuning completo con 100k pasos alcanza ~0.998 log loss y 52% de exactitud con GPUs A100 de 40GB."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf12_r.add_paragraph()
    p.text = "\n• Corrida en Hardware Local (RTX 5050 Laptop 8GB):"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p = tf12_r.add_paragraph()
    p.text = "  - Al calibrar la formulación siamés latente sin fine-tuning de 30 horas, el modelo obtiene Log Loss = 1.0886 y Accuracy = 39.96% en Test.\n  - Supera la línea base trivial (1.0972), pero evidencia que un transformer requiere épocas completas de ajuste de pesos en sus 12 capas para superar a XGBoost.\n  - Swap bias obtenido: 0.000000 (perfecto)."
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_MUTED

    # SLIDE 13: Tabla Comparativa General E2E
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Comparativa E2E de los 4 Modelos del Proyecto", "Fase 3: Discusión Comparativa", "Pacheco", 13)

    # Tabla grande
    rows, cols = 6, 7
    table_shape = s13.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.733), Inches(4.2))
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(2.2) # Modelo
    table.columns[1].width = Inches(1.8) # Responsable
    table.columns[2].width = Inches(1.5) # Log Loss
    table.columns[3].width = Inches(1.4) # Accuracy
    table.columns[4].width = Inches(1.4) # F1 Macro
    table.columns[5].width = Inches(1.7) # Swap Bias
    table.columns[6].width = Inches(1.733) # Tiempo Entreno

    headers = ["Modelo / Arquitectura", "Responsable", "Log Loss ↓", "Accuracy ↑", "F1 Macro ↑", "Swap Bias ↓", "Tiempo Entreno"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = SLATE_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    data_rows = [
        ["Línea Base Trivial (Proporciones)", "Referencia", "1.0972", "34.95%", "0.1726", "0.0000", "< 1 ms"],
        ["Modelo 1: Regresión Logística (B)", "Mia Fuentes", "1.0253", "47.77%", "0.4505", "0.0005", "1.2 s"],
        ["Modelo 2: XGBoost + Estilo + BTD", "Angel Esquit", "0.9909 ★", "50.67% ★", "0.4958 ★", "0.0121", "24.5 s"],
        ["Modelo 3: Embeddings + Híbrido", "Roberto Barreda", "1.0143", "48.57%", "0.4732", "0.0753", "19.0 min"],
        ["Modelo 4: DeBERTa-v3 Siamés", "Javier Pacheco", "1.0886", "39.96%", "0.3551", "0.0000 ★", "23.6 min"],
    ]

    for i, row in enumerate(data_rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            if "★" in val:
                cell.fill.fore_color.rgb = RGBColor(236, 253, 245) # Light emerald
            else:
                cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else LIGHT_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10.5)
            if j >= 2:
                p.alignment = PP_ALIGN.CENTER
            if "★" in val or j == 0:
                p.font.bold = True
            p.font.color.rgb = TEXT_MAIN

    # Callout abajo de la tabla
    add_card(s13, 0.8, 5.9, 11.733, 1.2, "", BORDER_COLOR, CARD_BG)
    tb_c = s13.shapes.add_textbox(Inches(1.0), Inches(6.0), Inches(11.3), Inches(1.0))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Veredicto Técnico Final: XGBoost con rating Bradley-Terry y rasgos de estilo es el modelo con mejor compromiso costo-beneficio (Log Loss 0.9909 en 24 segundos). El diseño siamés de DeBERTa demuestra la máxima consistencia posicional (Swap Bias = 0), y los Embeddings logran la mayor detección de empates (F1 Tie = 0.356)."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    # SLIDE 14: Aplicación Streamlit E2E
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Aplicación Web Interactiva Streamlit E2E", "Fase 4: Despliegue y UX/UI", "Pacheco", 14)

    add_card(s14, 0.8, 1.4, 5.6, 5.4, "Arquitectura y Buenas Prácticas de UX/UI")
    tb14_l = s14.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf14_l = tb14_l.text_frame
    tf14_l.word_wrap = True

    p = tf14_l.paragraphs[0]
    p.text = "Cumplimiento Estricto de la Rúbrica:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    p = tf14_l.add_paragraph()
    p.text = "1. Preprocesamiento Invisible (15 pts):\n   El usuario ingresa texto sin formato y la app calcula longitudes, diferencias logarítmicas, variables de estilo y ratings BTD automáticamente en memoria."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf14_l.add_paragraph()
    p.text = "\n2. Exploración Dinámica de Datos:\n   Filtros interactivos de longitud, visualizaciones Plotly con teoría de color y opción para colapsar o mostrar métricas técnicas."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf14_l.add_paragraph()
    p.text = "\n3. Clasificación Multimodelo en Vivo:\n   Permite comparar las probabilidades de Regresión Logística y XGBoost sobre casos reales ingresados al instante."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    add_card(s14, 6.8, 1.4, 5.7, 5.4, "Organización de las 4 Pestañas de la App")
    tb14_r = s14.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf14_r = tb14_r.text_frame
    tf14_r.word_wrap = True

    tabs_info = [
        ("1. Resumen Ejecutivo:", "KPIs clave de la arena (57,406 batallas, 64 LLMs, 30.8% empates) y tabla comparativa de modelos."),
        ("2. Explorar los Datos:", "Gráficos de dispersión interactivos, correlaciones y distribución de ganadores según longitud y modelo."),
        ("3. Resultados de Modelos:", "Matrices de confusión, curvas ROC, calibración y análisis de sesgo A/B swap."),
        ("4. Probar un Caso Nuevo:", "Consola de prueba con 3 ejemplos precargados ('¿Qué es París?', 'Rechazo vs Ayuda', 'Sinónimos') y feedback explicativo de estilo."),
    ]
    for title, desc in tabs_info:
        p = tf14_r.add_paragraph() if tf14_r.paragraphs[0].text else tf14_r.paragraphs[0]
        p.text = title
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = SPEAKER_COLORS["Pacheco"]
        p_d = tf14_r.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_MUTED

    # SLIDE 15: Conclusiones del Proyecto
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Conclusiones y Respuestas a las Preguntas Científicas", "Fase 5: Cierre", "Pacheco", 15)

    add_card(s15, 0.8, 1.4, 5.6, 5.4, "Logros Frente a los Objetivos Iniciales")
    tb15_l = s15.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.6))
    tf15_l = tb15_l.text_frame
    tf15_l.word_wrap = True

    p = tf15_l.paragraphs[0]
    p.text = "1. ¿La longitud es todo lo que importa?"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf15_l.add_paragraph()
    p.text = "No. Aunque la longitud reduce el log loss a 1.065, el estilo (código, viñetas) y el rating Elo BTD bajan la pérdida a 0.9909. La identidad del modelo y la presentación aportan una señal inmensa."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf15_l.add_paragraph()
    p.text = "\n2. ¿Los Embeddings superan a los modelos clásicos?"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf15_l.add_paragraph()
    p.text = "Sí, pero solo en arquitectura híbrida (Log Loss 1.014 vs 1.025 de LogReg). Su mayor mérito radica en detectar empates semánticos (F1 Tie 0.356 vs 0.272)."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    add_card(s15, 6.8, 1.4, 5.7, 5.4, "Recomendaciones y Trabajo Futuro")
    tb15_r = s15.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.6))
    tf15_r = tb15_r.text_frame
    tf15_r.word_wrap = True

    p = tf15_r.paragraphs[0]
    p.text = "3. ¿Vale la pena el costo computacional de DeBERTa?"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p = tf15_r.add_paragraph()
    p.text = "En producción industrial con clusters A100 sí, pero para aplicaciones ágiles, un ensamble XGBoost + BTD ofrece 95% del beneficio a 1/500 del costo de inferencia."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf15_r.add_paragraph()
    p.text = "\n4. Ensamble Óptimo Propuesto:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p = tf15_r.add_paragraph()
    p.text = "El sistema ideal es un ensamble ponderado: 60% XGBoost (BTD + estilo) + 40% Embeddings Semánticos. Combina la fuerza del ranking Elo con la detección de paráfrasis semántica."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    # SLIDE 16: Preguntas y Cierre
    s16 = prs.slides.add_slide(blank_layout)
    bg16 = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg16.fill.solid()
    bg16.fill.fore_color.rgb = NAVY
    bg16.line.color.rgb = NAVY

    t_box = s16.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.5))
    tf16 = t_box.text_frame
    tf16.word_wrap = True
    p = tf16.paragraphs[0]
    p.text = "¡Muchas Gracias por su Atención!"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    p_sub = tf16.add_paragraph()
    p_sub.text = "Espacio para Preguntas, Dudas y Demostración en Vivo de Streamlit"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = CYAN_ACCENT
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.space_before = Pt(16)

    p_links = tf16.add_paragraph()
    p_links.text = "Código Versionado en GitHub · Artefactos en Google Drive · App disponible localmente"
    p_links.font.size = Pt(13)
    p_links.font.color.rgb = RGBColor(148, 163, 184)
    p_links.alignment = PP_ALIGN.CENTER
    p_links.space_before = Pt(24)

    out_path = ROOT / "presentacion_proyecto2_lmsys.pptx"
    prs.save(str(out_path))
    print(f"Presentación guardada exitosamente en {out_path}")
    return out_path

if __name__ == "__main__":
    create_deck()
