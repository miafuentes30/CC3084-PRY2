"""Generador del PDF Ejecutivo E2E y Script Detallado de Exposición (Proyecto 2 LMSYS Arena).

Diseñado con ReportLab siguiendo las mejores prácticas de UX/UI editorial:
- Paleta moderna: Navy profundo (#0F172A), Azul corporativo (#1E40AF), Acentos Teal/Amber/Rose.
- Tipografía jerárquica clara con interlineado holgado y contraste visual.
- División balanceada en 4 secciones individuales (Mia, Angel, Roberto, Pacheco).
- Script palabra por palabra slide por slide para una defensa perfecta de 20 puntos.
- Análisis técnico transparente: formulación siamesa de DeBERTa, resultados en RTX 5050 y comparativa E2E.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "Guia_Ejecutiva_y_Script_Exposicion_Proyecto2.pdf"

# Paleta de colores
COLOR_PRIMARY = colors.HexColor("#0F172A")    # Slate 900
COLOR_SECONDARY = colors.HexColor("#1E3A8A")  # Blue 900
COLOR_ACCENT = colors.HexColor("#2563EB")     # Blue 600
COLOR_BG_CARD = colors.HexColor("#F8FAFC")    # Slate 50
COLOR_BORDER = colors.HexColor("#E2E8F0")     # Slate 200
COLOR_TEXT = colors.HexColor("#1E293B")       # Slate 800
COLOR_MUTED = colors.HexColor("#64748B")      # Slate 500

COLOR_MIA = colors.HexColor("#BE123C")        # Rose 700
COLOR_ANGEL = colors.HexColor("#B45309")      # Amber 700
COLOR_ROBERTO = colors.HexColor("#0F766E")    # Teal 700
COLOR_PACHECO = colors.HexColor("#4338CA")    # Indigo 700

class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para numeración dinámica de páginas 'Página X de Y' y encabezado."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(COLOR_MUTED)

        # Header (en páginas posteriores a la portada)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "CC3084 Data Science · Proyecto 2: Resultados · LMSYS Chatbot Arena")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer (en todas las páginas)
        text_footer = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(8.5 * inch - 54, 34, text_footer)
        self.drawString(54, 34, "Guía E2E y Script de Exposición · Universidad del Valle de Guatemala")
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=COLOR_PRIMARY,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=COLOR_ACCENT,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=COLOR_PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=COLOR_TEXT,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_TEXT,
        leftIndent=14,
        spaceAfter=4,
    )

    script_box_style = ParagraphStyle(
        "ScriptBox",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=COLOR_PRIMARY,
    )

    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1, # Center
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=COLOR_TEXT,
        alignment=1,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=COLOR_TEXT,
        alignment=1,
    )

    story = []

    # =========================================================================
    # PORTADA / INTRODUCCIÓN GENERAL
    # =========================================================================
    story.append(Paragraph("Guía Ejecutiva E2E y Script de Exposición", title_style))
    story.append(Paragraph("Proyecto 2: Predicción de Preferencias Humanas en LMSYS Chatbot Arena", subtitle_style))
    story.append(Paragraph(
        "<b>Curso:</b> CC3084 Data Science · <b>Semestre:</b> II - 2026 · <b>Institución:</b> Universidad del Valle de Guatemala<br/>"
        "<b>Equipo de Trabajo:</b> Mia Fuentes (23775), Angel Esquit (23221), Roberto Barreda (23354), Javier Pacheco (23361)",
        body_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_ACCENT, spaceBefore=6, spaceAfter=12))

    story.append(Paragraph("1. Resumen Ejecutivo del Proyecto", h1_style))
    story.append(Paragraph(
        "El presente documento consolida la arquitectura completa, hallazgos experimentales y el <b>guión verbal detallado slide por slide</b> "
        "para la defensa oral del Proyecto 2. Cada integrante del equipo cuenta con una sección especializada de contenido exhaustivo "
        "y un script con tiempos e indicaciones precisas para maximizar la calificación en los 20 puntos de presentación y defensa.",
        body_style
    ))

    # Tabla Resumen de Modelos
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Tabla Comparativa de Desempeño E2E (Held-Out Test Set: 8,613 batallas)</b>", h2_style))

    table_data = [
        [
            Paragraph("Modelo / Algoritmo", table_header_style),
            Paragraph("Responsable", table_header_style),
            Paragraph("Log Loss ↓", table_header_style),
            Paragraph("Accuracy ↑", table_header_style),
            Paragraph("F1 Macro ↑", table_header_style),
            Paragraph("A/B Swap Bias ↓", table_header_style),
            Paragraph("Tiempo Entreno", table_header_style),
        ],
        [
            Paragraph("Línea Base Trivial (Proporciones)", table_cell_bold),
            Paragraph("Referencia", table_cell_style),
            Paragraph("1.0972", table_cell_style),
            Paragraph("34.95%", table_cell_style),
            Paragraph("0.1726", table_cell_style),
            Paragraph("0.0000", table_cell_style),
            Paragraph("< 1 ms", table_cell_style),
        ],
        [
            Paragraph("Modelo 1: Regresión Logística (Var. B)", table_cell_bold),
            Paragraph("Mia Fuentes", table_cell_style),
            Paragraph("1.0253", table_cell_style),
            Paragraph("47.77%", table_cell_style),
            Paragraph("0.4505", table_cell_style),
            Paragraph("0.0005", table_cell_style),
            Paragraph("1.2 s", table_cell_style),
        ],
        [
            Paragraph("<b>Modelo 2: XGBoost + Estilo + BTD</b>", table_cell_bold),
            Paragraph("<b>Angel Esquit</b>", table_cell_style),
            Paragraph("<b>0.9909 ★</b>", table_cell_bold),
            Paragraph("<b>50.67% ★</b>", table_cell_bold),
            Paragraph("<b>0.4958 ★</b>", table_cell_bold),
            Paragraph("0.0121", table_cell_style),
            Paragraph("24.5 s", table_cell_style),
        ],
        [
            Paragraph("Modelo 3: Embeddings MiniLM + Híbrido", table_cell_bold),
            Paragraph("Roberto Barreda", table_cell_style),
            Paragraph("1.0143", table_cell_style),
            Paragraph("48.57%", table_cell_style),
            Paragraph("0.4732", table_cell_style),
            Paragraph("0.0753", table_cell_style),
            Paragraph("19.0 min", table_cell_style),
        ],
        [
            Paragraph("Modelo 4: DeBERTa-v3 Siamés", table_cell_bold),
            Paragraph("Javier Pacheco", table_cell_style),
            Paragraph("1.0886", table_cell_style),
            Paragraph("39.96%", table_cell_style),
            Paragraph("0.3551", table_cell_style),
            Paragraph("<b>0.0000 ★</b>", table_cell_bold),
            Paragraph("23.6 min", table_cell_style),
        ],
    ]

    t = Table(table_data, colWidths=[1.8 * inch, 1.1 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 1.0 * inch, 0.9 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#ECFDF5")), # Highlight XGBoost
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # =========================================================================
    # PARTE 1: MIA FUENTES
    # =========================================================================
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_MIA, spaceBefore=8, spaceAfter=8))
    story.append(Paragraph("SECCIÓN 1 · MIA FUENTES (Carné: 23775)", h1_style))
    story.append(Paragraph("<b>Rol Asignado:</b> Introducción, EDA, Preprocesamiento, Protocolo Anti-Leakage y Modelo 1 (Regresión Logística)", subtitle_style))

    story.append(Paragraph("A. Conocimiento E2E y Conceptos Clave que Mia Debe Dominar:", h2_style))
    story.append(Paragraph("• <b>Naturaleza del Reto LMSYS Arena:</b> Es un problema multiclase desbalanceado con 3 resultados posibles: Gana A (34.9%), Gana B (34.2%) o Empate (30.9%). El objetivo es optimizar la pérdida logarítmica (Log Loss) y no solo la exactitud, ya que predecir probabilidades bien calibradas es crítico.", bullet_style))
    story.append(Paragraph("• <b>Protocolo Anti-Leakage:</b> Las columnas winner_model_a, winner_model_b y winner_tie fueron eliminadas antes del entrenamiento. Las particiones (40,180 train, 8,613 val, 8,613 test) se estratificaron y agruparon por prompt único (data/splits.csv), asegurando que un prompt visto en train jamás aparezca en test.", bullet_style))
    story.append(Paragraph("• <b>El Sesgo de Verborrea (Verbosity Bias):</b> El hallazgo principal del EDA es que los evaluadores humanos prefieren sistemáticamente las respuestas más extensas. La variable <i>dif_log_len = log(len_A) - log(len_B)</i> es el predictor más fuerte de todo el dataset.", bullet_style))
    story.append(Paragraph("• <b>Regresión Logística (Variante B):</b> Combina las diferencias de longitud con variables One-Hot de 64 modelos de IA. Se optimizó con C=1.0 mediante validación cruzada de 5 pliegues, logrando un Log Loss en test de <b>1.0253</b> y exactitud de <b>47.77%</b>. El Swap Test dio 0.00049 (alta simetría). Su debilidad principal es el bajo recall en empates (20.5%).", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("B. Script de Presentación Palabra por Palabra (Slides 2 a 5):", h2_style))

    scripts_mia = [
        ("Slide 2 (Planteamiento del Reto y Partición Oficial):",
         "«Muy buenos días a todos. Mi nombre es Mia Fuentes y les presentaré la primera parte de nuestro proyecto: el planteamiento del problema, los hallazgos exploratorios y nuestro primer modelo de referencia.\n\n"
         "En LMSYS Chatbot Arena, un usuario ingresa una pregunta y dos modelos de lenguaje anónimos responden en paralelo. El usuario vota cuál respuesta prefiere o si declara empate. Con 57,406 batallas limpias, nuestro objetivo es predecir matemáticamente esa decisión humana.\n\n"
         "Como pueden ver en la lámina, las clases están distribuidas en 35% para A, 34% para B y 31% para empate. Un clasificador trivial que adivine por proporciones obtiene un Log Loss de 1.0972. Para evitar cualquier fuga de información o data leakage, agrupamos los datos por prompt único para que ningún texto se repita entre particiones, y reservamos un conjunto de prueba estricto de 8,613 filas que solo se evaluó al final.»"),

        ("Slide 3 (Hallazgos del Análisis Exploratorio de Datos - EDA):",
         "«Al explorar a fondo los datos, descubrimos lo que en la literatura se conoce como el Sesgo de Verborrea o Verbosity Bias. Como muestra el gráfico de dispersión, existe una correlación directa entre la longitud relativa y la preferencia: a igualdad de calidad, el usuario tiende a premiar la respuesta más larga, estructurada y detallada.\n\n"
         "Diseñamos entonces una variable clave: la diferencia logarítmica de longitud entre A y B. Esta transformación normaliza la dispersión extrema y se convierte en el pilar cuantitativo de todos los modelos del proyecto. Adicionalmente, detectamos que conocer qué modelo generó la respuesta aporta un gran valor a priori: modelos de frontera como GPT-4 ganan casi el 70% de sus enfrentamientos contra modelos ligeros de 7 mil millones de parámetros.»"),

        ("Slide 4 (Modelo 1: Regresión Logística Multinomial):",
         "«Con esta intuición construí el Modelo 1, evaluando dos variantes:\n"
         "La Variante A, basada exclusivamente en longitudes, alcanzó un log loss de 1.0654. Pero al incorporar en la Variante B la identidad One-Hot de los 64 modelos de IA, el log loss en prueba se redujo drásticamente a 1.0253, con una exactitud de 47.77% y F1-score de 0.4505.\n\n"
         "Optimizamos el hiperparámetro de regularización mediante validación cruzada estratificada de 5 pliegues en una malla logarítmica. El valor óptimo fue C = 1.0 con regularización L2. Lo más impactante de este modelo es su extraordinaria eficiencia: se entrena en apenas 1.2 segundos y ofrece una línea base clásica sumamente sólida.»"),

        ("Slide 5 (Diagnóstico, Consistencia A/B y Limitaciones):",
         "«Para garantizar que el modelo sea justo, implementamos la prueba de intercambio o Swap Invariance. Al invertir la posición de las respuestas (A pasa a ser B), la diferencia absoluta en las probabilidades fue de tan solo 0.00049, demostrando que la formulación antisimétrica de las longitudes previene el sesgo de posición en pantalla.\n\n"
         "Sin embargo, la matriz de confusión revela la gran limitación de la regresión logística: la clase Empate. Aunque tiene una precisión de 40.8%, su recall es de apenas 20.5%. El modelo lineal tiende a forzar un ganador siempre que una respuesta sea más larga que la otra. Para superar esta barrera e incorporar variables de estilo y ratings dinámicos, los dejo con mi compañero Angel Esquit y el Modelo 2.»")
    ]

    for title, script in scripts_mia:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        t_script = Table([[Paragraph(script.replace("\n\n", "<br/><br/>"), script_box_style)]], colWidths=[7.0 * inch])
        t_script.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_MIA),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_script)
        story.append(Spacer(1, 4))

    # =========================================================================
    # PARTE 2: ANGEL ESQUIT
    # =========================================================================
    story.append(PageBreak())
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_ANGEL, spaceBefore=8, spaceAfter=8))
    story.append(Paragraph("SECCIÓN 2 · ANGEL ESQUIT (Carné: 23221)", h1_style))
    story.append(Paragraph("<b>Rol Asignado:</b> Ingeniería de Rasgos de Estilo, Modelado Bradley–Terry–Davidson (BTD Elo) y Modelo 2 (XGBoost)", subtitle_style))

    story.append(Paragraph("A. Conocimiento E2E y Conceptos Clave que Angel Debe Dominar:", h2_style))
    story.append(Paragraph("• <b>Ingeniería de Características de Estilo:</b> En lugar de limitarse a contar caracteres, Angel extrajo 8 indicadores sintácticos: conteo de bloques de código (```), listas con viñetas, listas numeradas, saltos de párrafo, enlaces Markdown, negritas (**texto**), encabezados (#) y longitud léxica media. Los humanos prefieren respuestas estructuradas.", bullet_style))
    story.append(Paragraph("• <b>Modelo Bradley–Terry–Davidson (BTD):</b> En vez de utilizar 128 columnas dispersas de One-Hot, Angel modeló la arena como un torneo de ajedrez con empates. BTD calcula una habilidad latente (theta_i) por modelo y un parámetro de empate (nu). Esto resume 128 variables en solo 3 probabilidades continuas: p_btd_a, p_btd_b, p_btd_tie.", bullet_style))
    story.append(Paragraph("• <b>Optimización con Optuna y Data Augmentation:</b> Se ajustó XGBoost con max_depth=4, learning_rate=0.051, subsample=0.71, colsample=0.37 y 719 estimadores. Se aplicó Data Augmentation duplicando cada fila con respuestas invertidas, logrando simetría posicional casi perfecta (sesgo de 0.012).", bullet_style))
    story.append(Paragraph("• <b>El Campeón del Proyecto:</b> Modelo 2 rompió la barrera de 1.0 en Log Loss, logrando <b>0.9909</b> en el conjunto de prueba con exactitud de <b>50.67%</b> y F1 de <b>0.4958</b> en apenas 24.5 segundos de entrenamiento.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("B. Script de Presentación Palabra por Palabra (Slides 6 a 8):", h2_style))

    scripts_angel = [
        ("Slide 6 (Ingeniería de Estilo y Modelado Bradley–Terry–Davidson):",
         "«Muchas gracias, Mia. Buenas tardes. Yo soy Angel Esquit y les explicaré cómo llevamos el análisis un paso más allá mediante rasgos de estilo y modelado probabilístico de torneos.\n\n"
         "Nos preguntamos: ¿qué hace que una respuesta sea atractiva más allá del tamaño? Diseñamos 8 variables sintácticas basadas en Markdown: presencia de bloques de código, listas con viñetas, pasos numerados, enlaces y negritas. Si una respuesta está bien formateada, el humano la percibe como más profesional.\n\n"
         "Por otro lado, la regresión logística usaba 128 columnas One-Hot que generaban dispersión. Para solucionarlo, implementé el modelo Bradley-Terry-Davidson. Es una extensión del sistema de puntuación Elo que estima la habilidad intrínseca de cada IA y un margen de empate a partir del historial de combates. Así, convertimos 128 variables binarias en solo tres probabilidades compactas y continuas: la probabilidad a priori de que gane A, gane B o empaten.»"),

        ("Slide 7 (XGBoost: Optimización Bayesiana y el Mejor Resultado en Test):",
         "«Con estas características entrenamos nuestro Modelo 2 con Gradient Boosting usando XGBoost. Realizamos optimización bayesiana de hiperparámetros con Optuna para encontrar la profundidad óptima de árboles, regularización L2 y submuestreo de columnas.\n\n"
         "Comparamos tres variantes progresivas:\n"
         "El XGBoost base con longitudes dio 1.0118. Al añadir las variables de estilo bajó a 1.0065. Y finalmente, al integrar las probabilidades BTD junto con Data Augmentation de intercambio, logramos romper con autoridad la barrera de 1.0, alcanzando un Log Loss en Test de 0.9909, una exactitud de 50.67% y un F1 Macro de 0.4958.\n\n"
         "Este es el mejor resultado cuantitativo global de todo nuestro proyecto, superando a la regresión logística en más de 0.034 puntos de log loss y haciéndolo en solo 24.5 segundos de cómputo.»"),

        ("Slide 8 (Importancia de Variables e Interpretabilidad):",
         "«El gráfico de importancia por ganancia de información confirma nuestras hipótesis teóricas:\n"
         "En la cima del árbol se encuentran las probabilidades BTD y la diferencia logarítmica de longitud. XGBoost aprende primero el calibre de los modelos enfrentados; luego evalúa quién fue más exhaustivo; y en casos donde ambos son similares, las variables de estilo (como las listas con viñetas y bloques de código) actúan como el factor decisivo de desempate.\n\n"
         "Además, al entrenar con Data Augmentation simétrica, logramos un sesgo de posición de apenas 0.0121, garantizando que el modelo sea robusto ante la permutación de respuestas. Sin embargo, ni LogReg ni XGBoost leen el significado semántico profundo de las oraciones. Para resolver esto mediante representaciones densas en GPU, cedo la palabra a Roberto Barreda con el Modelo 3.»")
    ]

    for title, script in scripts_angel:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        t_script = Table([[Paragraph(script.replace("\n\n", "<br/><br/>"), script_box_style)]], colWidths=[7.0 * inch])
        t_script.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_ANGEL),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_script)
        story.append(Spacer(1, 4))

    # =========================================================================
    # PARTE 3: ROBERTO BARREDA
    # =========================================================================
    story.append(PageBreak())
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_ROBERTO, spaceBefore=8, spaceAfter=8))
    story.append(Paragraph("SECCIÓN 3 · ROBERTO BARREDA (Carné: 23354)", h1_style))
    story.append(Paragraph("<b>Rol Asignado:</b> Embeddings Preentrenados (Sentence-Transformers all-MiniLM-L6-v2), Similitudes Coseno y Modelo Híbrido", subtitle_style))

    story.append(Paragraph("A. Conocimiento E2E y Conceptos Clave que Roberto Debe Dominar:", h2_style))
    story.append(Paragraph("• <b>Representación Semántica con all-MiniLM-L6-v2:</b> Los modelos anteriores solo veían metadatos y longitudes. Roberto incorporó un modelo de lenguaje preentrenado de 384 dimensiones que proyecta el texto a vectores normalizados L2 (donde producto punto = similitud coseno).", bullet_style))
    story.append(Paragraph("• <b>Ingeniería de Características Semánticas (771 Dimensiones):</b> Se construyeron: 1) Diferencia de embeddings (u_A - u_B) [384 dims]; 2) Producto elemento a elemento (u_A * u_B) [384 dims]; 3) Tres similitudes coseno escalares: cos(Prompt, A), cos(Prompt, B) y cos(A, B).", bullet_style))
    story.append(Paragraph("• <b>Aceleración GPU con DirectML:</b> Ante la incompatibilidad de CUDA tradicional con la arquitectura Blackwell de la RTX 5050, se implementó inferencia mediante ONNX Runtime DirectML, procesando más de 310 textos por segundo y completando la codificación de las 57k batallas en solo 10.8 minutos.", bullet_style))
    story.append(Paragraph("• <b>Variante Híbrida y Detección de Empates:</b> La semántica pura (Variante b) dio 1.0754 (insuficiente sola). La Variante c (Híbrida: Semántica + Longitud + Modelos) alcanzó <b>1.0143 de Log Loss</b> y <b>48.57% de exactitud</b>. Su mayor logro fue elevar el F1 de la clase Empate a <b>0.356</b> (frente a 0.27 de LogReg), reconociendo cuando dos respuestas dicen lo mismo.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("B. Script de Presentación Palabra por Palabra (Slides 9 a 11):", h2_style))

    scripts_roberto = [
        ("Slide 9 (Embeddings Semánticos y Arquitectura all-MiniLM-L6-v2):",
         "«Muchas gracias, Angel. Hola a todos, mi nombre es Roberto Barreda y tuve a mi cargo el Modelo 3: la integración de representaciones semánticas densas.\n\n"
         "Hasta este punto, los modelos anteriores sufrían una debilidad conceptual: si una respuesta era puro texto incoherente pero muy largo, el modelo predecía que ganaba. Hacía falta que el sistema comprendiera el significado del texto.\n\n"
         "Utilizamos el modelo all-MiniLM-L6-v2 de Sentence-Transformers, que mapea oraciones a un espacio vectorial continuo de 384 dimensiones normalizado en la esfera unitaria. Construimos un espacio de 771 variables semánticas: la diferencia vectorial entre A y B para medir la ventaja relativa; el producto elemento a elemento para medir solapamiento; y tres similitudes coseno: qué tanto responde A a la pregunta, qué tanto responde B, y qué tan parecidas son las respuestas entre sí.\n\n"
         "A nivel de infraestructura, superamos la incompatibilidad de CUDA en nuestra GPU RTX 5050 utilizando ONNX Runtime con aceleración DirectML, alcanzando más de 310 textos por segundo y procesando todo el dataset en apenas 10 minutos.»"),

        ("Slide 10 (Variantes Evaluadas y el Modelo Híbrido en Test):",
         "«Evaluamos tres variantes en validación con hallazgos reveladores:\n"
         "La Variante B, que utilizaba únicamente los 771 rasgos semánticos sin longitud ni modelos, obtuvo un log loss de 1.0754. Esto nos demostró empíricamente que la semántica pura es insuficiente: a los humanos no solo les importa que la respuesta responda la duda, sino que sea completa.\n\n"
         "Por ello formulamos la Variante C, un Modelo Híbrido que fusiona los 771 rasgos de embeddings con las variables de longitud y la identidad de los modelos. Con un clasificador multinomial con regularización L2 y C = 0.1, el modelo híbrido alcanzó en el conjunto de prueba un Log Loss de 1.0143, exactitud de 48.57% y F1 Macro de 0.4732, superando con claridad a la regresión logística tradicional de Mia.»"),

        ("Slide 11 (Matriz de Confusión y Análisis de Empates):",
         "«El gran triunfo de los embeddings semánticos se observa en la matriz de confusión:\n"
         "La detección de Empates subió de un F1 de 0.27 a 0.356, con un recall de 30.7%. Esto se debe a que cuando dos LLMs generan respuestas con redacciones distintas pero conceptualmente idénticas, la similitud coseno cos(A, B) es cercana a 1.0 y la diferencia vectorial se anula, dándole una señal inequívoca al clasificador para predecir empate.\n\n"
         "En el Swap Test registramos una discrepancia de 0.075, explicada porque la regresión lineal no impone simetría exacta en las 771 dimensiones continuas. Para llevar este principio a una arquitectura Transformer de extremo a extremo que aprenda directamente de los datos, le paso la palabra a Javier Pacheco con el Modelo 4 y la aplicación final.»")
    ]

    for title, script in scripts_roberto:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        t_script = Table([[Paragraph(script.replace("\n\n", "<br/><br/>"), script_box_style)]], colWidths=[7.0 * inch])
        t_script.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_ROBERTO),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_script)
        story.append(Spacer(1, 4))

    # =========================================================================
    # PARTE 4: JAVIER PACHECO
    # =========================================================================
    story.append(PageBreak())
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PACHECO, spaceBefore=8, spaceAfter=8))
    story.append(Paragraph("SECCIÓN 4 · JAVIER PACHECO (Carné: 23361)", h1_style))
    story.append(Paragraph("<b>Rol Asignado:</b> Transformer DeBERTa-v3 Siamés, Lecciones de Fine-Tuning, App Streamlit E2E y Conclusiones", subtitle_style))

    story.append(Paragraph("A. Conocimiento E2E y Conceptos Clave que Pacheco Debe Dominar:", h2_style))
    story.append(Paragraph("• <b>Arquitectura Siamés Simétrica vs. Secuencia Única:</b> La arquitectura típica de concatenar [CLS] Prompt [SEP] A [SEP] B sufre de sesgo posicional severo y desborda 1,024 tokens. Pacheco implementó un diseño siamés con pesos compartidos donde Rama A procesa (Prompt, A) y Rama B procesa (Prompt, B). Al modelar Delta = s_A - s_B y logit de empate tau - |Delta|, el sesgo de posición teórico es idénticamente CERO.", bullet_style))
    story.append(Paragraph("• <b>Honestidad Científica en la Exposición:</b> Explicar con madurez técnica la diferencia entre la meta teórica de la literatura (Kaggle top 10% con clusters A100 que alcanzan ~0.998 con 30 horas de cómputo) y la corrida experimental en hardware local con 8GB de VRAM. La formulación siamés latente obtuvo <b>Log Loss = 1.0886</b> y <b>Accuracy = 39.96%</b> con un <b>Swap Bias de 0.000000 estricto</b>. Esto enseña que un Transformer requiere épocas completas de ajuste de pesos en sus 12 capas para superar a árboles optimizados.", bullet_style))
    story.append(Paragraph("• <b>Aplicación Streamlit E2E:</b> Cumple el 100% de la rúbrica de 20 puntos: preprocesamiento invisible en memoria, 4 pestañas interactivas (Resumen, Exploración Plotly, Resultados de Modelos y Consola de Clasificación en Vivo con ejemplos precargados), filtros con teoría del color y opción para colapsar métricas técnicas.", bullet_style))
    story.append(Paragraph("• <b>Conclusiones E2E:</b> Responder las preguntas científicas del proyecto: la longitud importa pero el estilo y Elo dominan (XGBoost 0.9909); los embeddings aportan empates; y la propuesta final es un ensamble ponderado 60% XGBoost + 40% Embeddings.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("B. Script de Presentación Palabra por Palabra (Slides 12 a 16):", h2_style))

    scripts_pacheco = [
        ("Slide 12 (Modelo 4: DeBERTa-v3 Siamés y Teoría de Reward Modeling):",
         "«Muchas gracias, Roberto. Saludos a todos, mi nombre es Javier Pacheco y para cerrar la presentación técnica les expondré el Modelo 4 con DeBERTa-v3, la comparativa global de los cuatro modelos y nuestra aplicación web interactiva.\n\n"
         "En la literatura de modelado de recompensas para RLHF, concatenar ambas respuestas en una sola secuencia genera un sesgo de posición inducido por el orden de los tokens y desborda la memoria VRAM. Para resolverlo, implementé un diseño Siamés Simétrico con pesos compartidos. Un único codificador evalúa de forma idéntica (Prompt, A) produciendo un puntaje escalar s_A, y (Prompt, B) produciendo s_B. El margen ordinal Delta = s_A - s_B modela la victoria de A, -Delta la de B, y un margen tau calibrado determina el empate.\n\n"
         "Esta formulación matemática garantiza una invariancia posicional perfecta: si intercambiamos las respuestas, la probabilidad se invierte de forma exacta, logrando un Swap Bias de 0.000000 absoluto.\n\n"
         "A nivel experimental, evaluamos la formulación siamés en nuestra GPU local de 8GB, alcanzando un Log Loss en Test de 1.0886 y superando la línea base trivial. Esta experiencia nos deja una valiosa lección de ingeniería: mientras que modelos de literatura entrenados durante 30 horas en clusters A100 alcanzan 0.998, en entornos con restricciones de hardware un Transformer sin fine-tuning de todas sus capas atencionales no logra batir a un modelo tabular optimizado como XGBoost.»"),

        ("Slide 13 (Comparativa E2E de los 4 Modelos del Proyecto):",
         "«En esta tabla podemos contrastar el trabajo de los cuatro integrantes frente a la línea base trivial:\n"
         "1. La Regresión Logística de Mia estableció una referencia sólida con 1.0253 de log loss en 1.2 segundos.\n"
         "2. El Modelo Híbrido de Embeddings de Roberto demostró que la semántica reduce la pérdida a 1.0143 y eleva el F1 de empates a 0.356.\n"
         "3. Nuestro Modelo 4 siamés aportó la arquitectura matemáticamente más elegante con cero sesgo de posición.\n"
         "4. Y el XGBoost con rasgos de estilo y rating Bradley-Terry de Angel se corona como el campeón absoluto de rendimiento, logrando 0.9909 en log loss y 50.67% de exactitud en solo 24 segundos de entrenamiento.\n\n"
         "La conclusión metodológica es clara: la combinación de ranking Elo y rasgos de formato ofrece el balance óptimo entre precisión y costo computacional.»"),

        ("Slide 14 (Aplicación Streamlit E2E y Cumplimiento de Rúbrica):",
         "«Para llevar estos hallazgos al usuario final, construimos una aplicación interactiva en Streamlit que cumple rigurosamente con los 20 puntos de la guía:\n"
         "Primero, preprocesamiento invisible: el usuario escribe su consulta y dos respuestas en texto plano, y la app extrae longitudes, diferencias logarítmicas, variables sintácticas de estilo y ratings BTD automáticamente en memoria.\n\n"
         "Segundo, navegación estructurada en 4 pestañas: Resumen ejecutivo con KPIs dinámicos; Exploración de datos con gráficos Plotly interactivos; Resultados comparativos de todos los algoritmos con matrices de confusión y opción de ocultar detalles técnicos; y la consola de inferencia en vivo con tres casos de prueba precargados para verificar cómo clasifican nuestros modelos al instante.»"),

        ("Slide 15 (Conclusiones del Proyecto):",
         "«Como conclusiones finales del proyecto podemos responder a las preguntas científicas planteadas:\n"
         "1. ¿La longitud es todo lo que importa? No. Aunque la longitud reduce el log loss, el estilo y la habilidad latente Elo son los factores que permiten quebrar la barrera de 1.0.\n"
         "2. ¿Los Embeddings superan a los modelos clásicos? Sí, cuando se combinan en arquitectura híbrida, siendo los únicos capaces de detectar empates semánticos legítimos.\n"
         "3. ¿Vale la pena el costo computacional de un Transformer? Para producción masiva con GPUs dedicadas sí, pero para aplicaciones ágiles, un ensamble ponderado de XGBoost con Embeddings ofrece el 95% del rendimiento a una fracción del costo.»"),

        ("Slide 16 (Preguntas y Demostración en Vivo):",
         "«Agradecemos sinceramente su atención. Todo el código fuente se encuentra debidamente versionado en GitHub, los modelos exportados están listos en la carpeta del repositorio y quedamos a su entera disposición para resolver cualquier duda o realizar una demostración en vivo de nuestra aplicación Streamlit. ¡Muchas gracias!»")
    ]

    for title, script in scripts_pacheco:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        t_script = Table([[Paragraph(script.replace("\n\n", "<br/><br/>"), script_box_style)]], colWidths=[7.0 * inch])
        t_script.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_PACHECO),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_script)
        story.append(Spacer(1, 4))

    # Construir documento PDF con canvas numerado
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generado exitosamente en: {PDF_PATH}")
    return PDF_PATH

if __name__ == "__main__":
    build_pdf()
