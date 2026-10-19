"""Paleta y estilo de gráficas de la app (método dataviz: slots categóricos en orden fijo).

Cada modelo conserva SIEMPRE su color, aunque se oculten otros. Los 4 primeros slots
categóricos de la paleta de referencia validan en pares adyacentes (CVD ΔE ≥ 8); aun así
las gráficas llevan etiquetas de valor visibles porque aqua y amarillo quedan bajo 3:1
de contraste sobre la superficie clara.
"""
from __future__ import annotations

import plotly.graph_objects as go

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# id -> (nombre, color)
MODELS = {
    "logreg": ("Regresión logística", "#2a78d6"),
    "xgboost": ("XGBoost", "#eb6834"),
    "embeddings": ("Embeddings + LR", "#1baf7a"),
    "deberta": ("DeBERTa-v3", "#eda100"),
}
MODEL_NAME = {k: v[0] for k, v in MODELS.items()}
MODEL_COLOR = {v[0]: v[1] for v in MODELS.values()}

# Resultado de la batalla (otro "trabajo" de color: identidad de la clase). Empate = neutro.
CLASS_NAMES = ["Gana A", "Gana B", "Empate"]
CLASS_COLOR = {"Gana A": "#6250d6", "Gana B": "#008300", "Empate": MUTED}

SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]]   # rojo ↔ gris ↔ azul

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.kpi {border: 1px solid #e1e0d9; border-radius: 10px; padding: 14px 16px; background: #fcfcfb;}
.kpi .l {font-size: 12px; color: #52514e; text-transform: uppercase; letter-spacing: .04em;}
.kpi .v {font-size: 28px; font-weight: 650; color: #0b0b0b; line-height: 1.2;}
.kpi .s {font-size: 12px; color: #52514e;}
.story {border-left: 4px solid #2a78d6; padding: 6px 14px; background: #f0efec; border-radius: 4px; margin: 6px 0 14px 0;}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; color:#0b0b0b; border:1px solid #c3c2b7;}
</style>
"""


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Estilo común: ejes recesivos, grilla de línea fina, tinta en tokens de texto."""
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_2, size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(size=15, color=INK), x=0.0, xanchor="left"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig
