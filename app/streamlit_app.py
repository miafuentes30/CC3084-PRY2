"""Aplicación Streamlit – Proyecto 2 (LMSYS Chatbot Arena).

    streamlit run app/streamlit_app.py

Pestañas: Resumen · Explorar los datos · Resultados de los modelos · Probar un caso nuevo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import data, inference, theme  # noqa: E402
from app.theme import CLASS_COLOR, CLASS_NAMES, MODEL_COLOR, MODEL_NAME, kpi, style  # noqa: E402

TRIVIAL_LL = 1.0986
st.set_page_config(page_title="Arena: ¿qué respuesta prefiere la gente?", page_icon="🏟️", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def story(text: str) -> None:
    st.markdown(f'<div class="story">{text}</div>', unsafe_allow_html=True)


clean = data.load_clean()
models_avail = data.available_models()

st.title("¿Qué respuesta prefiere la gente?")
st.caption("LMSYS Chatbot Arena · CC3084 Data Science · Proyecto 2 — predicción de la preferencia humana entre dos LLMs")

if clean is None:
    st.error("Falta el dataset limpio. Descarga `train.csv` a `data/raw/`, ejecuta el notebook 02 "
             "(genera `data/processed/train_clean.csv`) y verifica que exista `data/splits.csv`.")

tab_resumen, tab_datos, tab_res, tab_pred = st.tabs(
    ["Resumen", "Explorar los datos", "Resultados de los modelos", "Probar un caso nuevo"])

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ════════════════════════════════════════════════════════════════════════════
with tab_resumen:
    story("**El problema.** Dos chatbots anónimos responden al mismo mensaje y una persona elige cuál "
          "prefiere (o declara empate). Con solo ver el texto y quién respondió, ¿podemos predecir su voto?")
    if clean is not None:
        c = st.columns(4)
        c[0].markdown(kpi("Batallas (limpias)", f"{len(clean):,}", "tras eliminar duplicados"), unsafe_allow_html=True)
        c[1].markdown(kpi("Modelos de IA", f"{pd.concat([clean.model_a, clean.model_b]).nunique()}", "enfrentados"), unsafe_allow_html=True)
        c[2].markdown(kpi("Empates", f"{(clean.winner == 2).mean():.1%}", "de las batallas"), unsafe_allow_html=True)
        c[3].markdown(kpi("Línea base", f"{TRIVIAL_LL:.4f}", "log loss de adivinar por proporciones"), unsafe_allow_html=True)

    if models_avail and clean is not None:
        rows = []
        for m in models_avail:
            mt = data.metrics_from_preds(m, "test")
            if mt:
                rows.append({"Modelo": MODEL_NAME[m], "Log loss": mt["log_loss"], "F1 macro": mt["f1_macro"], "Accuracy": mt["accuracy"]})
        if rows:
            res = pd.DataFrame(rows).sort_values("Log loss").reset_index(drop=True)
            best = res.iloc[0]
            st.subheader("Qué encontramos")
            story(f"Sobre el conjunto de **prueba** (nunca usado para ajustar nada), el mejor modelo es "
                  f"**{best['Modelo']}** con log loss **{best['Log loss']:.4f}** "
                  f"(adivinar por proporciones da {TRIVIAL_LL:.4f}; menor es mejor). "
                  f"Acierta {best['Accuracy']:.1%} de las veces, frente a ~35 % de la clase mayoritaria.")
            st.dataframe(res.style.format({"Log loss": "{:.4f}", "F1 macro": "{:.4f}", "Accuracy": "{:.2%}"})
                         .highlight_min(subset=["Log loss"], color="#cde2fb")
                         .highlight_max(subset=["F1 macro", "Accuracy"], color="#cde2fb"),
                         hide_index=True, width="stretch")
    st.markdown(
        "**Cómo navegar:** *Explorar los datos* cuenta qué patrones tiene la preferencia humana; "
        "*Resultados* compara los cuatro algoritmos (puedes ocultar los que no quieras ver); "
        "*Probar un caso nuevo* clasifica un prompt y dos respuestas que tú escribas.")

# ════════════════════════════════════════════════════════════════════════════
# EXPLORAR LOS DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab_datos:
    if clean is None:
        st.info("Genera el dataset limpio para explorar los datos.")
    else:
        view = st.radio("Conjunto", ["Todo (entrenamiento + validación + prueba)", "Solo entrenamiento"],
                        horizontal=True, label_visibility="collapsed")
        d = clean if view.startswith("Todo") else clean[clean.split == "train"]

        # 1. ¿Cómo se reparten los votos?
        st.subheader("1 · ¿Cómo se reparten los votos?")
        story("Los tres resultados están casi equilibrados: no hay una clase dominante que un modelo pueda "
              "adivinar sin mirar el contenido. Gana A y gana B son prácticamente iguales, así que la "
              "<b>posición</b> (izquierda/derecha) no decide por sí sola.")
        cnt = d["resultado"].value_counts().reindex(CLASS_NAMES)
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[CLASS_COLOR[k] for k in cnt.index],
                               text=[f"{v:,}<br>{v/cnt.sum():.1%}" for v in cnt.values], textposition="outside",
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_yaxes(range=[0, cnt.max() * 1.18], title="Batallas")
        show(style(fig, 330, legend=False).update_layout(title="Resultado de la batalla"))

        # 2. Longitud
        st.subheader("2 · ¿Gana la respuesta más larga?")
        story("Al dividir las batallas según cuánto más larga es la respuesta A que la B, el voto se inclina "
              "hacia la más larga: cuando A es mucho más larga gana A más a menudo, y al revés. "
              "Es la señal simple más fuerte, y la que aprovechan todos los modelos.")
        q = d.dropna(subset=["dif_len"]).copy()
        q["tramo"] = pd.qcut(q["dif_len"], 10, duplicates="drop")
        tab = (q.groupby("tramo", observed=True)["resultado"].value_counts(normalize=True).unstack()
               .reindex(columns=CLASS_NAMES))
        mid = q.groupby("tramo", observed=True)["dif_len"].median()
        fig = go.Figure()
        for cls in CLASS_NAMES:
            fig.add_scatter(x=mid.values, y=tab[cls].values, mode="lines+markers", name=cls,
                            line=dict(color=CLASS_COLOR[cls], width=2), marker=dict(size=8, line=dict(width=2, color=theme.SURFACE)),
                            hovertemplate=f"{cls}: " + "%{y:.1%}<br>Dif. de longitud (A − B): %{x:,.0f} car.<extra></extra>")
        fig.add_vline(x=0, line_color=theme.AXIS, line_dash="dot")
        fig.update_yaxes(tickformat=".0%", title="% de batallas del tramo")
        fig.update_xaxes(title="Diferencia de longitud A − B (caracteres, mediana por decil)")
        show(style(fig, 380).update_layout(title="Resultado según la diferencia de longitud"))

        # 3. Empates
        st.subheader("3 · ¿Cuándo hay empates?")
        story("Los empates se concentran cuando las dos respuestas tienen una longitud parecida entre sí. "
              "Si una es mucho más larga que la otra, es más fácil elegir.")
        d2 = d.copy()
        d2["tramo_abs"] = pd.qcut(d2["log_dif_abs"].rank(method="first"), 8, labels=False)
        g = d2.groupby("tramo_abs").agg(emp=("winner", lambda s: (s == 2).mean()),
                                         dif=("dif_len", lambda s: s.abs().median()), n=("winner", "size"))
        fig = go.Figure(go.Bar(x=[f"{v:,.0f}" for v in g["dif"]], y=g["emp"], marker_color=CLASS_COLOR["Empate"],
                               text=[f"{v:.1%}" for v in g["emp"]], textposition="outside",
                               hovertemplate="Diferencia |A − B| ≈ %{x} car.<br>Empates: %{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", title="% de empates", range=[0, g["emp"].max() * 1.2])
        fig.update_xaxes(title="Diferencia absoluta de longitud |A − B| (caracteres, mediana por octil)")
        show(style(fig, 340, legend=False).update_layout(title="Tasa de empates según qué tan distintas son las respuestas"))

        # 4. Modelos
        st.subheader("4 · ¿Importa quién responde?")
        story("Algunos modelos ganan claramente más que otros. Por eso la identidad del modelo (y su puntaje "
              "Bradley–Terry–Davidson) aporta información. El color de cada barra separa victorias, empates y derrotas.")
        a = d[[cfg for cfg in ["model_a", "winner"]]].rename(columns={"model_a": "modelo"})
        a["res"] = a["winner"].map({0: "Gana", 1: "Pierde", 2: "Empata"})
        b = d[["model_b", "winner"]].rename(columns={"model_b": "modelo"})
        b["res"] = b["winner"].map({1: "Gana", 0: "Pierde", 2: "Empata"})
        long = pd.concat([a, b])
        mt = long.groupby("modelo")["res"].value_counts(normalize=True).unstack().fillna(0)
        n_games = long.groupby("modelo").size()
        top_n = st.slider("Modelos a mostrar (más batallas primero)", 5, int(mt.shape[0]), min(15, int(mt.shape[0])))
        keep = n_games.sort_values(ascending=False).head(top_n).index
        mt = mt.loc[keep].sort_values("Gana")
        fig = go.Figure()
        for col, color in [("Gana", "#2a78d6"), ("Empata", theme.MUTED), ("Pierde", "#eb6834")]:
            fig.add_bar(y=mt.index, x=mt[col], orientation="h", name=col, marker=dict(color=color, line=dict(width=2, color=theme.SURFACE)),
                        text=[f"{v:.0%}" for v in mt[col]], textposition="inside", insidetextanchor="middle",
                        hovertemplate="%{y} — " + col + ": %{x:.1%}<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickformat=".0%", title="% de sus batallas")
        show(style(fig, max(380, 24 * top_n + 120)).update_layout(title="Resultado de cada modelo de IA en sus batallas"))

        # 5. Correlaciones
        st.subheader("5 · ¿Qué variables se relacionan con el voto?")
        story("Correlación de cada variable numérica con tres indicadores (¿ganó A?, ¿ganó B?, ¿empate?). "
              "Azul = más de la variable, más de ese resultado; rojo = lo contrario. "
              "Las diferencias de longitud empujan A y B en direcciones opuestas.")
        ind = pd.DataFrame({"Gana A": (d.winner == 0).astype(float), "Gana B": (d.winner == 1).astype(float),
                            "Empate": (d.winner == 2).astype(float)})
        nums = ["prompt_len", "resp_a_len", "resp_b_len", "dif_len", "prompt_words", "resp_a_words", "resp_b_words", "dif_words", "n_turns"]
        cm = pd.DataFrame({k: [d[v].corr(ind[k]) for v in nums] for k in ind}, index=nums)
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale=theme.DIVERGING, zmid=0,
                                   zmin=-0.25, zmax=0.25, text=np.round(cm.values, 3), texttemplate="%{text}",
                                   colorbar=dict(title="r"), hovertemplate="%{y} vs %{x}: r = %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 400, legend=False).update_layout(title="Correlación con el resultado"))

        # 6. Explorador libre
        st.subheader("6 · Explora una variable")
        num_labels = {"resp_a_len": "Longitud de la respuesta A", "resp_b_len": "Longitud de la respuesta B",
                      "prompt_len": "Longitud del prompt", "resp_a_words": "Palabras de A", "resp_b_words": "Palabras de B",
                      "prompt_words": "Palabras del prompt", "n_turns": "Turnos de la conversación", "dif_len": "Dif. de longitud A − B"}
        var = st.selectbox("Variable", list(num_labels), format_func=num_labels.get)
        cut = st.slider("Recortar valores extremos (percentil máximo)", 90, 100, 99)
        x = d[[var, "resultado"]].dropna()
        if var != "dif_len":
            x = x[x[var] <= x[var].quantile(cut / 100)]
        else:
            lo, hi = x[var].quantile((100 - cut) / 200), x[var].quantile(1 - (100 - cut) / 200)
            x = x[(x[var] >= lo) & (x[var] <= hi)]
        fig = go.Figure()
        for cls in CLASS_NAMES:
            fig.add_box(y=x.loc[x.resultado == cls, var], name=cls, marker_color=CLASS_COLOR[cls], boxmean=True)
        fig.update_yaxes(title=num_labels[var])
        show(style(fig, 380, legend=False).update_layout(title=f"{num_labels[var]} según el resultado"))
        with st.expander("Ver los datos (muestra de 500 filas)"):
            st.dataframe(d.drop(columns=["resultado", "log_dif_abs", "total_len"]).sample(min(500, len(d)), random_state=42), width="stretch")

# ════════════════════════════════════════════════════════════════════════════
# RESULTADOS DE LOS MODELOS
# ════════════════════════════════════════════════════════════════════════════
with tab_res:
    if not models_avail or clean is None:
        st.info("No hay predicciones en `results/preds/` o falta el dataset limpio para evaluarlas.")
    else:
        top = st.columns([3, 2])
        chosen = top[0].multiselect("Modelos a mostrar (puedes ocultar los que no quieras ver)",
                                    models_avail, default=models_avail, format_func=MODEL_NAME.get)
        split = top[1].radio("Conjunto de evaluación", ["test", "val"], format_func={"test": "Prueba", "val": "Validación"}.get, horizontal=True)
        if not chosen:
            st.warning("Elige al menos un modelo.")
        else:
            mets = {m: data.metrics_from_preds(m, split) for m in chosen}
            names = [MODEL_NAME[m] for m in chosen]
            cols = [MODEL_COLOR[n] for n in names]
            story("Todas las métricas se recalculan aquí con el mismo código a partir de las probabilidades "
                  "guardadas por cada modelo, así la comparación es justa. La métrica principal es el "
                  "<b>log loss</b> (menor es mejor): premia que las probabilidades sean honestas, no solo acertar.")

            # Resumen en tarjetas
            kc = st.columns(len(chosen))
            for col, m in zip(kc, chosen):
                mm = mets[m]
                col.markdown(kpi(MODEL_NAME[m], f"{mm['log_loss']:.4f}",
                                 f"log loss · F1 {mm['f1_macro']:.3f} · acc {mm['accuracy']:.1%}"), unsafe_allow_html=True)

            # Tres métricas, un eje cada una
            fig = make_subplots(rows=1, cols=3, subplot_titles=["Log loss (menor es mejor)", "F1 macro (mayor es mejor)", "Accuracy (mayor es mejor)"],
                                horizontal_spacing=0.09)
            for i, (key, fmt) in enumerate([("log_loss", ".4f"), ("f1_macro", ".3f"), ("accuracy", ".1%")], start=1):
                vals = [mets[m][key] for m in chosen]
                fig.add_bar(x=names, y=vals, marker_color=cols, row=1, col=i, showlegend=False,
                            text=[format(v, fmt) for v in vals], textposition="outside",
                            hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>")
                lo = min(vals + ([TRIVIAL_LL] if key == "log_loss" else []))
                hi = max(vals + ([TRIVIAL_LL] if key == "log_loss" else []))
                pad = (hi - lo) * 0.5 + 0.02
                fig.update_yaxes(range=[max(0, lo - pad) if key != "log_loss" else lo - pad, hi + pad * 0.6], row=1, col=i)
            fig.add_hline(y=TRIVIAL_LL, line_dash="dash", line_color="#d03b3b", row=1, col=1,
                          annotation_text="línea base 1.0986", annotation_font_color=theme.INK_2, annotation_position="top left")
            fig.update_xaxes(tickangle=-30)
            show(style(fig, 440, legend=False).update_layout(title="Rendimiento general"))

            # Por clase
            st.subheader("¿Dónde aciertan y dónde fallan?")
            story("Los modelos distinguen razonablemente quién gana, pero los <b>empates</b> casi nunca se predicen "
                  "como clase más probable: son los votos más ambiguos. El recall por clase lo muestra.")
            metric_sel = st.radio("Métrica por clase", ["f1", "recall", "precision"], horizontal=True,
                                  format_func={"f1": "F1", "recall": "Recall", "precision": "Precisión"}.get)
            fig = go.Figure()
            for m in chosen:
                pc = mets[m]["per_class"]
                v = [pc["model_a"][metric_sel], pc["model_b"][metric_sel], pc["tie"][metric_sel]]
                fig.add_bar(name=MODEL_NAME[m], x=CLASS_NAMES, y=v, marker=dict(color=MODEL_COLOR[MODEL_NAME[m]], line=dict(width=2, color=theme.SURFACE)),
                            text=[f"{t:.2f}" for t in v], textposition="outside",
                            hovertemplate="%{x} · " + MODEL_NAME[m] + ": %{y:.3f}<extra></extra>")
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1], title=metric_sel)
            show(style(fig, 380).update_layout(title="Rendimiento por clase"))

            # Matrices de confusión
            st.subheader("Matrices de confusión")
            cc = st.columns(min(len(chosen), 2))
            for i, m in enumerate(chosen):
                df = data.load_preds(m, split)
                from sklearn.metrics import confusion_matrix
                cmx = confusion_matrix(df["winner"], df[["p_a", "p_b", "p_tie"]].to_numpy().argmax(1), labels=[0, 1, 2])
                cmn = cmx / cmx.sum(axis=1, keepdims=True)
                fig = go.Figure(go.Heatmap(z=cmn, x=CLASS_NAMES, y=CLASS_NAMES, colorscale=[[0, "#f0efec"], [1, "#0d366b"]],
                                           zmin=0, zmax=1, text=[[f"{v:.1%}<br>({c:,})" for v, c in zip(r, rc)] for r, rc in zip(cmn, cmx)],
                                           texttemplate="%{text}", showscale=False,
                                           hovertemplate="Real %{y} → predicho %{x}: %{z:.1%}<extra></extra>"))
                fig.update_yaxes(autorange="reversed", title="Real")
                fig.update_xaxes(title="Predicho")
                cc[i % 2].plotly_chart(style(fig, 360, legend=False).update_layout(title=f"{MODEL_NAME[m]} (% por fila)"),
                                       width="stretch", config={"displaylogo": False})

            # Calibración
            st.subheader("¿Se puede confiar en las probabilidades?")
            story("Si un modelo dice 40 %, debería acertar ~40 % de las veces. Cuanto más cerca de la diagonal, "
                  "mejor calibrado. Aquí se agrupan las tres probabilidades de cada batalla.")
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=theme.MUTED, dash="dash"), name="Calibración perfecta")
            for m in chosen:
                df = data.load_preds(m, split)
                p = df[["p_a", "p_b", "p_tie"]].to_numpy().ravel()
                y = np.eye(3)[df["winner"].to_numpy()].ravel()
                bins = np.clip((p * 12).astype(int), 0, 11)
                g = pd.DataFrame({"b": bins, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
                g = g[g.n >= 30]
                fig.add_scatter(x=g.p, y=g.y, mode="lines+markers", name=MODEL_NAME[m], line=dict(color=MODEL_COLOR[MODEL_NAME[m]], width=2),
                                marker=dict(size=8, line=dict(width=2, color=theme.SURFACE)),
                                hovertemplate=MODEL_NAME[m] + "<br>Predicho: %{x:.1%}<br>Real: %{y:.1%}<extra></extra>")
            fig.update_xaxes(title="Probabilidad predicha", tickformat=".0%", range=[0.1, 0.65])
            fig.update_yaxes(title="Frecuencia real", tickformat=".0%", range=[0.1, 0.65])
            show(style(fig, 420).update_layout(title="Curva de calibración"))

            # Eficiencia y sesgo
            st.subheader("Costo y sesgo de posición")
            ce = st.columns(2)
            ts = {m: data.train_seconds(m) for m in chosen}
            ts = {k: v for k, v in ts.items() if v}
            fig = go.Figure(go.Bar(y=[MODEL_NAME[m] for m in ts], x=list(ts.values()), orientation="h",
                                   marker_color=[MODEL_COLOR[MODEL_NAME[m]] for m in ts],
                                   text=[f"{v:,.0f} s" for v in ts.values()], textposition="outside",
                                   hovertemplate="%{y}: %{x:,.1f} s<extra></extra>"))
            fig.update_xaxes(type="log", title="Segundos (escala log)")
            ce[0].plotly_chart(style(fig, 330, legend=False).update_layout(title="Tiempo de entrenamiento"), width="stretch", config={"displaylogo": False})
            sw = {m: data.swap_diff(m) for m in chosen}
            sw = {k: v for k, v in sw.items() if v is not None}
            fig = go.Figure(go.Bar(y=[MODEL_NAME[m] for m in sw], x=list(sw.values()), orientation="h",
                                   marker_color=[MODEL_COLOR[MODEL_NAME[m]] for m in sw],
                                   text=[f"{v:.4f}" for v in sw.values()], textposition="outside",
                                   hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            fig.update_xaxes(title="Diferencia media al intercambiar A↔B (0 = sin sesgo)")
            ce[1].plotly_chart(style(fig, 330, legend=False).update_layout(title="Sesgo de posición"), width="stretch", config={"displaylogo": False})
            st.caption("El tiempo de XGBoost es el del ajuste final; la búsqueda de hiperparámetros se reporta aparte en su JSON. "
                       "El de Embeddings incluye la codificación del texto.")

            # Ablación
            abl = []
            for m in chosen:
                meta = data.load_meta(m)
                if m == "logreg":
                    v = meta.get("validation_metrics", {})
                    for k, lab in [("variant_a", "(a) Solo longitud/turnos"), ("variant_b", "(b) + identidad de modelos")]:
                        if k in v:
                            abl.append((MODEL_NAME[m], lab, v[k]["log_loss"]))
                else:
                    for lab, v in meta.get("variants_comparison", {}).items():
                        abl.append((MODEL_NAME[m], lab if isinstance(lab, str) else str(lab), v.get("log_loss")))
            abl = [a for a in abl if a[2] is not None]
            if abl:
                st.subheader("¿Qué aporta cada ingrediente?")
                story("Cada modelo se probó con variantes que añaden información paso a paso (log loss en validación, menor es mejor).")
                fig = go.Figure()
                for name in dict.fromkeys(a[0] for a in abl):
                    sub = [a for a in abl if a[0] == name]
                    fig.add_bar(name=name, x=[s[1] for s in sub], y=[s[2] for s in sub],
                                marker=dict(color=MODEL_COLOR[name], line=dict(width=2, color=theme.SURFACE)),
                                text=[f"{s[2]:.4f}" for s in sub], textposition="outside",
                                hovertemplate=name + " · %{x}: %{y:.4f}<extra></extra>")
                lo = min(a[2] for a in abl)
                fig.update_yaxes(range=[lo - 0.03, TRIVIAL_LL + 0.01], title="Log loss (validación)")
                fig.add_hline(y=TRIVIAL_LL, line_dash="dash", line_color="#d03b3b", annotation_text="línea base", annotation_font_color=theme.INK_2)
                fig.update_layout(barmode="group")
                show(style(fig, 400).update_layout(title="Variantes de cada modelo"))

            # Tabla completa
            with st.expander("Ver tabla completa de métricas"):
                tbl = pd.DataFrame([{"Modelo": MODEL_NAME[m], "Log loss": mets[m]["log_loss"], "F1 macro": mets[m]["f1_macro"],
                                     "Accuracy": mets[m]["accuracy"], "Mejora vs. base": mets[m]["baseline_comparison"]["log_loss_improvement"],
                                     "Recall empate": mets[m]["per_class"]["tie"]["recall"]} for m in chosen])
                st.dataframe(tbl, hide_index=True, width="stretch")

            # Detalle de un modelo
            with st.expander("Detalle de un modelo (hiperparámetros y variables importantes)"):
                one = st.selectbox("Modelo", chosen, format_func=MODEL_NAME.get)
                meta = data.load_meta(one)
                st.json(meta.get("hyperparameters") or meta.get("chosen_hyperparameters") or {}, expanded=False)
                imp = meta.get("feature_importance_top")
                if imp:
                    fi = pd.DataFrame(imp, columns=["Variable", "Importancia"]).iloc[::-1]
                    fig = go.Figure(go.Bar(x=fi["Importancia"], y=fi["Variable"], orientation="h", marker_color=MODEL_COLOR[MODEL_NAME[one]]))
                    show(style(fig, 520, legend=False).update_layout(title="Variables más importantes"))

# ════════════════════════════════════════════════════════════════════════════
# PROBAR UN CASO NUEVO
# ════════════════════════════════════════════════════════════════════════════
EXAMPLES = {
    "Respuesta detallada vs. respuesta corta": (
        "What is the capital of France and why is it historically important?",
        "The capital of France is Paris. It has been the political and cultural center of the country for centuries: "
        "it was the seat of the monarchy, the stage of the 1789 Revolution, and today hosts the government, major museums "
        "such as the Louvre, and landmarks like the Eiffel Tower.\n\n- Political center\n- Cultural capital\n- Major economic hub",
        "Paris."),
    "Rechazo vs. ayuda": (
        "How do I pick a good password manager?",
        "I'm sorry, but as an AI I cannot recommend specific products.",
        "Look for end-to-end encryption, open-source audits, cross-device sync, and a strong master-password policy. "
        "Popular choices include Bitwarden and 1Password."),
    "Dos respuestas parecidas": (
        "Give me a synonym for 'happy'.",
        "A synonym for happy is joyful.",
        "A synonym for happy is cheerful."),
}

with tab_pred:
    st.subheader("Clasifica un caso nuevo")
    story("Escribe un mensaje y las respuestas de dos chatbots. La app prepara el texto sola (longitudes, estilo, "
          "puntajes de los modelos) y cada modelo estima la probabilidad de que gane A, gane B o haya empate.")
    ex = st.selectbox("Cargar un ejemplo", ["—"] + list(EXAMPLES))
    if ex != "—":
        st.session_state["p_prompt"], st.session_state["p_a"], st.session_state["p_b"] = EXAMPLES[ex]
    prompt = st.text_area("Mensaje del usuario (prompt)", key="p_prompt", height=90)
    ca, cb = st.columns(2)
    resp_a = ca.text_area("Respuesta A", key="p_a", height=180)
    resp_b = cb.text_area("Respuesta B", key="p_b", height=180)
    known = sorted(set(clean.model_a) | set(clean.model_b)) if clean is not None else []
    with st.expander("Opcional: ¿qué modelos de IA escribieron las respuestas?"):
        st.caption("Si no lo sabes, deja «Desconocido»: los modelos que usan la identidad la ignorarán.")
        m1, m2 = st.columns(2)
        model_a = m1.selectbox("Modelo de A", [inference.UNKNOWN] + known)
        model_b = m2.selectbox("Modelo de B", [inference.UNKNOWN] + known)

    if st.button("Clasificar", type="primary"):
        if not (prompt.strip() and resp_a.strip() and resp_b.strip()):
            st.warning("Completa el mensaje y las dos respuestas.")
        else:
            st.session_state["pred"] = inference.predict_all(prompt, resp_a, resp_b, model_a, model_b)
            st.session_state["pred_txt"] = (resp_a, resp_b)

    probs = st.session_state.get("pred")
    if probs:
        names = list(probs)
        fig = go.Figure()
        for j, cls in enumerate(CLASS_NAMES):
            vals = [probs[m][j] for m in names]
            fig.add_bar(name=cls, y=[MODEL_NAME[m] for m in names], x=vals, orientation="h",
                        marker=dict(color=CLASS_COLOR[cls], line=dict(width=2, color=theme.SURFACE)),
                        text=[f"{v:.1%}" for v in vals], textposition="inside", insidetextanchor="middle",
                        textfont=dict(color="white"), hovertemplate="%{y} · " + cls + ": %{x:.1%}<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Probabilidad")
        show(style(fig, 120 + 70 * len(names)).update_layout(title="Predicción de cada modelo"))

        mean = np.mean([probs[m] for m in names], axis=0)
        k = int(mean.argmax())
        verdict = ["la respuesta A", "la respuesta B", "un empate"][k]
        icon = ["🅰️", "🅱️", "🤝"][k]
        story(f"{icon} Promediando los modelos disponibles, lo más probable es <b>{verdict}</b> "
              f"({mean[k]:.1%}). Recuerda: con tan poca señal en los datos, ninguna probabilidad es muy extrema.")
        if len(names) > 1:
            st.caption("Desacuerdo entre modelos (diferencia máxima en la probabilidad de A): "
                       f"{max(probs[m][0] for m in names) - min(probs[m][0] for m in names):.1%}")

        with st.expander("Qué «vio» el modelo"):
            ra, rb = st.session_state["pred_txt"]
            sa, sb = inference.style_summary(ra), inference.style_summary(rb)
            tbl = pd.DataFrame({"A": sa, "B": sb})
            tbl.loc["caracteres"] = [len(ra), len(rb)]
            tbl.loc["palabras"] = [len(ra.split()), len(rb.split())]
            st.dataframe(tbl.rename(index=lambda s: s.replace("n_", "").replace("_", " ")), width="stretch")

    miss = [m for m in data.MODEL_IDS if m not in inference.load_models()]
    if miss:
        st.markdown("**Modelos sin inferencia en vivo**")
        for m in miss:
            st.markdown(f"- <span class='chip'>{MODEL_NAME[m]}</span> {inference.UNAVAILABLE.get(m, 'Falta el artefacto del modelo en `models/`.')} "
                        "Sus resultados sí están en la pestaña *Resultados*.", unsafe_allow_html=True)
