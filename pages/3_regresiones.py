# pages/3_regresiones.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from utils.data_loader import load_all_data

st.title("Regresión")

# ===================== CARGA Y LIMPIEZA =====================
RAW = load_all_data()

def _to_city_map(obj):
    """Convierte la salida de load_all_data en {ciudad: DataFrame}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columna fantasma
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # normalizar price a numérico
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return df

raw_city_map = _to_city_map(RAW)
dfs_ciudades = {k: clean_city_df(v) for k, v in raw_city_map.items()}

CITY_PALETTE = {
    "Barcelona": "#FF5A5F",  # Coral
    "Budapest":  "#F2C94C",  # Amarillo
    "Hawai":     "#1E90FF",  # Turquesa
    "Hawái":     "#1E90FF",  # (por si sale con acento)
    "Boston":    "#484848",  # Gris
    "Cambridge": "#00A699",  # Aqua
}
def city_badge(nombre: str) -> str:
    color = CITY_PALETTE.get(nombre, "#999999")
    return f"""
    <span style="
        background:{color};
        color:white;
        padding:4px 10px;
        border-radius:999px;
        font-weight:600;
        font-size:0.9rem;
        display:inline-block;
        margin-bottom:0.5rem;
    ">
        {nombre}
    </span>
    """
# ===================== SIDEBAR =====================
ciudades_reg_sel = st.sidebar.multiselect(
    "Ciudades para regresión",
    options=list(dfs_ciudades.keys()),
    default=[k for k in list(dfs_ciudades.keys())[:2]],
    max_selections=4
)

tipo_reg = st.sidebar.radio(
    "Tipo de regresión",
    options=[
        "Regresión lineal simple",
        "Regresión lineal múltiple",
        "Regresión logística"
    ],
    index=0
)

if not ciudades_reg_sel:
    st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    st.stop()

# ===================== REGRESIÓN LINEAL (simple / múltiple) =====================
if tipo_reg in ("Regresión lineal simple", "Regresión lineal múltiple"):
    n = len(ciudades_reg_sel)
    cols = st.columns(min(3, n))

    for i, ciudad in enumerate(ciudades_reg_sel):
        if i > 0 and i % 3 == 0:
            cols = st.columns(min(3, n - i))

        df_ciudad = dfs_ciudades[ciudad]
        with cols[i % 3]:
            st.markdown(city_badge(ciudad), unsafe_allow_html=True)

            if "price" not in df_ciudad.columns:
                st.info("Falta columna 'price'.")
                continue

            # ---------- Lineal simple: price ~ accommodates ----------
            if tipo_reg == "Regresión lineal simple":
                if "accommodates" not in df_ciudad.columns:
                    st.info("Falta columna 'accommodates'.")
                    continue

                tmp = df_ciudad[["accommodates", "price"]].astype(float).dropna()
                if len(tmp) < 3:
                    st.info("Datos insuficientes para ajustar el modelo.")
                    continue

                x = tmp["accommodates"].to_numpy()
                y = tmp["price"].to_numpy()
                a, b = np.polyfit(x, y, 1)  # β1, β0
                y_pred = a * x + b
                r2 = r2_score(y, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{r2:.3f}")
                m2.metric("Pendiente (β1)", f"{a:.3f}")
                m3.metric("Intercepto (β0)", f"{b:.2f}")

                fig = px.scatter(
                    tmp, x="accommodates", y="price",
                    labels={"accommodates": "Accommodates", "price": "Price"},
                    title="Price ~ Accommodates (Regresión lineal)"
                )
                x_line = np.linspace(x.min(), x.max(), 50)
                fig.add_trace(go.Scatter(x=x_line, y=a * x_line + b, mode="lines", name="Ajuste"))
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # ---------- Lineal múltiple ----------
            elif tipo_reg == "Regresión lineal múltiple":
                num_cols = df_ciudad.select_dtypes(include="number").columns.tolist()
                num_cols = [c for c in num_cols if not c.lower().startswith("unnamed")]
                candidatas = [c for c in num_cols if c != "price"]

                if len(candidatas) < 2:
                    st.info("Se requieren al menos 2 variables numéricas distintas a 'price' en el dataset.")
                    continue

                default_preds = []
                if "accommodates" in candidatas:
                    default_preds.append("accommodates")
                for c in candidatas:
                    if c not in default_preds:
                        default_preds.append(c)
                    if len(default_preds) >= 2:
                        break

                predictores = st.multiselect(
                    "Variables explicativas (elige 2 o más)",
                    options=candidatas,
                    default=default_preds,
                    key=f"preds_mult_{ciudad}"
                )

                if len(predictores) < 2:
                    st.warning("Selecciona al menos 2 variables para ajustar la regresión múltiple.")
                    continue

                tmp = df_ciudad[predictores + ["price"]].dropna()
                if len(tmp) < len(predictores) + 1:
                    st.info("Datos insuficientes después de eliminar NA para ajustar el modelo.")
                    continue

                X = tmp[predictores].to_numpy(dtype=float)
                y = tmp["price"].to_numpy(dtype=float)

                modelo = LinearRegression()
                modelo.fit(X, y)

                r2 = modelo.score(X, y)
                intercepto = modelo.intercept_
                coefs = modelo.coef_

                m1, m2, m3 = st.columns(3)
                m1.metric("R² (múltiple)", f"{r2:.3f}")
                m2.metric("N° predictores", f"{len(predictores)}")
                m3.metric("Intercepto (β0)", f"{intercepto:.2f}")

                coef_df = pd.DataFrame({
                    "Variable": predictores,
                    "Coeficiente (β)": coefs
                })
                st.dataframe(coef_df, use_container_width=True)

                # Efecto parcial de una variable manteniendo las demás en su media
                var_plot = "accommodates" if "accommodates" in predictores else predictores[0]

                fig = px.scatter(
                    tmp[[var_plot, "price"]], x=var_plot, y="price",
                    labels={var_plot: var_plot, "price": "Price"},
                    title=f"Price ~ {var_plot} (Regresión múltiple, otros = media)"
                )

                x_line = np.linspace(tmp[var_plot].min(), tmp[var_plot].max(), 50)
                medias = tmp[predictores].mean()
                X_line = np.tile(medias.to_numpy(), (50, 1))
                idx_plot = predictores.index(var_plot)
                X_line[:, idx_plot] = x_line
                y_line = modelo.predict(X_line)

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Predicción (otros = media)"
                ))
                st.plotly_chart(fig, use_container_width=True)


# ===================== REGRESIÓN LOGÍSTICA =====================
if tipo_reg == "Regresión logística":
    st.markdown("## Regresión logística")

    ciudad_ref = ciudades_reg_sel[0]
    df_ref = dfs_ciudades[ciudad_ref]

    if "price" not in df_ref.columns:
        st.warning("No se puede hacer regresión logística: falta columna 'price'.")
    else:

        posibles_vars = [
            "accommodates", "bedrooms", "bathrooms",
            "minimum_nights", "availability_365",
            "number_of_reviews", "estimated_occupancy_l365d",
            "estimated_revenue_l365d", "review_scores_rating",
            "reviews_per_month", "host_is_superhost"
        ]

        vars_disponibles = [v for v in posibles_vars if v in df_ref.columns]

        predictores_log = st.sidebar.multiselect(
            "Variables explicativas (logística)",
            options=vars_disponibles,
            default=vars_disponibles[:3],
            key="logit_vars_airbnb",
        )

        if len(predictores_log) == 0:
            st.info("Selecciona al menos una variable explicativa.")
        else:

            n = len(ciudades_reg_sel)
            cols_log = st.columns(min(4, n))

            for i, ciudad in enumerate(ciudades_reg_sel):

                if i > 0 and i % 4 == 0:
                    cols_log = st.columns(min(4, n - i))

                with cols_log[i % 4]:
                    st.markdown(city_badge(ciudad), unsafe_allow_html=True)

                    df_city = dfs_ciudades[ciudad].copy()

                    if "price" not in df_city.columns:
                        st.info("Falta price.")
                        continue

                    # ---- host_is_superhost → binario ----
                    if "host_is_superhost" in df_city.columns:
                        mapa = {"t":1,"f":0,"T":1,"F":0,"True":1,"False":0,True:1,False:0}
                        df_city["host_is_superhost_bin"] = (
                            df_city["host_is_superhost"].map(mapa).fillna(0).astype(int)
                        )
                        predictores_uso = [
                            "host_is_superhost_bin" if p == "host_is_superhost" else p
                            for p in predictores_log
                        ]
                    else:
                        predictores_uso = predictores_log

                    # ---- Target ----
                    umbral_local = df_city["price"].median()
                    df_city["high_price"] = (df_city["price"] > umbral_local).astype(int)

                    cols_needed = predictores_uso + ["high_price"]
                    df_city = df_city[cols_needed].dropna()

                    if df_city.empty:
                        st.info("Datos insuficientes.")
                        continue

                    X = df_city[predictores_uso].astype(float).values
                    y = df_city["high_price"].values

                    if y.sum() == 0 or y.sum() == len(y):
                        st.info("Solo una clase presente.")
                        continue

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    modelo = LogisticRegression(max_iter=1200)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    # ---------------- Métricas ----------------
                    cm = confusion_matrix(y_test, y_pred)

                    acc = accuracy_score(y_test, y_pred)

                    prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
                    rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                    f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

                    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
                    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
                    f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

                    st.markdown("### Métricas")

                    st.metric("Accuracy", f"{acc:.3f}")

                    st.markdown("---") 

                    # FILA 1: clase 1
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision (1)", f"{prec_1:.3f}")
                    c2.metric("Recall (1)", f"{rec_1:.3f}")
                    c3.metric("F1 Score (1)", f"{f1_1:.3f}")

                    # FILA 2: clase 0 (debajo)
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Precision (0)", f"{prec_0:.3f}")
                    cc2.metric("Recall (0)", f"{rec_0:.3f}")
                    cc3.metric("F1 Score (0)", f"{f1_0:.3f}")

                    # ---------------- Matriz de confusión ----------------
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()

                    # [[TP, FP],
                    #  [FN, TN]]
                    cm_display = np.array([[tp, fp],
                                           [fn, tn]])

                    cm_df = pd.DataFrame(
                        cm_display,
                        index=["Pred 1 (caro)", "Pred 0 (barato)"],   # filas = predicción
                        columns=["Real 1 (caro)", "Real 0 (barato)"]  # columnas = realidad
                    )

                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=["#DEDEDE", "#FFB3B3", "#FF5A5F", "#00999F"],
                    )

                    fig_cm.update_layout(
                        title="Matriz de confusión",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )

                    st.plotly_chart(fig_cm, use_container_width=True)

