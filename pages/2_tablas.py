# pages/2_tablas_comparativas.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_all_data

# ===================== Título =====================

# ===================== Carga y limpieza =====================
RAW = load_all_data()

def _to_city_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # elimina columna fantasma
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # normaliza price
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return df

raw_city_map = _to_city_map(RAW)
dfs_ciudades = {k: clean_city(v) for k, v in raw_city_map.items()}

CITY_PALETTE = {
    "Barcelona": "#FF5A5F",  # Coral
    "Budapest":  "#F2C94C",  # Amarillo
    "Hawái":     "#1E90FF",  # Turquesa
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

def city_badge_light(nombre: str) -> str:
    color = CITY_PALETTE.get(nombre, "#999999")
    return f"""
    <span style="
        background:{color}22;     /* color muy clarito */
        color:{color};
        padding:4px 10px;
        border-radius:6px;
        font-weight:600;
        font-size:0.9rem;
        display:inline-block;
    ">
        {nombre}
    </span>
    """


# ===================== Variables principales =====================
PRIMARY_VARS = [
    "price", "accommodates", "bedrooms", "beds",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "availability_30", "availability_365",
    "estimated_occupancy_l365d", "minimum_nights", "maximum_nights",
]

def available_primary_vars(df: pd.DataFrame):
    cols = [c for c in df.select_dtypes(include="number").columns if not c.lower().startswith("unnamed")]
    ban = {"id", "latitude", "longitude"}
    return [c for c in PRIMARY_VARS if c in cols and c not in ban]

# ===================== Sidebar =====================
ciudades_sel = st.sidebar.multiselect(
    "Selecciona las ciudades a comparar",
    options=list(dfs_ciudades.keys()),
    default=[k for k in list(dfs_ciudades.keys())[:2]],
    max_selections=4,
    key="cmp_ciudades",
)
if not ciudades_sel:
    st.warning("Selecciona al menos una ciudad para comparar.")
    st.stop()

# Intersección de variables principales disponibles
common_primary = set(available_primary_vars(dfs_ciudades[ciudades_sel[0]]))
for c in ciudades_sel[1:]:
    common_primary &= set(available_primary_vars(dfs_ciudades[c]))
pool_cols = sorted(common_primary) or available_primary_vars(dfs_ciudades[ciudades_sel[0]])

if not pool_cols:
    st.warning("No se encontraron variables principales numéricas comunes.")
    st.stop()

cols_sel = st.sidebar.multiselect(
    "Columnas a evaluar (variables principales)",
    options=pool_cols,
    default=pool_cols[:6],
    key="cmp_cols",
)
if not cols_sel:
    st.warning("Selecciona al menos una columna.")
    st.stop()

# ===================== Métricas (sin %Missing ni orden) =====================
METRIC_MAP = {
    "count":  ("Conteo",  lambda s: s.count()),
    "mean":   ("Media",   lambda s: s.mean()),
    "median": ("Mediana", lambda s: s.median()),
    "std":    ("DesvEst", lambda s: s.std(ddof=1)),
    "min":    ("Mín",     lambda s: s.min()),
    "p25":    ("P25",     lambda s: s.quantile(0.25)),
    "p75":    ("P75",     lambda s: s.quantile(0.75)),
    "max":    ("Máx",     lambda s: s.max()),
}

metrics_sel = st.sidebar.multiselect(
    "Métricas",
    options=list(METRIC_MAP.keys()),
    default=["count", "median", "mean", "std"],
    key="cmp_metrics",
)
if not metrics_sel:
    st.warning("Selecciona al menos una métrica.")
    st.stop()

# ===================== Tablas por ciudad =====================
n = len(ciudades_sel)
cols_grid = st.columns(min(4, n))

for i, ciudad in enumerate(ciudades_sel):
    if i > 0 and i % 4 == 0:
        cols_grid = st.columns(min(4, n - i))
    df_city = dfs_ciudades[ciudad]

    with cols_grid[i % 4]:
        st.markdown(city_badge(ciudad), unsafe_allow_html=True)

        filas = []
        for c in cols_sel:
            s = df_city[c]
            fila = {}
            for k in metrics_sel:
                name, fn = METRIC_MAP[k]
                try:
                    fila[name] = fn(s)
                except Exception:
                    fila[name] = np.nan
            filas.append(pd.Series(fila, name=c))
        tabla = pd.DataFrame(filas)
        st.dataframe(tabla, use_container_width=True)

# ===================== Matriz comparativa =====================
st.markdown("---")
st.markdown("### Matriz comparativa por métrica")

metric_for_matrix = st.selectbox(
    "Selecciona la métrica para la matriz comparativa",
    options=metrics_sel,
    index=(metrics_sel.index("median") if "median" in metrics_sel else 0),
    key="matrix_metric",
)
metric_for_matrix_name = METRIC_MAP[metric_for_matrix][0]

rows = []
for c in cols_sel:
    row = {"variable": c}
    for ciudad in ciudades_sel:
        s = dfs_ciudades[ciudad][c]
        try:
            row[ciudad] = METRIC_MAP[metric_for_matrix][1](s)
        except Exception:
            row[ciudad] = np.nan
    rows.append(row)

matriz = pd.DataFrame(rows).set_index("variable")
if ciudades_sel:
    matriz = matriz.sort_values(by=ciudades_sel[0], ascending=False)

# ---  Colorear columnas por ciudad (suave) ---
def highlight_city(col):
    if col.name in ciudades_sel:
        base = CITY_PALETTE.get(col.name, "#e0e0e0")  # usa tu paleta
        # color MUY clarito para no ensuciar la tabla
        return [f"background-color: {base}22" for _ in col]
    return [""] * len(col)

styler = matriz.style.apply(highlight_city, axis=0)

st.dataframe(styler, use_container_width=True)

# ===================== Descarga =====================
outs = []
for ciudad in ciudades_sel:
    df_city = dfs_ciudades[ciudad]
    filas = []
    for c in cols_sel:
        s = df_city[c]
        fila = {"ciudad": ciudad, "variable": c}
        for k in metrics_sel:
            name, fn = METRIC_MAP[k]
            try:
                fila[name] = fn(s)
            except Exception:
                fila[name] = np.nan
        filas.append(fila)
    outs.append(pd.DataFrame(filas))
export_df = pd.concat(outs, ignore_index=True)

st.download_button(
    "📥 Descargar resumen por ciudad (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="tablas_comparativas_resumen.csv",
    mime="text/csv",
)

# ===================== Hallazgos (sobrio y visual) =====================
def quick_findings(ciudades):
    lines = []
    for c in ciudades:
        df = dfs_ciudades[c]
        med_price = df["price"].median() if "price" in df.columns else np.nan
        rating = df["review_scores_rating"].mean() if "review_scores_rating" in df.columns else np.nan
        occ = df["estimated_occupancy_l365d"].mean() if "estimated_occupancy_l365d" in df.columns else np.nan
        reviews = df["number_of_reviews"].median() if "number_of_reviews" in df.columns else np.nan
        badge = city_badge(c)
        txt = f"{badge} — Mediana precio: {med_price:,.0f} USD"

        if not np.isnan(rating):  txt += f", Rating promedio: {rating:.1f}"
        if not np.isnan(reviews): txt += f", Mediana # reviews: {reviews:,.0f}"
        if not np.isnan(occ):     txt += f", Ocupación anual estimada: {occ:.1f}%"
        lines.append(txt)
    return lines

hallazgos = quick_findings(ciudades_sel)
if hallazgos:
    st.markdown("### 💡 Hallazgos principales")
    st.markdown(
        """
        <style>
        .hallazgo-box {
            background: #f6f7f9;
            border-left: 6px solid #e15b5b;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for h in hallazgos:
        st.markdown(f"<div class='hallazgo-box'>{h}</div>", unsafe_allow_html=True)
