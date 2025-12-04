import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import load_all_data

# =====================
# CARGA DE DATOS
# =====================
RAW = load_all_data()

def _to_city_map(obj):
    """Convierte load_all_data en {ciudad: DataFrame}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawai", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city(df: pd.DataFrame, is_budapest: bool = False) -> pd.DataFrame:
    """Limpia columnas fantasma y convierte price a float.
       Si is_budapest=True, convierte de HUF a USD."""
    df = df.copy()

    # quitar columnas tipo "Unnamed: 0"
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df.drop(columns=[c], inplace=True)

    if "price" in df.columns:
        s = df["price"].astype(str)

        # quitar símbolos de moneda, comas, etc. pero dejar dígitos, punto y guión
        s = s.str.replace(r"[^\d\.\-]", "", regex=True)

        # valores raros con más de un '-' (ej. "9.5-12") los mandamos a NaN
        mask_bad = s.str.count("-") > 1
        s[mask_bad] = np.nan

        price = pd.to_numeric(s, errors="coerce")

        # conversión HUF → USD solo para Budapest
        if is_budapest:
            HUF_TO_USD = 1 / 370  # ≈ 0.00270 USD por HUF
            price = price * HUF_TO_USD

        df["price"] = price

    return df

raw_city_map = _to_city_map(RAW)

dfs_ciudades = {
    name: clean_city(df, is_budapest=(name == "Budapest"))
    for name, df in raw_city_map.items()
}
CITY_PALETTE = {
    "Barcelona": "#FF5A5F",  # Coral
    "Budapest":  "#F2C94C",  # Amarillo
    "Hawai":     "#1E90FF",  # Turquesa
    "Boston":    "#484848",  # Gris
    "Cambridge": "#00A699",  # Aqua
}


# =====================
# FILTROS LATERALES
# =====================
st.sidebar.header("Filtros")

ciudades = st.sidebar.multiselect(
    "Ciudades (máx 5)",
    options=list(dfs_ciudades.keys()),
    default=list(dfs_ciudades.keys())
)

if len(ciudades) == 0:
    st.warning("Selecciona al menos 1 ciudad.")
    st.stop()

df_list = []
for c in ciudades:
    tmp = dfs_ciudades[c].copy()
    tmp["city"] = c
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

if "price" not in df.columns:
    st.error("No existe la columna 'price' en tus datos.")
    st.stop()

df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# ---------- Filtro room_type: MULTISELECT ----------
if "room_type" in df.columns:
    room_unique = sorted(df["room_type"].dropna().unique().tolist())
    room_opts = ["Todos"] + room_unique

    room_sel = st.sidebar.multiselect(
        "Room type",
        options=room_opts,
        default=["Todos"]
    )

    # Quitamos "Todos" para filtrar solo si hay tipos específicos
    seleccion_room = [r for r in room_sel if r != "Todos"]

    if seleccion_room:
        df = df[df["room_type"].isin(seleccion_room)]
else:
    room_sel = []

# ---------- Filtro property_type ----------
if "property_type" in df.columns:
    top_props = df["property_type"].value_counts().head(25).index.tolist()
    prop_opts = ["Todos"] + top_props + ["Otros (fuera del Top 25)"]
    prop = st.sidebar.selectbox("Property type", prop_opts, index=0)
    if prop == "Otros (fuera del Top 25)":
        df = df[~df["property_type"].isin(top_props)]
    elif prop != "Todos":
        df = df[df["property_type"] == prop]
else:
    prop = "N/A"

# ---------- Slider de precio ----------
p01 = float(df["price"].quantile(0.01))
p99 = float(df["price"].quantile(0.99))
pmin = int(max(0, np.floor(p01)))
pmax = int(np.ceil(p99))
rango = st.sidebar.slider("Rango de precio (USD)", pmin, pmax, (pmin, pmax))
df = df[df["price"].between(rango[0], rango[1])]

if df.empty:
    st.warning("Con esos filtros no quedaron datos.")
    st.stop()

# =====================
# SELECTORES DE VISTA
# =====================
views = [
    "1) Distribución de precio por ciudad (Boxplot)",
    "2) Histograma de precio por ciudad (Overlay)",
    "3) Precio por ciudad y room_type (Barras)",
    "4) Precio vs accommodates por ciudad (Línea de medianas/promedios)",
    "5) Barrio más caro por ciudad (Top neighbourhood)",
    "6) Métrica por ciudad (availability/occupancy/revenue/rating)",
]

col1, col2 = st.columns([1.3, 1])
with col1:
    view = st.selectbox("Vista comparativa", views, index=0)
with col2:
    agg = st.selectbox("Agregación", ["Mediana", "Promedio"], index=0)

agg_fn = np.nanmedian if agg == "Mediana" else np.nanmean
H = 400  # altura más pequeña para evitar scroll

# =====================
# GRÁFICAS EN CONTENEDORES
# =====================

if view.startswith("1)"):
    fig = px.box(
        df, x="city", y="price", color="city",
        points=False,
        labels={"city": "Ciudad", "price": "Precio (USD)"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
        )
    fig.update_layout(
        height=H,
        legend_title_text="Ciudad",
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.subheader("Distribución de precio por ciudad")
        st.plotly_chart(fig, use_container_width=True)

elif view.startswith("2)"):
    fig = px.histogram(
        df, x="price", color="city",
        nbins=60, barmode="overlay", opacity=0.35,
        labels={"price": "Precio (USD)", "city": "Ciudad"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
    )
    fig.update_layout(
        height=H,
        legend_title_text="Ciudad",
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.subheader("Histograma de precio por ciudad")
        st.plotly_chart(fig, use_container_width=True)

elif view.startswith("3)"):
    if "room_type" not in df.columns:
        st.error("No existe la columna 'room_type' en tus datos.")
        st.stop()

    g = df.groupby(["city", "room_type"], as_index=False).agg(price=("price", agg_fn))
    fig = px.bar(
        g, x="city", y="price", color="room_type",
        barmode="group",
        labels={"city": "Ciudad", "price": "Precio (USD)", "room_type": "Room type"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
    )
    fig.update_layout(
        height=H,
        legend_title_text="Room type",
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.subheader(f"{agg} de precio por ciudad y room_type")
        st.plotly_chart(fig, use_container_width=True)

elif view.startswith("4)"):
    if "accommodates" not in df.columns:
        st.error("No existe la columna 'accommodates'.")
        st.stop()

    tmp = df.dropna(subset=["accommodates"]).copy()
    tmp["accommodates_bin"] = tmp["accommodates"].clip(lower=1, upper=12).astype(int)
    g = tmp.groupby(["city", "accommodates_bin"], as_index=False).agg(price=("price", agg_fn))

    fig = px.line(
        g, x="accommodates_bin", y="price", color="city",
        markers=True,
        labels={"accommodates_bin": "Accommodates", "price": "Precio (USD)", "city": "Ciudad"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
    )
    fig.update_layout(
        height=H,
        legend_title_text="Ciudad",
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.subheader(f"{agg} de precio vs accommodates por ciudad")
        st.plotly_chart(fig, use_container_width=True)

elif view.startswith("5)"):
    if "neighbourhood_cleansed" not in df.columns:
        st.error("No existe la columna 'neighbourhood_cleansed'.")
        st.stop()

    g = (
        df.dropna(subset=["neighbourhood_cleansed"])
          .groupby(["city", "neighbourhood_cleansed"], as_index=False)
          .agg(price=("price", agg_fn))
    )

    idx = g.groupby("city")["price"].idxmax()
    top = g.loc[idx].sort_values("price", ascending=False)

    fig = px.bar(
        top, x="city", y="price", color="city",
        hover_data=["neighbourhood_cleansed"],
        labels={"city": "Ciudad", "price": "Precio (USD)"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
    )
    fig.update_layout(
        height=H,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.subheader(f"Barrio más caro por ciudad ({agg})")
        st.plotly_chart(fig, use_container_width=True)

elif view.startswith("6)"):
    candidates = []
    for c in [
        "availability_30",
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "reviews_per_month",
        "review_scores_rating"
    ]:
        if c in df.columns:
            candidates.append(c)

    if len(candidates) == 0:
        st.error("No encontré columnas de métricas (availability/occupancy/revenue/rating) en tus datos.")
        st.stop()

    metric = st.selectbox("Métrica", candidates, index=0)

    g = (
        df.dropna(subset=[metric])
          .groupby("city", as_index=False)
          .agg(value=(metric, agg_fn))
          .sort_values("value", ascending=False)
    )

    fig = px.bar(
        g, x="city", y="value", color="city",
        labels={"city": "Ciudad", "value": f"{agg} de {metric}"},
        color_discrete_map=CITY_PALETTE,
        category_orders={"city": list(CITY_PALETTE.keys())}
    )
    fig.update_layout(
        height=H,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10)
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)










