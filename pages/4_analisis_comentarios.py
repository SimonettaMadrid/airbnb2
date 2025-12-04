# pages/4_analisis_comentarios.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from collections import Counter

from utils.data_loader import load_all_data


# =====================
# CARGA Y LIMPIEZA
# =====================
RAW = load_all_data()

def _to_city_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawai", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city_df(df: pd.DataFrame, is_budapest: bool = False) -> pd.DataFrame:
    """
    Limpia columnas Unnamed y convierte 'price' a float.
    Si is_budapest=True, convierte de HUF a USD.
    """
    df = df.copy()
    # Quitar columnas Unnamed
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df.drop(columns=[col], inplace=True)

    # price -> numérico
    if "price" in df.columns:
        s = df["price"].astype(str)

        # quitar símbolos de moneda / comas, dejando solo dígitos, punto y guión
        s = s.str.replace(r"[^\d\.\-]", "", regex=True)

        # valores raros con más de un '-' (ej. "9.5-12") -> NaN
        mask_bad = s.str.count("-") > 1
        s[mask_bad] = np.nan

        price = pd.to_numeric(s, errors="coerce")

        # conversión HUF -> USD solo para Budapest
        if is_budapest:
            HUF_TO_USD = 1 / 370  # ~0.0027 USD por HUF
            price = price * HUF_TO_USD

        df["price"] = price

    return df

city_map_raw = _to_city_map(RAW)

# aplicar limpieza por ciudad, con conversión para Budapest
dfs_ciudades = {
    k: clean_city_df(v, is_budapest=(k == "Budapest"))
    for k, v in city_map_raw.items()
}

CITY_PALETTE = {
    "Barcelona": "#FF5A5F",  # Coral
    "Budapest":  "#F2C94C",  # Amarillo
    "Hawai":     "#1E90FF",  # Turquesa
    "Hawái":     "#1E90FF",  # (por si sale con acento)
    "Boston":    "#484848",  # Gris
    "Cambridge": "#00A699",  # Aqua
}
ROOM_TYPE_LABELS_ES = {
    "Entire home/apt": "Alojamiento entero",
    "Private room":    "Habitación privada",
    "Shared room":     "Habitación compartida",
    "Hotel room":      "Habitación de hotel",}
# =====================
# FUNCIONES AUXILIARES
# =====================
def parse_amenities_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell)
    # Intentar lista tipo Python
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    # Fallback: separar por comas
    s = s.strip("{}[]()")
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p]

def ensure_amenities_list(df_city):
    if "amenities" in df_city.columns and "amenities_list" not in df_city.columns:
        df_city["amenities_list"] = df_city["amenities"].apply(parse_amenities_cell)
    return df_city

# =====================
# SIDEBAR
# =====================
st.sidebar.header("Filtros")

# --- Ciudades ---
ciudades_sel = st.sidebar.multiselect(
    "Ciudades",
    options=list(dfs_ciudades.keys()),
    default=list(dfs_ciudades.keys())
)

if not ciudades_sel:
    st.warning("Selecciona al menos una ciudad.")
    st.stop()

# --- Top 10 barrios por ciudad (para filtro) ---
neigh_options = []
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "neighbourhood_cleansed" in dfc.columns:
        top10 = dfc["neighbourhood_cleansed"].value_counts().head(10).index
        neigh_options.extend([f"{ciudad} | {barrio}" for barrio in top10])

barrios_sel = st.sidebar.multiselect(
    "Barrios (Top 10 por ciudad)",
    options=neigh_options,
    default=[]
)

# convertimos selección a dict {ciudad: set(barrios)}
barrios_dict: dict[str, set] = {}
for item in barrios_sel:
    if " | " in item:
        ciudad, barrio = item.split(" | ", 1)
        barrios_dict.setdefault(ciudad, set()).add(barrio)

# --- Universo de amenidades (para filtro) ---
amenity_counter = Counter()
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "amenities" in dfc.columns:
        dfc = ensure_amenities_list(dfc)
        for lst in dfc["amenities_list"]:
            if isinstance(lst, list):
                amenity_counter.update(lst)

amenity_options = [a for a, _ in amenity_counter.most_common(30)]
amenidades_sel = st.sidebar.multiselect("Amenidades (filtro)", amenity_options)

# --- Universo de room types ---
room_types = set()
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "room_type" in dfc.columns:
        room_types.update(dfc["room_type"].dropna().unique().tolist())

room_types = sorted(room_types)

# Opciones en español para mostrar en el sidebar
rt_to_es = ROOM_TYPE_LABELS_ES
es_to_rt = {v: k for k, v in rt_to_es.items()}

options_es = [rt_to_es.get(rt, rt) for rt in room_types]

room_types_sel_es = st.sidebar.multiselect(
    "Tipo de alojamiento",
    options=options_es,
    default=[],
)

# Convertimos lo que el usuario eligió (en español) de regreso al valor original del dataset
room_types_sel = [es_to_rt.get(opt, opt) for opt in room_types_sel_es]


# --- Rango rating global ---
ratings_min, ratings_max = [], []
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "review_scores_rating" in dfc.columns and dfc["review_scores_rating"].notna().any():
        ratings_min.append(dfc["review_scores_rating"].min())
        ratings_max.append(dfc["review_scores_rating"].max())

if ratings_min and ratings_max:
    r1, r2 = st.sidebar.slider(
        "Rango rating",
        float(np.floor(min(ratings_min))),
        float(np.ceil(max(ratings_max))),
        (float(np.floor(min(ratings_min))), float(np.ceil(max(ratings_max))))
    )
else:
    r1, r2 = None, None

# --- Rango de accommodates (número de huéspedes) ---
acc_mins, acc_maxs = [], []
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "accommodates" in dfc.columns and dfc["accommodates"].notna().any():
        acc_mins.append(dfc["accommodates"].min())
        acc_maxs.append(dfc["accommodates"].max())

if acc_mins and acc_maxs:
    acc_min_global = int(np.floor(min(acc_mins)))
    acc_max_global = int(np.ceil(max(acc_maxs)))
    acc1, acc2 = st.sidebar.slider(
        "Número de huéspedes (accommodates)",
        acc_min_global,
        acc_max_global,
        (acc_min_global, acc_max_global)
    )
else:
    acc1, acc2 = None, None

# --- Rango de precio global (ya con Budapest en USD) ---
price_mins, price_maxs = [], []
for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad]
    if "price" in dfc.columns and dfc["price"].notna().any():
        price_mins.append(dfc["price"].quantile(0.01))
        price_maxs.append(dfc["price"].quantile(0.99))

if price_mins and price_maxs:
    pmin_global = int(max(0, np.floor(min(price_mins))))
    pmax_global = int(np.ceil(max(price_maxs)))
    price1, price2 = st.sidebar.slider(
        "Rango de precio (USD)",
        pmin_global,
        pmax_global,
        (pmin_global, pmax_global)
    )
else:
    price1, price2 = None, None

# =====================
# APLICAR FILTROS POR CIUDAD
# =====================
filtered_city_dfs = {}

for ciudad in ciudades_sel:
    dfc = dfs_ciudades[ciudad].copy()

    # rating
    if r1 is not None and "review_scores_rating" in dfc.columns:
        dfc = dfc[
            (dfc["review_scores_rating"] >= r1)
            & (dfc["review_scores_rating"] <= r2)
        ]

    # accommodates
    if acc1 is not None and "accommodates" in dfc.columns:
        dfc = dfc[
            (dfc["accommodates"] >= acc1) &
            (dfc["accommodates"] <= acc2)
        ]

    # precio
    if price1 is not None and "price" in dfc.columns:
        dfc = dfc[
            (dfc["price"] >= price1) &
            (dfc["price"] <= price2)
        ]

    # room type
    if room_types_sel and "room_type" in dfc.columns:
        dfc = dfc[dfc["room_type"].isin(room_types_sel)]

    # barrios seleccionados (solo si hay para esa ciudad)
    if barrios_dict.get(ciudad) and "neighbourhood_cleansed" in dfc.columns:
        dfc = dfc[dfc["neighbourhood_cleansed"].isin(barrios_dict[ciudad])]

    # amenidades
    if amenidades_sel and "amenities" in dfc.columns:
        dfc = ensure_amenities_list(dfc)
        dfc = dfc[dfc["amenities_list"].apply(
            lambda lst: all(a in lst for a in amenidades_sel)
        )]

    # para colorear por ciudad
    dfc["city"] = ciudad

    filtered_city_dfs[ciudad] = dfc

if all(len(df_city) == 0 for df_city in filtered_city_dfs.values()):
    st.warning("No hay datos después de aplicar filtros.")
    st.stop()

# DF combinado (para gráficas globales)
combined_df = pd.concat(filtered_city_dfs.values(), ignore_index=True)

# =====================
# TARJETAS DE ROOM TYPE (CONTEO DE ALOJAMIENTOS)
# =====================
 
if "room_type" in combined_df.columns and not combined_df.empty:
    room_counts = combined_df["room_type"].value_counts()
    cols_kpi = st.columns(len(room_counts))

    for col, (rt, cnt) in zip(cols_kpi, room_counts.items()):
        with col:
            label_es = ROOM_TYPE_LABELS_ES.get(rt, rt)
            st.metric(label=label_es, value=int(cnt))
else:
    st.info("No hay información de tipo de alojamiento para los filtros actuales.")


st.markdown("---")

# =====================
# GRID DE 4 CONTENEDORES (2x2)
# =====================
row1_col1, row1_col2 = st.columns(2, gap="small")
row2_col1, row2_col2 = st.columns(2, gap="small")

H = 260

# ---------- CONTENEDOR 1: TOP AMENIDADES ----------
with row1_col1:
    with st.container(border=True):
        st.subheader("Top amenidades")

        amenity_rows = []

        for ciudad in ciudades_sel:
            dfc = filtered_city_dfs[ciudad]
            if dfc.empty or "amenities" not in dfc.columns:
                continue

            dfc = ensure_amenities_list(dfc)
            cnt = Counter()
            for lst in dfc["amenities_list"]:
                if isinstance(lst, list):
                    cnt.update(lst)

            for amenity, count in cnt.most_common(5):
                amenity_rows.append({"city": ciudad, "amenity": amenity, "count": count})

        if not amenity_rows:
            st.info("No hay amenidades para mostrar.")
        else:
            amen_df = pd.DataFrame(amenity_rows)
            fig_amen = px.bar(
                amen_df,
                x="amenity",
                y="count",
                color="city",
                barmode="group",
                labels={"amenity": "Amenidad", "count": "Frecuencia", "city": "Ciudad"},
                color_discrete_map=CITY_PALETTE
            )
            fig_amen.update_layout(
                xaxis_tickangle=-45,
                margin=dict(t=25, b=50, l=10, r=10),
                height=H
            )
            st.plotly_chart(fig_amen, use_container_width=True)

# ---------- CONTENEDOR 2: TOP BARRIOS ----------
with row1_col2:
    with st.container(border=True):
        st.subheader("Top barrios")

        rows_barrios = []

        for ciudad in ciudades_sel:
            dfc = filtered_city_dfs[ciudad]
            if dfc.empty or "neighbourhood_cleansed" not in dfc.columns:
                continue

            top_b = dfc["neighbourhood_cleansed"].value_counts().head(5)
            for barrio, n in top_b.items():
                rows_barrios.append({"city": ciudad, "barrio": barrio, "count": int(n)})

        if rows_barrios:
            barrios_df = pd.DataFrame(rows_barrios)
            fig_barrios = px.bar(
                barrios_df,
                x="barrio",
                y="count",
                color="city",
                barmode="group",
                labels={"barrio": "Barrio", "count": "N° alojamientos", "city": "Ciudad"},
                color_discrete_map=CITY_PALETTE, 
            )
            fig_barrios.update_layout(
                xaxis_tickangle=-45,
                margin=dict(t=25, b=50, l=10, r=10),
                height=H
            )
            st.plotly_chart(fig_barrios, use_container_width=True)
        else:
            st.info("No encontré información de barrios para las ciudades seleccionadas.")

# ---------- CONTENEDOR 3: REVIEWS vs PRECIO ----------
with row2_col1:
    with st.container(border=True):
        st.subheader("Reseñas vs precio")

        if "number_of_reviews" not in combined_df.columns or "price" not in combined_df.columns:
            st.info("Necesito 'number_of_reviews' y 'price' para esta gráfica.")
        else:
            df_scatter = combined_df.dropna(subset=["number_of_reviews", "price"]).copy()
            if df_scatter.empty:
                st.info("No hay datos suficientes para esta gráfica.")
            else:
                # recortar outliers
                p99_price = df_scatter["price"].quantile(0.99)
                p99_reviews = df_scatter["number_of_reviews"].quantile(0.99)
                df_scatter = df_scatter[
                    (df_scatter["price"] <= p99_price) &
                    (df_scatter["number_of_reviews"] <= p99_reviews)
                ]

                fig_reviews = px.scatter(
                    df_scatter,
                    x="number_of_reviews",
                    y="price",
                    color="city",
                    labels={
                        "number_of_reviews": "Número de reseñas",
                        "price": "Precio (USD)",
                        "city": "Ciudad"
                    },color_discrete_map=CITY_PALETTE
                )
                fig_reviews.update_layout(
                    margin=dict(t=25, b=40, l=10, r=10),
                    height=H
                )
                st.plotly_chart(fig_reviews, use_container_width=True)

# ---------- CONTENEDOR 4: MAPA DETALLADO DE UBICACIONES ----------
with row2_col2:
    with st.container(border=True):
        st.subheader("Mapa de ubicaciones de Airbnb")

        if "latitude" not in combined_df.columns or "longitude" not in combined_df.columns:
            st.info("No encontré las columnas 'latitude' y 'longitude'.")
        else:
            df_map = combined_df.dropna(subset=["latitude", "longitude"]).copy()

            if df_map.empty:
                st.info("No hay datos de ubicación después de los filtros.")
            else:
                # limitar puntos si hay demasiados
                if len(df_map) > 5000:
                    df_map = df_map.sample(5000, random_state=42)

                center_lat = df_map["latitude"].mean()
                center_lon = df_map["longitude"].mean()

                fig_map = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    color="city",
                    hover_name="name",
                    hover_data={
                        "price": True,
                        "review_scores_rating": True,
                        "city": False,
                        "latitude": False,
                        "longitude": False,
                    },
                    zoom=10,
                    labels={
                        "price": "Precio (USD)",
                        "review_scores_rating": "Rating",
                        "city": "Ciudad"
                    },color_discrete_map=CITY_PALETTE,
                )
                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_center={"lat": center_lat, "lon": center_lon},
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=H,
                    legend_title_text="Ciudad",
                )
                st.plotly_chart(fig_map, use_container_width=True)

# =====================
# CONTENEDOR EXTRA ABAJO: COMENTARIOS POR BARRIO
# =====================
with st.container(border=True):
    st.subheader("Comentarios por barrio (detalle)")

    for ciudad in ciudades_sel:
        dfc = filtered_city_dfs[ciudad]
        if dfc.empty or "neighbourhood_cleansed" not in dfc.columns or "neighborhood_overview" not in dfc.columns:
            continue

        st.markdown(f"### {ciudad}")

        top_barrios = dfc["neighbourhood_cleansed"].value_counts().head(5).index

        for barrio in top_barrios:
            df_b = dfc[dfc["neighbourhood_cleansed"] == barrio].dropna(subset=["neighborhood_overview"])
            if df_b.empty:
                continue

            sort_cols = []
            if "review_scores_rating" in df_b.columns:
                sort_cols.append("review_scores_rating")
            if "number_of_reviews" in df_b.columns:
                sort_cols.append("number_of_reviews")
            if sort_cols:
                df_b = df_b.sort_values(sort_cols, ascending=False)

            top_comments = df_b.head(5)

            with st.expander(f"{barrio} — Top comentarios"):
                cols_show = []
                if "name" in df_b.columns:
                    cols_show.append("name")
                if "review_scores_rating" in df_b.columns:
                    cols_show.append("review_scores_rating")
                if "number_of_reviews" in df_b.columns:
                    cols_show.append("number_of_reviews")
                cols_show.append("neighborhood_overview")

                df_comm = top_comments[cols_show].rename(columns={
                    "name": "Anuncio",
                    "review_scores_rating": "Rating",
                    "number_of_reviews": "N° reviews",
                    "neighborhood_overview": "Comentario de barrio"
                })
                st.dataframe(df_comm, use_container_width=True, hide_index=True)










    