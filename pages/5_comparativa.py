import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import load_all_data

# ===================== ESTILOS =====================
st.markdown(
    """
    <style>
    /* ==================== TITULOS 20PX ==================== */

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-size: 20px !important;
    }

    /* st.title */
    div[data-testid="stAppViewContainer"] h1 {
        font-size: 20px !important;
    }

    /* st.header */
    div[data-testid="stHeader"] {
        font-size: 20px !important;
    }

    /* st.subheader */
    div[data-testid="stSubheader"] {
        font-size: 20px !important;
    }

    /* ==================== ESTILO DE CUADROS ==================== */

    /* KPIs (st.metric) */
    div[data-testid="stMetric"] {
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 8px 12px;
        background-color: white;
    }

    /* Contenedor de gráficas Plotly */
    div[data-testid="stPlotlyChart"] {
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 8px 8px 2px 8px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

###### DATOS
RAW = load_all_data()

def _to_city_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        default_keys = ["Barcelona", "Cambridge", "Boston", "Hawái", "Budapest"][:len(obj)]
        return dict(zip(default_keys, obj))
    raise ValueError("load_all_data() debe regresar dict o lista/tupla de DataFrames.")

def clean_city(df: pd.DataFrame, is_budapest: bool = False) -> pd.DataFrame:
    """
    Limpia columnas fantasma y convierte 'price' a float.
    Si is_budapest=True, convierte de HUF a USD.
    """
    df = df.copy()

    # Quitar columnas tipo "Unnamed: 0"
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df.drop(columns=[c], inplace=True)

    if "price" in df.columns:
        # Convertir a string para limpieza
        s = df["price"].astype(str)

        # Quitar símbolos de moneda, comas, etc. (dejamos dígitos, punto y guión)
        s = s.str.replace(r"[^\d\.\-]", "", regex=True)

        # Valores raros con más de un '-' (ej. "9.5-12") -> NaN
        mask_bad = s.str.count("-") > 1
        s[mask_bad] = np.nan

        # A numérico
        price = pd.to_numeric(s, errors="coerce")

        # Conversión HUF -> USD solo para Budapest
        if is_budapest:
            HUF_TO_USD = 1 / 370  # ≈ 0.00270 USD por HUF
            price = price * HUF_TO_USD

        df["price"] = price

    return df

raw_city_map = _to_city_map(RAW)

# Limpieza por ciudad, con conversión de moneda solo en Budapest
dfs_ciudades = {}
for nombre, df_city in raw_city_map.items():
    dfs_ciudades[nombre] = clean_city(df_city, is_budapest=(nombre == "Budapest"))

PRIMARY_VARS = [
    "price", "accommodates", "bedrooms", "beds",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "availability_30", "availability_365",
    "estimated_occupancy_l365d", "minimum_nights", "maximum_nights",
]
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
        font-size:0.85rem;
        display:inline-block;
        ">
        {nombre}
    </span>
    """

###### ===================== Helpers =====================
def guess_superhost_col(df: pd.DataFrame):
    for c in df.columns:
        if "superhost" in c.lower():
            return c
    return None

###### SIDEBAR
st.sidebar.title("Filtros")

lista_ciudades = list(dfs_ciidades.keys()) if 'dfs_ciidades' in globals() else list(dfs_ciudades.keys())
# (por si acaso, pero realmente usamos dfs_ciudades)
lista_ciudades = list(dfs_ciudades.keys())

ciudades_sel = st.sidebar.multiselect(
    "Ciudades",
    options=lista_ciudades,
    default=lista_ciudades,
)

if not ciudades_sel:
    st.warning("Selecciona al menos una ciudad en el sidebar.")
    st.stop()

dfs_filtered = {c: dfs_ciudades[c] for c in ciudades_sel}
df_global = pd.concat(dfs_filtered.values(), ignore_index=True)

vars_disponibles = [
    v for v in PRIMARY_VARS
    if any(v in df.columns for df in dfs_filtered.values())
]

vars_sel = st.sidebar.multiselect(
    "Variables (líneas)",
    options=vars_disponibles,
    default=["bedrooms", "beds"][:len(vars_disponibles)],
    key="vars_line",
)

###### KPIs
###### KPIs

def nombre_metrica_ocupacion(col_occ: str) -> str:
    mapping = {
        "estimated_occupancy_l365d": "Ocupación anual estimada",
        "availability_365": "Disponibilidad anual (días)",
        "reviews_per_month": "Reseñas por mes",
    }
    return mapping.get(col_occ, col_occ)

# Recorremos ciudades de 2 en 2 para hacer filas
for i in range(0, len(ciudades_sel), 2):
    ciudades_fila = ciudades_sel[i:i+2]
    cols_fila = st.columns(len(ciudades_fila))

    for col, ciudad in zip(cols_fila, ciudades_fila):
        dfc = dfs_filtered[ciudad]

        # ---- CÁLCULO DE KPIs ----
        total_listings = len(dfc)
        avg_price = dfc["price"].mean() if "price" in dfc.columns else np.nan

        col_occ = None
        for cand in ["estimated_occupancy_l365d", "availability_365", "reviews_per_month"]:
            if cand in dfc.columns:
                col_occ = cand
                break
        avg_occ = dfc[col_occ].mean() if col_occ else np.nan

        # ---- TARJETA POR CIUDAD DENTRO DE LA COLUMNA ----
        with col:
            with st.container(border=True):
                st.markdown(city_badge(ciudad), unsafe_allow_html=True)


                k1, k2, k3 = st.columns(3)

                # KPI 1: Listings
                with k1:
                    st.metric("Listings", f"{total_listings:,}")

                # KPI 2: Precio promedio
                with k2:
                    if not np.isnan(avg_price):
                        st.metric("Precio prom. (USD)", f"{avg_price:,.0f}")
                    else:
                        st.metric("Precio prom. (USD)", "N/D")

                # KPI 3: Ocupación / Actividad
                with k3:
                    if col_occ and not np.isnan(avg_occ):
                        nombre_m = nombre_metrica_ocupacion(col_occ)
                        st.metric(nombre_m, f"{avg_occ:,.2f}")
                    else:
                        st.metric("Actividad", "N/D")


###### GRÁFICAS
with st.container():
    col_line, col_bar, col_host = st.columns(3)

    # ----- LÍNEAS POR CIUDAD -----
    with col_line:
        st.subheader("Líneas por ciudad")

        if vars_sel:
            filas = []
            for ciudad, dfc in dfs_filtered.items():
                for var in vars_sel:
                    if var in dfc.columns:
                        filas.append(
                            {
                                "Ciudad": ciudad,
                                "Variable": var,
                                "Valor": dfc[var].mean(),
                            }
                        )
            df_line = pd.DataFrame(filas)
            if not df_line.empty:
                fig_line = px.line(
                    df_line,
                    x="Ciudad",
                    y="Valor",
                    color="Variable",
                    markers=True,
                )
                fig_line.update_layout(
                    xaxis_title="Ciudad",
                    yaxis_title="Valor promedio",
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No hay datos numéricos para las variables seleccionadas.")
        else:
            st.info("Selecciona al menos una variable en el sidebar.")

    # ----- TOP 10 BARRIOS -----
    with col_bar:
        st.subheader("Top 10 barrios por ciudad")

        barrio_col = "neighbourhood_cleansed"

        if barrio_col in df_global.columns and "price" in df_global.columns:
            filas_barrio = []
            for ciudad, dfc in dfs_filtered.items():
                if barrio_col in dfc.columns and "price" in dfc.columns:
                    top10 = dfc[barrio_col].value_counts().head(10).index
                    tmp = (
                        dfc[dfc[barrio_col].isin(top10)]
                        .groupby(barrio_col)["price"]
                        .mean()
                        .reset_index()
                        .rename(
                            columns={
                                barrio_col: "Barrio",
                                "price": "Precio_promedio",
                            }
                        )
                    )
                    tmp["Ciudad"] = ciudad
                    filas_barrio.append(tmp)

            if filas_barrio:
                df_barrios = pd.concat(filas_barrio, ignore_index=True)

                fig_bar = px.bar(
                    df_barrios,
                    x="Barrio",
                    y="Precio_promedio",
                    color="Ciudad",
                    barmode="group",
                    color_discrete_map=CITY_PALETTE,      # 👈 siempre mismos colores por ciudad
                )

                fig_bar.update_layout(
                    xaxis_tickangle=-60,
                    xaxis_title="Barrio",
                    yaxis_title="Precio promedio (USD)",
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No se pudieron calcular precios por barrio.")
        else:
            st.info("No se encontró la columna 'neighbourhood_cleansed' o 'price'.")

    # ----- HOST vs SUPERHOST -----
    with col_host:
        st.subheader("Host vs Superhost")

        super_col = guess_superhost_col(df_global)

        if super_col:
            filas_host = []
            for ciudad, dfc in dfs_filtered.items():
                if super_col in dfc.columns:
                    counts = dfc[super_col].fillna("No info").value_counts()
                    for val, cnt in counts.items():
                        val_str = str(val).lower()
                        es_super = val_str in [
                            "t", "true", "1", "yes", "y", "si", "sí", "superhost",
                        ]
                        tipo = "Superhost" if es_super else "Host normal"
                        filas_host.append(
                            {
                                "Ciudad": ciudad,
                                "TipoHost": tipo,
                                "Anuncios": cnt,
                            }
                        )

            if filas_host:
                df_host = pd.DataFrame(filas_host)
                fig_host = px.bar(
                    df_host,
                    x="Ciudad",
                    y="Anuncios",
                    color="TipoHost",
                    barmode="group",
                )
                fig_host.update_layout(
                    xaxis_title="Ciudad",
                    yaxis_title="Anuncios",
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_host, use_container_width=True)
            else:
                st.info("No hay datos suficientes de host/superhost.")
        else:
            st.info("No se encontró columna de superhost.")





