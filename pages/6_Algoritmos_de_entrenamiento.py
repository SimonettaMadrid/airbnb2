# app.py
# Dashboard (1 página) para comparar KMeans vs DBSCAN en datos de Airbnb por ciudad
# Ejecuta: streamlit run app.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42

# Cambia nombres a los tuyos (en ./data/)
CITY_FILES = {
    "Barcelona": "Barcelona_Limpios.csv",
    "Hawái": "Hawai_Limpios.csv",
}

BASE_NUM_COLS = [
    "price",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "availability_365",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "days_since_last_review",
    "host_tenure_days",
]

CAT_CANDIDATES = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_is_superhost",
    "instant_bookable",
]

LOG1P_COLS = ["price", "minimum_nights", "maximum_nights", "number_of_reviews", "reviews_per_month"]

# KMeans
K_VALUES = list(range(2, 13))
SILHOUETTE_SAMPLE = 5000

# DBSCAN
K_FOR_KDIST = 10
DB_MIN_SAMPLES_VALUES = [5, 10, 20]
EPS_Q_LOW, EPS_Q_HIGH, EPS_N = 0.90, 0.99, 8
MAX_NOISE = 0.55

# PCA
USE_PCA = True
PCA_N_COMPONENTS = 0.95  # keep 95% variance


# -----------------------------
# UI polish
# -----------------------------
def _apply_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 0.75rem; padding-bottom: 0.75rem; max-width: 1600px;}
          [data-testid="stMetricValue"] {font-size: 1.10rem;}
          [data-testid="stMetricLabel"] {font-size: 0.85rem;}
          footer {visibility: hidden;}
          #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Data + features
# -----------------------------
def _cap_iqr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        if x.notna().sum() < 10:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = x.clip(lo, hi)
    return out


def clean_airbnb_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates()

    # Booleans
    for c in ["host_is_superhost", "instant_bookable"]:
        if c in df.columns:
            df[c] = df[c].map({"t": 1, "f": 0, True: 1, False: 0})

    # Dates (si existen)
    for c in ["last_scraped", "host_since", "first_review", "last_review"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "last_scraped" in df.columns and "host_since" in df.columns:
        df["host_tenure_days"] = (df["last_scraped"] - df["host_since"]).dt.days

    if "last_scraped" in df.columns and "last_review" in df.columns:
        df["days_since_last_review"] = (df["last_scraped"] - df["last_review"]).dt.days

    # Ensure numeric
    for c in BASE_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def pick_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Returns: (num_cluster_cols, cat_cols, log_map)."""
    num_cols = [c for c in BASE_NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_CANDIDATES if c in df.columns]

    # Reduce cardinality (top N)
    if "property_type" in df.columns and "property_type" in cat_cols:
        top = df["property_type"].astype("object").value_counts(dropna=False).head(25).index
        df.loc[~df["property_type"].isin(top), "property_type"] = "Other"

    if "neighbourhood_cleansed" in df.columns and "neighbourhood_cleansed" in cat_cols:
        top = df["neighbourhood_cleansed"].astype("object").value_counts(dropna=False).head(30).index
        df.loc[~df["neighbourhood_cleansed"].isin(top), "neighbourhood_cleansed"] = "Other"

    # IQR cap + log1p for skewed vars
    df2 = _cap_iqr(df, num_cols)
    log_map: Dict[str, str] = {}
    for c in LOG1P_COLS:
        if c in df2.columns:
            newc = f"log_{c}"
            df2[newc] = np.log1p(np.clip(pd.to_numeric(df2[c], errors="coerce").to_numpy(), 0, None))
            log_map[c] = newc

    df[df2.columns] = df2[df2.columns]

    num_cluster = [log_map.get(c, c) for c in num_cols]
    for c in num_cluster:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in cat_cols:
        df[c] = df[c].astype("object")

    return num_cluster, cat_cols, log_map


def build_preprocessor(num_cluster: List[str], cat_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cluster),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
    )
    return pre


@st.cache_data(show_spinner=False)
def preprocess_city(df_raw: pd.DataFrame) -> Dict[str, object]:
    """
    Devuelve SOLO tipos serializables por pickle (para st.cache_data).
    """
    df = clean_airbnb_df(df_raw)
    num_cluster, cat_cols, _ = pick_feature_sets(df)

    pre = build_preprocessor(num_cluster, cat_cols)
    X0 = pre.fit_transform(df)
    if not isinstance(X0, np.ndarray):
        X0 = np.asarray(X0)
    X0 = X0.astype(np.float32)

    X = X0
    explained = 1.0
    if USE_PCA:
        pca_model = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
        X = pca_model.fit_transform(X0).astype(np.float32)
        explained = float(np.sum(pca_model.explained_variance_ratio_))

    # 2D (para scatter)
    if X.shape[1] >= 2:
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        p2 = pca2.fit_transform(X)
    else:
        p2 = np.concatenate([X, np.zeros((X.shape[0], 1), dtype=X.dtype)], axis=1)

    return {
        "df": df,
        "X": X,
        "pca_2d": p2,
        "pca_explained": explained,
        "num_story": [c for c in BASE_NUM_COLS if c in df.columns],
        "cat_story": [c for c in CAT_CANDIDATES if c in df.columns],
    }


# -----------------------------
# KMeans
# -----------------------------
@st.cache_data(show_spinner=False)
def kmeans_experiments(X: np.ndarray) -> pd.DataFrame:
    n = X.shape[0]
    if n <= SILHOUETTE_SAMPLE:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=SILHOUETTE_SAMPLE, replace=False)
    Xs = X[idx]

    rows = []
    for k in K_VALUES:
        mb = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=2048, n_init=10)
        labels = mb.fit_predict(X)

        sil = np.nan
        if len(np.unique(labels[idx])) > 1:
            sil = float(silhouette_score(Xs, labels[idx]))

        ch = float(calinski_harabasz_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))

        rows.append({"k": int(k), "inertia": float(mb.inertia_), "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": dbi})

    return pd.DataFrame(rows)


def choose_best_k(km_exp: pd.DataFrame) -> int:
    tmp = km_exp.copy()
    tmp["sil_fill"] = tmp["silhouette"].fillna(-1.0)
    return int(tmp.sort_values(["sil_fill", "davies_bouldin"], ascending=[False, True]).iloc[0]["k"])


@st.cache_data(show_spinner=False)
def fit_kmeans(X: np.ndarray, best_k: int) -> Tuple[np.ndarray, Dict[str, float]]:
    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30)
    labels = model.fit_predict(X)

    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
    ch = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
    dbi = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")

    return labels, {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": dbi}


# -----------------------------
# DBSCAN
# -----------------------------
@st.cache_data(show_spinner=False)
def k_distance_curve(X: np.ndarray, k: int = K_FOR_KDIST) -> Tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists_knn, _ = nn.kneighbors(X)
    kth = np.sort(dists_knn[:, -1])
    eps_values = np.quantile(kth, np.linspace(EPS_Q_LOW, EPS_Q_HIGH, EPS_N))
    return kth, eps_values


@st.cache_data(show_spinner=False)
def dbscan_experiments(X: np.ndarray, eps_values: np.ndarray) -> pd.DataFrame:
    rows = []
    for ms in DB_MIN_SAMPLES_VALUES:
        for eps in eps_values:
            model = DBSCAN(eps=float(eps), min_samples=int(ms), n_jobs=-1)
            labels = model.fit_predict(X)

            n_noise = int(np.sum(labels == -1))
            noise_pct = float(n_noise / len(labels))
            n_clusters = int(len([c for c in np.unique(labels) if c != -1]))

            sil = np.nan
            ch = np.nan
            dbi = np.nan

            if n_clusters >= 2 and (len(labels) - n_noise) >= 200:
                mask = labels != -1
                Xnn = X[mask]
                ynn = labels[mask]
                if len(np.unique(ynn)) >= 2:
                    sil = float(silhouette_score(Xnn, ynn))
                    ch = float(calinski_harabasz_score(Xnn, ynn))
                    dbi = float(davies_bouldin_score(Xnn, ynn))

            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": int(ms),
                    "n_clusters": n_clusters,
                    "noise_pct": noise_pct,
                    "silhouette_non_noise": sil,
                    "calinski_harabasz_non_noise": ch,
                    "davies_bouldin_non_noise": dbi,
                }
            )

    return pd.DataFrame(rows)


def choose_best_dbscan(db_grid: pd.DataFrame) -> Tuple[float, int, Dict[str, float]]:
    cand = db_grid[(db_grid["n_clusters"] >= 2) & (db_grid["noise_pct"] <= MAX_NOISE)].copy()
    if len(cand) == 0:
        cand = db_grid[db_grid["n_clusters"] >= 2].copy()

    cand["sil_fill"] = cand["silhouette_non_noise"].fillna(-1.0)
    best = cand.sort_values(["sil_fill", "noise_pct", "n_clusters"], ascending=[False, True, False]).iloc[0]

    info = {
        "n_clusters": float(best["n_clusters"]),
        "noise_pct": float(best["noise_pct"]),
        "silhouette_non_noise": float(best["silhouette_non_noise"]),
        "calinski_non_noise": float(best["calinski_harabasz_non_noise"]),
        "davies_bouldin_non_noise": float(best["davies_bouldin_non_noise"]),
    }
    return float(best["eps"]), int(best["min_samples"]), info


@st.cache_data(show_spinner=False)
def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, Dict[str, float]]:
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=-1)
    labels = model.fit_predict(X)

    n_noise = int(np.sum(labels == -1))
    noise_pct = float(n_noise / len(labels))
    n_clusters = int(len([c for c in np.unique(labels) if c != -1]))

    return labels, {"n_clusters": float(n_clusters), "noise_pct": noise_pct}


# -----------------------------
# Explicación por métricas
# -----------------------------
def _robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    if pd.isna(iqr) or iqr == 0:
        sd = x.std()
        if pd.isna(sd) or sd == 0:
            return (x - med) * 0.0
        return (x - med) / sd
    return (x - med) / iqr


def top_category_modes(df: pd.DataFrame, label_col: str, cat_cols: List[str]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    work = df.copy()
    work[label_col] = work[label_col].astype(int)

    for cl in sorted(work[label_col].unique()):
        sub = work[work[label_col] == cl]
        cdict: Dict[str, str] = {}
        for c in [cc for cc in cat_cols if cc in work.columns]:
            vc = sub[c].astype("object").fillna("NaN").value_counts(normalize=True).head(1)
            if len(vc) == 0:
                continue
            val = vc.index[0]
            pct = float(vc.iloc[0]) * 100.0
            cdict[c] = f"{val} ({pct:.0f}%)"
        out[int(cl)] = cdict
    return out


def cluster_metrics_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = df.copy()
    work[label_col] = work[label_col].astype(int)

    candidates = [
        ("price", "precio_med"),
        ("accommodates", "capacidad_med"),
        ("bedrooms", "recamaras_med"),
        ("beds", "camas_med"),
        ("bathrooms", "banos_med"),
        ("minimum_nights", "min_noches_med"),
        ("availability_365", "disp_365_med"),
        ("number_of_reviews", "reviews_med"),
        ("reviews_per_month", "reviews_mes_med"),
        ("review_scores_rating", "rating_med"),
        ("host_tenure_days", "ant_host_dias_med"),
        ("days_since_last_review", "dias_ult_review_med"),
    ]
    cols = [c for c, _ in candidates if c in work.columns]

    out = (
        work.groupby(label_col)
        .agg(n=(label_col, "size"), **{f"{c}_median": (c, "median") for c in cols})
        .reset_index()
        .rename(columns={label_col: "cluster"})
        .sort_values("cluster")
    )

    rename_map = {"cluster": "cluster", "n": "n"}
    for c, pretty in candidates:
        if f"{c}_median" in out.columns:
            rename_map[f"{c}_median"] = pretty
    out = out.rename(columns=rename_map)

    # rounding
    for c in out.columns:
        if c in ["cluster", "n"]:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if c in ["precio_med", "disp_365_med", "reviews_med", "ant_host_dias_med", "dias_ult_review_med"]:
            out[c] = out[c].round(0)
        else:
            out[c] = out[c].round(2)

    out["cluster"] = out["cluster"].astype(int)
    out["n"] = out["n"].astype(int)
    return out


# -----------------------------
# Plots
# -----------------------------
def fig_pca_scatter(p2: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    d = pd.DataFrame({"PC1": p2[:, 0], "PC2": p2[:, 1], "cluster": labels.astype(int).astype(str)})
    fig = px.scatter(d, x="PC1", y="PC2", color="cluster", opacity=0.70, title=title)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320, legend_title_text="cluster")
    return fig


def fig_cluster_sizes(labels: np.ndarray, title: str) -> go.Figure:
    vc = pd.Series(labels).value_counts().sort_index()
    d = pd.DataFrame({"cluster": vc.index.astype(int).astype(str), "n": vc.values})
    fig = px.bar(d, x="cluster", y="n", title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def fig_line(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=y))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=300)
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return fig


def fig_kdist(kth: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=kth, mode="lines", name="k-dist"))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=300)
    fig.update_xaxes(title="Puntos ordenados")
    fig.update_yaxes(title="Distancia al k-ésimo vecino")
    return fig


def fig_cluster_signature_bars(
    df: pd.DataFrame,
    label_col: str,
    cluster_id: int,
    numeric_cols: List[str],
    title: str,
    top_n: int = 8,
) -> go.Figure:
    work = df.copy()
    num = [c for c in numeric_cols if c in work.columns]
    if len(num) == 0:
        return go.Figure()

    z_all = {c: _robust_z(work[c]) for c in num}
    if "cluster_id_tmp" in work.columns:
        work = work.drop(columns=["cluster_id_tmp"])

    work["cluster_id_tmp"] = work[label_col].astype(int)
    mask = work["cluster_id_tmp"] == int(cluster_id)
    if mask.sum() == 0:
        return go.Figure()

    deltas = []
    for c in num:
        dz = float(z_all[c][mask].median())
        if not math.isnan(dz):
            deltas.append((c, dz))

    deltas = sorted(deltas, key=lambda t: abs(t[1]), reverse=True)[:top_n]
    d = pd.DataFrame({"variable": [c.replace("_", " ") for c, _ in deltas], "delta": [dz for _, dz in deltas]})
    d = d.sort_values("delta")

    fig = px.bar(d, x="delta", y="variable", orientation="h", title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=280)
    fig.update_xaxes(title="Diferencia vs típico (robust z)")
    fig.update_yaxes(title="")
    return fig


def fig_cluster_bubble(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    work = df.copy()
    if "accommodates" not in work.columns or "price" not in work.columns:
        return go.Figure()

    g = (
        work.groupby(label_col)
        .agg(
            n=("price", "size"),
            price_med=("price", "median"),
            acc_med=("accommodates", "median"),
        )
        .reset_index()
    )
    g[label_col] = g[label_col].astype(int).astype(str)

    fig = px.scatter(
        g,
        x="acc_med",
        y="price_med",
        size="n",
        color=label_col,
        title=title,
        labels={"acc_med": "Capacidad mediana", "price_med": "Precio mediano", label_col: "cluster"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=280, legend_title_text="cluster")
    return fig


# -----------------------------
# IO
# -----------------------------
def _load_city_df(city: str) -> Optional[pd.DataFrame]:
    data_dir = Path.cwd() / "data"
    fp = data_dir / CITY_FILES.get(city, "")
    if fp.exists():
        return pd.read_csv(fp)
    return None


def _metric_if(col: str, row: Dict, label: str, fmt: str) -> Optional[Tuple[str, str]]:
    if col not in row:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return label, format(v, fmt)


def render_cluster_metrics(row: Dict) -> None:
    # Primera fila (lo esencial)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Anuncios", f"{int(row.get('n', 0))}")
        if "precio_med" in row:
            st.metric("Precio mediano", f"{row['precio_med']:.0f}")
    with c2:
        if "capacidad_med" in row:
            st.metric("Capacidad mediana", f"{row['capacidad_med']:.0f}")
        if "rating_med" in row and not pd.isna(row["rating_med"]):
            st.metric("Rating mediano", f"{row['rating_med']:.2f}")
    with c3:
        if "disp_365_med" in row:
            st.metric("Disponibilidad 365", f"{row['disp_365_med']:.0f}")
        if "min_noches_med" in row:
            st.metric("Mínimo noches", f"{row['min_noches_med']:.0f}")

    # Segunda fila (contexto)
    c4, c5, c6 = st.columns(3)
    with c4:
        if "recamaras_med" in row:
            st.metric("Recámaras (med.)", f"{row['recamaras_med']:.0f}")
        if "banos_med" in row:
            st.metric("Baños (med.)", f"{row['banos_med']:.0f}")
    with c5:
        if "reviews_med" in row:
            st.metric("Reviews (med.)", f"{row['reviews_med']:.0f}")
        if "reviews_mes_med" in row and not pd.isna(row["reviews_mes_med"]):
            st.metric("Reviews/mes (med.)", f"{row['reviews_mes_med']:.2f}")
    with c6:
        if "ant_host_dias_med" in row:
            st.metric("Antigüedad host (días)", f"{row['ant_host_dias_med']:.0f}")
        if "dias_ult_review_med" in row and not pd.isna(row["dias_ult_review_med"]):
            st.metric("Días desde últ. review", f"{row['dias_ult_review_med']:.0f}")


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Clustering Airbnb (KMeans vs DBSCAN)", layout="wide")
    _apply_css()



    st.divider()

    city = st.selectbox("Ciudad", list(CITY_FILES.keys()), index=0)

    df_raw = _load_city_df(city)
    if df_raw is None:
        st.error(f"No encontré el archivo. Colócalo en ./data/ con el nombre: {CITY_FILES[city]}")
        st.stop()

    prep = preprocess_city(df_raw)
    df = prep["df"]
    X = prep["X"]
    pca_2d = prep["pca_2d"]
    pca_expl = float(prep["pca_explained"])
    numeric_for_story = list(prep["num_story"])
    cat_for_story = list(prep["cat_story"])

    # Clustering
    km_exp = kmeans_experiments(X)
    best_k = choose_best_k(km_exp)
    km_labels, km_metrics = fit_kmeans(X, best_k)

    kth, eps_values = k_distance_curve(X, k=K_FOR_KDIST)
    db_grid = dbscan_experiments(X, eps_values)
    best_eps, best_ms, best_info = choose_best_dbscan(db_grid)
    db_labels, db_metrics = fit_dbscan(X, best_eps, best_ms)

    # Attach labels
    df_km = df.copy()
    df_km["cluster_kmeans"] = km_labels.astype(int)

    df_db = df.copy()
    df_db["cluster_dbscan"] = db_labels.astype(int)

    # Tables + categories
    km_table = cluster_metrics_table(df_km, "cluster_kmeans")
    db_table = cluster_metrics_table(df_db, "cluster_dbscan")
    km_cats = top_category_modes(df_km, "cluster_kmeans", cat_for_story)
    db_cats = top_category_modes(df_db, "cluster_dbscan", cat_for_story)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Anuncios", f"{len(df):,}".replace(",", " "))
    with k2:
        st.metric("Vars numéricas", str(len(numeric_for_story)))
    with k3:
        st.metric("PCA var explicada", f"{pca_expl*100:.1f}%")
    with k4:
        st.metric("KMeans k*", str(best_k))
    with k5:
        st.metric("DBSCAN ruido", f"{db_metrics['noise_pct']*100:.1f}%")

    st.divider()

    tab_km, tab_db = st.tabs(["KMeans", "DBSCAN"])

    # -------- KMeans --------
    with tab_km:
        vis, expl = st.tabs(["Visión (gráficas)", "Clústeres (métricas)"])

        with vis:
            a, b = st.columns(2)
            with a:
                st.plotly_chart(fig_pca_scatter(pca_2d, km_labels, f"{city} — PCA 2D (KMeans)"), use_container_width=True)
            with b:
                st.plotly_chart(fig_cluster_sizes(km_labels, f"{city} — Tamaño de clúster (KMeans)"), use_container_width=True)

            c, d = st.columns(2)
            with c:
                st.plotly_chart(fig_line(km_exp, "k", "inertia", f"{city} — Codo (Inercia)"), use_container_width=True)
            with d:
                st.plotly_chart(fig_line(km_exp, "k", "silhouette", f"{city} — Silhouette vs k"), use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Silhouette (final)", f"{km_metrics['silhouette']:.3f}")
            with m2:
                st.metric("Calinski-Harabasz", f"{km_metrics['calinski_harabasz']:.1f}")
            with m3:
                st.metric("Davies-Bouldin", f"{km_metrics['davies_bouldin']:.3f}")

        with expl:
            left, right = st.columns([1, 1.15])
            with left:
                st.subheader("Clúster seleccionado (KMeans)")
                cl = st.selectbox("Clúster", km_table["cluster"].tolist(), index=0, key="kmeans_cluster")
                row = km_table.loc[km_table["cluster"] == int(cl)].iloc[0].to_dict()
                render_cluster_metrics(row)

                st.caption("Por qué se separa así: KMeans agrupa anuncios por similitud (distancias) usando variables numéricas escaladas y categorías codificadas.")
                st.caption("Categorías más comunes en este clúster")
                cats = km_cats.get(int(cl), {})
                if cats:
                    st.json(cats, expanded=False)
                else:
                    st.write("—")

            with right:
                st.subheader("Qué variables diferencian este clúster")
                st.plotly_chart(
                    fig_cluster_signature_bars(df_km, "cluster_kmeans", int(cl), numeric_for_story, f"{city} — Firma del clúster {int(cl)} (KMeans)"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    fig_cluster_bubble(df_km, "cluster_kmeans", f"{city} — Clústeres (capacidad vs precio, KMeans)"),
                    use_container_width=True,
                )

            st.subheader("Resumen por clúster (medianas)")
            st.dataframe(km_table, use_container_width=True, height=240)

    # -------- DBSCAN --------
    with tab_db:
        vis, expl = st.tabs(["Visión (gráficas)", "Clústeres (métricas)"])

        with vis:
            a, b = st.columns(2)
            with a:
                st.plotly_chart(fig_pca_scatter(pca_2d, db_labels, f"{city} — PCA 2D (DBSCAN; -1=ruido)"), use_container_width=True)
            with b:
                st.plotly_chart(fig_cluster_sizes(db_labels, f"{city} — Tamaño de clúster (DBSCAN; -1=ruido)"), use_container_width=True)

            c, d = st.columns(2)
            with c:
                st.plotly_chart(fig_kdist(kth, f"{city} — k-distance (k={K_FOR_KDIST})"), use_container_width=True)
            with d:
                g = db_grid.copy()
                g["eps"] = g["eps"].round(3)
                pivot = g.pivot_table(index="min_samples", columns="eps", values="n_clusters", aggfunc="first").fillna(0)
                fig = px.imshow(pivot, aspect="auto", title=f"{city} — #clústeres (DBSCAN) por eps/min_samples")
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
                st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("eps*", f"{best_eps:.3f}")
            with m2:
                st.metric("min_samples*", str(best_ms))
            with m3:
                st.metric("#clústeres (sin ruido)", str(int(best_info["n_clusters"])))
            with m4:
                st.metric("Ruido", f"{best_info['noise_pct']*100:.1f}%")

        with expl:
            left, right = st.columns([1, 1.15])
            with left:
                st.subheader("Clúster seleccionado (DBSCAN)")
                cl = st.selectbox("Clúster (incluye -1=ruido)", db_table["cluster"].tolist(), index=0, key="dbscan_cluster")
                row = db_table.loc[db_table["cluster"] == int(cl)].iloc[0].to_dict()
                render_cluster_metrics(row)

                if int(cl) == -1:
                    st.info("Ruido (-1) = anuncios atípicos: están aislados y no pertenecen a zonas densas.")
                st.caption("Por qué se separa así: DBSCAN agrupa por densidad. Si no hay suficientes vecinos cercanos, el anuncio se queda como ruido (-1).")

                st.caption("Categorías más comunes en este clúster")
                cats = db_cats.get(int(cl), {})
                if cats:
                    st.json(cats, expanded=False)
                else:
                    st.write("—")

            with right:
                st.subheader("Qué variables diferencian este clúster")
                st.plotly_chart(
                    fig_cluster_signature_bars(df_db, "cluster_dbscan", int(cl), numeric_for_story, f"{city} — Firma del clúster {int(cl)} (DBSCAN)"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    fig_cluster_bubble(df_db, "cluster_dbscan", f"{city} — Clústeres (capacidad vs precio, DBSCAN)"),
                    use_container_width=True,
                )

            st.subheader("Resumen por clúster (medianas)")
            st.dataframe(db_table, use_container_width=True, height=240)

    st.caption(f"Datos: ./data/{CITY_FILES[city]}  |  RandomState={RANDOM_STATE}  |  PCA={USE_PCA} ({PCA_N_COMPONENTS})")


if __name__ == "__main__":
    main()
