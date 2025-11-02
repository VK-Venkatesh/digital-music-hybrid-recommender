# Time Series Model
import os
import io
import json
import math
import time
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hybrid Recommender (Digital Music)", layout="wide")
st.title("🎵 Hybrid Recommender System — Digital Music")
st.caption("Popular ➜ Content-based ➜ Collaborative ➜ Hybrid. Cleaned, encoded, and ready to recommend.")

DEFAULT_REVIEWS_PATH = ",json file path"
DEFAULT_META_PATH = "meta__.jsonl file path"

@st.cache_data(show_spinner=False)
def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False
def _read_jsonl(fp) -> pd.DataFrame:
    """Read JSON Lines into a DataFrame (handles file-like or path)."""
    records = []
    if isinstance(fp, (str, os.PathLike)):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    try:
                        records.append(json.loads(line.rstrip(",\n ")))
                    except Exception:
                        continue
    else:
        for raw in fp:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json.loads(line.rstrip(",\n ")))
                except Exception:
                    continue
    return pd.DataFrame.from_records(records)

def _flatten_categories(cats):
    if cats is None:
        return ""
    if isinstance(cats, (list, tuple)):
        if len(cats) == 0:
            return ""
        try:
            flat = []
            for sub in cats:
                if isinstance(sub, (list, tuple)):
                    flat.extend([str(x) for x in sub])
                else:
                    flat.append(str(sub))

            seen = set()
            ordered = [c for c in flat if not (c in seen or seen.add(c))]
            return " ".join(ordered)
        except Exception:
            return " ".join(map(str, cats))
    return str(cats)

@st.cache_data(show_spinner=True)
def load_and_clean_data(
    reviews_path: Optional[str],
    meta_path: Optional[str],
    min_user_interactions: int = 3,
    min_item_interactions: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if reviews_path is None and meta_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame()
    meta = pd.DataFrame()

    # ---------- Reviews ----------
    if reviews_path is not None:
        df = _read_jsonl(reviews_path)

    if "reviewerID" not in df.columns:
        for c in df.columns:
            if c.lower() in ["reviewerid", "user", "user_id", "uid"]:
                df = df.rename(columns={c: "reviewerID"})
                break
    if "asin" not in df.columns:
        for c in df.columns:
            if c.lower() in ["asin", "item", "productid", "product_id", "pid"]:
                df = df.rename(columns={c: "asin"})
                break
    if "overall" not in df.columns:
        for c in df.columns:
            if c.lower() in ["rating", "stars", "score"]:
                df = df.rename(columns={c: "overall"})
                break

    keep_cols = [c for c in ["reviewerID", "asin", "overall", "unixReviewTime"] if c in df.columns]
    df = df[keep_cols].copy()

    for c in ["reviewerID", "asin"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "overall" in df.columns:
        df = df[~df["overall"].isna()]
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        df = df[~df["overall"].isna()]

    if {"reviewerID", "asin", "unixReviewTime"}.issubset(df.columns):
        df = df.sort_values("unixReviewTime").drop_duplicates(["reviewerID", "asin"], keep="last")
    elif {"reviewerID", "asin"}.issubset(df.columns):
        df = df.drop_duplicates(["reviewerID", "asin"], keep="last")

    if "reviewerID" in df.columns and "asin" in df.columns:
        user_counts = df["reviewerID"].value_counts()
        item_counts = df["asin"].value_counts()
        df = df[df["reviewerID"].isin(user_counts[user_counts >= min_user_interactions].index)]
        df = df[df["asin"].isin(item_counts[item_counts >= min_item_interactions].index)]

    # ---------- Meta ----------
    if meta_path is not None:
        meta = _read_jsonl(meta_path)

# ...existing code...
    if not meta.empty:
        # Robustly rename possible ID columns to 'asin'
        meta_columns_lower = {c.lower(): c for c in meta.columns}
        if "asin" not in meta.columns:
            for possible in ["asin", "item", "productid", "product_id", "pid", "id"]:
                if possible in meta_columns_lower:
                    meta = meta.rename(columns={meta_columns_lower[possible]: "asin"})
                    break
        if "asin" not in meta.columns:
            raise ValueError("Meta file must contain an 'asin' column (or equivalent). Please check your file.")
# ...existing code...
# ...existing code...
    if df.empty or meta.empty or "asin" not in meta.columns:
        st.warning("No data after cleaning, or 'asin' column missing in meta file. Try lowering the filtering thresholds or verify file formats.")
        st.stop()
# ...existing code...
        if "title" not in meta.columns:
            for c in meta.columns:
                if c.lower() in ["title", "name"]:
                    meta = meta.rename(columns={c: "title"})
                    break

        if "categories" in meta.columns:
            meta["categories_str"] = meta["categories"].apply(_flatten_categories)
        else:
            meta["categories_str"] = ""

        if "description" in meta.columns:
            meta["description_str"] = meta["description"].apply(
                lambda x: " ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
            )
        else:
            meta["description_str"] = ""

        keep_meta = ["asin", "title", "categories_str", "description_str"]
        keep_meta = [c for c in keep_meta if c in meta.columns]

        if "asin" in keep_meta:
            meta = meta[keep_meta].drop_duplicates("asin", keep="first")
        else:
            meta = meta[keep_meta].copy()  # drop_duplicates only if asin exists

        for c in ["asin", "title"]:
            if c in meta.columns:
                meta[c] = meta[c].astype(str).str.strip()

    # ---------- Merge ----------
    if "asin" in df.columns and "asin" in meta.columns:
        merged = pd.merge(df, meta, on="asin", how="left")
    else:
        merged = df.copy()
        # add empty cols for consistency
        for col in ["title", "categories_str", "description_str"]:
            merged[col] = ""

    if "title" in merged.columns:
        merged["title"] = merged["title"].fillna("")
    if "categories_str" in merged.columns:
        merged["categories_str"] = merged["categories_str"].fillna("")
    if "description_str" in merged.columns:
        merged["description_str"] = merged["description_str"].fillna("")

    return df.reset_index(drop=True), meta.reset_index(drop=True), merged.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def popularity_table(df: pd.DataFrame, meta: pd.DataFrame, min_votes: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["asin", "title", "v", "R", "WR"])

    grp = df.groupby("asin").agg(v=("overall", "count"), R=("overall", "mean")).reset_index()
    C = grp["R"].mean() if not grp.empty else 0.0
    m = max(min_votes, int(grp["v"].quantile(0.80))) if len(grp) > 0 else min_votes
    grp["WR"] = (grp["v"]/(grp["v"]+m))*grp["R"] + (m/(grp["v"]+m))*C

    titles = meta[["asin", "title"]].drop_duplicates("asin") if "title" in meta.columns else None
    if titles is not None:
        grp = grp.merge(titles, on="asin", how="left")

    grp = grp.sort_values(["WR", "v"], ascending=False).reset_index(drop=True)
    return grp[["asin", "title", "v", "R", "WR"]]

@st.cache_resource(show_spinner=True)
def build_content_model(meta: pd.DataFrame, max_features: int = 50000):
    if meta.empty:
        return None, None, None

    text = (
        (meta["title"] if "title" in meta.columns else pd.Series("", index=meta.index)).fillna("")
        + " "
        + (meta["categories_str"] if "categories_str" in meta.columns else pd.Series("", index=meta.index)).fillna("")
        + " "
        + (meta["description_str"] if "description_str" in meta.columns else pd.Series("", index=meta.index)).fillna("")
    ).astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.8,
        min_df=2,
        max_features=max_features,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(text)
    # Use NearestNeighbors with cosine metric for efficiency on sparse vectors
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)
    return vectorizer, X, nn


def content_recommend(
    asin: str,
    meta: pd.DataFrame,
    vectorizer,
    X,
    nn,
    k: int = 20
) -> pd.DataFrame:
    if asin not in set(meta["asin"].tolist()):
        return pd.DataFrame(columns=["asin", "title", "score"])
    idx_map = {a:i for i,a in enumerate(meta["asin"])}
    idx = idx_map.get(asin, None)
    if idx is None:
        return pd.DataFrame(columns=["asin", "title", "score"])


    xq = X[idx]
    # Find neighbors
    distances, indices = nn.kneighbors(xq, n_neighbors=min(k+1, X.shape[0]))
    distances = distances.flatten()
    indices = indices.flatten()

    recs = []
    for d, i in zip(distances, indices):
        if i == idx:
            continue
        score = 1.0 - d
        a = meta.iloc[i]["asin"]
        t = meta.iloc[i]["title"] if "title" in meta.columns else ""
        recs.append((a, t, float(score)))
        if len(recs) >= k:
            break

    out = pd.DataFrame(recs, columns=["asin", "title", "score"])
    return out

@st.cache_resource(show_spinner=True)
def build_collab_matrices(df: pd.DataFrame):
    if df.empty:
        return None, None, None, None, None

    users = pd.Categorical(df["reviewerID"])
    items = pd.Categorical(df["asin"])
    user_index = pd.Series(range(len(users.categories)), index=users.categories, name="u_index")
    item_index = pd.Series(range(len(items.categories)), index=items.categories, name="i_index")

    row = users.codes
    col = items.codes
    data = df["overall"].astype(float).values

    # Center per-user to reduce bias
    user_means = pd.Series(data).groupby(row).mean()
    centered = data - np.array([user_means[r] for r in row])

    R = csr_matrix((centered, (row, col)), shape=(len(user_index), len(item_index)))
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(R.T)
    svd = TruncatedSVD(n_components=min(64, max(2, int(min(R.shape)-1)))) if min(R.shape) > 2 else None
    R_t = R.T
    X_latent = svd.fit_transform(R_t) if svd is not None else None

    return R, nn, svd, X_latent, item_index


def collab_recommend(
    asin: str,
    df: pd.DataFrame,
    R,
    nn: NearestNeighbors,
    svd: Optional[TruncatedSVD],
    X_latent: Optional[np.ndarray],
    item_index: pd.Series,
    k: int = 20
) -> pd.DataFrame:
    if R is None or nn is None or item_index is None:
        return pd.DataFrame(columns=["asin", "title", "score"])

    inv_item_index = {i:a for a,i in item_index.items()}
    inv_lookup = {v:k for k,v in item_index.items()}

    i = item_index.get(asin, None)
    if pd.isna(i):
        return pd.DataFrame(columns=["asin", "title", "score"])
    i = int(i)

    # NN on sparse item vectors
    distances, indices = nn.kneighbors(R.T[i], n_neighbors=min(k+1, R.T.shape[0]))
    distances = distances.flatten()
    indices = indices.flatten()

    recs = []
    for d, idx in zip(distances, indices):
        if idx == i:
            continue
        score = 1.0 - d
        a = inv_item_index.get(idx, None)
        if a is None:
            continue
        recs.append((a, float(score)))
        if len(recs) >= k:
            break
    if X_latent is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(X_latent[i:i+1], X_latent).flatten()
        order = np.argsort(-sims)
        latent_recs = []
        for idx in order:
            if idx == i:
                continue
            a = inv_item_index.get(idx, None)
            if a is None:
                continue
            latent_recs.append((a, float(sims[idx])))
            if len(latent_recs) >= k:
                break
        scores = {}
        for a, s in recs:
            scores[a] = max(scores.get(a, 0.0), s)
        for a, s in latent_recs:
            scores[a] = max(scores.get(a, 0.0), s)
        recs = list(scores.items())

    out = pd.DataFrame(recs, columns=["asin", "score"]).sort_values("score", ascending=False)
    return out
def hybrid_recommend(
    asin: str,
    content_df: pd.DataFrame,
    collab_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    w_content: float = 0.5,
    w_collab: float = 0.5,
    w_pop: float = 0.15,
    k: int = 20
) -> pd.DataFrame:
    def _norm(series):
        if series.empty:
            return series
        scaler = MinMaxScaler()
        vals = scaler.fit_transform(series.to_numpy().reshape(-1,1)).flatten()
        return pd.Series(vals, index=series.index)

    c = content_df.copy()
    if not c.empty:
        c["c_score"] = _norm(c["score"])
        c = c[["asin", "c_score"]]

    cb = collab_df.copy()
    if not cb.empty:
        cb["cf_score"] = _norm(cb["score"])
        cb = cb[["asin", "cf_score"]]

    p = pop_df.copy()
    if not p.empty:
        p["pop_score"] = _norm(p["WR"])
        p = p[["asin", "pop_score"]]

    dfs = [d for d in [c, cb, p] if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame(columns=["asin", "hybrid_score"])
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="asin", how="outer")
    merged = merged.fillna(0.0)

    merged["hybrid_score"] = w_content*merged.get("c_score", 0) + w_collab*merged.get("cf_score", 0) + w_pop*merged.get("pop_score", 0)
    merged = merged.sort_values("hybrid_score", ascending=False).head(k)
    return merged[["asin", "hybrid_score"]]
st.sidebar.header("Settings")

source_choice = st.sidebar.selectbox(
    "Load data from",
    ["Default paths (if present)", "Upload files (.jsonl)"],
    index=0
)

min_user_interactions = st.sidebar.slider("Min interactions per user", 1, 20, 3, 1)
min_item_interactions = st.sidebar.slider("Min interactions per item", 1, 50, 5, 1)
min_votes_for_pop = st.sidebar.slider("Min votes for popularity floor (m)", 1, 200, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Weights")
w_content = st.sidebar.slider("Content weight", 0.0, 1.0, 0.45, 0.05)
w_collab = st.sidebar.slider("Collaborative weight", 0.0, 1.0, 0.45, 0.05)
w_pop = st.sidebar.slider("Popularity weight (bonus)", 0.0, 1.0, 0.10, 0.05)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 1)
reviews_file = None
meta_file = None

if source_choice == "Default paths (if present)":
    detected_reviews = DEFAULT_REVIEWS_PATH if file_exists(DEFAULT_REVIEWS_PATH) else None
    detected_meta = DEFAULT_META_PATH if file_exists(DEFAULT_META_PATH) else None
    if detected_reviews is None and file_exists("/mnt/data/Digital_Music.jsonl"):
        detected_reviews = "/mnt/data/Digital_Music.jsonl"
    if detected_meta is None and file_exists("/mnt/data/meta_Digital_Music.jsonl"):
        detected_meta = "/mnt/data/meta_Digital_Music.jsonl"

    if detected_reviews is None or detected_meta is None:
        st.info("Default files not found. Please upload both review and meta files in the section below.")
    else:
        st.success(f"Using reviews: {detected_reviews}\n\nUsing meta: {detected_meta}")
        reviews_file = detected_reviews
        meta_file = detected_meta

if source_choice == "Upload files (.jsonl)" or (reviews_file is None or meta_file is None):
    c1, c2 = st.columns(2)
    with c1:
        up_reviews = st.file_uploader("Upload Reviews JSONL", type=["jsonl"], key="reviews")
    with c2:
        up_meta = st.file_uploader("Upload Meta JSONL", type=["jsonl"], key="meta")

    if up_reviews is not None and up_meta is not None:
        reviews_file = io.StringIO(up_reviews.getvalue().decode("utf-8"))
        meta_file = io.StringIO(up_meta.getvalue().decode("utf-8"))

if reviews_file is None or meta_file is None:
    st.stop()

df, meta, merged = load_and_clean_data(
    reviews_file, meta_file,
    min_user_interactions=min_user_interactions,
    min_item_interactions=min_item_interactions,
)

if df.empty or meta.empty:
    st.warning("No data after cleaning. Try lowering the filtering thresholds or verify file formats.")
    st.stop()

pop_df = popularity_table(df, meta, min_votes=min_votes_for_pop)
vectorizer, X_content, nn_content = build_content_model(meta)
R, nn_collab, svd, X_latent, item_index = build_collab_matrices(df)
tab_pop, tab_content, tab_collab, tab_hybrid = st.tabs([" Popular", "Content-based", "Collaborative", "Hybrid"])

with tab_pop:
    st.subheader("Top Popular Products (Bayesian weighted)")
    st.dataframe(pop_df.head(200), use_container_width=True)

with st.sidebar:
    st.markdown("---")
    st.subheader("Pick a Product")
    if pop_df.empty:
        if "asin" in meta.columns:
            options = meta["asin"].tolist()
        else:
            st.warning("No 'asin' column found in meta data. Please check your meta file format.")
            options = []
    else:
        top_pop_list = pop_df.head(200)[["asin", "title"]].fillna("")
        options = [f'{a} · {t[:80]}' if isinstance(t, str) else a for a,t in top_pop_list.values]
    selected = st.selectbox("Select an item (ASIN · Title)", options) if len(options) else None

def _extract_asin(option: Optional[str]) -> Optional[str]:
    if option is None:
        return None
    if "·" in option:
        return option.split("·", 1)[0].strip()
    return option.split()[0].strip()
# ...existing code...
    if "asin" not in meta.columns:
        st.write("Meta columns:", list(meta.columns))  # Add this line for debugging
        raise ValueError("Meta file must contain an 'asin' column (or equivalent). Please check your file.")
# ...existing code...

selected_asin = _extract_asin(selected) if selected else None
selected_title = None
if selected_asin:
    tt = meta.loc[meta["asin"] == selected_asin, "title"]
    selected_title = tt.iloc[0] if len(tt) else ""

with tab_content:
    st.subheader("Content-based Recommendations (TF‑IDF + cosine)")
    if selected_asin is None:
        st.info("Select a product in the sidebar to see content-based recommendations.")
    else:
        st.caption(f"Selected: *{selected_asin} — {selected_title}*")
        if vectorizer is None or nn_content is None:
            st.warning("Content model unavailable.")
        else:
            c_recs = content_recommend(selected_asin, meta, vectorizer, X_content, nn_content, k=top_k)
            # attach titles + popularity for context
            c_recs = c_recs.merge(meta[["asin", "title"]], on="asin", how="left")
            c_recs = c_recs.merge(pop_df[["asin", "v", "WR"]], on="asin", how="left")
            st.dataframe(c_recs[["asin", "title", "score", "v", "WR"]], use_container_width=True)

with tab_collab:
    st.subheader("Collaborative Recommendations (item‑based cosine + SVD blend)")
    if selected_asin is None:
        st.info("Select a product in the sidebar to see collaborative recommendations.")
    else:
        st.caption(f"Selected: *{selected_asin} — {selected_title}*")
        if R is None or nn_collab is None:
            st.warning("Collaborative model unavailable.")
        else:
            cb_recs = collab_recommend(selected_asin, df, R, nn_collab, svd, X_latent, item_index, k=top_k)
            cb_recs = cb_recs.merge(meta[["asin", "title"]], on="asin", how="left")
            cb_recs = cb_recs.merge(pop_df[["asin", "v", "WR"]], on="asin", how="left")
            st.dataframe(cb_recs[["asin", "title", "score", "v", "WR"]], use_container_width=True)

with tab_hybrid:
    st.subheader("Hybrid Recommendations (weighted blend)")
    if selected_asin is None:
        st.info("Select a product in the sidebar to see hybrid recommendations.")
    else:
        c_recs = content_recommend(selected_asin, meta, vectorizer, X_content, nn_content, k=top_k*3) if vectorizer is not None else pd.DataFrame(columns=["asin","score"])
        cb_recs = collab_recommend(selected_asin, df, R, nn_collab, svd, X_latent, item_index, k=top_k*3) if R is not None else pd.DataFrame(columns=["asin","score"])

        hybrid = hybrid_recommend(
            selected_asin, c_recs, cb_recs, pop_df,
            w_content=w_content, w_collab=w_collab, w_pop=w_pop, k=top_k
        )
        hybrid = hybrid.merge(meta[["asin", "title"]], on="asin", how="left")
        hybrid = hybrid.merge(pop_df[["asin", "v", "WR"]], on="asin", how="left")
        st.dataframe(hybrid[["asin", "title", "hybrid_score", "v", "WR"]], use_container_width=True)
with st.expander("Notes & Data Health Checks"):
    st.markdown("<!-- Data health checks or notes can go here. -->")