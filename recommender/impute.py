# recommender/impute.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def _initial_fill_for_clustering(wide: pd.DataFrame) -> pd.DataFrame:
    # Fill NaN with per-item mean (then global mean) ONLY for KMeans input
    work = wide.copy()
    item_means = work.mean(axis=0, skipna=True)
    filled = work.fillna(item_means)
    g = np.nanmean(work.values.astype(float))
    return filled.fillna(g)

def _rating_bounds(wide: pd.DataFrame) -> Tuple[float, float]:
    vals = wide.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0, 5.0
    return float(np.nanmin(vals)), float(np.nanmax(vals))

def impute_wide_cluster_knn(
    wide: pd.DataFrame, k: int = 5, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill NaNs in `wide` using cluster-aware KNN-style averaging:
      1) KMeans on item-mean-filled matrix to get clusters of similar users
      2) For each missing (u,m), use cosine-sim weighted average of same-cluster users' ratings for m
         over co-rated overlap. Fallbacks: cluster movie mean -> global movie mean -> global mean.
    Returns: (filled_wide, user_clusters_df[user_id, cluster_id])
    """
    assert isinstance(wide, pd.DataFrame)
    work = wide.copy()
    filled_for_kmeans = _initial_fill_for_clustering(work)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(filled_for_kmeans.values)

    user_clusters = pd.DataFrame(
        {"user_idx": np.arange(len(work)), "cluster_id": labels}
    )
    user_clusters["user_id"] = user_clusters["user_idx"] + 1
    user_clusters = user_clusters[["user_id", "cluster_id"]]

    filled = work.copy()
    item_means_global = work.mean(axis=0, skipna=True)
    global_mean = float(np.nanmean(work.values))
    lo, hi = _rating_bounds(work)

    for u in range(len(work)):
        row = work.iloc[u]
        cluster = labels[u]
        cluster_idx = np.where(labels == cluster)[0]
        cluster_mat = work.iloc[cluster_idx].copy()

        missing = row.index[row.isna()]
        if len(missing) == 0:
            continue

        # Cosine similarities to others in cluster based on co-rated overlap
        sims = np.zeros(len(cluster_idx))
        for j, v in enumerate(cluster_idx):
            if v == u:
                sims[j] = 0.0
            else:
                a = row.values
                b = work.iloc[v].values
                mask = (~np.isnan(a)) & (~np.isnan(b))
                if mask.sum() == 0:
                    sims[j] = 0.0
                else:
                    sa = a[mask].reshape(1, -1)
                    sb = b[mask].reshape(1, -1)
                    sims[j] = float(cosine_similarity(sa, sb)[0, 0])

        for m in missing:
            rated_mask = cluster_mat[m].notna().values
            pred = np.nan

            if rated_mask.any():
                weights = sims[rated_mask]
                vals = cluster_mat[m].values[rated_mask]
                denom = weights.sum()
                if denom > 0:
                    pred = float(np.dot(weights, vals) / denom)
                else:
                    pred = float(cluster_mat[m].mean(skipna=True))

            if np.isnan(pred):
                if not np.isnan(item_means_global[m]):
                    pred = float(item_means_global[m])
                else:
                    pred = global_mean if not np.isnan(global_mean) else 0.0

            pred = max(lo, min(hi, pred))
            filled.at[filled.index[u], m] = round(pred, 1)

    return filled, user_clusters

def save_imputation_artifacts(filled: pd.DataFrame, user_clusters: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filled.to_excel(out_dir / "filled_dataset.xlsx", index=False)
    user_clusters.to_csv(out_dir / "user_clusters.csv", index=False)


# ---- Single prediction functions for real-time endpoint ----

from typing import Optional, Dict, Any

def _single_prediction_cluster_knn(
    wide: pd.DataFrame,
    user_row: int,
    item_col: str,
    k: int = 5,
    random_state: int = 42
) -> float:
    """
    Predict a single missing rating at (user_row, item_col) using the same
    cluster-aware cosine-weighted neighbor logic used in impute_wide_cluster_knn.
    Does NOT mutate 'wide'.
    """
    work = wide  # alias, do not copy for performance – we won't assign
    filled_for_kmeans = _initial_fill_for_clustering(work)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(filled_for_kmeans.values)

    row = work.iloc[user_row]
    cluster = labels[user_row]
    cluster_idx = np.where(labels == cluster)[0]
    cluster_mat = work.iloc[cluster_idx]

    # rating bounds for clipping
    lo, hi = _rating_bounds(work)

    # ----- cosine sims to cluster users on co-rated overlap -----
    sims = np.zeros(len(cluster_idx))
    a = row.values
    for j, v in enumerate(cluster_idx):
        if v == user_row:
            sims[j] = 0.0
            continue
        b = work.iloc[v].values
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if mask.sum() == 0:
            sims[j] = 0.0
        else:
            sa = a[mask].reshape(1, -1)
            sb = b[mask].reshape(1, -1)
            sims[j] = float(cosine_similarity(sa, sb)[0, 0])

    # ----- weighted avg over users in cluster who rated this item -----
    rated_mask = cluster_mat[item_col].notna().values
    pred = np.nan
    if rated_mask.any():
        weights = sims[rated_mask]
        vals = cluster_mat[item_col].values[rated_mask]
        denom = weights.sum()
        if denom > 0:
            pred = float(np.dot(weights, vals) / denom)
        else:
            pred = float(cluster_mat[item_col].mean(skipna=True))

    # ----- fallbacks: global item mean -> global mean -----
    if np.isnan(pred):
        item_mean = work[item_col].mean(skipna=True)
        if not np.isnan(item_mean):
            pred = float(item_mean)
        else:
            global_mean = float(np.nanmean(work.values))
            pred = global_mean if not np.isnan(global_mean) else 0.0

    return round(max(lo, min(hi, pred)), 1)


def predict_one(
    wide: pd.DataFrame,
    user_row: int,
    item_col: str,
    k: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Public helper to predict a rating for (user_row, item_col).
    If the cell already has an observed rating, we return it and mark observed=True.
    Otherwise we compute a predictive value and observed=False.
    """
    val = wide.iloc[user_row][item_col]
    if pd.notna(val):
        lo, hi = _rating_bounds(wide)
        return {
            "observed": True,
            "value": round(float(max(lo, min(hi, float(val)))), 1),
            "method": "observed"
        }

    pred = _single_prediction_cluster_knn(
        wide=wide,
        user_row=user_row,
        item_col=item_col,
        k=k,
        random_state=random_state
    )
    return {
        "observed": False,
        "value": round(float(pred), 1),
        "method": "cluster_knn_cosine"
    }