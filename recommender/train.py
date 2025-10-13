# recommender/train.py
from __future__ import annotations
import json, sys, time, logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

from .data_io import load_client_excel, load_best_dataset
from .matrix import build_sparse, svd_factors
from .clustering import fit_kmeans_silhouette
from .features import compute_basic_stats, build_training_frame
from .impute import impute_wide_cluster_knn, save_imputation_artifacts


# ---------- logging ----------
LOGGER_NAME = "train"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)


def _make_xgb(n_estimators: int) -> XGBRegressor:
    """XGBoost config compatible with xgboost==2.x (no callbacks/early_stopping in fit)."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",   # switch to "gpu_hist" if you have CUDA
        n_jobs=0,
        random_state=42,
        eval_metric="rmse",   # set here (constructor)
    )


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _choose_n_estimators(X: np.ndarray, y: np.ndarray, seed: int) -> int:
    """
    Manual CV over a small grid of n_estimators.
    Avoids callbacks/early_stopping to stay compatible with XGB 2.x variants.
    """
    start = time.perf_counter()
    if len(X) < 20:
        logger.info("Dataset tiny (%d rows) → using default n_estimators=200", len(X))
        return 200

    grid: List[int] = [200, 400, 800]  # expand if needed: [200, 400, 600, 800, 1000]
    if len(X) >= 60:
        n_splits = 5
    elif len(X) >= 30:
        n_splits = 3
    else:
        n_splits = 2

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_rmse, best_n = None, None

    logger.info("CV: trying n_estimators grid %s with %d-fold CV", grid, n_splits)
    for n in grid:
        rmses = []
        fold_i = 0
        for tr_idx, va_idx in kf.split(X):
            fold_i += 1
            model = _make_xgb(n_estimators=n)
            model.fit(X[tr_idx], y[tr_idx], verbose=False)
            yhat = model.predict(X[va_idx])
            rmse = _rmse(y[va_idx], yhat)
            rmses.append(rmse)
            logger.info("  - n=%d | fold %d/%d RMSE=%.5f", n, fold_i, n_splits, rmse)
        mean_rmse = float(np.mean(rmses)) if rmses else float("inf")
        logger.info("  → n=%d | mean RMSE=%.5f", n, mean_rmse)
        if (best_rmse is None) or (mean_rmse < best_rmse):
            best_rmse, best_n = mean_rmse, n

    dur = time.perf_counter() - start
    logger.info("Selected n_estimators=%d (CV best RMSE=%.5f) in %.2fs", best_n or 400, best_rmse, dur)
    return int(best_n or 400)


def train_all(excel_path: str, models_dir: str, n_components: int = 20, random_state: int = 42) -> Dict:
    t0 = time.perf_counter()
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==== Training start ====")
    logger.info("Excel: %s", excel_path)
    logger.info("Models dir: %s", models_dir.resolve())

    # 1) Load dataset (prefer filled_dataset.xlsx if exists)
    step = time.perf_counter()
    out_dir = Path(excel_path).parent
    filled_path = out_dir / "filled_dataset.xlsx"
    
    if filled_path.exists():
        logger.info("Found existing filled dataset, using: %s", filled_path)
        # Load the filled dataset directly
        ds = load_client_excel(excel_path)  # Still need mappings and structure
        filled_wide = pd.read_excel(filled_path)
        
        # Convert filled wide to ratings format for training
        filled_long = []
        for u_idx, user_row in enumerate(filled_wide.iterrows()):
            user_id = u_idx + 1  # 1-based
            for m_col, rating in user_row[1].items():
                if pd.notna(rating):
                    movie_id = ds.movie_map[m_col]
                    filled_long.append({'user_id': user_id, 'movie_id': movie_id, 'rating': float(rating)})
        
        filled_ratings_df = pd.DataFrame(filled_long)
        logger.info("Using existing filled dataset: %d total ratings", len(filled_ratings_df))
        
    else:
        logger.info("No filled dataset found, creating from: %s", excel_path)
        ds = load_client_excel(excel_path)
        logger.info("Loaded Excel. Users=%d, Movies=%d, Ratings(rows)=%d",
                    ds.users_df["user_id"].max(), ds.movies_df["movie_id"].max(), len(ds.ratings_df))
        logger.info("Wide matrix shape: %s", tuple(ds.wide_df.shape))
        
        # --- Impute missing ratings (client deliverable) ---
        filled_wide, user_clusters = impute_wide_cluster_knn(ds.wide_df, k=5, random_state=random_state)
        save_imputation_artifacts(filled_wide, user_clusters, out_dir)
        logger.info("Imputation complete. Saved filled_dataset.xlsx and user_clusters.csv in %s", out_dir.resolve())
        
        # Convert filled wide back to long format for training
        filled_long = []
        for u_idx, user_row in enumerate(filled_wide.iterrows()):
            user_id = u_idx + 1  # 1-based
            for m_col, rating in user_row[1].items():
                if pd.notna(rating):  # Should all be filled now, but safety check
                    movie_id = ds.movie_map[m_col]
                    filled_long.append({'user_id': user_id, 'movie_id': movie_id, 'rating': float(rating)})
        
        filled_ratings_df = pd.DataFrame(filled_long)
        logger.info("Using newly created filled ratings for training: %d total ratings (was %d observed)", 
                    len(filled_ratings_df), len(ds.ratings_df))
    
    logger.info("Step 1 done in %.2fs", time.perf_counter() - step)
    # ---------------------------------------------------

    # 2) Build sparse rating matrix & SVD latent factors
    step = time.perf_counter()
    n_users = int(ds.users_df["user_id"].max())
    n_items = int(ds.movies_df["movie_id"].max())
    R = build_sparse(filled_ratings_df, n_users, n_items)
    nonzero = R.nnz
    density = nonzero / (R.shape[0] * R.shape[1]) if R.shape[0] and R.shape[1] else 0.0
    logger.info("Sparse matrix: shape=%s, nnz=%d, density=%.6f", tuple(R.shape), nonzero, density)
    Uf, If = svd_factors(R, n_components=n_components, random_state=random_state)
    logger.info("SVD done. n_components=%d | Uf=%s | If=%s", Uf.shape[1], tuple(Uf.shape), tuple(If.shape))
    logger.info("Step 2 done in %.2fs", time.perf_counter() - step)

    # 3) Cluster users on the filled wide matrix  
    step = time.perf_counter()
    X_users = filled_wide.values  # Use predictively filled data instead of zeros
    kmeans = fit_kmeans_silhouette(X_users, k_list=(3, 4, 5, 6, 7), random_state=random_state)
    logger.info("KMeans fitted. n_clusters=%d", getattr(kmeans, "n_clusters", -1))
    logger.info("Step 3 done in %.2fs", time.perf_counter() - step)

    # 4) Global/user/item stats (using filled ratings)
    step = time.perf_counter()
    stats = compute_basic_stats(filled_ratings_df)
    logger.info("Stats computed: global_mean=%.5f | users=%d | items=%d",
                stats["global_mean"], len(stats["user_mean"]), len(stats["item_mean"]))
    logger.info("Step 4 done in %.2fs", time.perf_counter() - step)

    # 5) Train XGB per cluster
    step = time.perf_counter()
    cluster_labels = kmeans.predict(X_users)
    users_df = ds.users_df.copy()
    users_df["cluster"] = cluster_labels
    reg_meta: Dict[str, dict] = {}
    clusters = sorted(np.unique(cluster_labels).tolist())
    logger.info("Training regressors per cluster: %s", clusters)

    for c in clusters:
        c_step = time.perf_counter()
        cluster_users = users_df.loc[users_df["cluster"] == c, "user_id"].tolist()
        rows_c = filled_ratings_df[filled_ratings_df["user_id"].isin(cluster_users)]
        logger.info("-- Cluster %s: users=%d | rating rows=%d", c, len(cluster_users), len(rows_c))
        if rows_c.empty:
            logger.info("   (skip: no rows)")
            continue

        Xc_df, yc = build_training_frame(rows_c, stats, Uf, If)
        Xc = Xc_df.values
        logger.info("   Feature matrix: X=%s | y=%s", tuple(Xc.shape), tuple(yc.shape))

        n_best = _choose_n_estimators(Xc, yc, seed=random_state)
        final_model = _make_xgb(n_estimators=n_best)
        final_model.fit(Xc, yc, verbose=False)

        path = models_dir / f"regressor_cluster_{c}.pkl"
        dump(final_model, path)
        reg_meta[str(c)] = {"rows": int(len(Xc)), "n_estimators": int(n_best)}
        logger.info("   Saved model: %s", path.name)
        logger.info("   Cluster %s done in %.2fs", c, time.perf_counter() - c_step)

    logger.info("Step 5 done in %.2fs", time.perf_counter() - step)

    # 6) Persist artifacts
    step = time.perf_counter()
    np.save(models_dir / "svd_user.npy", Uf)
    np.save(models_dir / "svd_item.npy", If)
    (models_dir / "svd_meta.json").write_text(json.dumps({"n_components": int(Uf.shape[1])}))
    dump(kmeans, models_dir / "kmeans.pkl")
    (models_dir / "stats.json").write_text(json.dumps(stats))
    (models_dir / "mappings.json").write_text(json.dumps({
        "user_map": ds.user_map,
        "movie_map": ds.movie_map,
        "movies": ds.movies_df.to_dict(orient="records"),
    }))
    (models_dir / "config.json").write_text(json.dumps({
        "excel_path": excel_path,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "model": "xgboost",
        "n_components": int(Uf.shape[1]),
    }))
    logger.info("Artifacts saved in %s", models_dir.resolve())
    logger.info("Step 6 done in %.2fs", time.perf_counter() - step)

    total = time.perf_counter() - t0
    summary = {"clusters": len(set(cluster_labels)), "regressors": reg_meta}
    logger.info("==== Training complete (%.2fs) ====", total)
    logger.info("Summary: %s", json.dumps(summary))
    return summary
