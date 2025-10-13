from __future__ import annotations
from typing import List, Dict, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from .data_io import load_client_excel, load_best_dataset
from .features import compute_basic_stats, build_feature_row

class InferenceEngine:
    def __init__(self, models_dir: str, excel_path: str):
        self.models_dir = Path(models_dir)
        self.excel_path = excel_path
        self._load_artifacts()

    def _load_artifacts(self):
        self.kmeans = load(self.models_dir/'kmeans.pkl')
        self.Uf = np.load(self.models_dir/'svd_user.npy', allow_pickle=False)
        self.If = np.load(self.models_dir/'svd_item.npy', allow_pickle=False)
        self.stats = json.loads((self.models_dir/'stats.json').read_text())
        self.mappings = json.loads((self.models_dir/'mappings.json').read_text())
        # load regressor files map
        self.regressors = {}
        for p in self.models_dir.glob('regressor_cluster_*.pkl'):
            c = int(p.stem.split('_')[-1])
            self.regressors[c] = load(p)

    def _get_cluster_for_user(self, u: int, wide_df: pd.DataFrame) -> int:
        # wide_df should already be filled if using load_best_dataset
        vec = wide_df.iloc[u-1].values.reshape(1, -1)
        return int(self.kmeans.predict(vec)[0])

    def recommend(self, user_id: int, top_n: int = 20) -> List[Dict]:
        # ✅ CRITICAL: Use original dataset to identify unwatched movies
        # Load filled dataset for clustering/modeling but original for identifying unwatched movies
        ds_filled = load_best_dataset(self.excel_path)  # For clustering and model features
        ds_original = load_client_excel(self.excel_path)  # For identifying unwatched movies
        
        n_users = ds_original.users_df['user_id'].max()
        n_items = ds_original.movies_df['movie_id'].max()
        if not (1 <= user_id <= n_users):
            raise ValueError(f"user_id {user_id} out of range 1..{n_users}")

        # Assign cluster using filled dataset (needed for clustering)
        cluster_id = self._get_cluster_for_user(user_id, ds_filled.wide_df)
        model = self.regressors.get(cluster_id)
        if model is None:
            # fallback: choose any existing regressor
            model = next(iter(self.regressors.values()))

        # ✅ CRITICAL: Find candidate movies from ORIGINAL dataset (unwatched movies)
        user_row_original = ds_original.wide_df.iloc[user_id-1]
        seen = set(user_row_original[user_row_original.notna()].index.map(ds_original.movie_map.get))
        candidates = [m for m in range(1, n_items+1) if m not in seen]

        # Predict scores
        preds = []
        for m in candidates:
            feats, names = build_feature_row(user_id, m, self.stats, self.Uf, self.If)
            score = float(model.predict([feats])[0])
            preds.append((m, score))

        preds.sort(key=lambda x: x[1], reverse=True)
        results = []
        for m, s in preds[:top_n]:
            title = next((r['title'] for r in self.mappings['movies'] if r['movie_id'] == m), f'movie_{m}')
            results.append({'movie_id': m, 'title': title})
        return results
