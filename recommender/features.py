from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def compute_basic_stats(ratings_df: pd.DataFrame) -> Dict:
    gmean = float(ratings_df['rating'].mean()) if len(ratings_df) else 0.0
    user_mean = ratings_df.groupby('user_id')['rating'].mean().to_dict()
    item_mean = ratings_df.groupby('movie_id')['rating'].mean().to_dict()
    user_cnt = ratings_df.groupby('user_id')['rating'].size().to_dict()
    item_cnt = ratings_df.groupby('movie_id')['rating'].size().to_dict()
    return {'global_mean': gmean, 'user_mean': user_mean, 'item_mean': item_mean, 'user_cnt': user_cnt, 'item_cnt': item_cnt}

def build_feature_row(u: int, m: int, stats: Dict, Uf: np.ndarray, If: np.ndarray) -> Tuple[np.ndarray, list]:
    g = stats['global_mean']
    um = stats['user_mean'].get(u, g)
    im = stats['item_mean'].get(m, g)
    uc = stats['user_cnt'].get(u, 0)
    ic = stats['item_cnt'].get(m, 0)
    uvec = Uf[u-1] if 0 <= (u-1) < len(Uf) else np.zeros(Uf.shape[1])
    ivec = If[m-1] if 0 <= (m-1) < len(If) else np.zeros(If.shape[1])
    # features: biases, counts, dot and concatenations
    features = [
        g,
        um - g,
        im - g,
        uc,
        ic,
        float(np.dot(uvec, ivec))
    ]
    # concat latent vectors (limited size)
    features.extend(uvec.tolist())
    features.extend(ivec.tolist())
    names = (['global_mean','user_bias','item_bias','user_cnt','item_cnt','latent_dot'] +
             [f'u_f{i}' for i in range(len(uvec))] + [f'i_f{i}' for i in range(len(ivec))])
    return np.asarray(features, dtype=float), names

def build_training_frame(ratings_df: pd.DataFrame, stats: Dict, Uf: np.ndarray, If: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in ratings_df.itertuples(index=False):
        feats, names = build_feature_row(int(r.user_id), int(r.movie_id), stats, Uf, If)
        rows.append((feats, float(r.rating)))
    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=float)
    return pd.DataFrame(X, columns=names), y
