from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def build_sparse(ratings_df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    rows = ratings_df['user_id'].values - 1
    cols = ratings_df['movie_id'].values - 1
    data = ratings_df['rating'].values.astype(float)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

def svd_factors(R: csr_matrix, n_components: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    # Center by global mean for stability
    from sklearn.utils.extmath import randomized_svd
    mask = R.copy()
    mask.data = np.ones_like(mask.data)
    total = R.data.sum()
    cnt = mask.data.sum()
    gmean = total / cnt if cnt > 0 else 0.0
    # Fill missing as zeros after centering
    dense = R.toarray()
    centered = dense - gmean
    U, S, Vt = randomized_svd(centered, n_components=min(n_components, min(centered.shape)-1), random_state=random_state)
    user_factors = U * S
    item_factors = Vt.T
    return user_factors, item_factors
