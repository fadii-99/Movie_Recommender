from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def fit_kmeans_silhouette(X: np.ndarray, k_list=(3,4,5,6,7), random_state: int = 42):
    best_k, best_score, best_model = None, -1.0, None
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        if sc > best_score:
            best_k, best_score, best_model = k, sc, km
    # Fallback to k=5 if silhouette failed
    if best_model is None:
        best_model = KMeans(n_clusters=5, random_state=random_state, n_init='auto').fit(X)
    return best_model
 
  

