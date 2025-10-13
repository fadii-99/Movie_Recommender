# Movie Recommender (Client-Focused, Cluster + ML)

This project implements a **user-personalized recommender** that:

- **Reads the client's XLSX** (wide format: each row is a user; many movie columns as ratings)
- **Cleans** using the client's approach
- **Clusters users (KMeans)** and **trains per-cluster regressors** (scikit-learn `HistGradientBoostingRegressor`)
- **Recommends movies for a given `user_id`** via FastAPI

> **Note:** We intentionally ignore the XLSX files from the other project. Only the **client's data format** is supported here.

## Quick start

### 1) Install

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Train

Assuming your client Excel is at `data/ratings.xlsx`:

python -m scripts.train_pipeline --excel data/client_ratings.xlsx --models_dir models/

### 3) Serve API

```bash
uvicorn app.main:app --reload
```

### 4) Get recommendations

```
GET /recommendations/{user_id}?top_n=20

curl -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d '{"user_id": 123}'

```

## Expected XLSX format (client-style)

- One sheet where **rows = users**.
- Columns may include metadata like `Enter name `, `Age`, `Select Interested Movie Genres`, etc.
- **Many movie columns** whose headers are **movie titles**; cells contain numeric ratings (or `not watch` / blank for missing).

We will:

- Drop non-rating columns
- Convert `"not watch"` → NaN
- Coerce remaining cells to numeric

## Artifacts

Training writes into `models/`:

- `kmeans.pkl` — user clustering model
- `regressor_cluster_{k}.pkl` — per-cluster regressors
- `svd_user.npy` / `svd_item.npy` / `svd_meta.json` — latent factors for users/items
- `stats.json` — global mean, user/item means & counts
- `mappings.json` — user and movie id maps, original headers
- `config.json` — config used for training

## Why SVD + Gradient Boosting?

We don’t have extra movie metadata here, so we derive **latent factors** from the rating matrix (SVD), then train a **per‑cluster regressor** on:

- global/user/movie bias features
- user/item activity counts
- latent user/item vectors

This yields personalized predictions while staying lightweight and easy to operate.
