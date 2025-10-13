from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil, os, json
from typing import Optional
import pandas as pd
from pydantic import BaseModel
from recommender.train import train_all
from recommender.infer import InferenceEngine
from recommender.data_io import load_client_excel, load_best_dataset, get_user_id_to_row_mapping
from recommender.impute import predict_one

app = FastAPI(title="Movie Recommender (Client)", version="1.0.0")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------- Config for prediction endpoint ----------
EXCEL_PATH = os.getenv("RATINGRING_CLIENT_EXCEL", "data/client_ratings.xlsx")
DEFAULT_K = int(os.getenv("RATINGRING_IMPUTE_K", "5"))
RANDOM_STATE = int(os.getenv("RATINGRING_RANDOM_STATE", "42"))

# ---------- App state for prediction ----------
class _State:
    wide_df = None
    users_df = None
    movies_df = None
    user_id_to_row = None
    movie_lookup = None  # maps id/title -> column name in wide_df

STATE = _State()

# ---------- Request/Response models ----------
class PredictRequest(BaseModel):
    user_id: int
    movie_id: int | None = None
    movie_title: str | None = None
    # k: int | None = None

class PredictResponse(BaseModel):
    movie_id: int | None = None
    movie_col: str
    value: float

@app.on_event("startup")
def _startup_load_excel():
    p = Path(EXCEL_PATH)
    if not p.exists():
        print(f"[startup] Client Excel not found at: {p.resolve()}. Prediction endpoint will be disabled until data is uploaded.")
        return

    # Use the best available dataset (filled if exists, otherwise original)
    ds = load_best_dataset(str(p))
    STATE.wide_df = ds.wide_df
    STATE.users_df = ds.users_df  # columns: user_id, maybe name/other
    STATE.movies_df = ds.movies_df  # columns: movie_id, title, etc.
    
    filled_path = p.parent / "filled_dataset.xlsx"
    if filled_path.exists():
        print(f"[startup] Using filled dataset: {filled_path}")
    else:
        print(f"[startup] Using original dataset (no filled dataset found yet)")
    

    # --- Build resolvers using proper user_id mapping ---
    STATE.user_id_to_row = get_user_id_to_row_mapping(STATE.users_df)

    # Map movie identifiers to wide_df column names.
    # Try exact column match on movie_id and title; otherwise assume titles are columns.
    colset = set(STATE.wide_df.columns.tolist())

    movie_map = {}

    if "movie_id" in STATE.movies_df.columns:
        for _, r in STATE.movies_df.iterrows():
            mid = int(r["movie_id"])
            title = str(r.get("title", mid))
            # Prefer exact column name match (id or title)
            if str(mid) in colset:
                movie_map[mid] = str(mid)
            elif title in colset:
                movie_map[mid] = title
            # Also allow lookup by title
            movie_map[title] = movie_map.get(mid, title if title in colset else None)

    else:
        # If you only have titles, map them directly
        for c in STATE.wide_df.columns:
            movie_map[c] = c

    STATE.movie_lookup = movie_map

    # Sanity log
    print(f"[startup] Loaded Excel: users={len(STATE.users_df)} movies={len(STATE.movies_df)} wide={STATE.wide_df.shape}")

# ---------- Resolver helpers ----------
def _resolve_user_row(user_id: int) -> int:
    if user_id not in STATE.user_id_to_row:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")
    return STATE.user_id_to_row[user_id]

def _resolve_movie_col(movie_id: int | None, movie_title: str | None) -> str:
    if movie_id is None and (movie_title is None or not movie_title.strip()):
        raise HTTPException(status_code=400, detail="Provide either movie_id or movie_title")

    # Prefer id if present
    if movie_id is not None:
        col = STATE.movie_lookup.get(int(movie_id))
        if col is not None:
            return col

    if movie_title is not None:
        col = STATE.movie_lookup.get(movie_title) or (movie_title if movie_title in STATE.wide_df.columns else None)
        if col is not None:
            return col

    raise HTTPException(status_code=404, detail="Movie not found in ratings columns")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
async def train_endpoint():
    # Use default Excel path
    dest = DATA_DIR / "client_ratings.xlsx"
    if not dest.exists():
        raise HTTPException(status_code=400, detail=f"Default ratings file not found at {dest}")
    
    # Train
    meta = train_all(str(dest), str(MODELS_DIR))
    
    # Refresh state for prediction endpoint
    _startup_load_excel()
    
    return JSONResponse({"message": "trained", "meta": meta})

# @app.post("/train")
# async def train_endpoint(file: UploadFile = File(...)):
#     # Save uploaded Excel
#     dest = DATA_DIR / "client_ratings.xlsx"
#     with dest.open("wb") as f:
#         shutil.copyfileobj(file.file, f)
#     # Train
#     meta = train_all(str(dest), str(MODELS_DIR))
    
#     # Refresh state for prediction endpoint
#     _startup_load_excel()
    
#     return JSONResponse({"message": "trained", "meta": meta})

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_n: int = Query(20, ge=1, le=100)):
    # Check if we have any data
    if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="No trained models found. Train first via /train.")
    
    if not DATA_DIR.exists():
        raise HTTPException(status_code=400, detail="No data directory found. Train first via /train.")
    
    # ✅ ALWAYS use original client_ratings.xlsx for recommendations
    # This ensures we identify truly unwatched movies (NaN in original data)
    client_path = DATA_DIR / "client_ratings.xlsx"
    
    if client_path.exists():
        dataset_path = str(client_path)
        print(f"[DEBUG] Using original dataset for recommendations: {dataset_path}")
    else:
        raise HTTPException(status_code=400, detail="No original dataset found. Train first via /train.")
    
    print(f"[DEBUG] Loading dataset from: {dataset_path}")
    
    # ✅ Pass the original client ratings file (not filled dataset)
    engine = InferenceEngine(str(MODELS_DIR), dataset_path)
    try:
        recs = engine.recommend(user_id=user_id, top_n=top_n)
        print(f"[DEBUG] Found {len(recs)} recommendations for user {user_id}")
        
        # Debug logging
        if not recs:
            print(f"[DEBUG] No recommendations found for user_id={user_id}")
            # Let's check if user exists - ✅ FIXED: Use client_ratings.xlsx file path
            try:
                ds = load_client_excel(dataset_path)  # Use the same file path
                user_mapping = get_user_id_to_row_mapping(ds.users_df)
                
                if user_id not in user_mapping:
                    available_users = list(user_mapping.keys())
                    raise HTTPException(
                        status_code=404, 
                        detail=f"User {user_id} not found. Available users: {available_users[:10]}..."
                    )
                else:
                    # User exists but no recommendations - let's debug why
                    user_row = user_mapping[user_id]
                    user_ratings = ds.wide_df.iloc[user_row]
                    unwatched_movies = user_ratings.index[user_ratings.isna()].tolist()
                    
                    print(f"[DEBUG] User {user_id} exists at row {user_row}")
                    print(f"[DEBUG] User has {len(user_ratings.dropna())} rated movies")
                    print(f"[DEBUG] User has {len(unwatched_movies)} unwatched movies")
                    print(f"[DEBUG] Sample unwatched: {unwatched_movies[:5]}")
                    print(f"[DEBUG] Models directory contents: {list(MODELS_DIR.glob('*.pkl'))}")
                    
            except Exception as e:
                print(f"[DEBUG] Error checking user existence: {e}")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[DEBUG] Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
    
    return {"user_id": user_id, "top_n": top_n, "items": recs}



@app.post("/predict-rating", response_model=PredictResponse)
def predict_rating(req: PredictRequest):
    if STATE.wide_df is None:
        raise HTTPException(status_code=500, detail="Ratings matrix not loaded. Upload data via /train first.")

    user_row = _resolve_user_row(req.user_id)
    movie_col = _resolve_movie_col(req.movie_id, req.movie_title)

    k = 5

    result = predict_one(
        wide=STATE.wide_df,
        user_row=user_row,
        item_col=movie_col,
        k=k,
        random_state=RANDOM_STATE
    )

    return PredictResponse(
        movie_id=req.movie_id,
        movie_col=movie_col,
        value=round(float(result["value"]), 1)
    )

# @app.get("/debug/data-cleaning")
# def debug_data_cleaning():
#     """Debug endpoint to check data cleaning process"""
#     try:
#         client_path = DATA_DIR / "client_ratings.xlsx"
#         if not client_path.exists():
#             return {"error": "No client_ratings.xlsx found"}
        
#         # Load raw data
#         raw_df = pd.read_excel(client_path)
        
#         # Apply cleaning
#         ds = load_client_excel(str(client_path))
        
#         return {
#             "raw_columns": list(raw_df.columns),
#             "raw_column_count": len(raw_df.columns),
#             "cleaned_columns": list(ds.wide_df.columns),
#             "cleaned_column_count": len(ds.wide_df.columns),
#             "filtered_out": [col for col in raw_df.columns if col not in ds.wide_df.columns],
#             "wide_shape": list(ds.wide_df.shape),
#             "sample_ratings_columns": list(ds.wide_df.columns)[:10]
#         }
        
#     except Exception as e:
#         return {"error": str(e)}

# @app.get("/debug/user/{user_id}")
# def debug_user_info(user_id: int):
#     """Debug endpoint to check user information"""
#     try:
#         client_path = DATA_DIR / "client_ratings.xlsx"
#         if not client_path.exists():
#             return {"error": "No client_ratings.xlsx found"}
            
#         ds = load_client_excel(str(client_path))
#         user_mapping = get_user_id_to_row_mapping(ds.users_df)
        
#         if user_id not in user_mapping:
#             available_users = list(user_mapping.keys())
#             return {
#                 "user_found": False,
#                 "user_id": user_id,
#                 "available_users": available_users[:20],
#                 "total_users": len(available_users)
#             }
        
#         user_row = user_mapping[user_id]
#         user_ratings = ds.wide_df.iloc[user_row]
#         unwatched_movies = user_ratings.index[user_ratings.isna()].tolist()
#         watched_movies = user_ratings.index[user_ratings.notna()].tolist()
        
#         # Check if models exist
#         model_files = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
        
#         return {
#             "user_found": True,
#             "user_id": user_id,
#             "user_row": user_row,
#             "total_movies": len(user_ratings),
#             "watched_movies": len(watched_movies),
#             "unwatched_movies": len(unwatched_movies),
#             "unwatched_sample": unwatched_movies[:10],
#             "watched_sample": watched_movies[:10],
#             "avg_rating": float(user_ratings.mean()) if len(watched_movies) > 0 else None,
#             "models_exist": len(model_files) > 0,
#             "model_files": [f.name for f in model_files]
#         }
        
#     except Exception as e:
#         return {"error": str(e)}
