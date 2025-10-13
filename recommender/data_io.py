from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

NON_RATING_COLS_DEFAULT = [
    'Enter name ', 'Age', 'Select Interested Movie Genres', 'Select Interested Movie Genres  ', 
    'Timestamp', 'Mobile no', 'user_id'
]

def is_non_rating_column(col_name: str) -> bool:
    """
    Check if a column should be excluded from rating columns.
    Handles various patterns of non-rating columns.
    """
    col_name = str(col_name).strip()
    
    # Exact matches
    if col_name in NON_RATING_COLS_DEFAULT:
        return True
    
    # Pattern matches
    patterns_to_exclude = [
        'unnamed:',  # Unnamed columns (e.g., 'Unnamed: 8', 'Unnamed: 51')
        'enter name',  # Name variations
        'age',
        'timestamp',
        'mobile',
        'phone',
        'email',
        'genre',
        'select interested',
        'gender'  # Gender column
    ]
    
    col_lower = col_name.lower()
    for pattern in patterns_to_exclude:
        if pattern in col_lower:
            return True
    
    # Exclude columns that are mostly empty or contain non-numeric data
    return False

@dataclass
class Dataset:
    users_df: pd.DataFrame       # columns: user_id, name (optional), ...
    movies_df: pd.DataFrame      # columns: movie_id, title
    ratings_df: pd.DataFrame     # columns: user_id, movie_id, rating
    wide_df: pd.DataFrame        # cleaned wide format (users x movies numeric)
    user_map: Dict[str, int]     # original name/id -> user_id
    movie_map: Dict[str, int]    # original column title -> movie_id

def clean_client_wide(df: pd.DataFrame, non_rating_cols: List[str] | None = None) -> Tuple[pd.DataFrame, List[str], pd.Series | None]:
    """Client-style cleaning, adapted from client's code.
    - Extract user_id column if present
    - Drop known non-rating columns (if present)
    - Replace 'not watch' with NaN
    - Coerce remaining ratings to numeric
    Returns (cleaned_df, rating_columns, user_id_series)
    """
    # Extract user_id if present before dropping non-rating columns
    user_id_series = None
    if 'user_id' in df.columns:
        user_id_series = df['user_id'].copy()
    
    # Filter out non-rating columns using improved logic
    rating_cols = []
    data_cleaned = df.copy()
    
    for col in df.columns:
        if is_non_rating_column(col):
            # Drop non-rating columns
            data_cleaned = data_cleaned.drop(columns=[col], errors='ignore')
        else:
            # Test if column contains mostly numeric rating data
            test_col = df[col].replace('not watch', np.nan)
            numeric_col = pd.to_numeric(test_col, errors='coerce')
            
            # Keep column if it has some valid numeric ratings
            if numeric_col.notna().sum() > 0:
                rating_cols.append(col)
            else:
                # Drop columns with no valid ratings
                data_cleaned = data_cleaned.drop(columns=[col], errors='ignore')
    
    # Clean the remaining rating columns
    data_cleaned = data_cleaned.replace('not watch', np.nan)
    for col in rating_cols:
        if col in data_cleaned.columns:
            data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
    
    # Final list of rating columns (after dropping)
    final_rating_cols = [col for col in rating_cols if col in data_cleaned.columns]
    
    return data_cleaned, final_rating_cols, user_id_series

def build_id_maps(wide_df: pd.DataFrame, original_df: pd.DataFrame, user_id_series: pd.Series | None = None) -> Tuple[dict, dict, List[int]]:
    # Build user IDs: prefer explicit user_id column, then names, then auto-generate
    if user_id_series is not None:
        # Use explicit user_id values from Excel
        user_ids = user_id_series.fillna(0).astype(int).tolist()
        # Ensure no zeros (replace with auto-generated if needed)
        for i, uid in enumerate(user_ids):
            if uid <= 0:
                user_ids[i] = max(user_ids) + 1 if user_ids else i + 1
        user_map = {f'row_{i}': uid for i, uid in enumerate(user_ids)}
        
    elif 'Enter name ' in original_df.columns:
        # Use names as identifiers but generate numeric IDs
        user_names = original_df['Enter name '].fillna('').astype(str).str.strip()
        user_ids = list(range(1, len(user_names) + 1))
        user_map = {name if name else f'user_{i}': uid for i, (name, uid) in enumerate(zip(user_names, user_ids))}
        
    else:
        # Auto-generate sequential IDs
        user_ids = list(range(1, len(wide_df) + 1))
        user_map = {f'user_{i}': uid for i, uid in enumerate(user_ids)}
    
    # Movie map from remaining columns
    movie_map = {title: j for j, title in enumerate(wide_df.columns, start=1)}
    
    return user_map, movie_map, user_ids

def wide_to_long(wide_df: pd.DataFrame, user_map: dict, movie_map: dict, user_ids: List[int]) -> pd.DataFrame:
    # melt wide into long ratings
    tmp = wide_df.copy()
    tmp['__row__'] = range(len(tmp))  # 0-based for indexing
    long = tmp.melt(id_vars='__row__', var_name='movie_title', value_name='rating')
    long = long.dropna(subset=['rating'])
    # map row index to actual user_id
    long['user_id'] = long['__row__'].map(lambda i: user_ids[i])
    long['movie_id'] = long['movie_title'].map(movie_map)
    
    # Drop rows where movie_id mapping failed (NaN values)
    long = long.dropna(subset=['movie_id'])
    
    # Convert to proper types
    long['user_id'] = long['user_id'].astype(int)
    long['movie_id'] = long['movie_id'].astype(int) 
    long['rating'] = long['rating'].astype(float)
    
    return long[['user_id','movie_id','rating']]

def load_client_excel(path: str) -> Dataset:
    try:
        raw = pd.read_excel(path)
        wide, rating_cols, user_id_series = clean_client_wide(raw)
        
        # Debug info
        print(f"[DEBUG] Raw columns: {len(raw.columns)}")
        print(f"[DEBUG] Clean columns: {len(wide.columns)}")
        print(f"[DEBUG] Rating columns: {len(rating_cols)}")
        
        user_map, movie_map, user_ids = build_id_maps(wide, raw, user_id_series)

        users_df = pd.DataFrame({'user_id': user_ids})
        if 'Enter name ' in raw.columns:
            users_df['name'] = raw['Enter name '].fillna('').astype(str)

        # Create movies_df with proper error handling
        movies_data = []
        for movie_title in wide.columns:
            if movie_title in movie_map:
                movies_data.append({
                    'movie_id': movie_map[movie_title],
                    'title': movie_title
                })
        
        movies_df = pd.DataFrame(movies_data)
        
        # Convert wide to long with better error handling
        ratings_df = wide_to_long(wide, user_map, movie_map, user_ids)
        
        print(f"[DEBUG] Final dataset - Users: {len(users_df)}, Movies: {len(movies_df)}, Ratings: {len(ratings_df)}")

        return Dataset(users_df=users_df, movies_df=movies_df, ratings_df=ratings_df,
                       wide_df=wide, user_map=user_map, movie_map=movie_map)
                       
    except Exception as e:
        print(f"[ERROR] Failed to load Excel: {e}")
        raise

def load_best_dataset(client_path: str) -> Dataset:
    """
    Load the best available dataset:
    1. If filled_dataset.xlsx exists in same directory as client_path, use it
    2. Otherwise, load the original client dataset
    """
    client_path = Path(client_path)
    filled_path = client_path.parent / "filled_dataset.xlsx"
    
    if filled_path.exists():
        # Load original for metadata, then replace wide_df with filled data
        ds = load_client_excel(str(client_path))
        filled_wide = pd.read_excel(filled_path)
        
        # Replace the wide_df with filled data and regenerate ratings_df
        ds.wide_df = filled_wide
        user_ids = ds.users_df['user_id'].tolist()  # Get the user_ids from original load
        ds.ratings_df = wide_to_long(filled_wide, ds.user_map, ds.movie_map, user_ids)
        
        return ds
    else:
        # Use original dataset
        return load_client_excel(str(client_path))

def get_user_id_to_row_mapping(users_df: pd.DataFrame) -> dict:
    """
    Create a mapping from user_id to row index for API lookups.
    This handles both synthetic (1,2,3...) and explicit user IDs from Excel.
    """
    return {int(uid): idx for idx, uid in enumerate(users_df['user_id'].tolist())}
