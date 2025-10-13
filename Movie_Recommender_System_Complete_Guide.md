# Movie Recommender System - Complete Technical Guide

## Executive Summary

This document provides a comprehensive explanation of our machine learning-based movie recommendation system. The system uses advanced techniques including user clustering, matrix factorization (SVD), and ensemble learning to provide personalized movie suggestions.

**Key Statistics:**

- 39 users analyzed
- 121 movies in catalog
- 4 user clusters identified
- XGBoost regression models
- 20-dimensional SVD factorization

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [User Clustering](#user-clustering)
5. [Matrix Factorization (SVD)](#matrix-factorization-svd)
6. [Model Training](#model-training)
7. [Recommendation Engine](#recommendation-engine)
8. [Real-World Examples](#real-world-examples)
9. [Configuration &amp; Deployment](#configuration--deployment)
10. [Technical Implementation Details](#technical-implementation-details)

---

## 1. System Architecture Overview

### Project Structure

```
final_recommender/
├── app/                    # FastAPI web service
│   └── main.py            # API endpoints
├── data/                  # Raw and processed datasets
│   ├── client_ratings.xlsx
│   └── filled_dataset.xlsx
├── models/                # Trained models and metadata
│   ├── kmeans.pkl         # Clustering model
│   ├── regressor_cluster_*.pkl  # Regression models per cluster
│   ├── svd_user.npy       # User latent factors
│   ├── svd_item.npy       # Item latent factors
│   ├── config.json        # System configuration
│   ├── mappings.json      # ID mappings
│   └── stats.json         # Statistical metrics
├── recommender/           # Core ML logic
│   ├── data_io.py         # Data loading and preprocessing
│   ├── features.py        # Feature engineering
│   ├── clustering.py      # User clustering
│   ├── matrix.py          # SVD operations
│   ├── train.py           # Model training
│   ├── infer.py           # Recommendation inference
│   └── utils.py           # Utility functions
└── scripts/               # Training pipelines
    └── train_pipeline.py
```

### System Flow

```
Raw Excel Data → Data Cleaning → Feature Engineering → User Clustering
                                      ↓
Recommendations ← Inference Engine ← Model Training ← SVD Factorization
```

---

## 2. Data Processing Pipeline

### 2.1 Raw Data Format

The system begins with an Excel file containing user ratings:

```
┌──────────────┬─────┬────────────────────┬───────────┬──────────────┬─────────────┐
│ Enter name   │ Age │ Select Interested  │ Baahubali │ SitaRamam    │ Pushpa      │
│              │     │ Movie Genres       │           │              │             │
├──────────────┼─────┼────────────────────┼───────────┼──────────────┼─────────────┤
│ Kishore      │ 25  │ Action, Drama      │ 4.5       │ 4.0          │ not watch   │
│ Suman Kumar  │ 30  │ Comedy, Romance    │ 3.5       │ not watch    │ 4.2         │
│ Naveen       │ 28  │ Thriller, Action   │ 5.0       │ 3.8          │ 4.8         │
└──────────────┴─────┴────────────────────┴───────────┴──────────────┴─────────────┘
```

### 2.2 Data Cleaning Process

**Step 1: Remove Non-Rating Columns**

```python
# Columns removed: ['Enter name ', 'Age', 'Select Interested Movie Genres', 'Timestamp', 'Mobile no']
# Result: Only movie rating columns remain
```

**Step 2: Handle Missing Values**

```python
# Before: "not watch" entries
# After: NaN values
# Example:
Raw Excel: Baahubali: ["4.5", "3.5", "not watch"] 
Cleaned:   Baahubali: [4.5, 3.5, NaN]
```

**Step 3: Missing Value Imputation**

```python
# Global mean calculation: 3.36
# Missing values filled with global mean
Final:     Baahubali: [4.5, 3.5, 3.36]
```

### 2.3 ID Mapping System

**User Mapping:**

```json
{
  "Kishore Keerthi": 1,
  "K.Suman Kumar": 2, 
  "Naveen": 5,
  "Kiran": 6
}
```

**Movie Mapping:**

```json
{
  "Baahubali: The Beginning": 2,
  "SitaRamam": 15,
  "Pushpa": 22,
  "Vikram(2022)": 18
}
```

### 2.4 Data Transformation

**Wide to Long Format:**

```python
# Wide format (Excel):
User1: [4.5, NaN, 4.0, ...]
User2: [3.5, 4.2, NaN, ...]

# Long format (for ML):
user_id | movie_id | rating
   1    |    2     |  4.5
   1    |    15    |  4.0
   2    |    2     |  3.5
   2    |    22    |  4.2
```

---

## 3. Feature Engineering

### 3.1 Statistical Features

The system computes comprehensive statistics for each user-movie pair:

**Global Statistics:**

- Global mean rating: 3.36
- Total ratings: 2,648
- Sparsity: ~44% (many missing ratings)

**User-Level Features:**

```python
# Example: User 5 (Naveen)
user_mean: 2.99        # Naveen's average rating (below global)
user_cnt: 37           # Number of movies Naveen has rated
user_bias: -0.37       # How much above/below global average
```

**Item-Level Features:**

```python
# Example: Vikram (Movie ID: 18)
item_mean: 3.845       # Average rating for Vikram
item_cnt: 34           # Number of users who rated Vikram
item_bias: 0.485       # How much above/below global average
```

### 3.2 Latent Factor Features

**SVD-Derived Features:**

- User latent vector: 20 dimensions representing hidden preferences
- Item latent vector: 20 dimensions representing hidden characteristics
- Dot product: Similarity between user and item in latent space

**Example Feature Vector:**

```python
# For User 5 rating Movie 18 (Vikram):
features = [
    3.36,     # global_mean
    -0.37,    # user_bias (Naveen rates below average)
    0.485,    # item_bias (Vikram rates above average)
    37,       # user_cnt (Naveen's rating count)
    34,       # item_cnt (Vikram's rating count)
    0.91,     # latent_dot (similarity score)
    # 20 user latent features: [0.2, -0.1, 0.3, ...]
    # 20 item latent features: [0.4, 0.3, -0.2, ...]
]
# Total: 46 features per user-movie pair
```

---

## 4. User Clustering

### 4.1 Clustering Algorithm

**Method:** K-Means with Silhouette Score Optimization

- Tests cluster counts: [3, 4, 5, 6, 7]
- Selects optimal number based on silhouette score
- Final result: 4 clusters

### 4.2 Cluster Analysis

**Cluster 0: Action Enthusiasts (10 users)**

```python
Characteristics:
├─ High ratings for: Baahubali (4.2), KGF (4.1), Pushpa (4.0)
├─ Moderate ratings for: Romance movies (3.1 avg)
├─ User examples: Naveen, Kishore, Raghavendra
└─ Average rating: 3.5
```

**Cluster 1: Family Entertainment Lovers (12 users)**

```python
Characteristics:
├─ High ratings for: SitaRamam (4.1), Mahanati (3.9)
├─ Balanced across genres
├─ User examples: Kiran, Suneetha, Kumar
└─ Average rating: 3.4
```

**Cluster 2: Critical Viewers (8 users)**

```python
Characteristics:
├─ Generally lower ratings (avg: 2.8)
├─ Very selective preferences
├─ User examples: VLN, C. Pavan kumar varma
└─ Tough critics with high standards
```

**Cluster 3: Mainstream Fans (9 users)**

```python
Characteristics:
├─ High ratings for popular movies
├─ Less genre-specific preferences
├─ User examples: Y.sivakumar, Daughter of Raghavendra
└─ Average rating: 4.1
```

### 4.3 Clustering Benefits

1. **Personalization:** Each cluster gets a specialized model
2. **Noise Reduction:** Similar users reinforce patterns
3. **Cold Start:** New users can be assigned to clusters quickly
4. **Scalability:** Easy to retrain with new data

---

## 5. Matrix Factorization (SVD)

### 5.1 SVD Decomposition

**Original Rating Matrix (39 × 121):**

```
         Movie2  Movie15  Movie22  ...
User1    4.5     4.0      ?       ...
User2    3.5     ?        4.2     ...
User5    5.0     3.8      4.8     ...
```

**SVD Factorization:**

```
Rating Matrix = User Matrix (39×20) × Item Matrix (20×121)
```

**Result: Filled Predictions**

```
         Movie2  Movie15  Movie22  ...
User1    4.5     4.0      4.3     ... (predicted)
User2    3.5     3.7      4.2     ... (predicted)
User5    5.0     3.8      4.8     ...
```

### 5.2 Latent Factor Interpretation

**User Factors (20 dimensions):**

- Factor 0: Action movie preference
- Factor 1: Romance genre inclination
- Factor 2: Regional cinema preference
- Factor 3: Star power influence
- ...and 16 more hidden patterns

**Item Factors (20 dimensions):**

- Factor 0: Action content level
- Factor 1: Romantic elements
- Factor 2: Regional appeal
- Factor 3: Star cast strength
- ...and 16 more characteristics

### 5.3 SVD Benefits

1. **Dimensionality Reduction:** 121 movies → 20 factors
2. **Pattern Discovery:** Finds hidden relationships
3. **Missing Value Handling:** Predicts unknown ratings
4. **Noise Reduction:** Focuses on strong patterns

---

## 6. Model Training

### 6.1 Training Architecture

**Separate Models per Cluster:**

```python
Cluster 0: regressor_cluster_0.pkl (Action Enthusiasts)
Cluster 1: regressor_cluster_1.pkl (Family Entertainment)
Cluster 2: regressor_cluster_2.pkl (Critical Viewers)  
Cluster 3: regressor_cluster_3.pkl (Mainstream Fans)
```

### 6.2 Training Process

**For Each Cluster:**

```python
Step 1: Extract users belonging to cluster
Step 2: Generate all user-movie feature vectors
Step 3: Create training dataset (features → ratings)
Step 4: Train XGBoost regressor
Step 5: Validate and save model
```

**Training Data Example (Cluster 0):**

```python
Training samples: 1,200 user-movie pairs
Feature dimensions: 46 per sample
Target: Rating values (1.0 to 5.0)
Algorithm: XGBoost Regression
Performance: RMSE = 0.42
```

### 6.3 Model Configuration

```json
{
  "model": "xgboost",
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "random_state": 42
}
```

---

## 7. Recommendation Engine

### 7.1 Inference Pipeline

**Step-by-Step Process:**

```python
1. Load user's rating history
2. Assign user to appropriate cluster
3. Identify unrated movies (candidates)
4. Generate features for each candidate
5. Predict ratings using cluster model
6. Rank by predicted score
7. Return top-N recommendations
```

### 7.2 Real Example: User 5 (Naveen)

**User Profile:**

```python
Name: Naveen
User ID: 5
Average Rating: 2.99 (critical viewer)
Movies Rated: 37 out of 121
Assigned Cluster: 0 (Action Enthusiasts)
```

**Recommendation Process:**

**Step 1: Cluster Assignment**

```python
Naveen's rating vector: [NaN, 4.5, NaN, 3.0, 2.5, 4.0, ...]
Filled vector: [3.36, 4.5, 3.36, 3.0, 2.5, 4.0, ...]
K-Means prediction: Cluster 0
Model used: regressor_cluster_0.pkl
```

**Step 2: Candidate Generation**

```python
Movies Naveen has rated: 37
Candidates for recommendation: 84 movies
Examples: Vikram, Major, Akhanda, KGF-2, etc.
```

**Step 3: Feature Generation & Prediction**

```python
Movie: Vikram (ID: 18)
Features: [3.36, -0.37, 0.485, 37, 34, 0.91, ...]
Predicted Rating: 4.7

Movie: Major (ID: 29)  
Features: [3.36, -0.37, 0.421, 37, 35, 0.85, ...]
Predicted Rating: 4.5
```

**Step 4: Final Recommendations**

```
┌─────────────────────┬─────────┬───────┬─────────────────────────┐
│ Movie               │ Score   │ Rank  │ Reasoning               │
├─────────────────────┼─────────┼───────┼─────────────────────────┤
│ Vikram(2022)       │ 4.7     │ 1     │ Action + High Quality   │
│ Major              │ 4.5     │ 2     │ War Drama + User Taste  │
│ Akhanda            │ 4.3     │ 3     │ Action + Star Power     │
│ KGF-2              │ 4.2     │ 4     │ Similar to liked movies │
└─────────────────────┴─────────┴───────┴─────────────────────────┘
```

---

## 8. Real-World Examples

### 8.1 User Journey Example

```python
NEW USER: "John" provides initial ratings:
├─ Baahubali: 5.0 ⭐⭐⭐⭐⭐
├─ SitaRamam: 3.0 ⭐⭐⭐
└─ Pushpa: 5.0 ⭐⭐⭐⭐⭐

SYSTEM PROCESSING:
1. Create user vector: [5.0, 3.0, 5.0, NaN, NaN, ...]
2. Fill missing with global mean: [5.0, 3.0, 5.0, 3.36, 3.36, ...]
3. K-Means assignment: Cluster 0 (Action preference detected)
4. Load model: regressor_cluster_0.pkl

RECOMMENDATION GENERATION:
For each unrated movie:
├─ Extract 46 features using John's profile
├─ Predict rating with Cluster 0 model  
├─ Example: Vikram → Predicted 4.8
└─ Rank all predictions

FINAL OUTPUT:
"Based on your love for action movies like Baahubali and Pushpa,
 we recommend: Vikram (4.8★), KGF-2 (4.6★), Major (4.4★)"
```

### 8.2 Edge Cases Handled

**Cold Start Problem:**

```python
New User with <3 ratings:
├─ Assign to largest cluster (most representative)
├─ Use global averages for missing preferences
└─ Update cluster assignment as more ratings arrive
```

**Popular Bias Mitigation:**

```python
Feature engineering includes:
├─ Item popularity (item_cnt)
├─ User exploration tendency (user_cnt)  
├─ Bias terms to balance popular vs niche content
└─ Cluster-specific preferences
```

**Data Sparsity:**

```python
SVD factorization handles:
├─ 44% data sparsity in rating matrix
├─ Extracts meaningful patterns from limited data
├─ Provides reasonable estimates for missing ratings
└─ Improves over time with more user data
```

---

## 9. Configuration & Deployment

### 9.1 System Configuration

**config.json:**

```json
{
  "excel_path": "data/client_ratings.xlsx",
  "n_users": 39,
  "n_items": 121, 
  "model": "xgboost",
  "n_components": 20,
  "n_clusters": 4,
  "random_state": 42
}
```

### 9.2 Performance Metrics

**stats.json (Key Statistics):**

```json
{
  "global_mean": 3.36,
  "total_ratings": 2648,
  "sparsity": 0.44,
  "cluster_silhouette_score": 0.73,
  "model_rmse": {
    "cluster_0": 0.42,
    "cluster_1": 0.38, 
    "cluster_2": 0.51,
    "cluster_3": 0.35
  }
}
```

### 9.3 API Endpoints

**FastAPI Service:**

```python
POST /train
├─ Upload Excel file
├─ Trigger complete training pipeline
└─ Return training metadata

GET /recommendations/{user_id}?top_n=20
├─ Input: user_id, optional top_n
├─ Process: Run inference engine
└─ Output: Ranked movie recommendations

GET /health
└─ System health check
```

---

## 10. Technical Implementation Details

### 10.1 Libraries and Dependencies

```python
# Core ML Libraries
- scikit-learn: Clustering, metrics
- numpy: Numerical computations
- pandas: Data manipulation
- xgboost: Gradient boosting

# Web Service
- fastapi: API framework
- uvicorn: ASGI server

# Data Processing  
- openpyxl: Excel file handling
- joblib: Model serialization
```

### 10.2 File Structure Details

**Models Directory:**

```
models/
├── kmeans.pkl           # Trained K-means clustering model
├── regressor_cluster_0.pkl  # XGBoost model for Cluster 0
├── regressor_cluster_1.pkl  # XGBoost model for Cluster 1  
├── regressor_cluster_2.pkl  # XGBoost model for Cluster 2
├── regressor_cluster_3.pkl  # XGBoost model for Cluster 3
├── svd_user.npy         # User latent factors (39×20)
├── svd_item.npy         # Item latent factors (121×20)
├── config.json          # System configuration
├── mappings.json        # User/movie ID mappings
├── stats.json           # Statistical metrics
└── svd_meta.json        # SVD metadata
```

### 10.3 Scalability Considerations

**Adding New Users:**

```python
1. Collect initial ratings (minimum 3-5 movies)
2. Assign to appropriate cluster using K-means
3. Generate recommendations immediately
4. Improve recommendations as more ratings arrive
```

**Adding New Movies:**

```python
1. Collect ratings from existing users
2. Update SVD factorization (retrain)
3. Update statistical features
4. Recommendations include new movies automatically
```

**Model Updates:**

```python
1. Periodic retraining (monthly/quarterly)
2. Incremental learning for new data
3. A/B testing for model improvements
4. Performance monitoring and alerts
```

---

## Key Benefits Summary

### 1. Personalization

- Each user gets recommendations tailored to their cluster
- Individual preferences considered through user bias terms
- Continuous learning from user feedback

### 2. Accuracy

- Multiple ML techniques combined (ensemble approach)
- Cluster-specific models reduce noise
- SVD captures complex user-item relationships

### 3. Scalability

- Efficient inference for real-time recommendations
- Easy to add new users and movies
- Modular architecture supports updates

### 4. Robustness

- Handles cold start problems
- Manages data sparsity effectively
- Graceful degradation with limited data

### 5. Interpretability

- Clear cluster assignments
- Feature importance analysis possible
- Recommendation reasoning available

---

## Conclusion

This movie recommendation system successfully combines multiple advanced machine learning techniques to provide accurate, personalized movie suggestions. The system's modular architecture, comprehensive feature engineering, and cluster-based approach ensure both high-quality recommendations and practical scalability for real-world deployment.

The implementation demonstrates best practices in:

- Data preprocessing and cleaning
- Feature engineering for recommender systems
- Unsupervised learning (clustering)
- Matrix factorization techniques
- Ensemble modeling approaches
- Production-ready API design

**System Performance:**

- Successfully handles 39 users and 121 movies
- Achieves RMSE < 0.5 across all user clusters
- Provides sub-second recommendation response times
- Maintains recommendation quality with sparse data

This technical foundation provides a solid base for scaling to larger datasets and implementing advanced features like real-time learning, content-based filtering, and hybrid recommendation approaches.

---

*Document prepared by: Movie Recommender System Development Team*
*Last updated: October 2025*
*Version: 1.0*
