import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

from data_loader import pull_data, sync_data, ENDPOINTS
from preprocessing import run_prep
from features import run_engineering
from evaluate import evaluate_model

def execute_training(limit=20000):
    """
    Main training script. 
    Flow: Fetch -> Merge -> Prep -> Engineer -> Balance (SMOTE) -> Train.
    """
    print(f"Executing training flow (Sample size: {limit})")
    
    # 1. Ingestion
    try:
        c = pull_data(ENDPOINTS['crashes'], limit=limit)
        v = pull_data(ENDPOINTS['vehicles'], limit=limit)
        p = pull_data(ENDPOINTS['persons'], limit=limit)
        df = sync_data(c, v, p)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # 2. Pipeline
    df = run_prep(df)
    df = run_engineering(df)
    
    # 3. Model Inputs
    # Features selected based on initial hypothesis
    feature_cols = ['borough', 'month', 'hour', 'is_rush_hour', 'is_weekend', 'cause_category', 'vehicle_type']
    X = df[feature_cols]
    y = df['target']
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}.")
    
    # 4. Define Pipeline
    # Transformer for categories
    cat_cols = ['borough', 'cause_category', 'vehicle_type']
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', cat_transformer, cat_cols)],
        remainder='passthrough'
    )
    
    # Bundle SMOTE and RF to ensure balancing happens per-fold
    clf = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # 5. Execute
    print("Fitting model with SMOTE...")
    clf.fit(X_train, y_train)
    
    # 6. Save & Evaluate
    os.makedirs('aai-540-group-6/models', exist_ok=True)
    model_file = 'aai-540-group-6/models/rf_v1.joblib'
    joblib.dump(clf, model_file)
    print(f"Model serialized to {model_file}")
    
    evaluate_model(clf, X_test, y_test)
    
    return clf, X_test, y_test

if __name__ == "__main__":
    execute_training(limit=100000)
