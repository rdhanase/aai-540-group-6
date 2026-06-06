import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

from data_loader import fetch_socrata_data, CRASHES_URL, VEHICLES_URL, PERSONS_URL, merge_datasets
from preprocessing import preprocess_pipeline
from features import feature_engineering_pipeline

def train_baseline_model(limit=20000):
    """
    Complete training pipeline: Ingestion -> Preprocessing -> Features -> SMOTE -> RF.
    """
    print(f"Starting training pipeline with limit={limit}...")
    
    # 1. Data Acquisition
    try:
        c_df = fetch_socrata_data(CRASHES_URL, limit=limit)
        v_df = fetch_socrata_data(VEHICLES_URL, limit=limit)
        p_df = fetch_socrata_data(PERSONS_URL, limit=limit)
        df = merge_datasets(c_df, v_df, p_df)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Preprocessing & Feature Engineering
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    
    # 3. Feature Selection
    # Selected based on hypothesis in Design Doc
    features = ['borough', 'crash_month', 'crash_day_of_week', 'crash_hour', 'factor_category']
    X = df[features]
    y = df['target']
    
    # 4. Train/Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 5. Pipeline Definition
    # We use imblearn.pipeline.Pipeline to correctly apply SMOTE during CV
    categorical_features = ['borough', 'factor_category']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # 6. Training
    print("Training Random Forest with SMOTE...")
    model_pipeline.fit(X_train, y_train)
    
    # 7. Persistence
    os.makedirs('aai-540-group-6/models', exist_ok=True)
    model_path = 'aai-540-group-6/models/rf_baseline.joblib'
    joblib.dump(model_pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # 8. Evaluation
    from evaluate import evaluate_model
    evaluate_model(model_pipeline, X_test, y_test)
    
    return model_pipeline, X_test, y_test

if __name__ == "__main__":
    # Run with a moderate limit for local validation
    train_baseline_model(limit=5000)
