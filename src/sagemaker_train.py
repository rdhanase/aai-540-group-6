import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Import internal modules (Assuming they are in the same directory or PYTHONPATH)
from preprocessing import preprocess_pipeline
from features import feature_engineering_pipeline

def train(args):
    """
    SageMaker training function.
    """
    print("Loading data from:", args.train)
    # SageMaker passes data as CSV files in the input directory
    # For simplicity in this template, we assume a single merged file or handle multiple
    # In a real pipeline, we might fetch from S3 directly or via SageMaker Processing
    
    # Example: list files in args.train
    files = [os.path.join(args.train, f) for f in os.listdir(args.train) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files found in {args.train}")
    
    df = pd.concat([pd.read_csv(f) for f in files])
    print(f"Data shape: {df.shape}")

    # 1. Preprocess & Feature Engineering
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    
    # 2. Feature Selection
    features = ['borough', 'crash_month', 'crash_day_of_week', 'crash_hour', 'factor_category']
    X = df[features]
    y = df['target']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Pipeline Definition
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
        ('classifier', RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        ))
    ])
    
    # 5. Training
    print(f"Training with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
    model_pipeline.fit(X_train, y_train)
    
    # 6. Save Model
    # SageMaker expects the model to be saved in /opt/ml/model
    model_output_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model_pipeline, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()
    train(args)
