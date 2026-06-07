import argparse
import os
import joblib
import pandas as pd
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Pull from local modules
from preprocessing import run_prep
from features import run_engineering

def model_fn(model_dir):
    """
    Deserializes and loads the model for SageMaker inference (Batch Transform).
    """
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """
    Parses the incoming request. We expect a CSV without a header.
    Converts it to a DataFrame with the correct column names so our pipeline works.
    """
    if request_content_type == 'text/csv':
        # Load the CSV line(s) into a dataframe
        df = pd.read_csv(io.StringIO(request_body), header=None)
        
        # SageMaker InputFilter $[1:] strips the ID column.
        # These names must match the order in our training features.
        feature_names = ['borough', 'month', 'hour', 'is_rush_hour', 'is_weekend', 'cause_category', 'vehicle_type']
        df.columns = feature_names
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Makes predictions using the loaded model pipeline.
    """
    return model.predict(input_data)

def run_train(args):
    """
    Entry point for SageMaker Training Job.
    Reads from /opt/ml/input/data/train and saves to /opt/ml/model.
    """
    print(f"Starting SM training job on data in {args.train_path}")
    
    # Load all CSVs in the train folder
    csvs = [os.path.join(args.train_path, f) for f in os.listdir(args.train_path) if f.endswith('.csv')]
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {args.train_path}")
    
    df = pd.concat([pd.read_csv(f) for f in csvs])
    print(f"Loaded {len(df)} rows.")

    # 1. Pipeline
    # Only run if raw columns are present. If from Feature Store, skip.
    if 'number_of_persons_injured' in df.columns:
        print("Raw data detected. Running full pipeline...")
        df = run_prep(df)
        df = run_engineering(df)
    else:
        print("Processed data detected (Feature Store). Skipping engineering...")
    
    # 2. Features
    # Ensure column names match our consolidated features engine
    feature_cols = ['borough', 'month', 'hour', 'is_rush_hour', 'is_weekend', 'cause_category', 'vehicle_type']
    X = df[feature_cols]
    y = df['target']
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Pipeline Setup
    cat_cols = ['borough', 'cause_category', 'vehicle_type']
    cat_tx = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', cat_tx, cat_cols)],
        remainder='passthrough'
    )
    
    # Integrated SMOTE + RF
    pipeline = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        ))
    ])
    
    # 5. Execute
    print(f"Fitting RF (trees={args.n_estimators}, depth={args.max_depth})...")
    pipeline.fit(X_train, y_train)
    
    # 6. Serialize
    out_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, out_path)
    print(f"Model stored at {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker specific args
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    run_train(parser.parse_args())
