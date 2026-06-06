import pandas as pd
import numpy as np

def extract_temporal_features(df):
    """
    Extracts time-based features from crash_date and crash_time.
    Fulfills requirements for temporal analysis and feature store compliance.
    """
    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        df['month'] = df['crash_date'].dt.month
        df['day_of_week'] = df['crash_date'].dt.day_name()
        df['is_weekend'] = (df['crash_date'].dt.dayofweek >= 5).astype(int)
    
    if 'crash_time' in df.columns:
        # Extract hour from HH:MM string
        df['crash_hour'] = pd.to_datetime(df['crash_time'], format='%H:%M', errors='coerce').dt.hour
        df['crash_hour'] = df['crash_hour'].fillna(-1).astype(int)
        
        # Rush Hour feature (7-9 AM, 4-6 PM)
        df['rush_hour'] = (
            df['crash_hour'].between(7, 9) | df['crash_hour'].between(16, 18)
        ).astype(int)
    
    return df

def group_contributing_factors(df):
    """
    Groups the 50+ specific contributing factors into predictive categories.
    Ensures SageMaker Feature Store compliance by using underscores.
    """
    factor_mapping = {
        'Distraction': ['Driver Inattention/Distraction', 'Passenger Distraction', 'Phone (hand-held)', 'Phone (hands-free)', 'Other Electronic Device'],
        'Intoxication': ['Alcohol Involvement', 'Drugs (illegal)', 'Prescription Medication'],
        'Human_Error': ['Driver Inexperience', 'Turning Improperly', 'Backing Unsafely', 'Following Too Closely', 'Failure to Yield Right-of-Way'],
        'Speed_Aggressive': ['Unsafe Speed', 'Aggressive Driving/Road Rage', 'Passing or Lane Usage Improper'],
        'Infrastructure': ['Pavement Slippery', 'Obstruction/Debris', 'Traffic Control Device Improper/Non-Working', 'View Obstructed/Limited'],
        'Vehicle_Defect': ['Brakes Defective', 'Steering Failure', 'Tire Failure/Inadequate', 'Other Lighting Defects']
    }
    
    # Flatten mapping
    flat_map = {v: k for k, values in factor_mapping.items() for v in values}
    
    if 'contributing_factor_vehicle_1' in df.columns:
        df['factor_category'] = df['contributing_factor_vehicle_1'].map(flat_map).fillna('Other_Unspecified')
    
    return df

def clean_categorical_features(df):
    """
    Cleans high-cardinality columns like vehicle types.
    """
    if 'vehicle_type_code1' in df.columns:
        df['vehicle_type'] = df['vehicle_type_code1'].fillna('Unknown').str.strip().str.upper()
        # Cap to top types to reduce noise
        top_types = df['vehicle_type'].value_counts().head(10).index
        df['vehicle_type'] = df['vehicle_type'].apply(lambda x: x if x in top_types else 'OTHER')
    
    return df

def feature_engineering_pipeline(df):
    """Full feature engineering sequence consolidated for local and SageMaker runs."""
    df = extract_temporal_features(df)
    df = group_contributing_factors(df)
    df = clean_categorical_features(df)
    return df
