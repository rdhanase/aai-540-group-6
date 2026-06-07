import pandas as pd
import numpy as np

def time_features(df):
    """
    Rip out hour, month, etc. from raw strings.
    Adding rush hour and weekend flags for better predictive power.
    """
    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        df['month'] = df['crash_date'].dt.month
        df['day_name'] = df['crash_date'].dt.day_name()
        df['is_weekend'] = (df['crash_date'].dt.dayofweek >= 5).astype(int)
    
    if 'crash_time' in df.columns:
        # Get hour from HH:MM
        df['hour'] = pd.to_datetime(df['crash_time'], format='%H:%M', errors='coerce').dt.hour
        df['hour'] = df['hour'].fillna(-1).astype(int)
        
        # Rush hour peak windows: morning (7-9) and evening (4-6)
        df['is_rush_hour'] = (
            df['hour'].between(7, 9) | df['hour'].between(16, 18)
        ).astype(int)
    
    return df

def categorize_factors(df):
    """
    Bucketing 50+ messy factors into 6 main categories.
    Makes the model more stable and the Feature Store cleaner.
    """
    lookup = {
        'Distraction': ['Driver Inattention/Distraction', 'Passenger Distraction', 'Phone (hand-held)', 'Phone (hands-free)', 'Other Electronic Device'],
        'DUI_Drugs': ['Alcohol Involvement', 'Drugs (illegal)', 'Prescription Medication'],
        'Human_Error': ['Driver Inexperience', 'Turning Improperly', 'Backing Unsafely', 'Following Too Closely', 'Failure to Yield Right-of-Way'],
        'Aggressive_Driving': ['Unsafe Speed', 'Aggressive Driving/Road Rage', 'Passing or Lane Usage Improper'],
        'Road_Conditions': ['Pavement Slippery', 'Obstruction/Debris', 'Traffic Control Device Improper/Non-Working', 'View Obstructed/Limited'],
        'Vehicle_Fail': ['Brakes Defective', 'Steering Failure', 'Tire Failure/Inadequate', 'Other Lighting Defects']
    }
    
    # Map raw strings to categories
    flat_map = {v: k for k, vals in lookup.items() for v in vals}
    
    if 'contributing_factor_vehicle_1' in df.columns:
        df['cause_category'] = df['contributing_factor_vehicle_1'].map(flat_map).fillna('Other_Unknown')
    
    return df

def vehicle_cleanup(df):
    """
    Standardize vehicle types and cap to Top 10 to avoid high-cardinality issues.
    """
    if 'vehicle_type_code1' in df.columns:
        df['vehicle_type'] = df['vehicle_type_code1'].fillna('Unknown').str.strip().str.upper()
        # Only keep the heavy hitters
        top_10 = df['vehicle_type'].value_counts().head(10).index
        df['vehicle_type'] = df['vehicle_type'].apply(lambda x: x if x in top_10 else 'OTHER')
    
    return df

def run_engineering(df):
    """Full feature engine for both training and online serving."""
    df = time_features(df)
    df = categorize_factors(df)
    df = vehicle_cleanup(df)
    return df
