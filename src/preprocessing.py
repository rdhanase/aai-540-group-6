import pandas as pd
import numpy as np

def create_target(df):
    """
    Defines the ML problem: Predict if a crash results in injury or fatality.
    Target: 1 if injured/killed > 0, else 0.
    """
    # Convert injury/killed columns to numeric, handling potential strings from API
    inj_cols = ['number_of_persons_injured', 'number_of_persons_killed']
    for col in inj_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['target'] = ((df['number_of_persons_injured'] > 0) | (df['number_of_persons_killed'] > 0)).astype(int)
    
    counts = df['target'].value_counts()
    print(f"Target distribution:\n{counts}")
    return df

def clean_data(df):
    """
    Basic cleaning: handling missing values and data types.
    """
    # Fill missing boroughs
    if 'borough' in df.columns:
        df['borough'] = df['borough'].fillna('UNKNOWN')
    
    # Fill missing contributing factors
    if 'contributing_factor_vehicle_1' in df.columns:
        df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].fillna('Unspecified')
    
    # Convert crash_date to datetime
    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'])
    
    # Drop redundant or high-cardinality columns not used in initial baseline
    cols_to_drop = [
        'location', 'on_street_name', 'off_street_name', 'cross_street_name',
        'number_of_persons_injured', 'number_of_persons_killed',
        'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
        'number_of_cyclist_injured', 'number_of_cyclist_killed',
        'number_of_motorist_injured', 'number_of_motorist_killed'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def preprocess_pipeline(df):
    """Full preprocessing sequence."""
    df = create_target(df)
    df = clean_data(df)
    return df
