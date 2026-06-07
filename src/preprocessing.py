import pandas as pd
import numpy as np

def prep_target(df):
    """
    Define the binary target. 
    We care if anyone was hurt or killed - safety first.
    """
    # API data can come in as strings, so cast everything to numeric to be safe
    cols = ['number_of_persons_injured', 'number_of_persons_killed']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 1 if injury/death, 0 if just property damage
    df['target'] = ((df['number_of_persons_injured'] > 0) | (df['number_of_persons_killed'] > 0)).astype(int)
    
    print("Class balance (0: Property, 1: Injury):")
    print(df['target'].value_counts(normalize=True))
    return df

def basic_cleaning(df):
    """
    Handle the messier parts of the NYC data.
    """
    # Fill missing boroughs - important for location analysis
    if 'borough' in df.columns:
        df['borough'] = df['borough'].fillna('UNKNOWN')
    
    # Contributing factors often missing for vehicle 1
    if 'contributing_factor_vehicle_1' in df.columns:
        df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].fillna('Unspecified')
    
    # Convert timestamps
    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'])
    
    # Drop stuff we don't need for the baseline model to save memory
    to_drop = [
        'location', 'on_street_name', 'off_street_name', 'cross_street_name',
        'number_of_persons_injured', 'number_of_persons_killed',
        'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
        'number_of_cyclist_injured', 'number_of_cyclist_killed',
        'number_of_motorist_injured', 'number_of_motorist_killed'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    
    return df

def run_prep(df):
    """Standard prep flow used by all notebooks."""
    df = prep_target(df)
    df = basic_cleaning(df)
    return df
