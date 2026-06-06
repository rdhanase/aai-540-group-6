import pandas as pd
import requests
import os

# Socrata API Endpoints for NYC Motor Vehicle Collisions
CRASHES_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
VEHICLES_URL = "https://data.cityofnewyork.us/resource/bm4k-52h4.json"
PERSONS_URL = "https://data.cityofnewyork.us/resource/f55k-p6yu.json"

def fetch_socrata_data(url, limit=50000):
    """
    Fetch data from NYC Open Data Socrata API.
    """
    print(f"Fetching data from {url} (limit={limit})...")
    params = {'$limit': limit}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception(f"Failed to fetch data from {url}: {response.status_code}")

def save_raw_data(df, filename, folder='aai-540-group-6/data/raw'):
    """Save the raw dataframe to a local CSV file."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"Saved {filename} to {path}")
    return path

def merge_datasets(crashes_df, vehicles_df, persons_df):
    """
    Merge the three datasets on collision_id.
    Ensures situational context from vehicles and persons is added to crashes.
    """
    print("Merging datasets on collision_id...")
    
    # Ensure collision_id is string for consistent merging
    for df in [crashes_df, vehicles_df, persons_df]:
        if 'collision_id' in df.columns:
            df['collision_id'] = df['collision_id'].astype(str)

    # Merge Crashes and Vehicles
    # One crash can have multiple vehicles; we'll keep all for now
    merged_df = crashes_df.merge(vehicles_df, on='collision_id', how='left', suffixes=('', '_veh'))
    
    # Merge with Persons
    merged_df = merged_df.merge(persons_df, on='collision_id', how='left', suffixes=('', '_pers'))
    
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    # Test fetch and merge with a small limit
    try:
        c_df = fetch_socrata_data(CRASHES_URL, limit=1000)
        v_df = fetch_socrata_data(VEHICLES_URL, limit=1000)
        p_df = fetch_socrata_data(PERSONS_URL, limit=1000)
        
        merged = merge_datasets(c_df, v_df, p_df)
        save_raw_data(merged, 'merged_sample.csv')
    except Exception as e:
        print(f"Error during data loading: {e}")
