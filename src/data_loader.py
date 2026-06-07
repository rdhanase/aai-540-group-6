import pandas as pd
import requests
import os

# NYC Open Data Socrata Endpoints
ENDPOINTS = {
    'crashes': "https://data.cityofnewyork.us/resource/h9gi-nx95.json",
    'vehicles': "https://data.cityofnewyork.us/resource/bm4k-52h4.json",
    'persons': "https://data.cityofnewyork.us/resource/f55k-p6yu.json"
}

def pull_data(url, limit=100000):
    """
    Grabs data from Socrata API. 
    Using a 100k limit to ensure stability in SageMaker.
    """
    print(f"Pulling from {url} (limit: {limit})")
    try:
        r = requests.get(url, params={'$limit': limit})
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        print(f"API request failed: {e}")
        raise

def sync_data(crashes, vehicles, persons):
    """
    Joins the three NYC datasets on collision_id.
    Standardizes ID types to avoid merge issues.
    """
    print("Syncing datasets on collision_id...")
    
    # Cast IDs to strings just in case
    for df in [crashes, vehicles, persons]:
        if 'collision_id' in df.columns:
            df['collision_id'] = df['collision_id'].astype(str)

    # Left join to keep all crashes even if vehicle/person info is missing
    merged = crashes.merge(vehicles, on='collision_id', how='left', suffixes=('', '_v'))
    merged = merged.merge(persons, on='collision_id', how='left', suffixes=('', '_p'))
    
    return merged

if __name__ == "__main__":
    # Quick sanity check
    try:
        c = pull_data(ENDPOINTS['crashes'], limit=1000)
        v = pull_data(ENDPOINTS['vehicles'], limit=1000)
        p = pull_data(ENDPOINTS['persons'], limit=1000)
        
        df = sync_data(c, v, p)
        
        out_dir = 'aai-540-group-6/data/raw'
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(f"{out_dir}/merged_sample.csv", index=False)
        print(f"Sample saved to {out_dir}/merged_sample.csv")
    except Exception as e:
        print(f"Script failed: {e}")
