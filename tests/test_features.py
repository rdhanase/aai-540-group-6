import pandas as pd
import pytest
from src.features import extract_temporal_features

def test_extract_temporal_features():
    data = {
        'crash_date': pd.to_datetime(['2024-01-01', '2024-05-15']),
        'crash_time': ['08:30', '23:45']
    }
    df = pd.DataFrame(data)
    df = extract_temporal_features(df)
    
    assert 'crash_month' in df.columns
    assert 'crash_day_of_week' in df.columns
    assert 'crash_hour' in df.columns
    
    assert list(df['crash_month']) == [1, 5]
    assert list(df['crash_hour']) == [8, 23]
