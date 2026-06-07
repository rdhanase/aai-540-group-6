import pandas as pd
import pytest
from src.preprocessing import create_target

def test_create_target():
    data = {
        'number_of_persons_injured': [0, 1, 0, 5],
        'number_of_persons_killed': [0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df = create_target(df)
    
    assert 'target' in df.columns
    assert list(df['target']) == [0, 1, 1, 1]

def test_create_target_with_strings():
    data = {
        'number_of_persons_injured': ['0', '2'],
        'number_of_persons_killed': ['0', '0']
    }
    df = pd.DataFrame(data)
    df = create_target(df)
    
    assert list(df['target']) == [0, 1]
