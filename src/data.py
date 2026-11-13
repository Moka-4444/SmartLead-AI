"""
Data loading, cleaning, and splitting utilities for SmartLead AI.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib

def load_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = pathlib.Path(__file__).parent.parent / "data" / "bank-additional-full.csv"
    
    df = pd.read_csv(csv_path, sep=';')
    return df

def clean_and_prepare_data(df: pd.DataFrame):
    # Check shape
    initial_count = len(df)
    
    # Remove duplicates with recorded count
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows. Clean dataset shape: {df.shape}")
    
    # Map target y: yes -> 1, no -> 0
    if 'y' in df.columns:
        df['target'] = df['y'].map({'yes': 1, 'no': 0})
        df = df.drop(columns=['y'])
    
    # Drop duration if present to prevent leakage
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
        
    return df, duplicates_removed

def get_train_val_test_splits(df: pd.DataFrame, random_state: int = 42):
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Stratified 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    
    # Split temp into 15% val and 15% test (50% of temp each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    print(f"Split sizes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test
