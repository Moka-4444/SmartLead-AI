"""
Schema definition and validation for SmartLead AI.
Defines features, types, categorical choices, default values, and strict validation rules.
Crucially excludes 'duration' to prevent pre-call target leakage.
"""

import pandas as pd


NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous"
]

MACRO_FEATURES = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome"
]

EXCLUDED_FEATURES = ["duration"]

ALL_FEATURES = NUMERIC_FEATURES + MACRO_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_CHOICES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"]
}

DEFAULT_VALUES = {
    "age": 39,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "wed",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 93.075,
    "cons.conf.idx": -47.1,
    "euribor3m": 1.405,
    "nr.employed": 5099.1
}

MACRO_DEFAULTS = {
    "emp.var.rate": -1.8,
    "cons.price.idx": 93.075,
    "cons.conf.idx": -47.1,
    "euribor3m": 1.405,
    "nr.employed": 5099.1
}

ALLOWED_ID_COLUMNS = ["customer_id", "id", "client_id"]

def validate_batch_dataframe(df):
    """
    Strict validation for batch inference CSV dataframe.
    Checks:
    - Non-empty
    - Rejects 'duration' (leakage protection)
    - Checks required features presence
    - Validates numeric boundaries and types
    - Validates categorical domains
    """
    errors = []
    
    if df.empty:
        return False, ["Uploaded CSV is empty."]
        
    # Check for target leakage feature: duration
    present_leakage = [col for col in EXCLUDED_FEATURES if col in df.columns]
    if present_leakage:
        errors.append(
            f"Data leakage feature detected: {present_leakage}. "
            "Call duration is unknown prior to marketing calls and is strictly forbidden."
        )
        return False, errors
        
    # Check missing required columns
    missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns ({len(missing_cols)}): {missing_cols}")
        return False, errors

    # Check numeric ranges and invalid values
    for col in NUMERIC_FEATURES + MACRO_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                errors.append(f"Column '{col}' must contain numeric values.")
        if df[col].isnull().any():
            errors.append(f"Column '{col}' contains NaN/Null values.")
            
    # Check categorical domains
    for col, choices in CATEGORICAL_CHOICES.items():
        unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
        invalid_vals = set(unique_vals) - set(choices)
        if invalid_vals:
            errors.append(f"Column '{col}' contains invalid categories: {list(invalid_vals)[:5]}")
            
    return (len(errors) == 0), errors
