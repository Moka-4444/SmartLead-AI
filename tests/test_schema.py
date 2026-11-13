"""
Tests for schema and feature constraints.
"""

import pytest
import pandas as pd
from src.schema import ALL_FEATURES, EXCLUDED_FEATURES, DEFAULT_VALUES

def test_duration_excluded():
    assert "duration" not in ALL_FEATURES
    assert "duration" in EXCLUDED_FEATURES

def test_batch_validation_rejects_duration():
    from src.schema import validate_batch_dataframe, DEFAULT_VALUES
    bad_data = pd.DataFrame([DEFAULT_VALUES])
    bad_data["duration"] = 120
    is_valid, errors = validate_batch_dataframe(bad_data)
    assert not is_valid
    assert any("duration" in err.lower() or "leakage" in err.lower() for err in errors)

def test_batch_validation_accepts_valid():
    from src.schema import validate_batch_dataframe, DEFAULT_VALUES
    good_data = pd.DataFrame([DEFAULT_VALUES])
    is_valid, errors = validate_batch_dataframe(good_data)
    assert is_valid
    assert len(errors) == 0

def test_default_values():
    for feat in ALL_FEATURES:
        assert feat in DEFAULT_VALUES
