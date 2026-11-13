"""
Tests for inference wrapper and model prediction output.
"""

import pytest
import pandas as pd
from src.predict import SmartLeadPredictor
from src.schema import DEFAULT_VALUES

def test_single_prediction():
    predictor = SmartLeadPredictor()
    res = predictor.predict_single(DEFAULT_VALUES)
    assert "probability" in res
    assert "prediction" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in [0, 1]

def test_predict_batch_duration_rejection():
    predictor = SmartLeadPredictor()
    bad_df = pd.DataFrame([DEFAULT_VALUES])
    bad_df["duration"] = 250
    with pytest.raises(ValueError) as excinfo:
        predictor.predict_batch(bad_df)
    assert "leakage" in str(excinfo.value).lower() or "duration" in str(excinfo.value).lower()

def test_batch_prediction():
    predictor = SmartLeadPredictor()
    df = pd.DataFrame([DEFAULT_VALUES, DEFAULT_VALUES])
    res_df = predictor.predict_batch(df)
    assert "conversion_probability" in res_df.columns
    assert "prediction_label" in res_df.columns
    assert len(res_df) == 2
