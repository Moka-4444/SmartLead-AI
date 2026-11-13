"""
Inference wrapper for SmartLead AI.
Loads persisted model and validates input schema.
"""

import pathlib
import json
import joblib
import pandas as pd
from src.schema import ALL_FEATURES, DEFAULT_VALUES, ALLOWED_ID_COLUMNS, validate_batch_dataframe

def get_persisted_threshold() -> float:
    metadata_path = pathlib.Path(__file__).parent.parent / "artifacts" / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
                return float(data.get("threshold", 0.19))
        except Exception:
            pass
    return 0.19

class SmartLeadPredictor:
    def __init__(self, model_path: pathlib.Path | str = None):
        if model_path is None:
            model_path = pathlib.Path(__file__).parent.parent / "models" / "champion_pipeline.joblib"
        self.model = joblib.load(model_path)
        self.default_threshold = get_persisted_threshold()
        
    def predict_single(self, input_data: dict, threshold: float = None) -> dict:
        actual_thresh: float = float(self.default_threshold if threshold is None else threshold)
            
        df = pd.DataFrame([input_data])
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = DEFAULT_VALUES.get(col, 0)
                
        df = df[ALL_FEATURES]
        prob = float(self.model.predict_proba(df)[:, 1][0])
        pred_label = int(prob >= actual_thresh)
        
        return {
            "probability": prob,
            "prediction": pred_label,
            "prediction_text": "Subscribed (Yes)" if pred_label == 1 else "Not Subscribed (No)",
            "threshold_used": actual_thresh
        }
        
    def predict_batch(self, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        actual_thresh: float = float(self.default_threshold if threshold is None else threshold)
            
        valid, errors = validate_batch_dataframe(df)
        if not valid:
            raise ValueError(f"Batch validation failed: {'; '.join(errors)}")
            
        inference_df = df[ALL_FEATURES].copy()
        probs = self.model.predict_proba(inference_df)[:, 1]
        preds = (probs >= actual_thresh).astype(int)
        
        # Keep allowed ID columns if present, drop PII/unwanted columns to avoid leakage/PII leaks
        id_cols = [c for c in ALLOWED_ID_COLUMNS if c in df.columns]
        result_df = df[id_cols + ALL_FEATURES].copy() if id_cols else df[ALL_FEATURES].copy()
        
        result_df["conversion_probability"] = probs
        result_df["prediction"] = preds
        result_df["prediction_label"] = ["Yes" if p == 1 else "No" for p in preds]
        result_df["threshold_used"] = actual_thresh
        return result_df
