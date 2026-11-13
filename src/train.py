"""
Training script for SmartLead AI.
Compares baseline, Logistic Regression, Random Forest, and HistGradientBoosting.
Persists champion pipeline and evaluation artifacts.
"""

import os
import tempfile
# Force all temp operations to D: drive to avoid C: drive space issues
# Ensure a writable tempdir without breaking cloud deployment
scratch_env = os.environ.get("TMPDIR") or os.environ.get("TEMP")
if not scratch_env and os.path.exists("D:/Hermes/cache/scratch"):
    scratch_env = "D:/Hermes/cache/scratch"
if scratch_env and os.path.exists(scratch_env):
    tempfile.tempdir = scratch_env

import json
import pathlib
import joblib
import datetime
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data import load_data, clean_and_prepare_data, get_train_val_test_splits
from src.features import get_preprocessing_pipeline
from src.evaluate import evaluate_predictions, get_curves
from src.schema import ALL_FEATURES

def train_models():
    print("Loading data...")
    raw_df = load_data()
    clean_df, dup_count = clean_and_prepare_data(raw_df)
    
    print("Splitting data (70/15/15 stratified)...")
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(clean_df)
    
    preprocessor = get_preprocessing_pipeline()
    
    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    fitted_pipelines = {}
    
    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        # Predict on validation
        val_probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_val).astype(float)
        val_preds = pipeline.predict(X_val)
        
        metrics = evaluate_predictions(y_val, val_preds, val_probs, threshold=0.5)
        results[name] = {
            "validation_metrics": metrics,
            "pipeline": pipeline
        }
        print(f"{name} Val AP: {metrics.get('average_precision', 0):.4f}, Val F1: {metrics['f1']:.4f}")
        
    # Select champion based on Validation Average Precision (AP) or F1
    champion_name = max(results.keys(), key=lambda k: results[k]["validation_metrics"].get("average_precision", results[k]["validation_metrics"]["f1"]))
    print(f"\n🏆 Selected Champion Model: {champion_name}")
    
    champion_pipeline = fitted_pipelines[champion_name]
    
    # Optimize threshold on validation set for F1
    val_probs = champion_pipeline.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = evaluate_predictions(y_val, None, val_probs, threshold=thresh)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            
    print(f"Optimized threshold on validation set: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
    
    # Evaluate champion on untouched test set
    test_probs = champion_pipeline.predict_proba(X_test)[:, 1]
    test_metrics_default = evaluate_predictions(y_test, None, test_probs, threshold=0.5)
    test_metrics_optimized = evaluate_predictions(y_test, None, test_probs, threshold=best_thresh)
    
    print(f"Test Metrics (optimized threshold {best_thresh}): Accuracy: {test_metrics_optimized['accuracy']:.4f}, F1: {test_metrics_optimized['f1']:.4f}, ROC-AUC: {test_metrics_optimized['roc_auc']:.4f}")
    
    # Save artifacts
    project_root = pathlib.Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    models_dir = project_root / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / "champion_pipeline.joblib"
    joblib.dump(champion_pipeline, model_path)
    print(f"Saved champion pipeline to {model_path}")
    
    # Save metrics summary
    metrics_summary = {
        "champion_model": champion_name,
        "optimized_threshold": best_thresh,
        "duplicates_removed": int(dup_count),
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "model_comparison": {
            name: results[name]["validation_metrics"] for name in results
        },
        "test_metrics_default": test_metrics_default,
        "test_metrics_optimized": test_metrics_optimized
    }
    
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Save curves for lab
    curves = get_curves(y_test, test_probs)
    with open(artifacts_dir / "evaluation_curves.json", "w") as f:
        json.dump(curves, f, indent=4)
        
    # Save feature schema
    schema_info = {
        "features": ALL_FEATURES,
        "numeric_features": ALL_FEATURES[:4],
        "macro_features": ALL_FEATURES[4:9],
        "categorical_features": ALL_FEATURES[9:],
        "excluded_features": ["duration"]
    }
    with open(artifacts_dir / "feature_schema.json", "w") as f:
        json.dump(schema_info, f, indent=4)
        
    # Save model metadata
    import sklearn
    import sys
    metadata = {
        "champion_model": champion_name,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "dataset_source": "UCI Bank Marketing (bank-additional-full.csv)",
        "threshold": best_thresh
    }
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("All training artifacts successfully generated and saved!")

if __name__ == "__main__":
    train_models()
