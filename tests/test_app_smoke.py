"""
Smoke tests for app artifacts and basic modules.
"""

import pathlib
import json

def test_artifacts_exist():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    models_dir = project_root / "models"
    
    assert (models_dir / "champion_pipeline.joblib").exists()
    assert (artifacts_dir / "metrics.json").exists()
    assert (artifacts_dir / "feature_schema.json").exists()
    assert (artifacts_dir / "model_metadata.json").exists()

def test_metrics_content():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    with open(artifacts_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
        
    assert "champion_model" in metrics
    assert "optimized_threshold" in metrics
    assert "test_metrics_optimized" in metrics
