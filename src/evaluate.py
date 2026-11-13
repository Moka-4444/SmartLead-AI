"""
Evaluation metrics calculation for SmartLead AI.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import numpy as np

def evaluate_predictions(y_true, y_pred, y_prob=None, threshold=0.5):
    if y_prob is not None:
        y_pred_thresh = (y_prob >= threshold).astype(int)
    else:
        y_pred_thresh = y_pred
        
    acc = accuracy_score(y_true, y_pred_thresh)
    prec = precision_score(y_true, y_pred_thresh, zero_division=0)
    rec = recall_score(y_true, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold)
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
            
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["average_precision"] = 0.0
            
    cm = confusion_matrix(y_true, y_pred_thresh)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Class support
    unique, counts = np.unique(y_true, return_counts=True)
    metrics["class_support"] = {int(k): int(v) for k, v in zip(unique, counts)}
    metrics["positive_prevalence"] = float(counts[1] / len(y_true)) if len(counts) > 1 else 0.0
    
    return metrics

def get_curves(y_true, y_prob):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    
    # Handle infinite threshold in raw ROC curve (standard JSON compliance)
    clean_roc_thresh = [None if np.isinf(t) else float(t) for t in roc_thresh]
    
    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": clean_roc_thresh
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresh.tolist()
        }
    }
