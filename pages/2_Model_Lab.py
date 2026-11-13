"""
Model Lab Page - SmartLead AI
"""

import streamlit as st
import json
import pathlib
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Lab | SmartLead AI", page_icon="📊", layout="wide")

st.markdown("# 📊 Model Lab & Evaluation")
st.markdown("Examine model performance metrics, model comparison, ROC/PR curves, and methodology.")

project_root = pathlib.Path(__file__).resolve().parent.parent
artifacts_dir = project_root / "artifacts"
metrics_path = artifacts_dir / "metrics.json"
curves_path = artifacts_dir / "evaluation_curves.json"

metrics_data = {}
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

curves_data = {}
if curves_path.exists():
    with open(curves_path, "r") as f:
        curves_data = json.load(f)

tab1, tab2, tab3 = st.tabs(["🏆 Model Comparison", "📈 ROC & PR Curves", "🛡️ Leakage & Methodology"])

with tab1:
    st.subheader("Model Performance Comparison (Validation Set)")
    comparison = metrics_data.get("model_comparison", {})
    if comparison:
        rows = []
        for model_name, m in comparison.items():
            rows.append({
                "Model": model_name,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "F1 Score": f"{m.get('f1', 0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                "Avg Precision (AP)": f"{m.get('average_precision', 0):.4f}"
            })
        st.table(pd.DataFrame(rows))
        
    st.subheader("Champion Untouched Test Metrics")
    test_opt = metrics_data.get("test_metrics_optimized", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{test_opt.get('accuracy', 0):.4f}")
    with col2:
        st.metric("Test Precision", f"{test_opt.get('precision', 0):.4f}")
    with col3:
        st.metric("Test Recall", f"{test_opt.get('recall', 0):.4f}")
    with col4:
        st.metric("Test F1", f"{test_opt.get('f1', 0):.4f}")

with tab2:
    st.subheader("Receiver Operating Characteristic (ROC) & Precision-Recall Curves")
    if curves_data:
        roc = curves_data.get("roc", {})
        fig_roc = px.line(
            x=roc.get("fpr", []), y=roc.get("tpr", []),
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            title="ROC Curve (Champion Model)"
        )
        fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, width="stretch")
        
        pr = curves_data.get("pr", {})
        fig_pr = px.line(
            x=pr.get("recall", []), y=pr.get("precision", []),
            labels={"x": "Recall", "y": "Precision"},
            title="Precision-Recall Curve (Champion Model)"
        )
        st.plotly_chart(fig_pr, width="stretch")

with tab3:
    st.subheader("🛡️ Target Leakage Prevention")
    st.markdown("""
    **Why is `duration` excluded?**
    In the UCI Bank Marketing dataset, `duration` represents the duration of the current call in seconds. 
    A call that lasts longer is highly correlated with a successful term deposit subscription (because if the customer subscribes, the call naturally takes longer). 
    However, **this information is unknown prior to dialing**. Including `duration` in a pre-call prediction model introduces severe **target leakage**, resulting in misleadingly inflated accuracy (~95%) that collapses in real-world deployment.
    
    SmartLead AI strictly enforces zero leakage by removing `duration` from all feature schemas.
    """)
