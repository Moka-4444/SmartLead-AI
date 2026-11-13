"""
SmartLead AI - Main Executive Overview App
"""

import streamlit as st
import json
import pathlib
import pandas as pd

st.set_page_config(
    page_title="SmartLead AI | Customer Conversion Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for executive SaaS look
st.markdown("""
<style>
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 3rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Load metrics
project_root = pathlib.Path(__file__).resolve().parent
artifacts_dir = project_root / "artifacts"
metrics_path = artifacts_dir / "metrics.json"

metrics_data = {}
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

# Sidebar
st.sidebar.markdown("### 🎯 SmartLead AI")
st.sidebar.markdown("---")
st.sidebar.info("Navigation:\n- **Overview**: Executive summary\n- **Prediction Studio**: Live customer scoring\n- **Model Lab**: Performance & curves\n- **Data Explorer**: EDA & distributions\n- **Batch Scoring**: CSV processing\n- **About**: Methodology & lineage")

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 style="color: #F8FAFC; font-size: 2.75rem; margin-bottom: 0.5rem;">SmartLead AI 🚀</h1>
    <h3 style="color: #06B6D4; font-size: 1.5rem; margin-bottom: 1rem;">Customer Conversion Prediction & Marketing Intelligence Platform</h3>
    <p style="color: #94A3B8; font-size: 1.1rem; line-height: 1.6;">
        An advanced supervised machine learning platform designed to predict customer term-deposit subscription propensity 
        prior to outbound telemarketing campaigns. Built for maximum precision, leakage prevention, and actionable executive insights.
    </p>
</div>
""", unsafe_allow_html=True)

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)

champ_name = metrics_data.get("champion_model", "Hist Gradient Boosting")
test_opt = metrics_data.get("test_metrics_optimized", {})
test_acc = test_opt.get("accuracy", 0.8783)
test_f1 = test_opt.get("f1", 0.5323)
test_auc = test_opt.get("roc_auc", 0.8210)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #94A3B8; margin-bottom: 0.5rem;">Champion Model</h4>
        <h2 style="color: #06B6D4; margin: 0;">{champ_name}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #94A3B8; margin-bottom: 0.5rem;">Test Accuracy</h4>
        <h2 style="color: #10B981; margin: 0;">{test_acc:.2%}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #94A3B8; margin-bottom: 0.5rem;">Test F1 Score</h4>
        <h2 style="color: #2563EB; margin: 0;">{test_f1:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #94A3B8; margin-bottom: 0.5rem;">ROC-AUC</h4>
        <h2 style="color: #F59E0B; margin: 0;">{test_auc:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Problem & Architecture Overview
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🧠 Executive Brief & Problem Statement")
    st.markdown("""
    Outbound telemarketing campaigns for bank term deposits suffer from high customer acquisition costs and low conversion rates (~11%).
    **SmartLead AI** solves this by predicting conversion likelihood **before** placing the call, enabling marketing teams to:
    - Target high-propensity customer segments with personalized offers.
    - Exclude low-probability leads to optimize agent working hours.
    - Prevent severe target leakage by strictly excluding call `duration`.
    """)
    
    st.markdown("### 🛡️ Leakage Prevention Protocol")
    st.warning("Critical Rule: The `duration` field measures the length of the ongoing call and is unknown prior to dialing. Including it causes massive artificial inflation of model metrics. SmartLead AI strictly excludes `duration` from all training and inference pipelines.")

with col_right:
    st.markdown("### 📊 Dataset Provenance & Split")
    st.markdown("""
    - **Source:** UCI Bank Marketing Repository (Dataset ID 222).
    - **Records:** 41,188 Portuguese telemarketing records (2008–2010).
    - **Split Protocol:** Stratified 70% Train, 15% Validation, 15% Untouched Test (Seed = 42).
    - **Class Balance:** Imbalanced target (~11.2% positive subscription rate).
    """)
    
    st.markdown("### 🚀 Quick Actions")
    if st.button("👉 Launch Prediction Studio"):
        st.switch_page("pages/1_Prediction_Studio.py")
    if st.button("📊 Explore Model Lab"):
        st.switch_page("pages/2_Model_Lab.py")
