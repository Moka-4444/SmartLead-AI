# 🚀 SmartLead AI: Customer Conversion Prediction & Marketing Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end, reproducible tabular machine learning system and executive analytics dashboard built for banking marketing intelligence. SmartLead AI predicts whether a client is likely to subscribe to a term deposit **before a telemarketing call is initiated**.

---

## 📌 Executive Summary & Key Highlights
- **Target Variable ($y$):** Binary classification (`1` = Subscribed, `0` = Not Subscribed).
- **Leakage Prevention:** Call duration (`duration`) is strictly excluded across preprocessing, model training, validation, and real-time/batch inference to prevent pre-call target leakage.
- **Dataset:** Official UCI Bank Marketing Dataset (ID 222, `bank-additional-full.csv`, 41,188 records).
- **Champion Model:** `HistGradientBoostingClassifier` evaluated against Logistic Regression, Random Forest, and a Dummy Baseline.
- **Decision Threshold Optimization:** Validation-set F1-optimized threshold (`0.19`) frozen before final untouched test set evaluation.

---

## 🏆 Verified Model Evaluation Metrics (Untouched Test Set)
Evaluated on **6,177 held-out test rows** (15% stratified split, seed 42):

| Metric | Dummy Baseline | Champion (HistGradientBoosting @ 0.19) | Default Threshold (0.50) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 88.73% | **87.83%** | 90.04% |
| **Precision (Subscribers)** | 0.00% | **45.69%** | 62.03% |
| **Recall (Subscribers)** | 0.00% | **63.75%** | 27.94% |
| **F1 Score** | 0.0000 | **0.5323** | 0.3855 |
| **ROC-AUC** | 0.5000 | **0.8210** | 0.8210 |
| **PR-AUC (Avg Precision)** | 0.1127 | **0.4684** | 0.4684 |

> **Business Insight:** While majority-class accuracy is ~88.7%, the baseline captures **0%** of subscribing customers. The tuned champion at threshold 0.19 captures **63.75% of actual subscribers**, enabling actionable telemarketing prioritization.

---

## 📂 Repository Architecture
```text
smartlead-ai/
├── app.py                      # Streamlit application entrypoint & Overview
├── pages/
│   ├── 1_Prediction_Studio.py  # Interactive lead scoring & what-if simulator
│   ├── 2_Model_Lab.py          # Baseline comparison, ROC/PR curves, leakage analysis
│   ├── 3_Data_Explorer.py      # Exploratory data analytics & distribution charts
│   ├── 4_Batch_Scoring.py      # Bulk CSV validation, batch prediction & download
│   └── 5_About.py              # Dataset provenance, citations, and architecture
├── src/
│   ├── schema.py               # Feature schemas, defaults, and strict CSV validation
│   ├── data.py                 # UCI dataset ingestion, cleaning & stratified splits
│   ├── features.py             # Scikit-learn ColumnTransformer preprocessing
│   ├── train.py                # Model training, validation selection, artifact persistence
│   ├── evaluate.py             # Metric calculations & ROC/PR curve generators
│   └── predict.py              # Single & batch inference pipeline wrappers
├── models/
│   └── champion_pipeline.joblib# Complete persisted preprocessing + classifier pipeline
├── artifacts/
│   ├── metrics.json            # Genuine test and validation metrics
│   ├── model_metadata.json     # Training timestamp, threshold, and environment metadata
│   ├── feature_schema.json     # 19 approved feature definitions
│   └── evaluation_curves.json  # Validated ROC and PR curve coordinate points
├── tests/
│   ├── test_schema.py          # Leakage prevention & schema tests
│   ├── test_inference.py       # Inference pipeline & batch rejection tests
│   └── test_app_smoke.py       # Artifact existence & smoke verification
├── scripts/
│   └── download_data.py        # Automated UCI dataset download script
├── requirements.txt            # Dependency specification
└── README.md
```

---

## 💻 Local Quickstart

### 1. Clone & Set Up Environment
```bash
git clone <YOUR_REPO_URL>
cd smartlead-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .\.venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
pytest -v
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this repository to **GitHub** (ensure `models/` and `artifacts/` are committed).
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **New App**.
3. Select your repository and set:
   - **Main file path:** `app.py`
   - **Branch:** `main` (or `master`)
4. Click **Deploy**.

---

## 📜 Dataset Citation & Attribution
- **Dataset:** UCI Bank Marketing Repository (Dataset ID 222).
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
