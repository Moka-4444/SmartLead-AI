import streamlit as st
import json
import pathlib

st.set_page_config(page_title="About & Methodology | SmartLead AI", page_icon="ℹ️", layout="wide")

st.markdown("# ℹ️ About SmartLead AI & Engineering Methodology")
st.markdown("Detailed dataset provenance, strict target leakage prevention protocols, validation methodology, and reproduction guidelines.")

tab1, tab2, tab3 = st.tabs(["📚 Dataset & Provenance", "🛡️ Machine Learning Methodology", "⚙️ Reproducibility & Audit"])

with tab1:
    st.subheader("Official Dataset Provenance")
    st.markdown("""
    - **Dataset Name:** UCI Bank Marketing (Dataset ID 222)
    - **Subset Used:** `bank-additional/bank-additional-full.csv` (41,188 records, 20 attributes + target `y`).
    - **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. *UCI Machine Learning Repository*. DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).
    - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
    - **Business Domain:** Portuguese telemarketing campaigns from May 2008 to November 2010 for long-term deposit subscriptions.
    """)
    
    st.subheader("Feature Dictionary (19 Pre-Call Attributes)")
    st.markdown("""
    1. **Customer Demographics:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.
    2. **Campaign Contacts:** `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`.
    3. **Macroeconomic Indicators:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
    """)

with tab2:
    st.subheader("Strict Leakage Prevention Protocol")
    st.markdown("""
    ### Why `duration` is Excluded
    In the raw UCI dataset, `duration` indicates the call duration in seconds. 
    **Critical rule:** This value is unknown before making the telemarketing call. Including `duration` would introduce massive target leakage, generating unrealistically high scores (e.g. 95%+ accuracy) that collapse in actual pre-call production inference. 
    SmartLead AI drops `duration` at ingestion, training, inference, and batch uploads.
    
    ### Splitting Protocol
    - **70% Training / 15% Validation / 15% Untouched Test** with stratification on target class.
    - **Random Seed:** 42 for exact reproducibility.
    - **Untouched Test Set:** Kept isolated until final evaluation of the champion model.
    """)

with tab3:
    st.subheader("Model Pipeline & Artifact Verification")
    project_root = pathlib.Path(__file__).resolve().parent.parent
    metadata_path = project_root / "artifacts" / "model_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        st.json(metadata)
    else:
        st.warning("Model metadata artifact not found.")

    st.subheader("Local Execution & Reproduction Commands")
    st.code("""
# 1. Activate venv
source .venv/bin/activate  # Or .venv\\Scripts\\activate on Windows

# 2. Run Test Suite
pytest -q

# 3. Launch App
streamlit run app.py
    """, language="bash")
