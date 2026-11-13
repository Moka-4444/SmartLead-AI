"""
Prediction Studio Page - SmartLead AI
"""

import streamlit as st
import pandas as pd
from src.predict import SmartLeadPredictor
from src.schema import CATEGORICAL_CHOICES, DEFAULT_VALUES, MACRO_DEFAULTS

st.set_page_config(page_title="Prediction Studio | SmartLead AI", page_icon="🎯", layout="wide")

st.markdown("# 🎯 Prediction Studio")
st.markdown("Configure customer profile, campaign details, and macroeconomic factors to predict term deposit subscription probability.")

@st.cache_resource
def get_predictor():
    return SmartLeadPredictor()

predictor = get_predictor()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Customer Demographics")
        age = st.slider("Age", 18, 95, DEFAULT_VALUES["age"])
        job = st.selectbox("Job", CATEGORICAL_CHOICES["job"], index=0)
        marital = st.selectbox("Marital Status", CATEGORICAL_CHOICES["marital"], index=1)
        education = st.selectbox("Education", CATEGORICAL_CHOICES["education"], index=6)
        default = st.selectbox("Credit in Default?", CATEGORICAL_CHOICES["default"], index=0)
        housing = st.selectbox("Housing Loan?", CATEGORICAL_CHOICES["housing"], index=1)
        loan = st.selectbox("Personal Loan?", CATEGORICAL_CHOICES["loan"], index=0)
        
    with col2:
        st.subheader("📞 Campaign & Contact Details")
        contact = st.selectbox("Contact Communication Type", CATEGORICAL_CHOICES["contact"], index=0)
        month = st.selectbox("Last Contact Month", CATEGORICAL_CHOICES["month"], index=3)
        day_of_week = st.selectbox("Last Contact Day of Week", CATEGORICAL_CHOICES["day_of_week"], index=2)
        campaign = st.number_input("Number of Contacts during Campaign", 1, 50, DEFAULT_VALUES["campaign"])
        pdays = st.number_input("Days Since Previous Contact (999 = Never)", 0, 999, DEFAULT_VALUES["pdays"])
        previous = st.number_input("Previous Contacts Before Campaign", 0, 20, DEFAULT_VALUES["previous"])
        poutcome = st.selectbox("Outcome of Previous Campaign", CATEGORICAL_CHOICES["poutcome"], index=1)
        
    with col3:
        st.subheader("📈 Macroeconomic Environment (Expert)")
        emp_var_rate = st.number_input("Employment Variation Rate (emp.var.rate)", -5.0, 5.0, MACRO_DEFAULTS["emp.var.rate"])
        cons_price_idx = st.number_input("Consumer Price Index (cons.price.idx)", 90.0, 100.0, MACRO_DEFAULTS["cons.price.idx"])
        cons_conf_idx = st.number_input("Consumer Confidence Index (cons.conf.idx)", -60.0, -20.0, MACRO_DEFAULTS["cons.conf.idx"])
        euribor3m = st.number_input("Euribor 3 Month Rate (euribor3m)", 0.0, 10.0, MACRO_DEFAULTS["euribor3m"])
        nr_employed = st.number_input("Number of Employees (nr.employed)", 4900.0, 5300.0, MACRO_DEFAULTS["nr.employed"])
        
        threshold = st.slider("Decision Threshold", 0.05, 0.90, 0.19, 0.01)

    submitted = st.form_submit_button("🔮 Predict Conversion Propensity")

if submitted:
    input_payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }
    
    result = predictor.predict_single(input_payload, threshold=threshold)
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric("Subscription Probability", f"{result['probability']:.2%}")
    with res_col2:
        st.metric("Classification Decision", result['prediction_text'])
    with res_col3:
        st.metric("Applied Threshold", f"{threshold:.2f}")
        
    if result['prediction'] == 1:
        st.success("🎯 **High Propensity Lead:** Customer exhibits strong likelihood of subscribing. Recommended for priority outreach.")
    else:
        st.info("ℹ️ **Low Propensity Lead:** Customer unlikely to subscribe at current threshold. Consider nurturing or alternate channels.")
