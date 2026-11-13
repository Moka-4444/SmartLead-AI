"""
Data Explorer Page - SmartLead AI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.data import load_data, clean_and_prepare_data

st.set_page_config(page_title="Data Explorer | SmartLead AI", page_icon="📁", layout="wide")

st.markdown("# 📁 Data Explorer & EDA")
st.markdown("Explore dataset statistics, class balance, demographic distributions, and campaign features.")

@st.cache_data
def get_clean_data():
    raw = load_data()
    clean, _ = clean_and_prepare_data(raw)
    return clean

df = get_clean_data()

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Clean Records", f"{len(df):,}")
with col2:
    pos_rate = df['target'].mean()
    st.metric("Subscription Rate (Positive Class)", f"{pos_rate:.2%}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Target & Demographics", "📞 Campaign Features", "📋 Raw Data Sample"])

with tab1:
    st.subheader("Class Balance (Subscription Target)")
    target_counts = df['target'].map({1: 'Yes (Subscribed)', 0: 'No (Not Subscribed)'}).value_counts().reset_index()
    target_counts.columns = ['Subscription', 'Count']
    fig_target = px.bar(target_counts, x='Subscription', y='Count', color='Subscription', title="Class Distribution")
    st.plotly_chart(fig_target, width="stretch")
    
    st.subheader("Age Distribution by Subscription")
    fig_age = px.histogram(df, x='age', color=df['target'].map({1: 'Yes', 0: 'No'}), barmode='overlay', title="Age Distribution")
    st.plotly_chart(fig_age, width="stretch")

with tab2:
    st.subheader("Job Category vs Subscription")
    job_sub = df.groupby('job')['target'].mean().reset_index()
    fig_job = px.bar(job_sub, x='job', y='target', title="Conversion Rate by Job Category", labels={'target': 'Conversion Rate'})
    st.plotly_chart(fig_job, width="stretch")
    
    st.subheader("Contact Communication Type vs Subscription")
    contact_sub = df.groupby('contact')['target'].mean().reset_index()
    fig_contact = px.bar(contact_sub, x='contact', y='target', title="Conversion Rate by Contact Type")
    st.plotly_chart(fig_contact, width="stretch")

with tab3:
    st.subheader("Dataset Sample (First 100 rows)")
    st.dataframe(df.head(100), width="stretch")
