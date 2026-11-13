import streamlit as st
import pandas as pd
import io
import pathlib
from src.predict import SmartLeadPredictor
from src.schema import ALL_FEATURES, DEFAULT_VALUES, ALLOWED_ID_COLUMNS, EXCLUDED_FEATURES, validate_batch_dataframe

st.set_page_config(page_title="Batch Scoring | SmartLead AI", page_icon="📁", layout="wide")

st.markdown("# 📁 Batch Scoring & Pipeline Processing")
st.markdown("Upload marketing contact lists in bulk to generate conversion probabilities and prioritize high-propensity leads.")

# 1. Download Template Section
st.subheader("1. Download Input CSV Template")
col_desc, col_btn = st.columns([3, 1])

with col_desc:
    st.info("The batch file must include all 19 pre-call features. Feature `duration` is strictly forbidden to prevent leakage.")

# Generate sample template
template_df = pd.DataFrame([DEFAULT_VALUES])
# Add a sample customer_id
template_df.insert(0, "customer_id", "CUST_1001")
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode('utf-8')

with col_btn:
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_bytes,
        file_name="smartlead_batch_template.csv",
        mime="text/csv"
    )

st.divider()

# 2. Upload and Score Section
st.subheader("2. Upload CSV for Scoring")

predictor = None
default_thresh = 0.19
try:
    predictor = SmartLeadPredictor()
    default_thresh = predictor.default_threshold
except Exception as e:
    st.error(f"Failed to load champion model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file (max 10MB)", type=["csv"])

col_thresh, col_stat = st.columns([1, 2])
with col_thresh:
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.90,
        value=float(default_thresh),
        step=0.01,
        help="Default is set to the model's validation-optimized threshold."
    )
    if threshold != default_thresh:
        st.caption(f"⚠️ Operating in What-If mode. Production validated threshold is: **{default_thresh:.2f}**")

if uploaded_file is not None:
    try:
        # Load CSV
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded rows:** {len(batch_df)} | **Columns:** {len(batch_df.columns)}")
        
        # Validate CSV
        is_valid, validation_errors = validate_batch_dataframe(batch_df)
        
        if not is_valid:
            st.error("❌ Batch validation failed:")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            if predictor is None:
                st.error("Model predictor is not initialized.")
                st.stop()
            st.success("✅ File schema validated successfully! Processing predictions...")
            
            # Predict
            scored_df = predictor.predict_batch(batch_df, threshold=threshold)
            
            # Summary Metrics
            total_leads = len(scored_df)
            subscribed_leads = int((scored_df["prediction"] == 1).sum())
            conversion_rate = (subscribed_leads / total_leads) * 100 if total_leads > 0 else 0
            avg_prob = scored_df["conversion_probability"].mean() * 100
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Processed", total_leads)
            kpi2.metric("Predicted Conversions", subscribed_leads)
            kpi3.metric("Projected Conversion Rate", f"{conversion_rate:.1f}%")
            kpi4.metric("Average Propensity", f"{avg_prob:.1f}%")
            
            st.subheader("Scored Preview (First 20 Records)")
            st.dataframe(scored_df.head(20), width="stretch")
            
            # Download Scored CSV
            out_buffer = io.StringIO()
            scored_df.to_csv(out_buffer, index=False)
            st.download_button(
                label="📥 Download Scored Results CSV",
                data=out_buffer.getvalue().encode('utf-8'),
                file_name="smartlead_scored_predictions.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
