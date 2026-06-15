import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    FAILURE_THRESHOLD,
    MODEL_DIR,
    DATA_PATH
)

# FAILURE_THRESHOLD = 0.50
# MODEL_DIR = Path("models")
# DATA_PATH = "data/predictive_maintenance.csv"

st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def load_models():
    with open(MODEL_DIR / "binary_failure_model.pkl", "rb") as f:
        binary_model = pickle.load(f)

    with open(MODEL_DIR / "failure_type_model.pkl", "rb") as f:
        failure_type_model = pickle.load(f)

    with open(MODEL_DIR / "failure_type_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return binary_model, failure_type_model, label_encoder, feature_columns


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def get_risk_level(probability):
    if probability < 0.20:
        return "Low Risk", "✅"
    elif probability < 0.50:
        return "Medium Risk", "🟡"
    elif probability < 0.80:
        return "High Risk", "🟠"
    else:
        return "Critical Risk", "🔴"


def get_recommendation(probability):
    if probability < 0.20:
        return "No immediate action required. Continue routine monitoring."
    elif probability < 0.50:
        return "Monitor the machine more closely and review sensor behavior."
    elif probability < 0.80:
        return "Schedule a maintenance inspection soon."
    else:
        return "Immediate maintenance inspection recommended."


df = load_data()

try:
    binary_model, failure_type_model, label_encoder, feature_columns = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run `python src/train_pipeline.py` first.")
    st.stop()


st.title("⚙️ Predictive Maintenance Dashboard")

st.write(
    "This app combines historical machine performance analysis with ML-based failure risk prediction."
)




st.subheader("Overall Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Failures", int(df["Target"].sum()))
col3.metric("Overall Failure Rate", f"{df['Target'].mean() * 100:.2f}%")

st.divider()

st.subheader("Failure Rate by Product Type")

failure_rate_by_type = (
    df.groupby("Type")["Target"]
    .mean()
    .mul(100)
    .reset_index()
)

failure_rate_by_type.columns = ["Product Type", "Failure Rate (%)"]

st.bar_chart(failure_rate_by_type.set_index("Product Type"))
st.dataframe(failure_rate_by_type, use_container_width=True)

st.divider()


st.subheader("Failure Types by Product Type")

failure_only_df = df[df["Failure Type"] != "No Failure"]

failure_type_by_product = (
    failure_only_df
    .groupby(["Type", "Failure Type"])
    .size()
    .reset_index(name="Count")
)

pivot_failure_type = failure_type_by_product.pivot(
    index="Type",
    columns="Failure Type",
    values="Count",
).fillna(0)

st.bar_chart(pivot_failure_type)
st.dataframe(failure_type_by_product, use_container_width=True)

st.divider()


st.subheader("Machine Selection")

col1, col2 = st.columns(2)

product_types = sorted(df["Type"].unique().tolist())

with col1:
    selected_type = st.selectbox(
        "Select Product Type",
        product_types
    )

filtered_df = df[df["Type"] == selected_type]

product_ids = sorted(filtered_df["Product ID"].unique().tolist())

with col2:
    selected_product_id = st.selectbox(
        "Select Product ID",
        product_ids
    )

machine_df = filtered_df[
    filtered_df["Product ID"] == selected_product_id
]

st.divider()

st.subheader("Selected Machine Summary")

st.subheader("Machine Selection")

col1, col2 = st.columns(2)

with col1:
    st.metric("Product Type", selected_type)

with col2:
    st.metric("Product ID", selected_product_id)

failure_rate = machine_df["Target"].mean() * 100
record_count = len(machine_df)

col1, col2, col3 = st.columns(3)

col1.metric("Records Available", record_count)
col2.metric("Historical Failure Rate", f"{failure_rate:.2f}%")
col3.metric("Actual Failure Count", int(machine_df["Target"].sum()))

st.divider() 

st.subheader("Average Sensor Behavior for Selected Machine")

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

avg_values = machine_df[sensor_cols].mean().reset_index()
avg_values.columns = ["Sensor", "Average Value"]

st.dataframe(avg_values, use_container_width=True)

st.divider()


st.subheader("ML Failure Risk Prediction")

st.write(
    "The user selects only Product Type and Product ID. "
    "The app uses the selected machine's existing sensor readings from the dataset for prediction."
)

if machine_df.empty:
    st.warning("No machine data found for this Product ID.")
else:
    selected_machine_row = machine_df.iloc[0]

    input_data = {}

    for feature in feature_columns:
        input_data[feature] = selected_machine_row[feature]

    input_df = pd.DataFrame([input_data])

    with st.expander("View model input data"):
        st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Failure Risk"):
        failure_probability = float(binary_model.predict_proba(input_df)[0][1])
        binary_prediction = int(failure_probability >= FAILURE_THRESHOLD)

        risk_level, risk_icon = get_risk_level(failure_probability)
        recommendation = get_recommendation(failure_probability)

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric("Failure Probability", f"{failure_probability:.2%}")
        col2.metric("Risk Level", f"{risk_icon} {risk_level}")
        col3.metric("Decision Threshold", f"{FAILURE_THRESHOLD:.2f}")

        st.progress(min(failure_probability, 1.0))

        if binary_prediction == 0:
            st.success("✅ Machine Status: NO FAILURE")
            st.write("The machine is currently predicted to be healthy.")

            st.subheader("Recommended Action")
            st.info(recommendation)

        else:
            failure_type_proba = failure_type_model.predict_proba(input_df)[0]
            failure_type_idx = int(failure_type_proba.argmax())
            predicted_failure_type = label_encoder.inverse_transform([failure_type_idx])[0]
            failure_type_confidence = float(failure_type_proba.max())

            top_failure_types = pd.DataFrame({
                "Failure Type": label_encoder.inverse_transform(
                    list(range(len(failure_type_proba)))
                ),
                "Confidence": failure_type_proba,
            }).sort_values("Confidence", ascending=False)

            st.error("⚠️ Machine Status: FAILURE DETECTED")

            col1, col2 = st.columns(2)

            col1.metric("Most Likely Failure Type", predicted_failure_type)
            col2.metric("Failure Type Confidence", f"{failure_type_confidence:.2%}")

            st.subheader("Top Failure Type Probabilities")
            st.bar_chart(top_failure_types.set_index("Failure Type"))
            st.dataframe(top_failure_types, use_container_width=True)

            st.subheader("Recommended Action")
            st.warning(recommendation)

        st.info(
            f"If failure probability is greater than or equal to {FAILURE_THRESHOLD}, "
            "the machine is classified as Failure."
        )


st.divider()
