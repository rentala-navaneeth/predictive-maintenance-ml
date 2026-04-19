# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.load_data import load_cmapss_data
from src.preprocess import add_rul, create_binary_label
from src.feature_engineering import create_rolling_features, drop_na_rows
from src.train import train_test_split_by_engine, prepare_features, train_models
from src.evaluate import classify_risk

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

st.title("🔧 Explainable Predictive Maintenance System")
st.markdown("Predict engine failure risk using sensor data with explainability")

# =========================
# LOAD + PREPROCESS DATA
# =========================
@st.cache_data
def load_data():
    df = load_cmapss_data("data/train_FD001.txt")
    df = add_rul(df)
    df = create_binary_label(df)
    df = create_rolling_features(df, window=10)
    df = drop_na_rows(df)
    return df

df = load_data()

# =========================
# TRAIN MODEL (CACHED)
# =========================
@st.cache_resource
def train_model(df):
    train_df, _ = train_test_split_by_engine(df)
    X_train, y_train = prepare_features(train_df)
    models = train_models(X_train, y_train)
    return models['Logistic Regression'], X_train

model, X_train = train_model(df)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

engine_ids = sorted(df['engine_id'].unique())
selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)

engine_data = df[df['engine_id'] == selected_engine]

cycle_max = int(engine_data['cycle'].max())

selected_cycle = st.sidebar.slider(
    "Select Cycle",
    min_value=10,
    max_value=cycle_max,
    value=cycle_max
)

selected_row = engine_data[engine_data['cycle'] == selected_cycle]

# =========================
# ENGINE OVERVIEW
# =========================
st.subheader(f"Engine {selected_engine} Overview")
colA, colB = st.columns(2)

colA.write(f"Total Cycles: {cycle_max}")
colB.write(f"Selected Cycle: {selected_cycle}")

# =========================
# SENSOR VISUALIZATION
# =========================
st.subheader("📈 Sensor Trend")

# Faster + cleaner filtering
sensor_cols = [col for col in df.columns if 'sensor_' in col and '_mean' not in col]
valid_sensors = [col for col in sensor_cols if df[col].std() > 0.01]

sensor_col = st.selectbox("Select Sensor to Visualize", valid_sensors)

fig, ax = plt.subplots()
ax.plot(engine_data['cycle'], engine_data[sensor_col])
ax.axvline(selected_cycle, color='red', linestyle='--', label='Selected Cycle')
ax.set_title(f"{sensor_col} Trend")
ax.set_xlabel("Cycle")
ax.set_ylabel("Value")
ax.legend()

st.pyplot(fig)

# =========================
# PREDICTION
# =========================
X_latest, _ = prepare_features(selected_row)

prob = model.predict_proba(X_latest)[0][1]
risk = classify_risk(prob)

# =========================
# DISPLAY RESULTS
# =========================
st.subheader("📊 Prediction Result")

col1, col2, col3 = st.columns(3)

col1.metric("Failure Probability", f"{prob:.2f}")
col2.metric("Risk Level", risk)

if risk == "High Risk":
    col3.error("⚠️ Immediate Maintenance Needed")
elif risk == "Medium Risk":
    col3.warning("⚠️ Monitor Closely")
else:
    col3.success("✅ Healthy")

# Progress bar
st.progress(float(prob))

# =========================
# SHAP EXPLANATION
# =========================
st.subheader("🔍 Key Feature Contributions")

scaler = model.named_steps['scaler']
lr_model = model.named_steps['model']

X_train_scaled = scaler.transform(X_train)
X_latest_scaled = scaler.transform(X_latest)

explainer = shap.LinearExplainer(lr_model, X_train_scaled[:500])
shap_values = explainer(X_latest_scaled)

feature_names = X_latest.columns
shap_vals = shap_values.values[0]

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_vals
})

shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
shap_df = shap_df.sort_values("Abs SHAP", ascending=False)
top_shap = shap_df.head(10)

fig, ax = plt.subplots()
ax.barh(top_shap["Feature"], top_shap["SHAP Value"])
ax.set_title("Top Feature Contributions")
ax.invert_yaxis()

st.pyplot(fig)

# =========================
# RAW DATA
# =========================
with st.expander("View Engine Data"):
    st.dataframe(engine_data.tail(20))