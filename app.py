import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Predictive Maintenance IoT",
    layout="centered"
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

DATA_PATH = "predictive_maintenance_dataset.csv"

FEATURE_COLUMNS = [
    "vibration",
    "acoustic",
    "temperature",
    "current",
    "IMF_1",
    "IMF_2",
    "IMF_3",
]


# =========================
# LOAD & TRAIN MODEL
# =========================

@st.cache_resource
def load_model():
    df = pd.read_csv(DATA_PATH).dropna()

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # Time-based split
    split_idx = int(0.7 * len(df))
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler


model, scaler = load_model()


# =========================
# STREAMLIT UI
# =========================

st.title("🔧 Predictive Maintenance – Industrial IoT")
st.write("Simulated industrial dashboard for machine health monitoring")

st.markdown("---")

# -------- Sensor Inputs --------
st.subheader("📥 Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    vibration = st.number_input("Vibration", value=0.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    imf1 = st.number_input("IMF 1", value=0.0)
    imf2 = st.number_input("IMF 2", value=0.0)

with col2:
    acoustic = st.number_input("Acoustic", value=0.0)
    current = st.number_input("Current (A)", value=1.0)
    imf3 = st.number_input("IMF 3", value=0.0)

st.markdown("---")

# -------- Prediction --------
if st.button("🔍 Predict Machine Health"):

    input_data = np.array([
        [vibration, acoustic, temperature, current, imf1, imf2, imf3]
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧠 Prediction Result")

    # Machine status
    if prediction == 0:
        st.success("🟢 Machine Status: NORMAL")
    else:
        st.error("🔴 Machine Status: FAILURE")

    # Risk level
    if probability < 0.3:
        st.success("🟢 Risk Level: LOW")
    elif probability < 0.6:
        st.warning("🟡 Risk Level: MEDIUM")
    else:
        st.error("🔴 Risk Level: HIGH")

    # Confidence bar
    st.subheader("📊 Failure Confidence")
    st.progress(min(int(probability * 100), 100))
    st.write(f"Failure Probability: **{probability:.2f}**")

    # -------- Sensor Health --------
    st.markdown("---")
    st.subheader("⚠️ Sensor Health Indicators")

    warning_flag = False

    if vibration > 0.5:
        st.warning("High vibration detected → Possible mechanical fault")
        warning_flag = True

    if temperature > 80:
        st.warning("High temperature detected → Possible overheating")
        warning_flag = True

    if current > 5:
        st.warning("High current detected → Possible electrical overload")
        warning_flag = True

    if not warning_flag:
        st.success("All sensor values are within normal operating range")

    # -------- Explanation --------
    st.markdown("---")
    st.info(
        "The prediction is influenced mainly by vibration, temperature, current, "
        "and IMF features, which are strong indicators of mechanical and electrical faults."
    )
