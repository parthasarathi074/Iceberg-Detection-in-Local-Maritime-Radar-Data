import streamlit as st
import pandas as pd
import pickle
import zipfile
import io
import os
import numpy as np

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.error("❌ Trained model file not found! Please add model.pkl to your project folder.")
        st.stop()

model = load_model()

st.set_page_config(page_title="Iceberg Detection in Local Maritime", layout="centered")
st.title("🧊 Iceberg Detection in Local Maritime")
st.markdown("Upload or input sensor data to predict whether it's an iceberg or not.")

uploaded_file = st.file_uploader("Upload a CSV file containing sensor data", type=["csv"])
manual_input = st.text_input("Enter comma-separated sensor values (e.g., 12.3, 45.6, 78.9):")

if st.button("🔍 Predict"):
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        X = data.values
        predictions = model.predict(X)
        data["Prediction"] = predictions
        st.success("✅ Predictions generated for uploaded file!")
        st.dataframe(data)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            with zf.open("predictions.csv", "w") as f:
                data.to_csv(f, index=False)
        buffer.seek(0)

        st.download_button(
            label="📦 Download Predictions ZIP",
            data=buffer,
            file_name="predictions.zip",
            mime="application/zip"
        )

    elif manual_input:
        try:
            values = np.array([list(map(float, manual_input.split(",")))])
            prediction = model.predict(values)[0]
            st.success(f"✅ Prediction: {'🧊 Iceberg' if prediction == 1 else '🌊 Not an Iceberg'}")
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.warning("Please upload a CSV file or enter sensor values manually.")
