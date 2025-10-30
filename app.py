import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Iceberg Detection", page_icon="🧊", layout="centered")

st.title("🧊 Iceberg Detection in Local Maritime")
st.write("Upload or input sensor data to predict whether it's an Iceberg or not.")

# Load model (replace with your actual model file)
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.warning("Model file not found! Please upload your trained model.pkl file.")
    model = None

user_input = st.text_input("Enter comma-separated sensor values (e.g. 12.3, 45.6, 78.9):")

if st.button("Predict"):
    if model is not None:
        try:
            values = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            st.success(f"Prediction: {'🧊 Iceberg' if prediction[0]==1 else '🚢 Ship'}")
        except:
            st.error("Please enter valid numeric values.")
    else:
        st.error("Model not loaded.")
