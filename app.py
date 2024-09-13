import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')

# User input section
st.title("Predict using the saved model")
input_data = st.text_input("Enter the input data (comma-separated)")

if st.button("Predict"):
    data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
    prediction = model.predict(data)
    st.write(f"Prediction: {prediction}")
