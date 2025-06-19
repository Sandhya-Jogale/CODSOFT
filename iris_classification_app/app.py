import streamlit as st
import numpy as np
import joblib

# Load pre-trained models and label encoder
log_reg = joblib.load("logistic_model.pkl")
knn = joblib.load("knn_model.pkl")
dt = joblib.load("dt_model.pkl")
le = joblib.load("label_encoder.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Species Prediction App")
st.markdown("Enter sepal and petal dimensions below to predict the Iris flower species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Model selection
model_choice = st.selectbox("Choose Classifier", ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree"])

# Map choice to model
model_map = {
    "Logistic Regression": log_reg,
    "K-Nearest Neighbors": knn,
    "Decision Tree": dt
}
model = model_map[model_choice]

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = le.inverse_transform(prediction)[0]
    st.success(f"The predicted Iris species is **{species}** 🌺")
