import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("movie_rating_predictor.pkl")

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")
st.title("🎥 Movie Rating Prediction App")
st.markdown("Enter the details of a movie below and get its predicted IMDb rating!")

# Input fields
genre = st.text_input("Genre (e.g. Action, Drama, Comedy)")
director = st.text_input("Director Name")
actor1 = st.text_input("Lead Actor")
actor2 = st.text_input("Supporting Actor 1")
actor3 = st.text_input("Supporting Actor 2")

# Predict button
if st.button("Predict Rating"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'Genre': genre,
        'Director': director,
        'Actor 1': actor1,
        'Actor 2': actor2,
        'Actor 3': actor3
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result
    st.success(f"🎯 Predicted IMDb Rating: **{prediction:.2f}**")
