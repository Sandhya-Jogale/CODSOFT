import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below to predict survival on the Titanic.")

# Input Features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex_input = st.selectbox("Sex", ['male', 'female'])
sex = 1 if sex_input == 'male' else 0

age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 250, 50)

sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

# Feature Engineering
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Embarked Encoding
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Age Group Encoding
agegroup = st.selectbox("Age Group", ['Child', 'Teen', 'Adult', 'Senior'])
agegroup_teen = 1 if agegroup == 'Teen' else 0
agegroup_adult = 1 if agegroup == 'Adult' else 0
agegroup_senior = 1 if agegroup == 'Senior' else 0

# Fare Bin Encoding
farebin = st.selectbox("Fare Category", ['Low', 'Mid', 'High', 'VeryHigh'])
farebin_mid = 1 if farebin == 'Mid' else 0
farebin_high = 1 if farebin == 'High' else 0
farebin_veryhigh = 1 if farebin == 'VeryHigh' else 0

# Input DataFrame
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'Fare': fare,
    'SibSp': sibsp,
    'Parch': parch,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Embarked_Q': embarked_q,
    'Embarked_S': embarked_s,
    'AgeGroup_Teen': agegroup_teen,
    'AgeGroup_Adult': agegroup_adult,
    'AgeGroup_Senior': agegroup_senior,
    'FareBin_Mid': farebin_mid,
    'FareBin_High': farebin_high,
    'FareBin_VeryHigh': farebin_veryhigh
}])

# Load model
try:
    model = joblib.load("titanic_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'titanic_model.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to **survive**.")
    else:
        st.error("❌ The passenger is **not likely to survive**.")
