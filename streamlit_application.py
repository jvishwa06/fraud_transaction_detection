import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("xgb_classifier_model.pkl")

rf = load_model()

st.title("Credit Card Fraud Detection App 🏦")
st.write("Enter transaction details to predict if it's fraudulent or not.")

amt = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=50.0)
gender = st.selectbox("Gender", ["Male", "Female"])
city_pop = st.number_input("City Population", min_value=0, max_value=10**7, value=10000)
distance = st.number_input("Distance (km) from Merchant", min_value=0.0, max_value=5000.0, value=10.0)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
fraud_merchant_pct = st.slider("Fraud Percentage of Merchant", min_value=0.0, max_value=100.0, value=5.0)
state_encoded = st.slider("State Frequency Encoding", min_value=0.0, max_value=1.0, value=0.05)
job_encoded = st.slider("Job Frequency Encoding", min_value=0.0, max_value=1.0, value=0.01)
day_period = st.selectbox("Transaction Time", ["Night", "Evening", "Afternoon", "Morning"], index=1)
trans_month = st.selectbox("Transaction Month", ["February", "January", "March", "May", "October", "September","November", "April", "August", "June", "July", "December"], index=0)

categories = [
    "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"
]
category = st.selectbox("Transaction Category", categories)

input_data = pd.DataFrame({
    "amt": [amt],
    "gender": [1 if gender == "Male" else 0],
    "city_pop": [city_pop],
    "distance": [distance],
    "age": [age],
    "fraud_merchant_pct": [fraud_merchant_pct],
    "state_encoded": [state_encoded],
    "job_encoded": [job_encoded],
    "day_period": [["Night", "Evening", "Afternoon", "Morning"].index(day_period)],
    "trans_month": [["February", "January", "March", "May", "October", "September","November", "April", "August", "June", "July", "December"].index(trans_month)]
})

for cat in categories:
    input_data[f"category_{cat}"] = [1 if cat == category else 0]

if st.button("Predict Fraudulence 🚀"):
    prediction = rf.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected! ⚠️")
    else:
        st.success("✅ Transaction is Safe!")