%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

st.title("Bank Marketing Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit (yes/no).")

# ----------------------
# MODEL SELECTION
# ----------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Logistic Regression (GridSearch)",
        "Random Forest (GridSearch)"
    ]
)

@st.cache_resource
def load_model(filename):
    return joblib.load(filename)

# Load the correct model
if model_choice == "Logistic Regression":
    model = load_model("bank_marketing_model_lr.joblib")
elif model_choice == "Random Forest":
    model = load_model("bank_marketing_model_rf.joblib")
elif model_choice == "Logistic Regression (GridSearch)":
    model = load_model("logistic_regression_model_withGridSearch.joblib")
elif model_choice == "Random Forest (GridSearch)":
    model = load_model("random_forest_model_withGridSearch.joblib")

# ----------------------
# INPUT FIELDS
# ----------------------
st.sidebar.header("Input Customer Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.sidebar.number_input("Balance", min_value=-5000, max_value=200000, value=500)
duration = st.sidebar.number_input("Duration (seconds)", min_value=0, max_value=5000, value=120)

job = st.sidebar.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown"
])

marital = st.sidebar.selectbox("Marital Status", [
    "single", "married", "divorced", "unknown"
])

education = st.sidebar.selectbox("Education", [
    "primary", "secondary", "tertiary", "unknown"
])

contact = st.sidebar.selectbox("Contact Type", [
    "cellular", "telephone"
])

input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "duration": [duration],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "contact": [contact]
})

# ----------------------
# PREDICTION
# ----------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"Prediction: YES (Probability: {proba:.2f})")
        else:
            st.error(f"Prediction: NO (Probability: {proba:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ----------------------
# AVAILABLE MODELS
# ----------------------
st.subheader("Available Models")
st.markdown("- Logistic Regression (Base)")
st.markdown("- Random Forest (Base)")
st.markdown("- Logistic Regression (GridSearch best model)")
st.markdown("- Random Forest (GridSearch best model)")
