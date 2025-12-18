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

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox(
    "Job",
    ["admin", "technician", "services", "management", "retired",
     "blue-collar", "unemployed", "entrepreneur", "housemaid",
     "self-employed", "student", "unknown"]
)
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)
default = st.selectbox("Credit in Default?", ["yes", "no"])
housing = st.selectbox("Housing Loan?", ["yes", "no"])
loan = st.selectbox("Personal Loan?", ["yes", "no"])
balance = st.number_input("Account Balance", value=0)
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
month = st.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)
duration = st.number_input("Call Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts (Campaign)", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact", min_value=0, value=0)
previous = st.number_input("Previous Contacts", min_value=0, value=0)
poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["success", "failure", "other", "unknown"]
)
input_data = pd.DataFrame([{
    "age": age,
    "balance": balance,
    "duration": duration,
}])


#df
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])
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
