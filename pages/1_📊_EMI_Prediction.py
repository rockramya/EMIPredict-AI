import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

CLASSIFICATION_MODEL_PATH = "models/xgboost_classification_model.pkl"
LABEL_ENCODER_PATH = "models/classification_label_encoder.pkl"
REGRESSION_MODEL_PATH = "models/xgboost_regression_model.pkl"

classification_model = joblib.load(
    CLASSIFICATION_MODEL_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

regression_model = joblib.load(
    REGRESSION_MODEL_PATH
)

st.set_page_config(
    page_title="EMI Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EMI Prediction")
st.write(
    "Enter the applicant's financial and personal information "
    "to assess EMI eligibility and estimate the maximum monthly EMI."
)

st.divider()

# --------------------------------------------------
# Personal Information
# --------------------------------------------------

st.subheader("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Education",
        [
            "High School",
            "Graduate",
            "Post Graduate",
            "Professional"
        ]
    )

with col2:
    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=3
    )

st.divider()

# --------------------------------------------------
# Employment & Income
# --------------------------------------------------

st.subheader("💼 Employment & Income")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_salary = st.number_input(
        "Monthly Salary (₹)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col2:
    employment_type = st.selectbox(
        "Employment Type",
        [
            "Private",
            "Government",
            "Self-employed"
        ]
    )

with col3:
    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.1
    )

col1, col2 = st.columns(2)

with col1:
    company_type = st.selectbox(
        "Company Type",
        [
            "Mid-size",
            "MNC",
            "Startup",
            "Large Indian",
            "Small"
        ]
    )

with col2:
    house_type = st.selectbox(
        "House Type",
        [
            "Rented",
            "Family",
            "Own"
        ]
    )

st.divider()

# --------------------------------------------------
# Household Expenses
# --------------------------------------------------

st.subheader("🏠 Household Expenses")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_rent = st.number_input(
        "Monthly Rent (₹)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

with col2:
    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=1
    )

with col3:
    school_fees = st.number_input(
        "School Fees (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    college_fees = st.number_input(
        "College Fees (₹)",
        min_value=0.0,
        value=2000.0,
        step=500.0
    )

with col2:
    travel_expenses = st.number_input(
        "Travel Expenses (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

with col3:
    groceries_utilities = st.number_input(
        "Groceries & Utilities (₹)",
        min_value=0.0,
        value=8000.0,
        step=500.0
    )

other_monthly_expenses = st.number_input(
    "Other Monthly Expenses (₹)",
    min_value=0.0,
    value=3000.0,
    step=500.0
)

st.divider()

# --------------------------------------------------
# Financial Information
# --------------------------------------------------

st.subheader("💳 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    existing_loans = st.selectbox(
        "Existing Loans",
        ["No", "Yes"]
    )

with col2:
    current_emi_amount = st.number_input(
        "Current EMI Amount (₹)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

with col3:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )

col1, col2 = st.columns(2)

with col1:
    bank_balance = st.number_input(
        "Bank Balance (₹)",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )

with col2:
    emergency_fund = st.number_input(
        "Emergency Fund (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

st.divider()

# --------------------------------------------------
# Loan Request
# --------------------------------------------------

st.subheader("🏦 Loan Request")

col1, col2, col3 = st.columns(3)

with col1:
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "Personal Loan EMI",
            "E-commerce Shopping EMI",
            "Education EMI",
            "Vehicle EMI",
            "Home Appliances EMI"
        ]
    )

with col2:
    requested_amount = st.number_input(
        "Requested Amount (₹)",
        min_value=0.0,
        value=200000.0,
        step=5000.0
    )

with col3:
    requested_tenure = st.number_input(
        "Requested Tenure (Months)",
        min_value=1,
        max_value=120,
        value=24
    )

st.divider()

# --------------------------------------------------
# Prediction button
# --------------------------------------------------

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict EMI Eligibility",
    type="primary",
    use_container_width=True
):

    # ----------------------------------------------
    # Feature engineering
    # ----------------------------------------------

    total_monthly_expenses = (
        monthly_rent
        + school_fees
        + college_fees
        + travel_expenses
        + groceries_utilities
        + other_monthly_expenses
    )

    expense_to_income = (
        total_monthly_expenses / monthly_salary
        if monthly_salary > 0 else 0
    )

    disposable_income = (
        monthly_salary
        - total_monthly_expenses
        - current_emi_amount
    )

    emi_to_income = (
        current_emi_amount / monthly_salary
        if monthly_salary > 0 else 0
    )

    savings_ratio = (
        emergency_fund / monthly_salary
        if monthly_salary > 0 else 0
    )

    bank_balance_to_income = (
        bank_balance / monthly_salary
        if monthly_salary > 0 else 0
    )

    total_financial_burden = (
        expense_to_income + emi_to_income
    )

    # Salary outlier flag
    salary_outlier = int(
        monthly_salary < 15000
        or monthly_salary > 200000
    )

    # ----------------------------------------------
    # Create input DataFrame
    # ----------------------------------------------

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,

        # Engineered features
        "salary_outlier": salary_outlier,
        "total_monthly_expenses": total_monthly_expenses,
        "expense_to_income": expense_to_income,
        "disposable_income": disposable_income,
        "emi_to_income": emi_to_income,
        "savings_ratio": savings_ratio,
        "bank_balance_to_income": bank_balance_to_income,
        "total_financial_burden": total_financial_burden
    }])

    # ----------------------------------------------
    # Make predictions
    # ----------------------------------------------

    prediction_encoded = classification_model.predict(
        input_data
    )

    prediction = label_encoder.inverse_transform(
        prediction_encoded.astype(int)
    )[0]

    predicted_emi = regression_model.predict(
        input_data
    )[0]

    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    st.divider()

    # ----------------------------------------------
# Display results
# ----------------------------------------------

st.divider()

st.subheader("🎯 Prediction Results")

# Get prediction probabilities
prediction_probabilities = classification_model.predict_proba(
    input_data
)[0]

class_names = label_encoder.classes_

probability_dict = dict(
    zip(class_names, prediction_probabilities)
)

prediction_confidence = probability_dict[prediction] * 100

# Result columns
col1, col2 = st.columns(2)

with col1:

    st.markdown("### EMI Eligibility")

    if prediction == "Eligible":
        st.success("🟢 Eligible")

    elif prediction == "High_Risk":
        st.warning("🟠 High Risk")

    else:
        st.error("🔴 Not Eligible")

    st.metric(
        "Prediction Confidence",
        f"{prediction_confidence:.2f}%"
    )

with col2:

    st.markdown("### Maximum Monthly EMI")

    st.metric(
        "Predicted EMI",
        f"₹{predicted_emi:,.2f}"
    )

# ----------------------------------------------
# Class probabilities
# ----------------------------------------------

st.markdown("### 📊 Eligibility Probability")

prob_col1, prob_col2, prob_col3 = st.columns(3)

with prob_col1:
    st.metric(
        "Eligible",
        f"{probability_dict['Eligible'] * 100:.2f}%"
    )

with prob_col2:
    st.metric(
        "High Risk",
        f"{probability_dict['High_Risk'] * 100:.2f}%"
    )

with prob_col3:
    st.metric(
        "Not Eligible",
        f"{probability_dict['Not_Eligible'] * 100:.2f}%"
    )

# ----------------------------------------------
# Financial summary
# ----------------------------------------------

st.markdown("### 💰 Financial Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "Monthly Salary",
        f"₹{monthly_salary:,.2f}"
    )

with summary_col2:
    st.metric(
        "Monthly Expenses",
        f"₹{total_monthly_expenses:,.2f}"
    )

with summary_col3:
    st.metric(
        "Disposable Income",
        f"₹{disposable_income:,.2f}"
    )

# ----------------------------------------------
# Model explanation
# ----------------------------------------------

st.markdown("### 💡 Assessment Summary")

if prediction == "Eligible":

    st.success(
        "The model predicts that the applicant is eligible "
        "for the requested EMI scenario."
    )

elif prediction == "High_Risk":

    st.warning(
        "The model identifies the applicant as high risk. "
        "Additional financial review may be appropriate."
    )

else:

    st.error(
        "The model predicts that the applicant is not eligible "
        "under the current financial profile."
    )

st.info(
    "The maximum monthly EMI shown above is the regression "
    "model's predicted affordable EMI estimate. It should be "
    "considered alongside the eligibility classification."
)