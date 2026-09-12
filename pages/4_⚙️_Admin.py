import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Admin",
    page_icon="⚙️",
    layout="wide"
)


# ==================================================
# PAGE HEADER
# ==================================================

st.title("⚙️ Admin Panel")

st.write(
    "Manage application data and platform settings "
    "for EMIPredict AI."
)

st.divider()


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==================================================
# LOAD ML MODELS
# ==================================================

CLASSIFICATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_classification_model.pkl"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification_label_encoder.pkl"
)

REGRESSION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_regression_model.pkl"
)


classification_model = joblib.load(
    CLASSIFICATION_MODEL_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

regression_model = joblib.load(
    REGRESSION_MODEL_PATH
)


# ==================================================
# LOAD DATA
# ==================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned_emi_dataset.csv"
)

df = pd.read_csv(DATA_PATH)


st.success(
    f"Dataset loaded successfully — {len(df):,} records."
)


# ==================================================
# ADMIN DASHBOARD
# ==================================================

st.header("📊 Dataset Management")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Total Columns",
        len(df.columns)
    )


with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


st.divider()


# ==================================================
# DATA PREVIEW
# ==================================================

st.subheader("📋 Data Preview")

st.dataframe(
    df.head(100),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==================================================
# DATASET INFORMATION
# ==================================================

st.subheader("🗂️ Dataset Information")


dataset_info = pd.DataFrame({

    "Column": df.columns,

    "Data Type": (
        df.dtypes.astype(str).values
    ),

    "Missing Values": (
        df.isnull().sum().values
    ),

    "Unique Values": [
        df[column].nunique()
        for column in df.columns
    ]

})


st.dataframe(
    dataset_info,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# SEARCH & FILTER RECORDS
# ==================================================

st.divider()

st.header("🔎 Search & Filter Records")


search_column = st.selectbox(
    "Select column to search",
    df.columns
)


search_value = st.text_input(
    f"Enter value for {search_column}"
)


if search_value:

    filtered_df = df[
        df[search_column]
        .astype(str)
        .str.contains(
            search_value,
            case=False,
            na=False
        )
    ]

    st.write(
        f"Found **{len(filtered_df):,}** matching records."
    )

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Enter a value above to search the dataset."
    )


# ==================================================
# CREATE NEW RECORD
# ==================================================

st.divider()

st.header("➕ Add New Record")


with st.form("add_record_form"):

    # ==================================================
    # PERSONAL INFORMATION
    # ==================================================

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )


    with col2:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single"
            ]
        )

        education = st.selectbox(
            "Education",
            [
                "Professional",
                "Graduate",
                "High School",
                "Post Graduate"
            ]
        )


    with col3:

        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=20,
            value=4
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=20,
            value=1
        )


    # ==================================================
    # EMPLOYMENT & INCOME
    # ==================================================

    st.subheader("Employment & Income")

    col1, col2, col3 = st.columns(3)


    with col1:

        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=1.0,
            value=30000.0
        )

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed"
            ]
        )


    with col2:

        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0
        )

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


    with col3:

        house_type = st.selectbox(
            "House Type",
            [
                "Rented",
                "Family",
                "Own"
            ]
        )

        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=8000.0
        )


    # ==================================================
    # HOUSEHOLD EXPENSES
    # ==================================================

    st.subheader("Household Expenses")

    col1, col2, col3 = st.columns(3)


    with col1:

        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=2000.0
        )

        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0
        )


    with col2:

        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=3000.0
        )

        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=8000.0
        )


    with col3:

        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=3000.0
        )

        current_emi_amount = st.number_input(
            "Current EMI Amount (₹)",
            min_value=0.0,
            value=3000.0
        )


    # ==================================================
    # FINANCIAL INFORMATION
    # ==================================================

    st.subheader("Financial Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        existing_loans = st.selectbox(
            "Existing Loans",
            [
                "Yes",
                "No"
            ]
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700
        )


    with col2:

        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=50000.0
        )

        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0.0,
            value=30000.0
        )


    # ==================================================
    # LOAN INFORMATION
    # ==================================================

    st.subheader("Loan Information")

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
            value=100000.0
        )


    with col3:

        requested_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=1,
            max_value=120,
            value=24
        )


    # ==================================================
    # SUBMIT BUTTON
    # ==================================================

    submitted = st.form_submit_button(
        "➕ Add Record"
    )


    # ==================================================
    # PROCESS NEW RECORD
    # ==================================================

    if submitted:

        # ----------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------

        if monthly_salary <= 0:

            st.error(
                "Monthly salary must be greater than ₹0."
            )

        elif credit_score < 300 or credit_score > 850:

            st.error(
                "Credit score must be between 300 and 850."
            )

        elif requested_amount <= 0:

            st.error(
                "Requested amount must be greater than ₹0."
            )

        else:

            # ------------------------------------------
            # FEATURE ENGINEERING
            # ------------------------------------------

            total_monthly_expenses = (
                monthly_rent
                + school_fees
                + college_fees
                + travel_expenses
                + groceries_utilities
                + other_monthly_expenses
            )


            expense_to_income = (
                total_monthly_expenses
                / monthly_salary
            )


            disposable_income = (
                monthly_salary
                - total_monthly_expenses
                - current_emi_amount
            )


            emi_to_income = (
                current_emi_amount
                / monthly_salary
            )


            savings_ratio = (
                emergency_fund
                / monthly_salary
            )


            bank_balance_to_income = (
                bank_balance
                / monthly_salary
            )


            total_financial_burden = (
                expense_to_income
                + emi_to_income
            )


            salary_outlier = int(
                monthly_salary < 15000
                or monthly_salary > 200000
            )


            # ------------------------------------------
            # MODEL INPUT
            # ------------------------------------------

            model_input = pd.DataFrame([{

                "age": age,

                "gender": gender,

                "marital_status": marital_status,

                "education": education,

                "monthly_salary": monthly_salary,

                "employment_type": employment_type,

                "years_of_employment": (
                    years_of_employment
                ),

                "company_type": company_type,

                "house_type": house_type,

                "monthly_rent": monthly_rent,

                "family_size": family_size,

                "dependents": dependents,

                "school_fees": school_fees,

                "college_fees": college_fees,

                "travel_expenses": travel_expenses,

                "groceries_utilities": (
                    groceries_utilities
                ),

                "other_monthly_expenses": (
                    other_monthly_expenses
                ),

                "existing_loans": existing_loans,

                "current_emi_amount": (
                    current_emi_amount
                ),

                "credit_score": credit_score,

                "bank_balance": bank_balance,

                "emergency_fund": emergency_fund,

                "emi_scenario": emi_scenario,

                "requested_amount": (
                    requested_amount
                ),

                "requested_tenure": (
                    requested_tenure
                ),

                "salary_outlier": salary_outlier,

                "total_monthly_expenses": (
                    total_monthly_expenses
                ),

                "expense_to_income": (
                    expense_to_income
                ),

                "disposable_income": (
                    disposable_income
                ),

                "emi_to_income": (
                    emi_to_income
                ),

                "savings_ratio": (
                    savings_ratio
                ),

                "bank_balance_to_income": (
                    bank_balance_to_income
                ),

                "total_financial_burden": (
                    total_financial_burden
                )

            }])


            # ------------------------------------------
            # CLASSIFICATION PREDICTION
            # ------------------------------------------

            predicted_class_encoded = (
                classification_model
                .predict(model_input)[0]
            )


            predicted_eligibility = (
                label_encoder
                .inverse_transform(
                    [predicted_class_encoded]
                )[0]
            )


            # ------------------------------------------
            # REGRESSION PREDICTION
            # ------------------------------------------

            predicted_max_emi = (
                regression_model
                .predict(model_input)[0]
            )


            predicted_max_emi = max(
                0,
                float(predicted_max_emi)
            )


            # ------------------------------------------
            # CREATE NEW RECORD
            # ------------------------------------------

            new_record = {

                "age": age,

                "gender": gender,

                "marital_status": marital_status,

                "education": education,

                "monthly_salary": monthly_salary,

                "employment_type": employment_type,

                "years_of_employment": (
                    years_of_employment
                ),

                "company_type": company_type,

                "house_type": house_type,

                "monthly_rent": monthly_rent,

                "family_size": family_size,

                "dependents": dependents,

                "school_fees": school_fees,

                "college_fees": college_fees,

                "travel_expenses": travel_expenses,

                "groceries_utilities": (
                    groceries_utilities
                ),

                "other_monthly_expenses": (
                    other_monthly_expenses
                ),

                "existing_loans": existing_loans,

                "current_emi_amount": (
                    current_emi_amount
                ),

                "credit_score": credit_score,

                "bank_balance": bank_balance,

                "emergency_fund": emergency_fund,

                "emi_scenario": emi_scenario,

                "requested_amount": (
                    requested_amount
                ),

                "requested_tenure": (
                    requested_tenure
                ),

                # ML-generated targets
                "emi_eligibility": (
                    predicted_eligibility
                ),

                "max_monthly_emi": (
                    predicted_max_emi
                )

            }


            # ------------------------------------------
            # ADD RECORD TO DATASET
            # ------------------------------------------

            new_row = pd.DataFrame(
                [new_record]
            )


            df = pd.concat(
                [
                    df,
                    new_row
                ],
                ignore_index=True
            )


            # ------------------------------------------
            # SAVE DATASET
            # ------------------------------------------

            df.to_csv(
                DATA_PATH,
                index=False
            )


            # ------------------------------------------
            # SUCCESS MESSAGE
            # ------------------------------------------

            st.success(
                "✅ New record added successfully!"
            )


            st.success(
                f"🎯 Predicted Eligibility: "
                f"**{predicted_eligibility}**"
            )


            st.success(
                f"💰 Predicted Maximum EMI: "
                f"**₹{predicted_max_emi:,.2f}**"
            )


            st.info(
                "The record and its ML-generated "
                "predictions have been saved to the "
                "cleaned dataset."
            )


# ============================================================
# UPDATE RECORD
# ============================================================

st.divider()

st.header("✏️ Update Existing Record")

st.write("Select a record using its row number, modify the details, and save the updated record.")

# Create a row-number column for selection
update_df = df.copy()
update_df["row_number"] = update_df.index

if len(update_df) > 0:

    selected_row = st.selectbox(
        "Select Row Number to Update",
        update_df["row_number"].tolist()
    )

    selected_record = df.loc[selected_row]

    st.subheader(f"Editing Record: Row {selected_row}")

    with st.form("update_record_form"):

        # ----------------------------------------------------
        # Personal Information
        # ----------------------------------------------------

        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            update_age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=int(selected_record["age"]),
                key="update_age"
            )

        with col2:
            update_gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=["Male", "Female"].index(selected_record["gender"])
                if selected_record["gender"] in ["Male", "Female"] else 0,
                key="update_gender"
            )

        with col3:
            update_marital_status = st.selectbox(
                "Marital Status",
                ["Married", "Single"],
                index=["Married", "Single"].index(selected_record["marital_status"])
                if selected_record["marital_status"] in ["Married", "Single"] else 0,
                key="update_marital_status"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            education_options = [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional"
            ]

            update_education = st.selectbox(
                "Education",
                education_options,
                index=education_options.index(selected_record["education"])
                if selected_record["education"] in education_options else 0,
                key="update_education"
            )

        with col2:
            update_family_size = st.number_input(
                "Family Size",
                min_value=1,
                max_value=20,
                value=int(selected_record["family_size"]),
                key="update_family_size"
            )

        with col3:
            update_dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=20,
                value=int(selected_record["dependents"]),
                key="update_dependents"
            )

        # ----------------------------------------------------
        # Employment Information
        # ----------------------------------------------------

        st.markdown("### 💼 Employment Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            update_salary = st.number_input(
                "Monthly Salary",
                min_value=1.0,
                value=float(selected_record["monthly_salary"]),
                step=1000.0,
                key="update_salary"
            )

        with col2:
            employment_options = [
                "Private",
                "Government",
                "Self-employed"
            ]

            update_employment_type = st.selectbox(
                "Employment Type",
                employment_options,
                index=employment_options.index(selected_record["employment_type"])
                if selected_record["employment_type"] in employment_options else 0,
                key="update_employment_type"
            )

        with col3:
            update_years_employment = st.number_input(
                "Years of Employment",
                min_value=0.0,
                max_value=50.0,
                value=float(selected_record["years_of_employment"]),
                step=1.0,
                key="update_years_employment"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            company_options = [
                "Mid-size",
                "MNC",
                "Startup",
                "Large Indian",
                "Small"
            ]

            update_company_type = st.selectbox(
                "Company Type",
                company_options,
                index=company_options.index(selected_record["company_type"])
                if selected_record["company_type"] in company_options else 0,
                key="update_company_type"
            )

        with col2:
            house_options = [
                "Rented",
                "Family",
                "Own"
            ]

            update_house_type = st.selectbox(
                "House Type",
                house_options,
                index=house_options.index(selected_record["house_type"])
                if selected_record["house_type"] in house_options else 0,
                key="update_house_type"
            )

        with col3:
            update_rent = st.number_input(
                "Monthly Rent",
                min_value=0.0,
                value=float(selected_record["monthly_rent"]),
                step=500.0,
                key="update_rent"
            )

        # ----------------------------------------------------
        # Household Expenses
        # ----------------------------------------------------

        st.markdown("### 🏠 Household Expenses")

        col1, col2, col3 = st.columns(3)

        with col1:
            update_school_fees = st.number_input(
                "School Fees",
                min_value=0.0,
                value=float(selected_record["school_fees"]),
                step=500.0,
                key="update_school_fees"
            )

        with col2:
            update_college_fees = st.number_input(
                "College Fees",
                min_value=0.0,
                value=float(selected_record["college_fees"]),
                step=500.0,
                key="update_college_fees"
            )

        with col3:
            update_travel = st.number_input(
                "Travel Expenses",
                min_value=0.0,
                value=float(selected_record["travel_expenses"]),
                step=500.0,
                key="update_travel"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            update_groceries = st.number_input(
                "Groceries & Utilities",
                min_value=0.0,
                value=float(selected_record["groceries_utilities"]),
                step=500.0,
                key="update_groceries"
            )

        with col2:
            update_other_expenses = st.number_input(
                "Other Monthly Expenses",
                min_value=0.0,
                value=float(selected_record["other_monthly_expenses"]),
                step=500.0,
                key="update_other_expenses"
            )

        with col3:
            update_current_emi = st.number_input(
                "Current EMI Amount",
                min_value=0.0,
                value=float(selected_record["current_emi_amount"]),
                step=500.0,
                key="update_current_emi"
            )

        # ----------------------------------------------------
        # Financial Information
        # ----------------------------------------------------

        st.markdown("### 💰 Financial Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            loan_options = ["Yes", "No"]

            update_existing_loans = st.selectbox(
                "Existing Loans",
                loan_options,
                index=loan_options.index(selected_record["existing_loans"])
                if selected_record["existing_loans"] in loan_options else 0,
                key="update_existing_loans"
            )

        with col2:
            update_credit_score = st.number_input(
                "Credit Score",
                min_value=300.0,
                max_value=850.0,
                value=float(selected_record["credit_score"]),
                step=1.0,
                key="update_credit_score"
            )

        with col3:
            update_bank_balance = st.number_input(
                "Bank Balance",
                min_value=0.0,
                value=float(selected_record["bank_balance"]),
                step=1000.0,
                key="update_bank_balance"
            )

        update_emergency_fund = st.number_input(
            "Emergency Fund",
            min_value=0.0,
            value=float(selected_record["emergency_fund"]),
            step=1000.0,
            key="update_emergency_fund"
        )

        # ----------------------------------------------------
        # Loan Information
        # ----------------------------------------------------

        st.markdown("### 🏦 Loan Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            scenario_options = [
                "Personal Loan EMI",
                "E-commerce Shopping EMI",
                "Education EMI",
                "Vehicle EMI",
                "Home Appliances EMI"
            ]

            update_emi_scenario = st.selectbox(
                "EMI Scenario",
                scenario_options,
                index=scenario_options.index(selected_record["emi_scenario"])
                if selected_record["emi_scenario"] in scenario_options else 0,
                key="update_emi_scenario"
            )

        with col2:
            update_requested_amount = st.number_input(
                "Requested Amount",
                min_value=1.0,
                value=float(selected_record["requested_amount"]),
                step=1000.0,
                key="update_requested_amount"
            )

        with col3:
            update_requested_tenure = st.number_input(
                "Requested Tenure (Months)",
                min_value=1,
                max_value=120,
                value=int(selected_record["requested_tenure"]),
                step=1,
                key="update_requested_tenure"
            )

        # ----------------------------------------------------
        # Submit
        # ----------------------------------------------------

        update_submitted = st.form_submit_button(
            "💾 Update Record",
            use_container_width=True
        )

    # ========================================================
    # PROCESS UPDATE
    # ========================================================

    if update_submitted:

        # Basic validation
        if update_salary <= 0:
            st.error("Monthly salary must be greater than 0.")

        elif not (300 <= update_credit_score <= 850):
            st.error("Credit score must be between 300 and 850.")

        elif update_requested_amount <= 0:
            st.error("Requested amount must be greater than 0.")

        else:

            # ------------------------------------------------
            # Feature Engineering
            # ------------------------------------------------

            total_monthly_expenses = (
                update_rent
                + update_school_fees
                + update_college_fees
                + update_travel
                + update_groceries
                + update_other_expenses
            )

            expense_to_income = (
                total_monthly_expenses / update_salary
            )

            disposable_income = (
                update_salary
                - total_monthly_expenses
                - update_current_emi
            )

            emi_to_income = (
                update_current_emi / update_salary
            )

            savings_ratio = (
                update_emergency_fund / update_salary
            )

            bank_balance_to_income = (
                update_bank_balance / update_salary
            )

            total_financial_burden = (
                expense_to_income + emi_to_income
            )

            salary_outlier = int(
                update_salary < 15000
                or update_salary > 200000
            )

            # ------------------------------------------------
            # Prepare input for ML models
            # ------------------------------------------------

            model_input = pd.DataFrame([{
                "age": update_age,
                "gender": update_gender,
                "marital_status": update_marital_status,
                "education": update_education,
                "monthly_salary": update_salary,
                "employment_type": update_employment_type,
                "years_of_employment": update_years_employment,
                "company_type": update_company_type,
                "house_type": update_house_type,
                "monthly_rent": update_rent,
                "family_size": update_family_size,
                "dependents": update_dependents,
                "school_fees": update_school_fees,
                "college_fees": update_college_fees,
                "travel_expenses": update_travel,
                "groceries_utilities": update_groceries,
                "other_monthly_expenses": update_other_expenses,
                "existing_loans": update_existing_loans,
                "current_emi_amount": update_current_emi,
                "credit_score": update_credit_score,
                "bank_balance": update_bank_balance,
                "emergency_fund": update_emergency_fund,
                "emi_scenario": update_emi_scenario,
                "requested_amount": update_requested_amount,
                "requested_tenure": update_requested_tenure,
                "salary_outlier": salary_outlier,
                "total_monthly_expenses": total_monthly_expenses,
                "expense_to_income": expense_to_income,
                "disposable_income": disposable_income,
                "emi_to_income": emi_to_income,
                "savings_ratio": savings_ratio,
                "bank_balance_to_income": bank_balance_to_income,
                "total_financial_burden": total_financial_burden
            }])

            # ------------------------------------------------
            # ML Predictions
            # ------------------------------------------------

            predicted_class_encoded = classification_model.predict(
                model_input
            )[0]

            predicted_eligibility = label_encoder.inverse_transform(
                [predicted_class_encoded]
            )[0]

            predicted_max_emi = max(
                0,
                float(regression_model.predict(model_input)[0])
            )

            # ------------------------------------------------
            # Update DataFrame
            # ------------------------------------------------

            df.loc[selected_row, "age"] = update_age
            df.loc[selected_row, "gender"] = update_gender
            df.loc[selected_row, "marital_status"] = update_marital_status
            df.loc[selected_row, "education"] = update_education
            df.loc[selected_row, "monthly_salary"] = update_salary
            df.loc[selected_row, "employment_type"] = update_employment_type
            df.loc[selected_row, "years_of_employment"] = update_years_employment
            df.loc[selected_row, "company_type"] = update_company_type
            df.loc[selected_row, "house_type"] = update_house_type
            df.loc[selected_row, "monthly_rent"] = update_rent
            df.loc[selected_row, "family_size"] = update_family_size
            df.loc[selected_row, "dependents"] = update_dependents
            df.loc[selected_row, "school_fees"] = update_school_fees
            df.loc[selected_row, "college_fees"] = update_college_fees
            df.loc[selected_row, "travel_expenses"] = update_travel
            df.loc[selected_row, "groceries_utilities"] = update_groceries
            df.loc[selected_row, "other_monthly_expenses"] = update_other_expenses
            df.loc[selected_row, "existing_loans"] = update_existing_loans
            df.loc[selected_row, "current_emi_amount"] = update_current_emi
            df.loc[selected_row, "credit_score"] = update_credit_score
            df.loc[selected_row, "bank_balance"] = update_bank_balance
            df.loc[selected_row, "emergency_fund"] = update_emergency_fund
            df.loc[selected_row, "emi_scenario"] = update_emi_scenario
            df.loc[selected_row, "requested_amount"] = update_requested_amount
            df.loc[selected_row, "requested_tenure"] = update_requested_tenure

            # Update engineered columns
            df.loc[selected_row, "total_monthly_expenses"] = total_monthly_expenses
            df.loc[selected_row, "expense_to_income"] = expense_to_income
            df.loc[selected_row, "disposable_income"] = disposable_income
            df.loc[selected_row, "emi_to_income"] = emi_to_income
            df.loc[selected_row, "savings_ratio"] = savings_ratio
            df.loc[selected_row, "bank_balance_to_income"] = bank_balance_to_income
            df.loc[selected_row, "total_financial_burden"] = total_financial_burden
            df.loc[selected_row, "salary_outlier"] = salary_outlier

            # Update ML-generated targets
            df.loc[selected_row, "emi_eligibility"] = predicted_eligibility
            df.loc[selected_row, "max_monthly_emi"] = predicted_max_emi

            # ------------------------------------------------
            # Save Updated Dataset
            # ------------------------------------------------

            df.to_csv(DATA_PATH, index=False)

            st.success("✅ Record updated successfully!")

            st.info(
                f"🎯 Updated Eligibility: **{predicted_eligibility}**"
            )

            st.info(
                f"💰 Updated Maximum EMI: **₹{predicted_max_emi:,.2f}**"
            )

            st.info(
                "The updated record and its ML-generated predictions "
                "have been saved to the cleaned dataset."
            )

# ============================================================
# DELETE RECORD
# ============================================================

st.divider()

st.header("🗑️ Delete Record")

st.write(
    "Select a record using its row number and delete it from the dataset."
)

# Create a copy with row numbers
delete_df = df.copy()
delete_df["row_number"] = delete_df.index

if len(delete_df) > 0:

    selected_delete_row = st.selectbox(
        "Select Row Number to Delete",
        delete_df["row_number"].tolist(),
        key="delete_row"
    )

    # Display selected record
    selected_delete_record = df.loc[selected_delete_row]

    st.subheader(f"Selected Record: Row {selected_delete_row}")

    # Display important information before deletion
    display_columns = [
        "age",
        "gender",
        "monthly_salary",
        "employment_type",
        "emi_scenario",
        "requested_amount",
        "emi_eligibility",
        "max_monthly_emi"
    ]

    st.dataframe(
        pd.DataFrame([selected_delete_record[display_columns]]),
        use_container_width=True
    )

    st.warning(
        "⚠️ This action will permanently remove the selected record "
        "from the cleaned dataset."
    )

    # Confirmation checkbox
    confirm_delete = st.checkbox(
        "I understand that this record will be permanently deleted.",
        key="confirm_delete"
    )

    # Delete button
    delete_button = st.button(
        "🗑️ Delete Selected Record",
        type="primary",
        use_container_width=True
    )

    if delete_button:

        if not confirm_delete:

            st.error(
                "Please confirm the deletion by checking the "
                "confirmation box."
            )

        else:

            # Remove selected record
            df = df.drop(index=selected_delete_row)

            # Reset index after deletion
            df = df.reset_index(drop=True)

            # Save updated dataset
            df.to_csv(DATA_PATH, index=False)

            st.success(
                "✅ Record deleted successfully!"
            )

            st.info(
                f"📊 Remaining records: **{len(df):,}**"
            )

            st.info(
                "The deleted record has been removed from the "
                "cleaned dataset."
            )