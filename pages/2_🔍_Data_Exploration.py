import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Data Exploration",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Data Exploration")
st.write(
    "Explore the cleaned financial dataset used for "
    "EMIPredict AI."
)

st.divider()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

DATA_PATH = "data/cleaned_emi_dataset.csv"

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

expense_columns = [
    "monthly_rent",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses"
]

# Total monthly expenses
df["total_monthly_expenses"] = df[expense_columns].sum(axis=1)

# Expense-to-income ratio
df["expense_to_income"] = (
    df["total_monthly_expenses"] / df["monthly_salary"]
)

# Disposable income
df["disposable_income"] = (
    df["monthly_salary"]
    - df["total_monthly_expenses"]
    - df["current_emi_amount"]
)

# Current EMI-to-income ratio
df["emi_to_income"] = (
    df["current_emi_amount"] / df["monthly_salary"]
)

# Emergency fund-to-income ratio
df["savings_ratio"] = (
    df["emergency_fund"] / df["monthly_salary"]
)

# Bank balance-to-income ratio
df["bank_balance_to_income"] = (
    df["bank_balance"] / df["monthly_salary"]
)

# Total financial burden
df["total_financial_burden"] = (
    df["expense_to_income"]
    + df["emi_to_income"]
)

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

st.divider()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(100),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Interactive Filters
# --------------------------------------------------

st.subheader("🎛️ Interactive Filters")

col1, col2, col3 = st.columns(3)

with col1:
    selected_gender = st.multiselect(
        "Gender",
        options=sorted(df["gender"].unique()),
        default=sorted(df["gender"].unique())
    )

with col2:
    selected_employment = st.multiselect(
        "Employment Type",
        options=sorted(df["employment_type"].unique()),
        default=sorted(df["employment_type"].unique())
    )

with col3:
    selected_scenario = st.multiselect(
        "EMI Scenario",
        options=sorted(df["emi_scenario"].unique()),
        default=sorted(df["emi_scenario"].unique())
    )

# Apply filters
filtered_df = df[
    (df["gender"].isin(selected_gender)) &
    (df["employment_type"].isin(selected_employment)) &
    (df["emi_scenario"].isin(selected_scenario))
]

st.metric(
    "Filtered Records",
    f"{len(filtered_df):,}"
)

st.divider()

# --------------------------------------------------
# EMI Eligibility Distribution
# --------------------------------------------------

st.subheader("📊 EMI Eligibility Distribution")

eligibility_counts = (
    filtered_df["emi_eligibility"]
    .value_counts()
    .rename_axis("Eligibility")
    .reset_index(name="Count")
)

st.bar_chart(
    eligibility_counts,
    x="Eligibility",
    y="Count"
)

# --------------------------------------------------
# EMI Eligibility Distribution
# --------------------------------------------------

st.subheader("📊 EMI Eligibility Distribution")

eligibility_counts = (
    filtered_df["emi_eligibility"]
    .value_counts()
    .rename_axis("Eligibility")
    .reset_index(name="Count")
)

st.bar_chart(
    eligibility_counts,
    x="Eligibility",
    y="Count"
)

# --------------------------------------------------
# EMI Scenario Distribution
# --------------------------------------------------

st.subheader("🏦 EMI Scenario Distribution")

scenario_counts = (
    filtered_df["emi_scenario"]
    .value_counts()
    .rename_axis("EMI Scenario")
    .reset_index(name="Count")
)

st.bar_chart(
    scenario_counts,
    x="EMI Scenario",
    y="Count"
)

st.divider()

# --------------------------------------------------
# Maximum Monthly EMI Analysis
# --------------------------------------------------

st.subheader("💰 Maximum Monthly EMI Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Maximum EMI",
        f"₹{filtered_df['max_monthly_emi'].mean():,.2f}"
    )

with col2:
    st.metric(
        "Median Maximum EMI",
        f"₹{filtered_df['max_monthly_emi'].median():,.2f}"
    )

with col3:
    st.metric(
        "Minimum Maximum EMI",
        f"₹{filtered_df['max_monthly_emi'].min():,.2f}"
    )

with col4:
    st.metric(
        "Maximum EMI",
        f"₹{filtered_df['max_monthly_emi'].max():,.2f}"
    )

st.markdown("#### Maximum EMI Distribution")

# Create EMI ranges
emi_bins = [
    0,
    2000,
    5000,
    10000,
    15000,
    20000,
    30000,
    50000,
    float("inf")
]

emi_labels = [
    "Below ₹2K",
    "₹2K–₹5K",
    "₹5K–₹10K",
    "₹10K–₹15K",
    "₹15K–₹20K",
    "₹20K–₹30K",
    "₹30K–₹50K",
    "Above ₹50K"
]

emi_distribution = (
    pd.cut(
        filtered_df["max_monthly_emi"],
        bins=emi_bins,
        labels=emi_labels,
        include_lowest=True
    )
    .value_counts()
    .reindex(emi_labels)
)

st.bar_chart(emi_distribution)

st.divider()

# --------------------------------------------------
# Credit Score Analysis
# --------------------------------------------------

st.subheader("🏦 Credit Score Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Credit Score",
        f"{filtered_df['credit_score'].mean():.0f}"
    )

with col2:
    st.metric(
        "Minimum Credit Score",
        int(filtered_df["credit_score"].min())
    )

with col3:
    st.metric(
        "Maximum Credit Score",
        int(filtered_df["credit_score"].max())
    )

with col4:
    st.metric(
        "Median Credit Score",
        f"{filtered_df['credit_score'].median():.0f}"
    )

st.markdown("#### Credit Score Distribution")

credit_bins = [300, 550, 650, 700, 750, 800, 851]

credit_labels = [
    "300–549 (Poor)",
    "550–649 (Fair)",
    "650–699 (Good)",
    "700–749 (Very Good)",
    "750–799 (Excellent)",
    "800–850 (Exceptional)"
]

credit_distribution = (
    pd.cut(
        filtered_df["credit_score"],
        bins=credit_bins,
        labels=credit_labels,
        include_lowest=True
    )
    .value_counts()
    .reindex(credit_labels)
)

st.bar_chart(credit_distribution)

st.divider()

# --------------------------------------------------
# Salary Analysis
# --------------------------------------------------

st.subheader("💼 Salary vs EMI Eligibility")

salary_bins = [
    0,
    25000,
    50000,
    75000,
    100000,
    150000,
    250000,
    float("inf")
]

salary_labels = [
    "Below ₹25K",
    "₹25K–₹50K",
    "₹50K–₹75K",
    "₹75K–₹1L",
    "₹1L–₹1.5L",
    "₹1.5L–₹2.5L",
    "Above ₹2.5L"
]

salary_df = filtered_df.copy()

salary_df["Salary Range"] = pd.cut(
    salary_df["monthly_salary"],
    bins=salary_bins,
    labels=salary_labels,
    include_lowest=True
)

salary_summary = (
    salary_df
    .groupby("Salary Range")["emi_eligibility"]
    .value_counts()
    .unstack(fill_value=0)
)

st.bar_chart(salary_summary)

st.divider()

# --------------------------------------------------
# Financial KPI Dashboard
# --------------------------------------------------

st.subheader("📊 Financial KPI Dashboard")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Average Salary",
        f"₹{filtered_df['monthly_salary'].mean():,.0f}"
    )

with kpi2:
    st.metric(
        "Average Expenses",
        f"₹{filtered_df['total_monthly_expenses'].mean():,.0f}"
    )

with kpi3:
    st.metric(
        "Average Disposable Income",
        f"₹{filtered_df['disposable_income'].mean():,.0f}"
    )

with kpi4:
    st.metric(
        "Average Current EMI",
        f"₹{filtered_df['current_emi_amount'].mean():,.0f}"
    )