import streamlit as st

# Page configuration
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# Main title
st.title("💰 EMIPredict AI")

st.subheader("Intelligent Financial Risk Assessment Platform")

st.write(
    """
    Welcome to EMIPredict AI.

    This platform uses Machine Learning to assess EMI eligibility
    and predict the maximum affordable monthly EMI based on
    financial and personal information.
    """
)

# Project capabilities
st.markdown("### 🚀 Platform Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **📊 EMI Eligibility**

        Predict whether an applicant is:
        - Eligible
        - High Risk
        - Not Eligible
        """
    )

with col2:
    st.success(
        """
        **💰 Maximum EMI**

        Predict the maximum monthly EMI
        an applicant can potentially afford.
        """
    )

with col3:
    st.warning(
        """
        **📈 Model Analytics**

        Explore model performance,
        evaluation metrics and insights.
        """
    )

st.divider()

st.markdown(
    """
    ### 📌 How to use

    Use the navigation menu on the left to access:

    - **EMI Prediction** – Generate real-time predictions
    - **Data Exploration** – Explore the financial dataset
    - **Model Performance** – View ML model results
    - **Admin** – Manage application settings
    """
)