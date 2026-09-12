# EMIPredict AI

## Intelligent Financial Risk Assessment Platform

EMIPredict AI is a machine learning-based financial risk assessment platform designed to evaluate EMI eligibility and estimate the maximum affordable monthly EMI for loan applicants.

The project combines Machine Learning, Feature Engineering, MLflow experiment tracking, and Streamlit to provide an interactive end-to-end financial risk assessment application.

---

## 🚀 Project Overview

The platform solves two machine learning problems:

1. **EMI Eligibility Classification**
   - Predicts whether an applicant is:
     - Eligible
     - High Risk
     - Not Eligible

2. **Maximum EMI Prediction**
   - Predicts the maximum monthly EMI amount that an applicant can reasonably afford.

The application provides separate interfaces for prediction, data exploration, model performance analysis, and administrative CRUD operations.

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │   Financial Dataset  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning &      │
                    │ Preprocessing        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ & EDA                │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Machine Learning          │
              │                                 │
              │ Classification + Regression     │
              └──────────────┬──────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ MLflow Tracking &    │
                    │ Model Registry       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Web App     │
                    └──────────────────────┘
