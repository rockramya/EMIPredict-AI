import streamlit as st
import pandas as pd
from pathlib import Path
import mlflow

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance")
st.write(
    "Compare the performance of classification and regression "
    "models developed for EMIPredict AI."
)

st.divider()

# ==================================================
# CLASSIFICATION
# ==================================================

st.header("🎯 EMI Eligibility Classification")

classification_data = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Balanced XGBoost"
    ],
    "Accuracy": [
        0.787624,
        0.934140,
        0.952816,
        0.885721
    ],
    "Precision": [
        0.919096,
        0.923218,
        0.945706,
        0.962456
    ],
    "Recall": [
        0.787624,
        0.934140,
        0.952816,
        0.885721
    ],
    "F1 Score": [
        0.838802,
        0.922145,
        0.934188,
        0.912852
    ],
    "ROC-AUC": [
        0.956447,
        0.989270,
        0.996127,
        0.994447
    ]
}

classification_df = pd.DataFrame(classification_data)

st.subheader("Classification Metrics")

st.dataframe(
    classification_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "ROC-AUC": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("Classification Model Comparison")

st.bar_chart(
    classification_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]
)

# ==================================================
# REGRESSION
# ==================================================

st.divider()

st.header("💰 Maximum Monthly EMI Regression")

regression_data = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "RMSE": [
        4024.988582,
        924.859315,
        770.326427
    ],
    "MAE": [
        2906.947016,
        211.535037,
        264.420615
    ],
    "R2": [
        0.725584,
        0.985511,
        0.989949
    ],
    "MAPE": [
        191.032968,
        5.408615,
        10.396039
    ]
}

regression_df = pd.DataFrame(regression_data)

st.subheader("Regression Metrics")

st.dataframe(
    regression_df.style.format({
        "RMSE": "₹{:,.2f}",
        "MAE": "₹{:,.2f}",
        "R2": "{:.4f}",
        "MAPE": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("Regression Model Comparison")

st.bar_chart(
    regression_df.set_index("Model")[
        ["RMSE", "MAE"]
    ]
)

# ==================================================
# SELECTED MODELS
# ==================================================

st.divider()

st.header("🏆 Selected Models")

col1, col2 = st.columns(2)

with col1:
    st.success(
        """
        ### 🎯 Classification

        **Selected Model: XGBoost**

        Accuracy: **95.28%**

        F1 Score: **93.42%**

        ROC-AUC: **99.61%**
        """
    )

with col2:
    st.success(
        """
        ### 💰 Regression

        **Selected Model: XGBoost**

        RMSE: **₹770.33**

        MAE: **₹264.42**

        R²: **0.9899**
        """
    )

st.info(
    "XGBoost was selected as the primary model for both tasks "
    "based on the overall evaluation results."
)

st.divider()

# ==================================================
# XGBoost Classification Diagnostics
# ==================================================

st.header("🔎 XGBoost Classification Diagnostics")

st.subheader("Confusion Matrix")

confusion_matrix_data = pd.DataFrame(
    [
        [14595, 31, 263],
        [1372, 118, 2007],
        [143, 4, 62427]
    ],
    index=[
        "Eligible",
        "High_Risk",
        "Not_Eligible"
    ],
    columns=[
        "Eligible",
        "High_Risk",
        "Not_Eligible"
    ]
)

st.dataframe(
    confusion_matrix_data,
    use_container_width=True
)

st.caption(
    "Rows represent the actual class and columns represent "
    "the predicted class."
)

st.subheader("Classification Report")

classification_report_data = pd.DataFrame({
    "Class": [
        "Eligible",
        "High_Risk",
        "Not_Eligible"
    ],
    "Precision": [
        0.905959,
        0.771242,
        0.964913
    ],
    "Recall": [
        0.980254,
        0.033743,
        0.997651
    ],
    "F1 Score": [
        0.941643,
        0.064658,
        0.981009
    ]
})

st.dataframe(
    classification_report_data.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

# ==================================================
# Confusion Matrix Heatmap
# ==================================================

st.subheader("🔥 Confusion Matrix Heatmap")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=confusion_matrix_data.columns,
    yticklabels=confusion_matrix_data.index,
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("XGBoost Classification - Confusion Matrix")

st.pyplot(fig)

plt.close(fig)


# ==================================================
# XGBoost Feature Importance
# ==================================================

st.divider()

st.header("🌟 XGBoost Feature Importance")

# Top 10 features from the trained model
feature_importance_data = pd.DataFrame({
    "Feature": [
        "Disposable Income",
        "Requested Amount",
        "Existing Loans - Yes",
        "Total Financial Burden",
        "Existing Loans - No",
        "Requested Tenure",
        "Current EMI Amount",
        "Vehicle EMI",
        "E-commerce Shopping EMI",
        "Salary Outlier"
    ],
    "Importance": [
        0.166125,
        0.093840,
        0.070580,
        0.057718,
        0.057628,
        0.053989,
        0.041011,
        0.039329,
        0.038495,
        0.036131
    ]
})

st.dataframe(
    feature_importance_data.style.format({
        "Importance": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("#### Top 10 Features")

st.bar_chart(
    feature_importance_data.set_index("Feature")
)

# ==================================================
# MLflow Experiment Tracking
# ==================================================

st.divider()

st.header("🔬 MLflow Experiment Tracking")

st.write(
    "MLflow is used to track model parameters, evaluation "
    "metrics, and experiment runs for EMIPredict AI."
)

try:

    # Project root directory
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # MLflow SQLite database
    mlflow_db_path = (PROJECT_ROOT / "mlflow.db").as_posix()

    # Connect MLflow to SQLite database
    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path}"
    )

    # Get experiment
    experiment = mlflow.get_experiment_by_name(
        "EMIPredict_AI"
    )

    if experiment is not None:

        # Retrieve experiment runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        st.subheader("📋 MLflow Runs")

        if not runs.empty:

            # Select useful columns
            display_columns = [
                "run_name",
                "tags.task",
                "tags.model_type",
                "status",
                "metrics.accuracy",
                "metrics.precision",
                "metrics.recall",
                "metrics.f1_score",
                "metrics.roc_auc",
                "metrics.rmse",
                "metrics.mae",
                "metrics.r2",
                "metrics.mape",
                "start_time"
            ]

            available_columns = [
                col for col in display_columns
                if col in runs.columns
            ]

            st.dataframe(
                runs[available_columns],
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"MLflow experiment loaded successfully. "
                f"{len(runs)} runs found."
            )

        else:

            st.warning(
                "No MLflow runs were found in this experiment."
            )

    else:

        st.warning(
            "MLflow experiment 'EMIPredict_AI' was not found."
        )

except Exception as e:

    st.error(
        f"Unable to load MLflow experiment: {e}"
    )

# ==================================================
# MLflow Model Registry
# ==================================================

st.divider()

st.header("🏛️ MLflow Model Registry")

st.write(
    "The Model Registry stores the selected production models "
    "and manages their versions."
)

try:

    # Connect to MLflow SQLite database
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    mlflow_db_path = (PROJECT_ROOT / "mlflow.db").as_posix()

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path}"
    )

    client = mlflow.tracking.MlflowClient()

    # Get registered model versions
    classification_versions = client.search_model_versions(
        "name='EMIPredict_Classification_Model'"
    )

    regression_versions = client.search_model_versions(
        "name='EMIPredict_Regression_Model'"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Classification Model")

        if classification_versions:

            version = classification_versions[0]

            st.success("Model Registered")

            st.write(
                f"**Model Name:** {version.name}"
            )

            st.write(
                f"**Version:** {version.version}"
            )

            st.write(
                f"**Run ID:** `{version.run_id}`"
            )

            st.write(
                "**Model:** XGBoost Classification"
            )

            st.write(
                "**Purpose:** EMI Eligibility Prediction"
            )

        else:

            st.warning(
                "Classification model is not registered."
            )

    with col2:

        st.subheader("💰 Regression Model")

        if regression_versions:

            version = regression_versions[0]

            st.success("Model Registered")

            st.write(
                f"**Model Name:** {version.name}"
            )

            st.write(
                f"**Version:** {version.version}"
            )

            st.write(
                f"**Run ID:** `{version.run_id}`"
            )

            st.write(
                "**Model:** XGBoost Regression"
            )

            st.write(
                "**Purpose:** Maximum Monthly EMI Prediction"
            )

        else:

            st.warning(
                "Regression model is not registered."
            )

except Exception as e:

    st.error(
        f"Unable to load Model Registry: {e}"
    )