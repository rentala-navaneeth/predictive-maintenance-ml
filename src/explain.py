# src/explain.py

import shap
import matplotlib.pyplot as plt


def explain_model(model_pipeline, X_train, X_test):
    """
    Generate SHAP explanations for a pipeline model.
    """

    # Extract scaler and model
    scaler = model_pipeline.named_steps['scaler']
    model = model_pipeline.named_steps['model']

    # Transform data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use SHAP Linear Explainer (best for Logistic Regression)
    explainer = shap.LinearExplainer(model, X_train_scaled)

    shap_values = explainer(X_test_scaled)

    return shap_values, X_test_scaled


def plot_summary(shap_values, X_test, feature_names=None):
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)


def plot_single_prediction(shap_values, index=0):
    """
    Explain one prediction
    """
    shap.plots.waterfall(shap_values[index])