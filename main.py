# main.py

import numpy as np

from src.load_data import load_cmapss_data
from src.preprocess import add_rul, create_binary_label
from src.feature_engineering import create_rolling_features, drop_na_rows
from src.train import train_test_split_by_engine, prepare_features, train_models
from src.evaluate import evaluate_all, classify_risk
from src.explain import explain_model, plot_summary


def main():
    # =========================
    # 1. LOAD + PREPROCESS
    # =========================
    df = load_cmapss_data("data/train_FD001.txt")

    df = add_rul(df)
    df = create_binary_label(df)

    df = create_rolling_features(df, window=10)
    df = drop_na_rows(df)

    print("Final Dataset Shape:", df.shape)

    # =========================
    # 2. TRAIN-TEST SPLIT
    # =========================
    train_df, test_df = train_test_split_by_engine(df)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # =========================
    # 3. PREPARE DATA
    # =========================
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # =========================
    # 4. TRAIN MODELS
    # =========================
    models = train_models(X_train, y_train)

    # =========================
    # 5. EVALUATION
    # =========================
    results = evaluate_all(models, X_test, y_test)

    print("\nModel Performance:\n")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    # =========================
    # 6. BEST MODEL (LR)
    # =========================
    best_model = models['Logistic Regression']

    # =========================
    # 7. SHAP EXPLANATION
    # =========================
    print("\nGenerating SHAP explanations...")

    shap_values, X_test_scaled = explain_model(best_model, X_train, X_test)

    plot_summary(
        shap_values,
        X_test_scaled,
        feature_names=X_test.columns
    )

    # =========================
    # 8. RISK CLASSIFICATION
    # =========================
    probs = best_model.predict_proba(X_test)[:, 1]

    print("\nSample Risk Predictions (Balanced View):\n")

    # Select examples from all risk zones
    low_idx = np.where(probs < 0.3)[0][:2]
    med_idx = np.where((probs >= 0.3) & (probs < 0.7))[0][:2]
    high_idx = np.where(probs >= 0.7)[0][:2]

    selected_indices = np.concatenate([low_idx, med_idx, high_idx])

    for i in selected_indices:
        risk = classify_risk(probs[i])
        print(f"Probability: {probs[i]:.2f} → {risk}")


if __name__ == "__main__":
    main()