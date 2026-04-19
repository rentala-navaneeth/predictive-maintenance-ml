# 🔧 Explainable Predictive Maintenance System

An end-to-end machine learning system that predicts industrial engine failure using time-series sensor data and provides interpretable insights using SHAP.

---

## 🚀 Project Overview

This project uses the NASA CMAPSS dataset to build a predictive maintenance system that forecasts whether an engine will fail within the next 30 operational cycles.

The system includes:
- Time-aware feature engineering
- Machine learning model comparison
- Explainability using SHAP
- Interactive Streamlit dashboard

---

## 🧠 Key Features

- 📊 **Time-Series Feature Engineering**
  - Rolling mean, standard deviation, min/max
  - Trend slope features to capture degradation

- ⚙️ **Machine Learning Models**
  - Logistic Regression (best performing)
  - Random Forest
  - XGBoost

- 📈 **Explainability**
  - SHAP-based feature importance
  - Identifies key sensors driving failure

- ⚠️ **Risk Classification**
  - Low / Medium / High risk categorization

- 🖥️ **Interactive UI (Streamlit)**
  - Engine selection
  - Cycle-based prediction
  - Sensor trend visualization
  - Feature contribution insights

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|----------|--------|---------|--------|
| Logistic Regression | 0.96     | 0.86     | 0.86   | 0.86    | 0.98   |
| Random Forest       | 0.95     | 0.84     | 0.84   | 0.84    | 0.98   |
| XGBoost             | 0.96     | 0.85     | 0.86   | 0.85    | 0.99   |

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib

---

## 📂 Project Structure
predictive-maintenance-ml/
│
├── data/
├── src/
│ ├── load_data.py
│ ├── preprocess.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ ├── explain.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py