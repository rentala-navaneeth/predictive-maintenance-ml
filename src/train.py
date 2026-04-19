# src/train.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_test_split_by_engine(df, test_size=0.3):
    """
    Split dataset by engine_id to avoid leakage.
    """
    engine_ids = df['engine_id'].unique()
    
    split_index = int(len(engine_ids) * (1 - test_size))
    
    train_engines = engine_ids[:split_index]
    test_engines = engine_ids[split_index:]
    
    train_df = df[df['engine_id'].isin(train_engines)]
    test_df = df[df['engine_id'].isin(test_engines)]
    
    return train_df, test_df


def prepare_features(df):
    """
    Prepare X and y.
    """
    drop_cols = ['engine_id', 'cycle', 'RUL', 'label']
    
    X = df.drop(columns=drop_cols)
    y = df['label']
    
    return X, y


def train_models(X_train, y_train):
    """
    Train multiple models.
    """
    models = {}

    # =========================
    # Logistic Regression (BEST MODEL)
    # =========================
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',   # 🔥 handles imbalance
            random_state=42
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    models['Logistic Regression'] = lr_pipeline

    # =========================
    # Random Forest
    # =========================
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'   # 🔥 important
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # =========================
    # XGBoost
    # =========================
    # Compute imbalance ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight  # 🔥 key improvement
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    return models