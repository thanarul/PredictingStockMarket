from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_random_forest_with_cv(df):
    """
    Random Forest Classifier + GridSearchCV.
    Matches CIS4020 report requirements.
    """

    # ============================
    # 1. Create binary target
    # ============================
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)

    # ============================
    # 2. Feature selection
    # ============================
    feature_cols = [
        "Close", "High", "Low", "Open", "Volume",
        "SMA", "EMA", "RSI", "MACD", "Signal", "Vol_Change"
    ]

    X = df[feature_cols]
    y = df["Target"]

    # ============================
    # 3. Train/Test split
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    # ============================
    # 4. Random Forest Hyperparameter Grid
    # ============================
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("🔧 Best Random Forest Parameters:", grid.best_params_)

    # ============================
    # 5. Final Evaluation
    # ============================
    preds = best_model.predict(X_test)

    print("\n🌲 RANDOM FOREST — FINAL TEST PERFORMANCE")
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # ============================
    # 6. Feature Importance
    # ============================
    importances = best_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n🔥 FEATURE IMPORTANCE (Descending):")
    for idx in sorted_idx:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")

    return best_model
