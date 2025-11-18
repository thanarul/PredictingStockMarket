from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_svm_with_cv(df):
    """
    SVM (RBF kernel) with StandardScaler + GridSearchCV cross-validation.
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
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    # ============================
    # 4. Scale features
    # ============================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ============================
    # 5. Cross-Validation Grid
    # ============================
    param_grid = {
        "C": [0.1, 1, 5, 10],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "kernel": ["rbf"]
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("🔧 Best SVM Parameters:", grid.best_params_)

    # ============================
    # 6. Final Evaluation
    # ============================
    preds = best_model.predict(X_test)

    print("\n📊 SVM — FINAL TEST PERFORMANCE")
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return best_model, scaler
