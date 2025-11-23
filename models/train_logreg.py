from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA   # <-- NEW
from sklearn.metrics import classification_report, accuracy_score
from utils.plot import evaluate_and_plot  # <-- NEW
import numpy as np

def train_logistic_with_cv(df):
    """
    Logistic Regression with StandardScaler + PCA + GridSearchCV cross-validation.
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
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ============================
    # 5. PCA (keep 95% variance)
    # ============================
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    print(f"📉 Logistic PCA components: {pca.n_components_}")
    print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # ============================
    # 6. Cross-Validation (GridSearch)
    # ============================
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=5000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("🔧 Best Logistic Regression Parameters:", grid.best_params_)

    # ============================
    # 7. Final evaluation
    # ============================
    preds = best_model.predict(X_test)

    print("\n📊 LOGISTIC REGRESSION — FINAL TEST PERFORMANCE")
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Confusion matrix + bar plot + metrics
    metrics_dict = evaluate_and_plot(y_test, preds, model_name="LogisticRegression")

    # Return model, scaler and PCA (so you can use them later if needed)
    return best_model, scaler, pca, metrics_dict
