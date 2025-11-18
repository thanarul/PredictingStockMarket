from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt

def create_improved_target(df):
    """Create ternary target: Down=0, Same=1, Up=2"""
    tomorrow_close = df["Close"].shift(-1)
    
    # Use small threshold to account for floating point precision
    threshold = 0.000  # 0.1% threshold for "same" price
    
    df["Target"] = 1  # Default to "same"
    df.loc[tomorrow_close > df["Close"] * (1 + threshold), "Target"] = 2  # Up
    
    
    print("🎯 Target distribution:")
    print(df["Target"].value_counts().sort_index())
    return df

def analyze_class_separation(df):
    """Analyze how well the classes are separated"""
    print("🎯 CLASS SEPARATION ANALYSIS")
    print("=" * 50)
    
    # Create target variable
    df = create_improved_target(df)
    
    # 1. Check class distribution
    print("1. CLASS DISTRIBUTION:")
    class_counts = df["Target"].value_counts().sort_index()
    print(class_counts)
    print(f"Class balance ratio: {class_counts.min() / class_counts.max():.3f}")
    
    # 2. Visualize feature distributions by class
    feature_cols = [col for col in df.select_dtypes(include=["float64", "int64"]).columns 
                   if col not in ["Target", "Close", "High", "Low", "Open", "Volume"]]
    
    # Plot distributions for key features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    key_features = feature_cols[:6]  # First 6 features
    for i, feature in enumerate(key_features):
        for target_class in sorted(df["Target"].unique()):
            class_data = df[df["Target"] == target_class][feature]
            axes[i].hist(class_data, alpha=0.6, label=f'Class {target_class}', bins=30)
        axes[i].set_title(f'{feature} by Class')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    return df, feature_cols

def check_linear_separability(X, y):
    """Check if classes are linearly separable using LDA"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y)
    
    plt.figure(figsize=(10, 6))
    for class_label in np.unique(y):
        plt.scatter(X_lda[y == class_label], np.zeros_like(X_lda[y == class_label]), 
                   alpha=0.6, label=f'Class {class_label}')
    plt.title("LDA Projection - Linear Separability Check")
    plt.xlabel("LDA Component 1")
    plt.legend()
    plt.show()
    
    # Calculate between-class variance
    print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
    
    return X_lda

def train_logistic_with_analysis(df):
    """Train logistic regression with comprehensive analysis"""
    
    # First analyze class separation
    df, feature_cols = analyze_class_separation(df)
    
    # Prepare features and target
    X = df[feature_cols].iloc[:-1]  # Remove last row (no target)
    y = df["Target"].iloc[:-1]
    
    print(f"\n2. FEATURE SPACE:")
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    print(f"Feature means by class:")
    for class_label in sorted(y.unique()):
        class_means = X[y == class_label].mean()
        print(f"Class {class_label}: {class_means.values[:3]}...")  # First 3 features
    
    # Check linear separability
    print(f"\n3. LINEAR SEPARABILITY ANALYSIS:")
    X_lda = check_linear_separability(X, y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"\n4. MODEL TRAINING:")
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train logistic regression
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"📊 Logistic Regression Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(y_test, test_preds))
    
    # Check if model is just predicting the majority class
    majority_class = y_train.value_counts().idxmax()
    majority_accuracy = (y_test == majority_class).mean()
    print(f"Majority class baseline: {majority_accuracy:.4f}")
    
    return model, scaler

