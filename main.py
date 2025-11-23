from utils.preprocess import load_stock_data
from indicators.indicators import add_indicators

from models.train_logreg import train_logistic_with_cv
from models.train_svm import train_svm_with_cv
from models.train_rf import train_random_forest_with_cv
from models.train_lstm import train_lstm_classifier


def main():
    df = load_stock_data("data/AMZN.csv")
    df = add_indicators(df)

    # # ======= Logistic Regression =======
    # logreg_model, logreg_scaler, logreg_pca, logreg_metrics = train_logistic_with_cv(df)
    # print("\nLogReg metrics:", logreg_metrics)

    # # ======= SVM =======
    # svm_model, svm_scaler, svm_pca, svm_metrics = train_svm_with_cv(df)
    # print("\nSVM metrics:", svm_metrics)

    # ======= Random Forest =======
    rf_model, rf_metrics = train_random_forest_with_cv(df)
    print("\nRF metrics:", rf_metrics)

    # ======= LSTM =======
    lstm_model, lstm_scaler, lstm_metrics = train_lstm_classifier(df)
    print("\nLSTM metrics:", lstm_metrics)


if __name__ == "__main__":
    main()
