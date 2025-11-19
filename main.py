from utils.preprocess import load_stock_data
from indicators.indicators import add_indicators
from models.train_logreg import train_logistic_with_cv
from models.train_rf import train_random_forest_with_cv
from models.train_svm import train_svm_with_cv
from models.train_lstm import train_lstm_classifier

def main():
    df = load_stock_data("data/AMZN.csv")
    df = add_indicators(df)
    print(df.head())
    print("Columns:", df.columns)


    #model, scaler =  train_logistic_with_cv(df)
    #model , scaler = train_svm_with_cv(df)
    #model  = train_random_forest_with_cv(df)
    model = train_lstm_classifier(df)

    print (model)

if __name__ == "__main__":
    main()
