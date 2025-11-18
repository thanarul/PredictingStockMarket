from utils.preprocess import load_stock_data
from indicators.indicators import add_indicators
from models.train_logreg import train_logistic_with_analysis

def main():
    df = load_stock_data("data/AMZN.csv")
    df = add_indicators(df)
    print(df.head())
    print("Columns:", df.columns)

    model, scaler = train_logistic_with_analysis(df)

if __name__ == "__main__":
    main()
