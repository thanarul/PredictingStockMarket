import yfinance as yf
import pandas as pd
import os
import time
import random

def download_and_save(ticker, start="2010-01-01"):
    print(f"📥 Downloading {ticker} from Yahoo Finance...")
    
    try:
        df = yf.download(ticker, start=start)
        
        # flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        path = f"data/{ticker}.csv"
        df.to_csv(path)
        
        print(f"✅ Saved {ticker} data to {path}")
        print(f"Downloaded {len(df)} rows")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "SPY"]
    
    for i, t in enumerate(tickers):
        download_and_save(t)
        
        # Add delay between requests (except after the last one)
        if i < len(tickers) - 1:
            delay = random.uniform(2, 5)  # Random delay between 2-5 seconds
            print(f"⏳ Waiting {delay:.1f} seconds...")
            time.sleep(delay)