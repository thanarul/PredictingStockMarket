def SMA(df, window=20):
    df["SMA"] = df["Close"].rolling(window).mean()

def EMA(df, window=10):
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()

def RSI(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

def MACD(df):
    fast = df["Close"].ewm(span=12, adjust=False).mean()
    slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = fast - slow
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

def Volume_Change(df):
    df["Vol_Change"] = df["Volume"].pct_change()

def add_indicators(df):
    SMA(df)
    EMA(df)
    RSI(df)
    MACD(df)
    Volume_Change(df)

    # drop NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)
    return df
