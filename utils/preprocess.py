import yfinance as yf
import pandas as pd
import numpy as np

def load_stock_data(file_path):
    """
    Load stock data from CSV file with comprehensive NaN handling and preprocessing.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed stock data
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # If multiindex, flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        print("✅ Data loaded successfully. Shape:", df.shape)
        
        # Display initial NaN summary
        print("\n📊 Initial NaN Summary:")
        nan_summary = df.isna().sum()
        nan_percentage = (df.isna().sum() / len(df)) * 100
        
        nan_info = pd.DataFrame({
            'NaN Count': nan_summary,
            'NaN Percentage': nan_percentage
        })
        print(nan_info[nan_info['NaN Count'] > 0])
        
        # Check for complete NaN columns
        complete_nan_cols = df.columns[df.isna().all()].tolist()
        if complete_nan_cols:
            print(f"\n❌ Dropping completely NaN columns: {complete_nan_cols}")
            df = df.drop(columns=complete_nan_cols)
        
        # Handle datetime index if present
        date_columns = ['Date', 'date', 'Datetime', 'datetime']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Set as index if it's the main date column
                if col in ['Date', 'Datetime']:
                    df = df.set_index(col)
                break
        
        # Stock-specific preprocessing
        df = preprocess_stock_data(df)
        
        # Final NaN check and cleaning
        df = final_nan_cleaning(df)
        
        print(f"\n✅ Preprocessing complete. Final shape: {df.shape}")
        print(f"📈 Data range: {df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def preprocess_stock_data(df):
    """
    Apply stock-specific preprocessing rules.
    """
    # Common stock price columns
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    volume_column = 'Volume'
    
    available_price_cols = [col for col in price_columns if col in df.columns]
    
    if available_price_cols:
        print(f"\n🎯 Processing stock price columns: {available_price_cols}")
        
        # Forward fill then backward fill for stock prices (maintain continuity)
        for col in available_price_cols:
            if df[col].isna().sum() > 0:
                print(f"   Filling NaN values in {col}")
                df[col] = df[col].ffill().bfill()
        
        # Handle Volume separately (can be 0 or NaN for non-trading days)
        if volume_column in df.columns:
            print(f"   Processing {volume_column} column")
            # Fill volume NaN with 0 (assuming no trading)
            df[volume_column] = df[volume_column].fillna(0)
    
    return df

def final_nan_cleaning(df):
    """
    Final comprehensive NaN cleaning.
    """
    initial_shape = df.shape
    
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # For remaining NaN values, use appropriate filling strategies
    for column in df.columns:
        if df[column].isna().sum() > 0:
            print(f"   Handling remaining NaN in {column}")
            
            # Numeric columns: use interpolation
            if pd.api.types.is_numeric_dtype(df[column]):
                # Try interpolation first, then forward/backward fill
                df[column] = df[column].interpolate(method='linear', limit_direction='both')
                df[column] = df[column].ffill().bfill()
            else:
                # Non-numeric columns: fill with mode or 'Unknown'
                if len(df[column].mode()) > 0:
                    df[column] = df[column].fillna(df[column].mode()[0])
                else:
                    df[column] = df[column].fillna('Unknown')
    
    rows_dropped = initial_shape[0] - df.shape[0]
    if rows_dropped > 0:
        print(f"📉 Dropped {rows_dropped} rows with all NaN values")
    
    # Final verification
    remaining_nans = df.isna().sum().sum()
    if remaining_nans == 0:
        print("🎉 All NaN values have been successfully handled!")
    else:
        print(f"⚠️  Warning: {remaining_nans} NaN values remain in the dataset")
        print("Remaining NaN summary:")
        print(df.isna().sum()[df.isna().sum() > 0])
    
    return df


