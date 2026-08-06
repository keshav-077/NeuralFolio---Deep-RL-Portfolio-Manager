# scripts/fetch_market_data.py

import yfinance as yf_lib
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

# --- MOVE THESE OUTSIDE THE FUNCTION ---
# Define your assets (Global variable, importable)
ASSETS = ['AAPL', 'MSFT', 'SPY', 'TLT', 'BTC-USD']

# Define FRED IDs for macroeconomic data (Global variable, importable)
FRED_IDS = {
    'DFF': 'Federal Funds Rate', # Daily Federal Funds Rate
    'CPIAUCSL': 'CPI',           # Consumer Price Index (All Urban Consumers, Seasonally Adjusted, Monthly)
    'VIXCLS': 'VIX'              # CBOE Volatility Index (VIX) from FRED
}
# ---------------------------------------

def fetch_market_data(start_date, end_date, filename):
    """
    Fetches market data, macroeconomic indicators (including VIX from FRED),
    for specified assets and time period, then saves it to a CSV file.
    """
    # No need to re-define assets and fred_ids here.
    # The function will use the global ASSETS and FRED_IDS defined above.

    print(f"--- Fetching market data for {ASSETS} from {start_date} to {end_date} ---")

    # 1. Fetch Asset Prices (Daily) using yf_lib.download()
    try:
        # Use the global ASSETS variable
        df_prices = yf_lib.download(ASSETS, start=start_date, end=end_date)['Close']
        df_prices.dropna(inplace=True)
        print(f"✅ Fetched {len(ASSETS)} asset prices.")
    except Exception as e:
        print(f"❌ Error fetching asset prices: {e}")
        return None # Return None on failure

    # 2. Fetch Macro Data (VIX, Federal Funds Rate, CPI) from FRED using pandas_datareader
    print("--- Fetching macroeconomic data from FRED ---")

    try:
        # FRED data can be tricky with exact date ranges, fetching a bit more to ensure coverage
        fred_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

        # Use the global FRED_IDS variable
        df_fred = pdr.DataReader(list(FRED_IDS.keys()), 'fred', start=fred_start_date, end=end_date)
        df_fred.rename(columns=FRED_IDS, inplace=True)
        print("✅ Fetched Federal Funds Rate, CPI, and VIX data from FRED.")
    except Exception as e:
        print(f"❌ Error fetching FRED data: {e}. Check FRED API access or ticker validity.")
        df_fred = pd.DataFrame() # Create empty dataframe if fetch fails

    # Combine all dataframes
    df_combined = df_prices.copy()

    # Merge FRED data (now includes VIX)
    if not df_fred.empty:
        df_combined = df_combined.merge(df_fred, left_index=True, right_index=True, how='left')

    # Handle missing macro data: forward-fill and then back-fill for initial NaNs
    # This loop now covers all FRED columns
    # Use the global FRED_IDS variable
    for col_name in FRED_IDS.values():
        if col_name in df_combined.columns:
            df_combined[col_name] = df_combined[col_name].ffill().bfill()
            # Drop rows if they still have NaN for macro data after fill
            df_combined.dropna(subset=[col_name], inplace=True) # Added dropna for robustness

    # Ensure all data is within the requested date range after merging/filling
    df_combined = df_combined.loc[start_date:end_date]
    df_combined.dropna(inplace=True) # Final dropna for any remaining NaNs

    if df_combined.empty:
        print("❌ Final combined dataframe is empty after merging and cleaning. Check date ranges and data availability.")
        return None # Return None on failure

    # Save to CSV if a filename is provided
    if filename:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.dirname(filename) == "":
            os.makedirs(output_dir, exist_ok=True)

        df_combined.to_csv(filename, index=True)
        print(f"\n✅ Data saved successfully to {filename}")

    print(f"Final data shape: {df_combined.shape}")
    print("Columns:", df_combined.columns.tolist())
    
    return df_combined # Return the DataFrame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch market and macroeconomic data.")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2020-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--filename", type=str, default="data/train.csv", help="Output CSV filename.")

    args = parser.parse_args()

    fetch_market_data(args.start, args.end, args.filename)