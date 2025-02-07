from fredapi import Fred
import yfinance as yf
import numpy as np
import pandas as pd
import os
import pandas_ta as ta  # Import Pandas TA for technical indicators
import matplotlib.pyplot as plt
import mplfinance as mpf
from pre_selection import composite_index, correlation_plots

start_date, end_date = '1990-01-01','2024-12-01'

# ========================================================
# ========================================================
# ========================================================

# List of stock symbols
ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM',
                  'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 'LRCX', 'KLAC',
                  'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'WDC']

# Initialize a dictionary to hold stock data
all_stock_data = {}

# Ensure the output directory exists
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Fetch OHLCV data for each ticker
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    stock_data.index = stock_data.index.tz_localize(None)  # Remove timezone
    all_stock_data[ticker_symbol] = stock_data  # Store the data in the dictionary

    # Loop through each stock's data and perform check for NaN values
    if stock_data.isnull().values.any():
        print(f"{ticker_symbol} contains NaN values.")
    else:
        pass

    '''
    # Plot using mplfinance
    mpf.plot(stock_data,
             type='line',
             volume=True,
             title=f'{ticker_symbol} OHLCV Candlestick Chart',
             style='charles',  # You can choose a style you prefer
             savefig=f"{output_dir}/{ticker_symbol}_plot.png",  # Save as PNG
             figsize=(10, 6))  # Set figure size

    print(f"Plot saved in the directory: {output_dir}/{ticker_symbol}_plot.png")
    '''


# ========================================================
# ========================================================
# ========================================================

# # Define FRED API key
API_KEY = 'ec9c3a532618d0109bc583602f15dc83'
fred = Fred(api_key=API_KEY)

# # Define macroeconomic series and fetch data
macro_series_fred = {
    'Total Vehicle Sales': 'TOTALSA', # 1976 to Dec 2024
    'Domestic Auto Production': 'DAUPSA', # Jan 1993 to Oct 2024
    '15-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE15US', # 1991 to 2025
    '15-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE30US', # 1977 to 2025
    'Employment Level': 'CE16OV', # 1948 to Dec 2024
    'Unemployment Rate': 'UNRATE', # 1948 to Dec 2024
    'Inflation, consumer prices for the United States': 'FPCPITOTLZGUSA',  # 1960 to Sep 2023
    'Federal Funds Effective Rate': 'FEDFUNDS', # 1954 to 2025
    'Trade Balance: Goods and Services, Balance of Payments Basis': 'BOPGSTB', # 1992 to Dec 2024
    'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average': 'CPIAUCNS', # 1913 to Dec 2024
    'M1': 'M1SL', # 1959 to Dec 2024
    'M2': 'M2SL', # 1959 to Dec 2024
    'Industrial Production: Total Index': 'INDPRO', # 1919 to Dec 2024
    #'US_Manufacturing_PMI': '',
    #'US_House_Sold': '',
    'S&P CoreLogic Case-Shiller U.S. National Home Price Index': 'CSUSHPISA', # 1987 to Nov 2024
    #'US_Housing_Market_Index': '',
    'New Privately-Owned Housing Units Started: Total Units': 'HOUST' # 1959 to Dec 2024
}

# Fetch macroeconomic data
macro_data_fred = {}
for name, series_id in macro_series_fred.items():
    macro_data_fred[name] = fred.get_series(series_id, start_date, end_date)

# Convert macroeconomic data to a single DataFrame and align with OHLCV
macro_df_fred = pd.DataFrame(macro_data_fred)
macro_df_fred.index = pd.to_datetime(macro_df_fred.index)
macro_df_fred = macro_df_fred.resample('D').ffill() # Convert to daily frequency using forward fill

# ========================================================
# ========================================================
# ========================================================
# 2000-08-23 - 2024-11-29 

# Define the ticker symbols for the commodities
macro_series_commodities = {
    'Gold': 'GC=F',  # Gold Futures
    'Crude Oil': 'CL=F',  # Crude Oil Futures
    'Brent Oil': 'BZ=F',  # Brent Oil Futures
    'Natural Gas': 'NG=F',  # Natural Gas Futures
    'Reformulated Blendstock Oil': 'RB=F'  # RBOB Gasoline Futures
}

# Fetch commodity data
macro_data_commidities = []
for name, ticker in macro_series_commodities.items():
    macro_data_commidities.append(yf.download(ticker, start=start_date, end=end_date)["Close"])

macro_df_commidities = pd.concat(macro_data_commidities, axis=1)
macro_df_commidities = macro_df_commidities.resample('D').ffill() # Convert to daily frequency using forward fill






# Align macroeconomic data with OHLCV
aligned_macro = macro_df_fred.reindex(stock_data.index).ffill().bfill()


composite_index_df = composite_index(all_stock_data, output_dir)
correlation_df = correlation_plots(composite_index_df, macro_df_commidities)




'''
# Align macroeconomic data with OHLCV
aligned_macro = macro_df.reindex(stock_data.index).ffill().bfill()

# Calculate new technical indicators
# # 5-Day Weighted Close Price (Weighted Close: (High + Low + 2*Close)/4)
combined_data = pd.concat([stock_data, aligned_macro], axis=1)

# Calculate Technical Indicators
combined_data['5WCLPRICE'] = (combined_data['High'] + combined_data['Low'] + 2 * combined_data['Close']) / 4
combined_data['5WCLPRICE'] = combined_data['5WCLPRICE'].rolling(window=5).mean()
combined_data['5MedPrice'] = combined_data['Close'].rolling(window=5).median()
combined_data['5AvgPrice'] = combined_data['Close'].rolling(window=5).mean()

# Drop rows with NaN values resulting from rolling calculations
combined_data.dropna(inplace=True)

# Calculate mean and std across all columns for normalization
mean = combined_data.mean()
std = combined_data.std()

# Ensure the output directory exists
output_dir = 'dataset/MSFT'
os.makedirs(output_dir, exist_ok=True)

# Save scaling information
np.savez(os.path.join(output_dir, 'var_scaler_info.npz'), mean=mean.values, std=std.values)

dates = combined_data.index
norm_time_marker = np.stack([
    np.full(len(dates), 0.5),  # Time of day (fixed for daily data)
    dates.weekday / 4.0,        # Day of week (normalized)
    (dates.day - 1) / (dates.to_series().groupby(dates.to_period("M")).transform("count") - 1),  # Day of month
    (dates.dayofyear - 1) / (dates.to_series().groupby(dates.to_period("Y")).transform("count") - 1)  # Day of year
], axis=1)

# Save the final combined data and normalized time markers
np.savez(os.path.join(output_dir, 'feature.npz'), norm_var=combined_data.values, norm_time_marker=norm_time_marker)

# Verify shapes
print("Final combined data shape:", combined_data.shape)  # Should have OHLCV + macro + new indicators columns
print("norm_time_marker shape:", norm_time_marker.shape)
print(f"mean shape: {mean.shape} and std shape {std.shape}")
print(f"mean value: \n {mean}")
print(f"std value: \n {std}")
print("First 10 rows of the downloaded data:")
print(combined_data.head(10))
print("Last 10 rows of the downloaded data:")
print(combined_data.tail(10))
print("Data successfully saved.")

'''