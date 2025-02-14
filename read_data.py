from fredapi import Fred
import yfinance as yf
import numpy as np
import pandas as pd
import os
import pandas_ta as ta # Import Pandas TA for technical indicators
import matplotlib.pyplot as plt
import mplfinance as mpf
#import talib
from pre_selection import composite_index, correlation_plots

import gc
gc.collect()


start_date, end_date = '2010-01-01','2022-12-31'
window = 30

# List of stock symbols
ticker_symbols = ['^SPX',  # S&P 500 Index and #############S&P 500 IT Sector Index
                  'AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 
                  'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 
                  'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 
                  'PTC', 'WDC']

# Initialize a dictionary to hold stock data
stock_data = {}

# Ensure the output directory exists
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Fetch OHLCV data for each ticker
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    stock_series = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    stock_series.index = stock_series.index.tz_localize(None)  # Remove timezone

    # Calculate the Simple Moving Average (SMA) for each feature
    for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
        stock_series[feature] = stock_series[feature].rolling(window=window).mean()

    stock_data[ticker_symbol] = stock_series  # Store the smoothed data in the dictionary
    '''
    # Plot using mplfinance
    mpf.plot(stock_series,
             type='line',
             volume=True,
             title=f'{ticker_symbol} OHLCV Candlestick Chart',
             style='charles',  # You can choose a style you prefer
             savefig=f"{output_dir}/{ticker_symbol}_plot.png",  # Save as PNG
             figsize=(10, 6))  # Set figure size

    print(f"Plot saved in the directory: {output_dir}/{ticker_symbol}_plot.png")
    '''

stock_df = pd.concat(stock_data, axis=1)
stock_df.index = pd.to_datetime(stock_df.index)
stock_df = stock_df.iloc[window:]



# Initialize a dictionary to hold stock data
stock_indicators_data = {}

# Fetch OHLCV data for each ticker and calculate indicators
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    stock_series = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Category 1: Overlap Indicators
    ma20 = stock_series['Close'].rolling(window=20).mean()
    stock_series['Bollinger_Bands_Upper'] = ma20 + 2 * stock_series['Close'].rolling(window=20).std()
    stock_series['Bollinger_Bands_Middle'] = ma20
    stock_series['Bollinger_Bands_Lower'] = ma20 - 2 * stock_series['Close'].rolling(window=20).std()

    stock_series['DEMA'] = ta.overlap.dema(stock_series['Close'], window=20)
    stock_series['Midpoint'] = ta.overlap.midpoint(stock_series['Close'], window=20)
    stock_series['Midpoint_Price'] = (stock_series['High'] + stock_series['Low']) / 2
    stock_series['T3_Moving_Average'] = ta.overlap.t3(stock_series['Close'], window=20)

    # Category 2: Momentum Indicators
    stock_series['ADX'] = ta.trend.adx(stock_series['High'], stock_series['Low'], stock_series['Close']).iloc[:, 0]
    stock_series['Absolute_Price_Oscillator'] = ta.momentum.apo(stock_series['Close'])
    aroon = ta.trend.aroon(stock_series['High'], stock_series['Low'], window=14)
    stock_series['Aroon_Up'] = aroon.iloc[:, 0]
    stock_series['Aroon_Down'] = aroon.iloc[:, 1]
    stock_series['Aroon_Oscillator'] = aroon.iloc[:, 2]
    stock_series['Balance_of_Power'] = ta.momentum.bop(stock_series['Open'], stock_series['High'], stock_series['Low'], stock_series['Close'])
    stock_series['CCI'] = ta.momentum.cci(stock_series['High'], stock_series['Low'], stock_series['Close'], window=20)
    stock_series['Chande_Momentum_Oscillator'] = ta.momentum.cmo(stock_series['Close'], window=14)    
    
    # Calculate 26-day EMA and 12-day EMA
    exp26 = stock_series['Close'].ewm(span=26, adjust=False).mean()
    exp12 = stock_series['Close'].ewm(span=12, adjust=False).mean()
    stock_series['MACD'] = exp12 - exp26
    stock_series['MACD_Signal'] = stock_series['MACD'].ewm(span=9, adjust=False).mean()
    stock_series['MACD_Histogram'] = stock_series['MACD'] - stock_series['MACD_Signal']
    stock_series['Money_Flow_Index'] = ta.volume.mfi(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'], window=14)

    # Category 3: Volatility Indicators
    #stock_series['Normalized_Average_True_Range'] = ta.volatility.natr(stock_series['High'], stock_series['Low'], stock_series['Close'], window=14)

    # Category 4: Volume Indicators
    stock_series['Chaikin_A/D_Line'] = ta.volume.ad(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'])
    stock_series['Chaikin_A/D_Oscillator'] = ta.volume.adosc(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'])

    # Category 5: Price Transform
    stock_series['Median_Price'] = ta.statistics.median(stock_series['Close'])
    stock_series['Typical_Price'] = ta.overlap.hlc3(stock_series['High'], stock_series['Low'], stock_series['Close'])
    stock_series['Weighted_Closing_Price'] = ta.overlap.wcp(stock_series['High'], stock_series['Low'], stock_series['Close'])
    
    '''
    # Category 6: Hilbert Transform indicators
    stock_series['Hilbert_Dominant_Cycle_Period'] = talib.HT_DCPERIOD(stock_series['Close'])
    stock_series['Hilbert_Dominant_Cycle_Phase'] = talib.HT_DCPHASE(stock_series['Close'])
    inphase, quadrature = talib.HT_PHASOR(stock_series['Close'])
    stock_series['Hilbert_Phasor_Components_Inphase'] = inphase
    stock_series['Hilbert_Phasor_Components_Quadrature'] = quadrature
    stock_series['Hilbert_SineWave'] = talib.HT_SINE(stock_series['Close'])
    stock_series['Hilbert_LeadSineWave'] = talib.HT_LEADSINE(stock_series['Close'])
    stock_series['Hilbert_Trend_vs_Cycle_Mode'] = talib.HT_TRENDMODE(stock_series['Close'])
    '''

    # Store the data in the dictionary
    stock_indicators_data[ticker_symbol] = stock_series.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


# Concatenate all stock data into a single DataFrame
stock_indicators_df = pd.concat(stock_indicators_data, axis=1)
stock_indicators_df.index = pd.to_datetime(stock_indicators_df.index)
#stock_indicators_df.index = stock_indicators_df.index.strftime('%d/%m/%Y')



# # Define FRED API key
API_KEY = 'ec9c3a532618d0109bc583602f15dc83'
fred = Fred(api_key=API_KEY)

# # Define macroeconomic series and fetch data
macro_series_fred = {
    'Total Vehicle Sales': 'TOTALSA', # 1976 to Dec 2024
    'Domestic Auto Production': 'DAUPSA', # Jan 1993 to Oct 2024
    '15-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE15US', # 1991 to 2025
    '30-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE30US', # 1977 to 2025
    'Employment Level': 'CE16OV', # 1948 to Dec 2024
    'Unemployment Rate': 'UNRATE', # 1948 to Dec 2024
    'Inflation, consumer prices for the United States': 'FPCPITOTLZGUSA',  # 1960 to Sep 2023
    'Federal Funds Effective Rate': 'FEDFUNDS', # 1954 to 2025
    'Trade Balance: Goods and Services, Balance of Payments Basis': 'BOPGSTB', # 1992 to Dec 2024
    'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average': 'CPIAUCNS', # 1913 to Dec 2024
    'M1': 'M1SL', # 1959 to Dec 2024
    'M2': 'M2SL', # 1959 to Dec 2024
    'Industrial Production: Total Index': 'INDPRO', # 1919 to Dec 2024
    'US_Manufacturing_PMI': 'AMTMNO', # to Dec 2024
    'New One Family Houses Sold': 'HSN1F', # to Dec 2024
    'S&P CoreLogic Case-Shiller U.S. National Home Price Index': 'CSUSHPISA', # 1987 to Nov 2024
    'All-Transactions House Price Index for the United States': 'USSTHPI', # to Q3 2024
    'New Privately-Owned Housing Units Started: Total Units': 'HOUST' # 1959 to Dec 2024
}

# Fetch macroeconomic data
macro_data_fred = {}
for name, series_id in macro_series_fred.items():
    macro_data_fred[name] = fred.get_series(series_id, start_date, end_date)

# Convert macroeconomic data to a single DataFrame
macro_df_fred = pd.DataFrame(macro_data_fred)
macro_df_fred.index = pd.to_datetime(macro_df_fred.index)
macro_df_fred = macro_df_fred.resample('D').ffill().ffill().bfill() # Convert to daily frequency using forward fill



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
macro_df_commidities = macro_df_commidities.resample('D').ffill().ffill().bfill() # Convert to daily frequency using forward fill


stock_indicators_df = stock_indicators_df.resample('D').ffill().ffill().bfill().loc[stock_df.index[0]:]
stock_indicators_df.index = stock_indicators_df.index.strftime('%d/%m/%Y')

macro_df_fred = macro_df_fred.reindex(stock_df.index).ffill().bfill()
macro_df_fred.index = macro_df_fred.index.strftime('%d/%m/%Y')

macro_df_commidities = macro_df_commidities.reindex(stock_df.index).ffill().bfill()
macro_df_commidities.index = macro_df_commidities.index.strftime('%d/%m/%Y')

stock_df.index = stock_df.index.strftime('%d/%m/%Y')



# Ensure the output directory exists
output_dir = 'csv'
os.makedirs(output_dir, exist_ok=True)
stock_df.to_csv(f"{output_dir}/stock_df.csv", index=True)
stock_indicators_df.to_csv(f"{output_dir}/stock_indicators_df.csv", index=True)
macro_df_fred.to_csv(f"{output_dir}/macro_df_fred.csv", index=True)
macro_df_commidities.to_csv(f"{output_dir}/macro_df_commidities.csv", index=True)


print("Shape of stock_df: ", stock_df.shape)
print("Is there any nan value in stock_df: ", stock_df.isna().any().any())
print("\n")

print("Shape of stock_indicators_df: ", stock_indicators_df.shape)
print("Is there any nan value in stock_indicators_df: ", stock_indicators_df.isna().any().any())
print("\n")


print("Shape of macro_df_fred: ", macro_df_fred.shape)
print("Is there any nan value in macro_df_fred: ", macro_df_fred.isna().any().any())
print("\n")

print("Shape of macro_df_commidities: ", macro_df_commidities.shape)
print("Is there any nan value in macro_df_commidities: ", macro_df_commidities.isna().any().any())
print("\n")

'''



composite_index_df = composite_index(all_stock_data, output_dir)
correlation_df = correlation_plots(composite_index_df, macro_df_commidities)

'''


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