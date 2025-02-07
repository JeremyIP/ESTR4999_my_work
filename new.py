import gc
gc.collect()
from fredapi import Fred
import yfinance as yf
import numpy as np
import pandas as pd
import os
import pandas_ta as ta  # Import Pandas TA for technical indicators
import matplotlib.pyplot as plt
import mplfinance as mpf

start_date, end_date = '1990-01-01','2024-12-01'

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

print(macro_df_commidities.index)
print(macro_df_commidities.head())
print(macro_df_commidities.tail())
