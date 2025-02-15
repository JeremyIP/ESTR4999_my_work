import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def composite_index(stock_df, output_dir):
    # Extract closing prices for all stocks using MultiIndex
    closing_prices = stock_df.xs('Close', level=1, axis=1)

    # Compute the composite index as the average of closing prices (or weighted average)
    composite_index = closing_prices.mean(axis=1)  # Equal-weighted average

    # Add the composite index to a new DataFrame for easier plotting/analysis
    composite_index_df = pd.DataFrame({'Composite_Index': composite_index})

    # Plot the composite index
    plt.figure(figsize=(12, 6))
    plt.plot(composite_index_df, label="Composite Index", color='blue', linewidth=2)
    '''
    composite_index_df.plot()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d')) 
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30)) 
    plt.gca().xaxis.set_tick_params(rotation = 30)  
    plt.gca().set_xbound(composite_index_df.index[0], composite_index_df.index[-1])
    '''
    plt.title("Composite Index of 20 Stocks")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.savefig(f"{output_dir}/composite_index_plot.png")
    plt.close()

    print(f"Plot saved in the directory: {output_dir}/composite_index_plot.png")

    return composite_index_df


def correlation_plots(composite_index_df, macro_df_commidities):
    
    return
