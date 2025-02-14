import pandas as pd
import matplotlib.pyplot as plt

def composite_index(all_stock_data, output_dir):
    # Combine closing prices of all stocks into a single DataFrame
    print(all_stock_data.index)
    #all_stock_data = all_stock_data.drop(columns=['^SPX']) #?????????????
    closing_prices = pd.DataFrame({symbol: data['Close'] for symbol, data in all_stock_data.items()})

    # Drop rows with missing values (if any stock is missing data on a specific date)
    closing_prices = closing_prices.dropna()

    # Compute the composite index as the average of closing prices (or weighted average)
    composite_index = closing_prices.mean(axis=1)  # Equal-weighted average

    # Add the composite index to a new DataFrame for easier plotting/analysis
    composite_index_df = pd.DataFrame({'Composite_Index': composite_index})

    # Plot the composite index
    plt.figure(figsize=(12, 6))
    plt.plot(composite_index_df, label="Composite Index", color='blue', linewidth=2)
    plt.title("Composite Index of 20 Stocks")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/composite_index_plot.png")
    plt.close()

    print(f"Plot saved in the directory: {output_dir}/composite_index_plot.png")

    return composite_index_df


def correlation_plots(composite_index_df, macro_df_commidities):
    
    return
