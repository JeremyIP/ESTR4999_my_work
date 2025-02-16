import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sb


def equal_weighted_composite_index(stock_df):

    closing_prices = stock_df.xs('Close', level=1, axis=1)
    equal_weighted_composite_index = closing_prices.mean(axis=1)  # Equal-weighted average

    # Create a DataFrame for the composite index
    equal_weighted_composite_index_df = pd.DataFrame({'Equally Weighted Composite_Index': equal_weighted_composite_index})
    output_dir = 'csv'
    equal_weighted_composite_index_df.to_csv(f"{output_dir}/equal_weighted_composite_index_df.csv", index=True)

    output_dir = 'plots'
    # Plot the composite index
    plt.figure(figsize=(12, 6))
    plt.plot(equal_weighted_composite_index_df.index, equal_weighted_composite_index_df['Equally Weighted Composite_Index'], label="Equally Weighted Composite Index", color='blue', linewidth=2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.gca().xaxis.set_tick_params(rotation=30)
    plt.title("Equally Weighted Composite Index of 20 Stocks")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.savefig(f"{output_dir}/equal_weighted_composite_index_plot.png")
    plt.close()

    return equal_weighted_composite_index_df


def cap_weighted_composite_index(stock_df):

    closing_prices = stock_df.xs('Close', level=1, axis=1)
    volumes = stock_df.xs('Volume', level=1, axis=1)

    # Calculate the market value (price * volume) for each stock
    market_values = closing_prices * volumes

    # Calculate the total market value for each timestamp
    total_market_value = market_values.sum(axis=0)

    # Calculate the weights based on the total market value
    weights = total_market_value / total_market_value.sum()
    print("weights", weights)

    # Calculate the weighted composite index as the weighted sum of stock prices
    cap_weighted_composite_index = (closing_prices * weights).sum(axis=1)

    # Create a DataFrame for the composite index
    cap_weighted_composite_index_df = pd.DataFrame({'Cap Weighted Composite_Index': cap_weighted_composite_index})
    output_dir = 'csv'
    cap_weighted_composite_index_df.to_csv(f"{output_dir}/cap_weighted_composite_index_df.csv", index=True)

    output_dir = 'plots'
    # Plot the composite index
    plt.figure(figsize=(12, 6))
    plt.plot(cap_weighted_composite_index_df.index, cap_weighted_composite_index_df['Cap Weighted Composite_Index'], label="Cap Weighted Composite Index", color='blue', linewidth=2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.gca().xaxis.set_tick_params(rotation=30)
    plt.title("Cap Weighted Composite Index of 20 Stocks")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.savefig(f"{output_dir}/cap_weighted_composite_index_plot.png")
    plt.close()

    return cap_weighted_composite_index_df


def equal_weighted_correlation_plots(composite_index_df, macro_df):

    correlation_df = pd.concat([composite_index_df, macro_df], axis=1).corr()
    print(correlation_df.iloc[:, 0].sort_values(ascending=False))

    output_dir = 'plots'
    plt.figure(figsize=(20, 15))
    dataplot = sb.heatmap(correlation_df, cmap="YlGnBu", annot=False)
    plt.legend()
    plt.savefig(f"{output_dir}/equal_weighted_correlation_plot.png")
    plt.close()

    print(f"Plot saved in the directory: {output_dir}/equal_weighted_correlation_plot.png")   

    output_dir = 'csv'
    correlation_df.to_csv(f"{output_dir}/equal_weighted_correlation_df.csv", index=True)

    return correlation_df


def cap_weighted_correlation_plots(composite_index_df, macro_df):

    correlation_df = pd.concat([composite_index_df, macro_df], axis=1).corr()
    print(correlation_df.iloc[:, 0].sort_values(ascending=False))

    output_dir = 'plots'
    plt.figure(figsize=(20, 15))
    dataplot = sb.heatmap(correlation_df, cmap="YlGnBu", annot=False)
    plt.legend()
    plt.savefig(f"{output_dir}/cap_weighted_correlation_plot.png")
    plt.close()

    print(f"Plot saved in the directory: {output_dir}/cap_weighted_correlation_plot.png")   

    output_dir = 'csv'
    correlation_df.to_csv(f"{output_dir}/cap_weighted_correlation_df.csv", index=True)

    return correlation_df