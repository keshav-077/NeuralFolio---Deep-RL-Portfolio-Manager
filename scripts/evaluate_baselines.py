import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os # Import os for directory creation

def buy_and_hold(df_assets, initial_balance=10000): # Renamed df to df_assets for clarity
    """
    Simulates the Buy and Hold strategy.

    Args:
        df_assets (pd.DataFrame): DataFrame with daily tradable asset prices ONLY.
        initial_balance (int): The starting capital.

    Returns:
        pd.Series: A Series containing the portfolio value for each day.
    """
    print("--- Simulating Buy and Hold ---")
    n_assets = len(df_assets.columns)

    # Invest an equal amount in each asset at the beginning
    initial_investment_per_asset = initial_balance / n_assets

    # Get the initial prices
    initial_prices = df_assets.iloc[0]

    # Calculate the number of shares bought for each asset
    # Handle potential division by zero if an asset price is 0 (though unlikely with real data)
    shares = initial_investment_per_asset / (initial_prices + 1e-8)

    # Calculate the portfolio value for each day
    portfolio_values = df_assets.dot(shares)

    print(f"Initial Investment: ${initial_balance:.2f}")
    print(f"Final Portfolio Value (Buy and Hold): ${portfolio_values.iloc[-1]:.2f}")

    return portfolio_values

def equally_weighted_rebalanced(df_assets, initial_balance=10000, rebalance_freq='M', transaction_cost_pct=0.001): # Renamed df to df_assets
    """
    Simulates an Equally Weighted Portfolio with periodic rebalancing.

    Args:
        df_assets (pd.DataFrame): DataFrame with daily tradable asset prices ONLY.
        initial_balance (int): The starting capital.
        rebalance_freq (str): The rebalancing frequency ('M' for monthly, 'Q' for quarterly).
        transaction_cost_pct (float): The transaction cost as a percentage.

    Returns:
        pd.Series: A Series containing the portfolio value for each day.
    """
    print(f"\n--- Simulating Equally Weighted Portfolio (Rebalanced {rebalance_freq}) ---")
    n_assets = len(df_assets.columns)

    # Set the initial weights to be equal
    weights = np.full(n_assets, 1/n_assets)

    portfolio_value = initial_balance
    portfolio_values = pd.Series(index=df_assets.index, dtype=float) # Explicitly set dtype

    last_rebalance_date = None

    for i, (date, prices) in enumerate(df_assets.iterrows()):
        # Store the portfolio value for the day before any changes
        portfolio_values[date] = portfolio_value

        # Determine if it's a rebalancing day
        # Rebalance on the first day of the new period (month, quarter) or if it's the very first day
        rebalance_this_day = False
        if i == 0: # Rebalance on the very first day
            rebalance_this_day = True
        elif rebalance_freq == 'M' and date.month != df_assets.index[i-1].month:
            rebalance_this_day = True
        # Add 'Q' for quarterly if needed, similar logic

        if rebalance_this_day:
            # Calculate the value of trades to rebalance
            target_asset_values = portfolio_value * (1/n_assets)
            current_asset_values = weights * portfolio_value
            trades = target_asset_values - current_asset_values

            # Apply transaction costs
            transaction_costs = np.sum(np.abs(trades)) * transaction_cost_pct
            portfolio_value -= transaction_costs

            # Reset weights to be equal
            weights = np.full(n_assets, 1/n_assets)
            last_rebalance_date = date

        # Calculate portfolio value for the *next* day before the market opens
        # Get price changes from today to the next trading day
        today_prices = prices # Already have prices for the current date
        next_day_index = df_assets.index.get_loc(date) + 1
        if next_day_index < len(df_assets):
            next_day_prices = df_assets.iloc[next_day_index]

            # Avoid division by zero
            price_change_ratio = next_day_prices / (today_prices + 1e-8)

            # Update portfolio value based on price changes
            portfolio_value = np.sum( (weights * portfolio_value) * price_change_ratio )

            # Update weights due to market drift
            new_asset_values = (weights * portfolio_value) * price_change_ratio
            # Avoid division by zero for total portfolio value
            if np.sum(new_asset_values) > 1e-8: # Check if total value is effectively non-zero
                weights = new_asset_values / np.sum(new_asset_values)
            else:
                weights = np.full(n_assets, 1/n_assets) # Default to equal or handle as error


    print(f"Initial Investment: ${initial_balance:.2f}")
    print(f"Final Portfolio Value (Equally Weighted): ${portfolio_values.iloc[-1]:.2f}")

    return portfolio_values.dropna()


def main():
    # Load the evaluation data (which contains both assets and macro data)
    full_eval_df = pd.read_csv('data/eval.csv', index_col='Date', parse_dates=True)

    # --- IMPORTANT: Filter ONLY asset columns for baselines ---
    asset_columns = ['AAPL', 'BTC-USD', 'MSFT', 'SPY', 'TLT'] # Define your actual tradable assets
    test_df_assets_only = full_eval_df[asset_columns]

    # --- Run Baseline Strategies ---
    bnh_values = buy_and_hold1(test_df_assets_only)
    ewp_values = equally_weighted_rebalanced(test_df_assets_only)

    # --- Plot the results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(bnh_values.index, bnh_values, label='Buy and Hold', color='blue', linewidth=2)
    ax.plot(ewp_values.index, ewp_values, label='Equally Weighted (Rebalanced Monthly)', color='green', linewidth=2)

    ax.set_title('Baseline Strategy Performance (2021-2023)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)

    # Format the y-axis to show currency
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()

    # Ensure results directory exists for saving plot
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'baseline_performance.png'))
    plt.show()

if __name__ == '__main__':
    main()