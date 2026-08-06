import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import all agent classes and the environment
from stable_baselines3 import PPO, SAC, TD3
from src.environment import PortfolioEnv

# --- Helper Functions ---
def evaluate_agent(env, model):
    """Runs a trained agent on a given environment."""
    obs, info = env.reset()
    terminated, truncated = False, False
    portfolio_values = [env.initial_balance]
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
    return pd.Series(portfolio_values, index=env.df.index[:len(portfolio_values)])

def buy_and_hold(df, initial_balance=10000):
    """Simulates the Buy and Hold strategy."""
    n_assets = len(df.columns)
    initial_investment_per_asset = initial_balance / n_assets
    initial_prices = df.iloc[0]
    shares = initial_investment_per_asset / initial_prices
    portfolio_values = df.dot(shares)
    return portfolio_values

def calculate_metrics(portfolio_values):
    """Calculates performance metrics from a portfolio value series."""
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    num_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/num_years) - 1 if num_years > 0 else 0
    daily_returns = portfolio_values.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
    rolling_max = portfolio_values.cummax()
    daily_drawdown = portfolio_values / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    return {
        "Total Return": f"{total_return:.2%}", "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}", "Max Drawdown": f"{max_drawdown:.2%}"
    }

# --- Main Stress Test Function ---
def run_stress_test(datafile_path, ppo_path, sac_path, td3_path, output_path):
    """
    Loads data and models, runs evaluations, and plots the comparison.
    """
    print(f"--- Running Stress Test on {datafile_path} ---")
    
    # 1. Load Data
    try:
        test_df = pd.read_csv(datafile_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {datafile_path}")
        return

    # Check for asset mismatch (e.g., 4 assets in 2008 data vs 5-asset models)
    # The standard models were trained on 5 assets (e.g., shape = 30 * 5 = 150)
    expected_assets = 5
    if test_df.shape[1] != expected_assets:
        print(f"⚠️ Warning: Models were trained on {expected_assets} assets, but this dataset has {test_df.shape[1]}.")
        print("Skipping agent evaluation for this dataset.")
        return

    # 2. Define Models to Evaluate
    models_to_evaluate = {
        "PPO Agent": (PPO, ppo_path),
        "SAC Agent": (SAC, sac_path),
        "TD3 Agent": (TD3, td3_path)
    }

    portfolio_values = {}
    metrics = {}
    
    # 3. Run Evaluations
    for name, (agent_type, model_path) in models_to_evaluate.items():
        if os.path.exists(model_path):
            print(f"--- Evaluating {name} ---")
            model = agent_type.load(model_path)
            env = PortfolioEnv(test_df)
            portfolio_values[name] = evaluate_agent(env, model)
            metrics[name] = calculate_metrics(portfolio_values[name])
        else:
            print(f"⚠️ Warning: Model file not found at {model_path}. Skipping.")

    # Evaluate Buy and Hold Baseline
    print("\n--- Evaluating Buy and Hold Baseline ---")
    bnh_values = buy_and_hold(test_df)
    portfolio_values["Buy and Hold"] = bnh_values
    metrics["Buy and Hold"] = calculate_metrics(bnh_values)
    
    # 4. Display Results
    print("\n--- Stress Test Performance Metrics ---")
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    # 5. Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {"PPO Agent": "red", "SAC Agent": "green", "TD3 Agent": "orange", "Buy and Hold": "blue"}
    for name, values in portfolio_values.items():
        ax.plot(values.index, values, label=name, color=colors.get(name, 'black'), linewidth=2)

    plot_title = f"Agent Stress Test: {os.path.basename(datafile_path).replace('.csv', '')}"
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)
    
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✅ Plot saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a stress test on trained RL portfolio agents.")
    
    parser.add_argument("--datafile", type=str, default="data/stress_test_2018.csv", help="Path to the market data CSV file for the test.")
    parser.add_argument("--ppopath", type=str, default="checkpoints/ppo_portfolio_model.zip", help="Path to the trained PPO model.")
    parser.add_argument("--sacpath", type=str, default="checkpoints/sac_portfolio_model.zip", help="Path to the trained SAC model.")
    parser.add_argument("--td3path", type=str, default="checkpoints/td3_portfolio_model.zip", help="Path to the trained TD3 model.")
    parser.add_argument("--output", type=str, default="results/stress_test_comparison.png", help="Path to save the output plot.")
    
    args = parser.parse_args()

    run_stress_test(
        datafile_path=args.datafile,
        ppo_path=args.ppopath,
        sac_path=args.sacpath,
        td3_path=args.td3path,
        output_path=args.output
    )