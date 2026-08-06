# scripts/compare_performance.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import TD3, PPO, SAC
from gymnasium import spaces
from matplotlib.ticker import FuncFormatter
from environment import PortfolioEnv
from evaluate_baselines import buy_and_hold, equally_weighted_rebalanced
from custom_policy import TransformerFeatureExtractor


def evaluate_agent(env, model):
    """
    Runs the trained agent on the environment and returns portfolio values.
    """
    obs, info = env.reset()
    terminated, truncated = False, False
    portfolio_values = [env.initial_balance]

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])

    # Align index with the actual steps taken
    # The first obs is at window_size, so index should start one step before
    valid_dates = env.df.index[env.window_size-1:]
    return pd.Series(portfolio_values, index=valid_dates[:len(portfolio_values)])


def calculate_metrics(portfolio_values, freq=252, rf=0.0):
    """
    Calculates key performance metrics from a series of portfolio values.
    """
    if len(portfolio_values) < 2:
        return { "Total Return": "N/A", "CAGR": "N/A", "Sharpe Ratio": "N/A", "Max Drawdown": "N/A" }

    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return { "Total Return": "0.00%", "CAGR": "0.00%", "Sharpe Ratio": "0.00", "Max Drawdown": "0.00%" }

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    num_years = (len(portfolio_values) - 1) / freq
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/num_years) - 1 if num_years > 0 else 0.0

    sharpe_ratio = np.sqrt(freq) * (returns.mean() - rf) / returns.std() if returns.std() > 0 else np.nan

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = np.sqrt(freq) * (returns.mean() - rf) / downside_std if downside_std > 0 else np.nan

    volatility = returns.std() * np.sqrt(freq)

    rolling_max = portfolio_values.cummax()
    drawdown = portfolio_values / rolling_max - 1.0
    max_drawdown = drawdown.min()

    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 and cagr != 0 else np.nan

    return {
        "Total Return": f"{total_return:.2%}", "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}", "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Volatility": f"{volatility:.2%}", "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": f"{calmar_ratio:.2f}"
    }


def main(test_data_path='data/eval.csv'):
    """
    Loads, evaluates, and plots all agent performances against baselines.
    """
    # Define Model Paths and Agent Types
    models_to_evaluate = {
        "SAC Agent Default (MLP)": (SAC, 'checkpoints/sac_portfolio_model.zip'),
        "PPO Agent (MLP)": (PPO, 'checkpoints/ppo_portfolio_model.zip'),
        "TD3 Agent (MLP)": (TD3, 'checkpoints/td3_portfolio_model.zip'),
        "TD3 Agent (Transformer)": (TD3, 'checkpoints/td3_transformer_model.zip')
    }

    # Load test data (this contains ALL columns - assets + macro)
    full_eval_df = pd.read_csv(test_data_path, index_col='Date', parse_dates=True)

    # Define your actual tradable asset columns
    asset_columns = ['AAPL', 'BTC-USD', 'MSFT', 'SPY', 'TLT']

    portfolio_values = {}
    metrics = {}

    # --- Run Evaluations for each RL Agent---
    for name, (agent_type, model_path) in models_to_evaluate.items():
        print(f"--- Evaluating {name} ---")
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model file not found at {model_path}. Skipping.")
            continue

        model = agent_type.load(model_path)
        env = PortfolioEnv(full_eval_df) # Pass the full DataFrame to the RL env
        portfolio_values[name] = evaluate_agent(env, model)
        metrics[name] = calculate_metrics(portfolio_values[name])

    # --- Evaluate Buy and Hold Baseline ---
    print("\n--- Evaluating Buy and Hold Baseline ---")

    bnh_values = buy_and_hold(full_eval_df[asset_columns])
    ewp_values = equally_weighted_rebalanced(full_eval_df[asset_columns])

    portfolio_values["Buy and Hold"] = bnh_values
    metrics["Buy and Hold"] = calculate_metrics(bnh_values)

    portfolio_values["Equally Weighted"] = ewp_values
    metrics["Equally Weighted"] = calculate_metrics(ewp_values)

    # --- Combine and Print Metrics ---
    print("\n--- Performance Metrics ---")
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_markdown(numalign="left", stralign="left"))

    # --- Plotting All Strategies ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        "PPO Agent (MLP)": "red",
        "SAC Agent Default (MLP)": "green",
        "TD3 Agent (MLP)": "orange",
        "TD3 Agent (Transformer)": "cyan", 
        "Buy and Hold": "blue",
        "Equally Weighted": "purple"
    }

    for name, values in portfolio_values.items():
        if name in portfolio_values: # Check if it was successfully evaluated
            ax.plot(values.index, values, label=name, color=colors.get(name, 'gray'), linewidth=2)

    ax.set_title('Agent Performance Comparison', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)

    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'final_performance_comparison_all_agents.png'))
    plt.show()

if __name__ == '__main__':
    main()