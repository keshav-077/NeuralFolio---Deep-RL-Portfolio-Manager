# scripts/predict_tomorrow.py

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from stable_baselines3 import SAC

# --- Imports ---
try:
    # Ensure we can find local scripts
    sys.path.append(os.getcwd())
except:
    pass

from fetch_market_data import fetch_market_data, ASSETS, FRED_IDS
from llm_analysis_rag import analyze_agent_decision

# --- Configuration ---
MODEL_PATH = "checkpoints/sac_portfolio_model.zip"
WINDOW_SIZE = 30
MACRO_COLS = list(FRED_IDS.values()) # ['Federal Funds Rate', 'CPI', 'VIX']

def get_latest_data_window(window_size=30):
    """
    Fetches live data and returns the last 'window_size' rows.
    """
    print("--- 🔄 Fetching Real-Time Data for Prediction ---")

    # Fetch a buffer to ensure we have enough data after cleaning
    lookback_days = window_size + 100
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    # We don't strictly need to save to a file for prediction, so filename=None
    df = fetch_market_data(start_date, end_date, filename=None)

    if df is None or len(df) < window_size:
        print(f"❌ Not enough data fetched. Got {len(df) if df is not None else 0} rows, needed {window_size}.")
        return None

    # Return exactly the last N rows (Observation Window)
    return df.iloc[-window_size:].copy()

def prepare_observation(data_window):
    """
    Normalizes data: Window / First_Row_of_Window
    """
    # Extract specific columns to guarantee order
    price_data = data_window[ASSETS].values
    macro_data = data_window[MACRO_COLS].values

    # Normalize
    norm_prices = price_data / (price_data[0] + 1e-8)
    norm_macro = macro_data / (macro_data[0] + 1e-8)

    # Concatenate and flatten for MLP input
    obs = np.concatenate([norm_prices, norm_macro], axis=1)
    return obs.flatten().astype(np.float32)

def get_allocations(action):
    """Applies Softmax to convert raw action to weights"""
    action = np.asarray(action).flatten()
    exp_action = np.exp(action)
    return exp_action / np.sum(exp_action)

def main():
    print(f"🚀 Prediction Job: {datetime.now().strftime('%Y-%m-%d')}")

    # 1. Get Data
    data_window = get_latest_data_window(WINDOW_SIZE)
    if data_window is None: return

    # 2. Prepare Obs
    obs = prepare_observation(data_window)

    # 3. Load MLP Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    print(f"Loading MLP SAC model...")
    model = SAC.load(MODEL_PATH)

    # 4. Predict
    action, _ = model.predict(obs, deterministic=True)
    weights = get_allocations(action)

    # 5. Format Allocations (THE FIX IS HERE)
    allocations = {}
    for i, asset in enumerate(ASSETS):
        allocations[asset] = float(weights[i]) # Explicit float() cast
    allocations['Cash'] = float(weights[-1])   # Explicit float() cast

    # 6. Output Results
    print("\n" + "="*40)
    print(f"🤖 SAC MLP MODEL RECOMMENDATION")
    print("="*40)
    for asset, weight in allocations.items():
        print(f"{asset:<10} : {weight:6.2%}")
    print("="*40)

    # 7. AI Risk Analyst
    print("\n🧠 Running AI Risk Analysis...")

    # Now this will work because all numbers are standard floats
    analysis = analyze_agent_decision(data_window, allocations)

    if isinstance(analysis, dict):
        print(f"\nStrategy:      {analysis.get('strategy_summary')}")
        print(f"Risk Level:    {analysis.get('risk_level')}")
        print(f"Justification: {analysis.get('justification')}")

        if analysis.get('risk_level') == 'High':
             print("\n⛔ BLOCKING TRADE: High Risk detected by AI Guardrail.")
        else:
             print("\n✅ TRADE APPROVED.")
    else:
        print(analysis)

if __name__ == "__main__":
    main()