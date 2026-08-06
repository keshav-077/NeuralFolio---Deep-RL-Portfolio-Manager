# scripts/app.py

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import json
import torch
from fetch_market_data import fetch_market_data, ASSETS, FRED_IDS
from llm_analysis_rag import analyze_agent_decision, analyze_historical_segment
from stable_baselines3 import SAC
from environment import PortfolioEnv
from evaluate_baselines import buy_and_hold, equally_weighted_rebalanced

# --- Configuration ---
MODEL_PATH = os.path.join("checkpoints", "sac_portfolio_model.zip")
# For HF Spaces: If model doesn't exist, show a warning
if not os.path.exists(MODEL_PATH):
    print("⚠️ WARNING: Model checkpoint not found. Some features will be limited.")
    print("   To use full features, train a model locally and upload checkpoints/")

WINDOW_SIZE = 30
MACRO_COLS = list(FRED_IDS.values())
DASHBOARD_DATA_PATH = os.path.join("data", "historical_dashboard_data.csv")


TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2020-12-31"

# Global variable for dashboard data needed for Tabs 3 & 4
DASHBOARD_DATA_DF = None

# Define Time Period mappings for the dropdown
TIME_PERIODS = {
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825,
    "Max Available": 9999 # Sentinel value for max
}

# =========================================
# Initialization Functions
# =========================================

def initialize_dashboard_data():
    """Fetches and loads historical data at startup for Tabs 3 & 4."""
    global DASHBOARD_DATA_DF
    print("--- Initializing Historical Data for Analyst/Simulation Tabs ---")

    # Fetching last 6 years to support longer analysis periods and simulation
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*6)).strftime('%Y-%m-%d')

    print(f"Fetching historical data from {start_date} to {end_date}...")
    # This might take a minute on first run
    fetch_market_data(start_date, end_date, DASHBOARD_DATA_PATH)

    if os.path.exists(DASHBOARD_DATA_PATH):
        DASHBOARD_DATA_DF = pd.read_csv(DASHBOARD_DATA_PATH, index_col=0, parse_dates=True)
        # Basic cleaning
        DASHBOARD_DATA_DF.dropna(how='all', inplace=True)
        # Calculate equal weight return for dashboard metrics
        asset_cols = [c for c in ASSETS if c in DASHBOARD_DATA_DF.columns]
        if asset_cols:
             DASHBOARD_DATA_DF['Daily_Ret_Eq'] = DASHBOARD_DATA_DF[asset_cols].pct_change().mean(axis=1)
        print(f"Data loaded successfully. Shape: {DASHBOARD_DATA_DF.shape}")
        print(f"Data range: {DASHBOARD_DATA_DF.index.min().date()} to {DASHBOARD_DATA_DF.index.max().date()}")
    else:
        print("❌ Failed to initialize historical data.")

# Initialize data at startup
try:
    initialize_dashboard_data()
except Exception as e:
    print(f"Warning: Data initialization failed. Error: {e}")


# =========================================
# Professional Metrics & Evaluation Functions
# =========================================

def evaluate_agent_pro(env, model):
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
    valid_dates = env.df.index[env.window_size-1:]
    return pd.Series(portfolio_values, index=valid_dates[:len(portfolio_values)])

def calculate_metrics_pro(portfolio_values, freq=252, rf=0.0):
    """
    Calculates key professional performance metrics from a series of portfolio values.
    """
    if len(portfolio_values) < 2:
        return {k: "N/A" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Max Drawdown", "Calmar Ratio"]}

    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
         return {k: "0.00%" if "%" in k else "0.00" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Max Drawdown", "Calmar Ratio"]}

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
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio
    }

# =========================================
# XAI: Feature Importance Function
# =========================================
def calculate_feature_importance(model, obs):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    if obs_tensor.dim() == 1: obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor.requires_grad_()
    
    actor = model.policy.actor
    baseline = torch.zeros_like(obs_tensor)
    steps = 50
    scaled_inputs = [baseline + (float(i) / steps) * (obs_tensor - baseline) for i in range(steps + 1)]
    
    grads = []
    for scaled_input in scaled_inputs:
        action_mean = actor(scaled_input)
        target_output = action_mean.sum()
        grad = torch.autograd.grad(outputs=target_output, inputs=scaled_input)[0]
        grads.append(grad)

    # --- Stack gradients first, then perform arithmetic ---
    stacked_grads = torch.stack(grads)
    avg_grads = (stacked_grads[:-1] + stacked_grads[1:]) / 2.0
    avg_grads = avg_grads.mean(dim=0)
    # -----------------------------------------------------------

    integrated_grads = (obs_tensor - baseline) * avg_grads
    importance_scores = integrated_grads.detach().cpu().numpy().flatten()
    
    feature_names = []
    for i in range(WINDOW_SIZE):
        for asset in ASSETS: feature_names.append(f"{asset}_t-{WINDOW_SIZE-1-i}")
    for i in range(WINDOW_SIZE):
        for macro in MACRO_COLS: feature_names.append(f"{macro}_t-{WINDOW_SIZE-1-i}")

    feature_importance_dict = dict(zip(feature_names, importance_scores))
    aggregated_importance = {}
    for base_feature in ASSETS + MACRO_COLS:
        total_imp = sum(abs(val) for key, val in feature_importance_dict.items() if key.startswith(base_feature))
        aggregated_importance[base_feature] = total_imp

    top_features = dict(sorted(aggregated_importance.items(), key=lambda item: item[1], reverse=True)[:8])

    fig = px.bar(x=list(top_features.values()), y=list(top_features.keys()), orientation='h',
        title="Top Influential Features (XAI)", labels={'x': 'Importance', 'y': 'Feature'},
        color=list(top_features.values()), color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10), height=300,
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Roboto", font_color="black"))
    return fig

# =========================================
# Tab 4 Logic: Historical Simulation 
# =========================================

def run_historical_simulation(start_date_str, end_date_str):
    """
    Runs the RL agent on historical data and compares to baselines using professional metrics.
    """
    if DASHBOARD_DATA_DF is None:
        return go.Figure(), "Data not initialized. Please restart app.", gr.update(visible=False)

    status_msg = "Preparing simulation..."
    yield go.Figure(), status_msg, gr.update(visible=False)

    try:
        # 1. Validate and Slice Data
        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
        except ValueError:
             yield go.Figure(), "Error: Invalid date format. Use YYYY-MM-DD.", gr.update(visible=False)
             return

        if start_date < DASHBOARD_DATA_DF.index.min() or end_date > DASHBOARD_DATA_DF.index.max():
             avail_start = DASHBOARD_DATA_DF.index.min().date()
             avail_end = DASHBOARD_DATA_DF.index.max().date()
             yield go.Figure(), f"Error: Selected dates outside available range ({avail_start} to {avail_end}).", gr.update(visible=False)
             return

        df_slice = DASHBOARD_DATA_DF.loc[start_date:end_date].copy()
        asset_cols_only = [c for c in ASSETS if c in df_slice.columns]

        if len(df_slice) < WINDOW_SIZE + 10:
             yield go.Figure(), "Error: Time period too short for simulation.", gr.update(visible=False)
             return

        # 2. Setup Environment and Agent
        status_msg = "Running RL Agent simulation..."
        yield go.Figure(), status_msg, gr.update(visible=False)

        env = PortfolioEnv(df_slice, WINDOW_SIZE, initial_balance=10000)

        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = SAC.load(MODEL_PATH)

        # 3. Run Simulation Loop & Get Values using Pro Function
        rl_portfolio_series = evaluate_agent_pro(env, model)

        # 4. Calculate Baselines using Pro Functions
        status_msg = "Calculating baselines and metrics..."
        yield go.Figure(), status_msg, gr.update(visible=False)

        # Pass only asset columns to baseline functions
        bnh_portfolio_series = buy_and_hold(df_slice[asset_cols_only], initial_balance=10000)
        # Realign B&H index to match RL agent's start date
        bnh_portfolio_series = bnh_portfolio_series.loc[rl_portfolio_series.index[0]:]
        # Normalize B&H starting value to match RL agent's start
        bnh_portfolio_series = bnh_portfolio_series / bnh_portfolio_series.iloc[0] * 10000

        eq_portfolio_series = equally_weighted_rebalanced(df_slice[asset_cols_only], initial_balance=10000)
        eq_portfolio_series = eq_portfolio_series.loc[rl_portfolio_series.index[0]:]
        eq_portfolio_series = eq_portfolio_series / eq_portfolio_series.iloc[0] * 10000

        # 5. Generate Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rl_portfolio_series.index, y=rl_portfolio_series, mode='lines', name='RL Agent (SAC)', line=dict(color='#10b981', width=3)))
        fig.add_trace(go.Scatter(x=bnh_portfolio_series.index, y=bnh_portfolio_series, mode='lines', name='Buy & Hold (SPY)', line=dict(color='#6b7280', dash='dash')))
        fig.add_trace(go.Scatter(x=eq_portfolio_series.index, y=eq_portfolio_series, mode='lines', name='Equal Weighted', line=dict(color='#a855f7', dash='dot')))

        fig.update_layout(
            title="Simulation: Strategy Performance Comparison ($10k Start)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # 6. Calculate Professional Metrics Table
        rl_m = calculate_metrics_pro(rl_portfolio_series)
        bnh_m = calculate_metrics_pro(bnh_portfolio_series)
        eq_m = calculate_metrics_pro(eq_portfolio_series)

        # Helper to format based on metric type
        def fmt(val, is_pct=True):
            if pd.isna(val): return "N/A"
            return f"{val:.2%}" if is_pct else f"{val:.2f}"

        metrics_data = {
            "Metric": ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Volatility (Ann.)", "Max Drawdown", "Calmar Ratio"],
            "RL Agent (SAC)": [fmt(rl_m["Total Return"]), fmt(rl_m["CAGR"]), fmt(rl_m["Sharpe Ratio"], False), fmt(rl_m["Sortino Ratio"], False), fmt(rl_m["Volatility"]), fmt(rl_m["Max Drawdown"]), fmt(rl_m["Calmar Ratio"], False)],
            "Buy & Hold (SPY)": [fmt(bnh_m["Total Return"]), fmt(bnh_m["CAGR"]), fmt(bnh_m["Sharpe Ratio"], False), fmt(bnh_m["Sortino Ratio"], False), fmt(bnh_m["Volatility"]), fmt(bnh_m["Max Drawdown"]), fmt(bnh_m["Calmar Ratio"], False)],
            "Equal Weighted": [fmt(eq_m["Total Return"]), fmt(eq_m["CAGR"]), fmt(eq_m["Sharpe Ratio"], False), fmt(eq_m["Sortino Ratio"], False), fmt(eq_m["Volatility"]), fmt(eq_m["Max Drawdown"]), fmt(eq_m["Calmar Ratio"], False)],
        }
        metrics_df = pd.DataFrame(metrics_data)

        # Format the dataframe as a markdown table for cleaner display
        metrics_md = metrics_df.to_markdown(index=False)
        final_metrics_display = f"### 📊 Professional Performance Metrics\n\n{metrics_md}"

        yield fig, "Simulation Complete.", final_metrics_display

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield go.Figure(), f"Error during simulation: {str(e)}", gr.update(visible=False)


# =========================================
# Tab 3 Logic: Historical Data Analyst
# =========================================

def run_historical_analysis(selected_assets, period_name):
    """Backend for Tab 3."""
    if DASHBOARD_DATA_DF is None or not selected_assets:
        return go.Figure(), "Please wait for data initialization or select assets."

    status_html = """<div style="color: #9ca3af;">🔄 Processing data and running AI analysis...</div>"""
    yield go.Figure(), status_html

    try:
        # 1. Filter Data by Time Period
        days = TIME_PERIODS.get(period_name, 365)
        cutoff_date = datetime.now() - timedelta(days=days)
        valid_assets = [a for a in selected_assets if a in DASHBOARD_DATA_DF.columns]
        if not valid_assets:
             yield go.Figure(), "Error: Selected assets not found in available data."
             return
        df_filtered = DASHBOARD_DATA_DF.loc[cutoff_date:, valid_assets].copy()
        if df_filtered.empty:
             yield go.Figure(), f"No data found for the selected period: {period_name}"
             return

        # 2. Generate Normalized Price Plot
        df_normalized = df_filtered / df_filtered.iloc[0] * 100
        fig = px.line(df_normalized, x=df_normalized.index, y=df_normalized.columns,
                      title=f"Performance Comparison: {period_name} (Base=100)",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Normalized Price", xaxis_title="Date", legend_title_text="", hovermode="x unified")

        # 3. Run AI Analysis
        analysis_text = analyze_historical_segment(df_filtered, valid_assets, period_name)
        formatted_analysis = f"### 🤖 AI Analyst Report: {period_name}\n\n{analysis_text}"
        yield fig, formatted_analysis

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield go.Figure(), f"### Error during analysis\n\n{str(e)}"


# =========================================
# Tab 2 Logic: Forecast & Analysis (XAI)
# =========================================

def get_latest_data_window(window_size=30):
    """Fetches latest data needed for prediction."""
    print("Fetching prediction data...")
    lookback_days = window_size + 150
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    temp_filename = os.path.join("data", "temp_gradio_prediction_data.csv")
    fetch_market_data(start_date, end_date, temp_filename)
    if not os.path.exists(temp_filename): raise Exception("Failed to fetch market data file.")
    df = pd.read_csv(temp_filename, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    if len(df) < window_size: raise Exception(f"Not enough clean data fetched for prediction.")
    return df.iloc[-window_size:].copy()

def prepare_observation(data_window):
    price_data = data_window[ASSETS].values
    macro_data = data_window[MACRO_COLS].values
    norm_prices = price_data / (price_data[0] + 1e-8)
    norm_macro = macro_data / (macro_data[0] + 1e-8)
    obs = np.concatenate([norm_prices, norm_macro], axis=1)
    # Return both flattened obs for model and raw obs for XAI
    return obs.flatten().astype(np.float32), obs.astype(np.float32), data_window

def predict_and_analyze():
    yield "Starting...", None, go.Figure(), "Loading..."
    try:
        data_window = get_latest_data_window(WINDOW_SIZE)
        flat_obs, raw_obs, df_window_for_analyst = prepare_observation(data_window)
        
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError("Model not found.")
        model = SAC.load(MODEL_PATH)
        
        # --- Pass the FLATTENED observation to XAI function ---
        # The XAI function logic expects an input that matches the model's input layer.
        yield "XAI Calc...", None, go.Figure(), "Calculating XAI..."
        xai_plot = calculate_feature_importance(model, flat_obs)

        action, _ = model.predict(flat_obs, deterministic=True)
        exp_act = np.exp(np.asarray(action).flatten())
        weights = exp_act / np.sum(exp_act)
        
        allocs = {ASSETS[i]: weights[i] for i in range(len(ASSETS))}
        allocs['Cash'] = weights[-1]
        alloc_df = pd.DataFrame(list(allocs.items()), columns=['Asset', 'Alloc'])
        alloc_df['Alloc'] = alloc_df['Alloc'].apply(lambda x: f"{x:.2%}")

        yield "AI Analysis...", alloc_df, xai_plot, "Running AI..."
        llm_allocs = {k: float(v) for k, v in allocs.items()}
        res = analyze_agent_decision(df_window_for_analyst, llm_allocs)
        
        if isinstance(res, dict):
            strat, risk, just, conf = res.get('strategy_summary','N/A'), res.get('risk_level','N/A').upper(), res.get('justification','N/A'), res.get('confidence_score','N/A')
            border_col = "#ef4444" if 'HIGH' in risk else "#10b981"
            bg_col = "#7f1d1d" if 'HIGH' in risk else "#064e3b"
            icon = "⛔" if 'HIGH' in risk else "🚀"
            status = "TRADE BLOCKED" if 'HIGH' in risk else "TRADE APPROVED"
            
            html = f"""<div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151;">
                <h3 style="margin-top: 0; color: #e5e7eb;">🤖 AI Report</h3>
                <p><strong>Strategy:</strong> <span style="color:#d1d5db">{strat}</span></p>
                <p><strong>Risk:</strong> <span style="color:{border_col}; font-weight:bold">{risk}</span></p>
                <p><strong>Reason:</strong> <span style="color:#d1d5db">{just}</span></p>
                <p><strong>Conf:</strong> <span style="color:#d1d5db">{conf}/10</span></p></div>
                <div style="background-color:{bg_col}; color:white; padding:15px; margin-top:10px; border-radius:12px; text-align:center; font-weight:bold;">{icon} {status}</div>"""
        else:
            html = f"<div style='color:red'>{str(res)}</div>"
        
        yield "Done", alloc_df, xai_plot, html
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}", None, go.Figure(), f"Error: {str(e)}"


# =========================================
# Tab 1 Logic: Live Dashboard (DUMMY DATA)
# =========================================
def get_dashboard_metrics():
    return "$135,400", "+3.07%"

def get_portfolio_history_plot():
    dates = pd.date_range(start="2023-01-01", periods=100)
    np.random.seed(42)
    rl_returns = np.random.normal(0.001, 0.01, 100)
    bnh_returns = np.random.normal(0.0005, 0.012, 100)
    rl_value = 10000 * np.cumprod(1 + rl_returns)
    bnh_value = 10000 * np.cumprod(1 + bnh_returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=rl_value, mode='lines', name='RL Agent (Live)', line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=dates, y=bnh_value, mode='lines', name='Benchmark', line=dict(color='#6b7280', dash='dash')))
    fig.update_layout(title="Portfolio Net Worth (Live Tracking)", xaxis_title="Date", yaxis_title="Net Worth ($)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def get_current_allocation_plot():
    labels = ASSETS + ['Cash']
    values = [0.25, 0.10, 0.30, 0.15, 0.05, 0.15]
    fig = px.pie(values=values, names=labels, title="Current Holdings Breakdown", color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition='inside', textinfo='percent+label', hole=.4)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=-0.1))
    return fig

def get_recent_transactions():
    data = [["2025-11-24", "Rebalance", "MULTIPLE", "N/A"], ["2025-11-24", "SELL", "SPY", "$4,500"], ["2025-11-24", "BUY", "TLT", "$4,200"], ["2025-11-21", "BUY", "BTC-USD", "$1,000"]]
    return pd.DataFrame(data, columns=["Date", "Type", "Asset", "Approx. Value"])


# =========================================
# Gradio Interface
# =========================================

custom_css = """
.metric-box { background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; text-align: center; }
.metric-label { font-size: 1.1em; color: #9ca3af; margin-bottom: 5px; }
.metric-value { font-size: 2.2em; font-weight: 700; color: #e5e7eb; }
.disclaimer-box { background-color: #374151; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; color: #d1d5db; font-size: 0.9em; margin-bottom: 20px; }
"""

# theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="slate", neutral_hue="zinc").set(
#     body_background_fill="#111827", block_background_fill="#1f2937", block_border_width="1px", block_border_color="#374151"
# )

with gr.Blocks(
    # theme=theme, css=custom_css,
               title="NeuralFolio — Deep RL Portfolio Manager") as demo:
    gr.HTML("""<script>function forceDark(){document.body.classList.add('dark');} forceDark(); setTimeout(forceDark, 500);</script>""")

    gr.Markdown("# 🧠 NeuralFolio — Deep RL & LLM Portfolio Manager")

    with gr.Tabs():
        # ================= TAB 1: DASHBOARD (RESTORED) =================
        with gr.TabItem("📊 Live Dashboard"):
            # Metrics Row
            with gr.Row():
                nw_val, dc_val = get_dashboard_metrics()
                with gr.Column(elem_classes=["metric-box"]):
                    gr.HTML(f"<div class='metric-label'>Current Net Worth</div><div class='metric-value'>{nw_val}</div>")
                with gr.Column(elem_classes=["metric-box"]):
                    gr.HTML(f"<div class='metric-label'>24h Change</div><div class='metric-value' style='color: #10b981;'>{dc_val}</div>")

            # Main Chart row
            with gr.Row():
                with gr.Column(scale=3):
                    history_chart = gr.Plot(value=get_portfolio_history_plot(), label="Net Worth History")

            # Bottom Row: Allocations and Transactions
            with gr.Row():
                with gr.Column(scale=1):
                    allocation_chart = gr.Plot(value=get_current_allocation_plot(), label="Current Allocation")
                with gr.Column(scale=2):
                    gr.Markdown("### Recent Transactions")
                    transactions_table = gr.Dataframe(value=get_recent_transactions(), interactive=False, wrap=True)

        # ================= TAB 2: FORECAST (UPDATED with XAI) =================
        with gr.TabItem("🔮 Forecast & AI Analysis"):
            gr.Markdown("### Generate Tomorrow's Portfolio Strategy")
            run_btn = gr.Button("🚀 Run Overnight Analysis", variant="primary", size="lg")
            status_output = gr.Textbox(label="System Status", placeholder="Ready...", interactive=False, lines=1)
            gr.Markdown("---")

            with gr.Row():
                # Left Column: Allocations & XAI Plot
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Suggested Position")
                    allocation_output = gr.Dataframe(headers=["Asset", "Allocation"], datatype=["str", "str"], interactive=False)

                    # NEW: XAI Feature Importance Plot
                    gr.Markdown("### 🧠 Why did the agent choose this?")
                    xai_output_plot = gr.Plot(label="Top Influential Factors (XAI)", show_label=False)

                # Right Column: AI Analysis Report
                with gr.Column(scale=3):
                    analysis_report_html = gr.HTML(label="AI Risk Analysis Report")

            # Updated click event with new XAI output
            run_btn.click(
                fn=predict_and_analyze,
                inputs=None,
                outputs=[status_output, allocation_output, xai_output_plot, analysis_report_html]
            )

        # ================= TAB 3: HISTORICAL DATA ANALYST =================
        with gr.TabItem("📅 Historical Data Analyst"):
            gr.Markdown("### Analyze Past Market Performance with AI")

            with gr.Row():
                with gr.Column(scale=1):
                    all_tickers_hist = ASSETS + list(FRED_IDS.values())
                    if DASHBOARD_DATA_DF is not None:
                        available_tickers_hist = [t for t in all_tickers_hist if t in DASHBOARD_DATA_DF.columns]
                    else:
                        available_tickers_hist = []
                    default_tickers_hist = available_tickers_hist[:3] if available_tickers_hist else []

                    asset_selector = gr.Dropdown(choices=available_tickers_hist, value=default_tickers_hist, multiselect=True, label="1. Select Assets")
                    period_selector = gr.Dropdown(choices=list(TIME_PERIODS.keys()), value="1 Year", label="2. Select Period")
                    analyze_btn = gr.Button("🔎 Run Analysis", variant="primary")

                with gr.Column(scale=3):
                    historical_plot = gr.Plot(label="Performance Plot")

            gr.Markdown("---")
            historical_analysis_md = gr.Markdown("### 🤖 AI Analyst Report\n\n*Click 'Run Analysis' to generate.*")

            analyze_btn.click(
                fn=run_historical_analysis,
                inputs=[asset_selector, period_selector],
                outputs=[historical_plot, historical_analysis_md]
            )

        # ================= TAB 4: HISTORICAL SIMULATION (UPDATED with Pro Metrics) =================
        with gr.TabItem("🔙 Historical Simulation"):
            gr.Markdown("### Backtest the RL Agent against Baselines")

            # Disclaimer Box
            gr.HTML(f"""
            <div class='disclaimer-box'>
                <strong>⚠️ IMPORTANT DISCLAIMER:</strong> The RL model was trained on data from approximately
                <strong>{TRAIN_START_DATE} to {TRAIN_END_DATE}</strong>. Running simulations outside or overlapping significantly
                with this period may not accurately reflect real-world performance (lookahead bias or out-of-distribution data).
                Use for educational purposes only.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
                    end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
                    sim_btn = gr.Button("▶️ Run Simulation", variant="primary")
                    sim_status = gr.Textbox(label="Status", interactive=False, lines=1)

                with gr.Column(scale=3):
                    sim_plot = gr.Plot(label="Simulation Performance")

            gr.Markdown("---")
            # Updated to Markdown component for better table formatting
            sim_metrics_md = gr.Markdown("### 📊 Professional Performance Metrics\n\n*Run simulation to see metrics.*")

            sim_btn.click(
                fn=run_historical_simulation,
                inputs=[start_date_input, end_date_input],
                outputs=[sim_plot, sim_status, sim_metrics_md]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)