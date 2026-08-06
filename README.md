![Banner](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🧠 NeuralFolio — Deep RL & LLM Portfolio Manager

An end-to-end research project that uses **Deep Reinforcement Learning** to train autonomous agents for multi-asset portfolio management, paired with a **Large Language Model** that acts as a risk analyst on top of the agent's decisions. Ships with an interactive **Gradio** dashboard for live tracking, forward-looking strategy generation, and historical backtesting.

The system trains and benchmarks three state-of-the-art DRL algorithms — **Proximal Policy Optimization (PPO)**, **Soft Actor-Critic (SAC)**, and **Twin Delayed DDPG (TD3)** — and pits them against a Buy & Hold baseline using a full suite of financial metrics (Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar).

The web app integrates **`Qwen/Qwen2.5-3B-Instruct`** as an AI Risk Analyst that produces a textual justification, a risk level, and a confidence score for every allocation the RL agent proposes.

> 🛑 *On free-tier hardware the LLM is swapped for `Qwen/Qwen2.5-1.5B-Instruct`. Analysis is slow — let it run in the background and come back to it.* 🛑
> 🛑 *The default agent was trained on data from 2015-01-01 to 2020-12-31. For live deployment, retrain on the most recent data.* 🛑

---

## 📜 Table of Contents

1. [📊 Data & Asset Selection](#-data--asset-selection)
2. [🎯 Benchmarking Against Baselines](#-benchmarking-against-baselines)
3. [🏆 Key Findings](#-key-findings)
4. [🧠 Comparative Analysis of Agent Strategies](#-comparative-analysis-of-agent-strategies)
5. [🔬 Research Notes: Why Simplicity Won](#-research-notes-why-simplicity-won)
6. [🖥️ The NeuralFolio Web App](#-the-neuralfolio-web-app)
7. [✅ Conclusion](#-conclusion)
8. [📂 Project Structure](#-project-structure)
9. [🚀 How to Run](#-how-to-run)

---

## 📊 Data & Asset Selection

Daily closing prices come from **Yahoo Finance** via `yfinance`. The observation space is augmented with key macroeconomic indicators from **FRED (Federal Reserve Economic Data)** — VIX, Federal Funds Rate, and CPI — so the agents can learn to adapt across market regimes.

**Realistic constraints:** the environment charges a **0.001%** transaction cost on the notional of every trade, forcing the agent to learn fee-aware strategies.

The portfolio is a deliberately diverse five-asset basket:

* **Growth Equities (AAPL, MSFT)** — high-growth, high-volatility tech.
* **Market Index (SPY)** — broad US equity exposure.
* **Safe Haven (TLT)** — 20+ Year US Treasury Bonds, defensive in downturns.
* **Alternative Asset (BTC-USD)** — non-traditional, high-volatility return source.

---

## 🎯 Benchmarking Against Baselines

An RL agent is only useful if it beats a naive strategy. The primary baseline is **Buy & Hold** (equal allocation at the start, never touched). Every trained agent is evaluated on a comprehensive set of risk-adjusted metrics.

![Baseline Performance](results/baseline_results.png)

---

## 🏆 Key Findings

Evaluated on out-of-sample 2021–2023 data, two champions emerged on different axes of performance.

#### Final Performance Comparison (2021–2023)

| Metric | **TD3 (Transformer)** | **SAC (MLP)** | Buy & Hold | PPO (MLP) | TD3 (MLP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Return** | 25.34% | **39.23%** | 32.76% | 22.85% | 22.07% |
| **CAGR** | 8.20% | **12.25%** | 9.96% | 7.45% | 7.21% |
| **Sharpe Ratio** | **0.61** | 0.56 | 0.59 | 0.41 | 0.42 |
| **Volatility** | **14.77%** | 27.47% | 19.06% | 25.90% | 23.00% |
| **Max Drawdown** | **-20.01%** | -29.08% | -28.82% | -44.26% | -40.50% |

![Main Performance Chart](results/final_performance_comparison_all_agents.png)
*Bitcoin was excluded from the comparison for a like-for-like equity/bond benchmark.*

### 🥇 TD3 (Transformer) — Master of Risk Management

The Transformer-based TD3 agent wins on every risk-adjusted metric. Lowest volatility (14.77%), smallest max drawdown (-20.01%). Its attention mechanism over the 30-day lookback lets it shift into defensive assets like TLT during the 2022 bear market.

### 🚀 SAC (MLP) — Aggressive Growth Engine

SAC achieved the highest absolute returns (39.23% total, 12.25% CAGR) but at 27.47% volatility. A high-conviction, mostly-static allocation to growth assets.

---

## 🧠 Comparative Analysis of Agent Strategies

Different algorithm + architecture combinations produce distinct "investment philosophies." The allocation charts make this visible.

### TD3 (Transformer): The Dynamic Hedger

The Transformer agent maintains a core equity position but actively rebalances into TLT during drawdowns — the source of its superior risk profile.

![TD3 Transformer Allocation](results/td3_transformer_allocation.png)

### SAC (MLP): The High-Conviction Aggressor

SAC converges to a near-static, high-risk, high-return allocation with little defensive exposure.

![SAC Allocation](results/sac_allocation.png)

### PPO (MLP): The Failed Active Trader

PPO churns the portfolio without generating alpha, producing the deepest max drawdown (-44.26%).

![PPO Allocation](results/ppo_allocation.png)

### TD3 (MLP): The Failed Static Allocator

The vanilla-MLP TD3 picks a static allocation that captures neither the growth of SAC nor the risk control of the Transformer.

![TD3 MLP Allocation](results/td3_allocation.png)

---

## 🔬 Research Notes: Why Simplicity Won

Hypotheses tested during the project, all of which *degraded* performance:

1. **More features are better** — adding RSI / MACD added noise.
2. **Memory models are better** — `RecurrentPPO` (LSTM) overfit.
3. **Regularization helps** — L1 and L2 both hurt.
4. **Longer context windows are better** — 60-day window hurt vs. 30.
5. **Transformers are better** — when *not* paired with a small dataset (Transformer was the eventual winner only because the lookback regime it could model was rich enough).

The takeaway: a simple MLP on normalized 30-day price data, with no fancy features, was the most robust architecture.

---

## 🖥️ The NeuralFolio Web App

The interactive dashboard is built with **Gradio** and integrates Qwen as an AI Risk Analyst.

*Live demo ->* [Hugging Face Space](https://huggingface.co/keshav775)
> 🛑 *The hosted demo uses a smaller LLM (`Qwen/Qwen2.5-1.5B-Instruct`) for free-tier hardware compatibility. Analysis is slow — let it run in the background.* 🛑
> 🛑 *The deployed agent was trained on 2015–2020 data. For a real production deployment, retrain on the most recent data.* 🛑

### Key Features

#### 1. Live Dashboard & Net Worth Tracking

Track holdings, recent transactions, and net worth evolution in real time.

![Live Dashboard](results/tab1.png)

#### 2. AI-Powered Strategy Forecast & Risk Analysis

Generate the next-day allocation from a trained RL agent. The LLM produces a Risk Analyst Report with confidence score and justification. Includes **Explainable AI (XAI)** feature-importance plots.

![AI Forecast and Risk Analysis](results/tab2.png)

#### 3. AI-Driven Historical Market Analyst

Compare historical performance of selected assets over custom timeframes, normalized to a base of 100. The system generates an AI Analyst Report on trends, volatility, and comparative performance.

![AI-Driven Historical Market Analyst](results/tab3.png)

#### 4. Historical Simulation & Backtesting

Run dynamic backtests of any trained RL agent against Buy & Hold and Equal-Weighted baselines on any historical window.

![Historical Simulation](results/tab4.png)

---

## ✅ Conclusion

Deep RL can discover sophisticated investment strategies — but the most successful agent was not the one that traded the most, it was the one that managed risk the best. Effective risk management, not hyperactive trading, is the durable edge.

---

## 📂 Project Structure

```bash
├── assets/             # Images for the README (banner)
├── checkpoints/        # Trained model weights (.zip)
├── data/               # Fetched CSV data
├── results/            # Generated plots and metrics
├── scripts/
│   ├── app.py                  # The Gradio web app (NeuralFolio)
│   ├── check_env.py            # Smoke test for the custom env
│   ├── custom_policy.py        # Custom policy networks (e.g. Transformer)
│   ├── environment.py          # Gymnasium environment
│   ├── evaluate_baselines.py   # Buy & hold and equal-weighted baselines
│   ├── evaluate.py             # Evaluate a single trained agent
│   ├── fetch_market_data.py    # Download historical data from yfinance
│   ├── llm_analysis_rag.py     # LLM-based risk analyst
│   ├── predict_tomorrow.py     # Generate next-day allocations
│   ├── stress_test.py          # Compare all agents on a dataset
│   ├── train.py                # Train an RL agent
│   ├── tune_sac.py             # Hyperparameter tuning for SAC
│   └── visualize_strategy.py   # Plot asset allocation of a trained agent
├── requirements.txt    # Python dependencies
├── DEPLOY.md           # Step-by-step Hugging Face Space deploy guide
└── README.md
```

## 🚀 How to Run

### Setup

```bash
git clone https://github.com/keshav-077/NeuralFolio
cd NeuralFolio
pip install -r requirements.txt
```

### Data Fetching

```bash
python scripts/fetch_market_data.py --start 2015-01-01 --end 2020-12-31 --filename data/train_data.csv
python scripts/fetch_market_data.py --start 2021-01-01 --end 2023-12-31 --filename data/eval_data.csv
```

### Training

```bash
# TD3 agent (default 20,000 timesteps)
python scripts/train.py --agent td3 --datafile data/train_data.csv

# SAC for longer
python scripts/train.py --agent sac --datafile data/train_data.csv --timesteps 50000
```

Trained models are saved to `checkpoints/` (e.g. `sac_portfolio_model.zip`).

### Evaluation & Visualization

```bash
# Compare all agents on eval data
python scripts/stress_test.py --datafile data/eval_data.csv

# Evaluate a single agent
python scripts/evaluate.py --agent td3 --checkpoint checkpoints/td3_portfolio_model.zip --datafile data/eval_data.csv

# Visualize an agent's strategy
python scripts/visualize_strategy.py --agent ppo --checkpoint checkpoints/ppo_portfolio_model.zip --datafile data/eval_data.csv
```

### Launch the App

```bash
python scripts/app.py
```

Open the URL printed in the console (default `http://localhost:7860`).

### Deploy to Hugging Face Spaces

See [DEPLOY.md](DEPLOY.md) for a step-by-step guide to deploy this Gradio app to your own HF Space under the `keshav775` account.

---

## 📜 License

MIT — see [LICENSE](LICENSE).
