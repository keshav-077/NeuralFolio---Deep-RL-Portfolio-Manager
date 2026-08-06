---
title: NeuralFolio
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 NeuralFolio — Deep RL & LLM Portfolio Manager

An AI-powered portfolio optimization system that uses **Deep Reinforcement Learning** (SAC, TD3, PPO) and a **Large Language Model** as an AI Risk Analyst.

## Features

- 📊 **Live Dashboard** - Real-time portfolio tracking
- 🔮 **AI Forecast** - Next-day allocation with LLM risk analysis
- 📅 **Historical Analyst** - Compare asset performance with AI insights  
- 🔙 **Backtesting** - Test RL agents vs baselines

## Tech Stack

- **RL**: Stable-Baselines3 (SAC, TD3, PPO)
- **LLM**: Qwen 2.5 (1.5B for free-tier, 3B for GPU)
- **Data**: Yahoo Finance + FRED macro indicators
- **Framework**: Gradio + PyTorch

## GitHub

Full code, training scripts, and documentation:  
**https://github.com/keshav-077/NeuralFolio---Deep-RL-Portfolio-Manager**

---

⚠️ **Note**: This demo runs on CPU (free-tier). The LLM analysis is slow (~30-60s per request). For faster inference, upgrade to GPU hardware.

⚠️ **Disclaimer**: This is a research project for educational purposes. Not financial advice.
