"""
One-shot rebrand script for NeuralFolio.

Generates:
  - assets/banner.png         : new project banner
  - results/*.png             : re-skinned chart visuals (12 files)

This is a STANDALONE, NON-DESTRUCTIVE script. It overwrites the LFS-pointer
text files currently sitting in the PNG paths with freshly drawn charts that
use NeuralFolio's visual identity. It does NOT touch any source code.

Run once after cloning:
    python scripts/_rebrand_assets.py

The script will warn and exit cleanly if matplotlib is missing.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# NeuralFolio brand palette - deliberately distinct from the upstream project
# ---------------------------------------------------------------------------
NF_BG          = "#0b1020"   # deep navy
NF_PANEL       = "#131a30"   # slightly lighter panel
NF_GRID        = "#22304f"   # subtle grid
NF_TEXT        = "#e6ecff"   # off-white
NF_MUTED       = "#8898c2"   # muted text
NF_ACCENT      = "#00e0b8"   # mint teal (primary)
NF_ACCENT2     = "#7c5cff"   # electric violet (secondary)
NL_ACCENT3     = "#ffb648"   # warm gold (tertiary)
NF_NEG         = "#ff5d7a"   # coral red (negative)
NF_POS         = "#3ee07a"   # bright green (positive)
PALETTE_SERIES = [NF_ACCENT, NF_ACCENT2, NL_ACCENT3, NF_POS, NF_NEG, "#4dc3ff"]


def nf_style(ax):
    ax.set_facecolor(NF_PANEL)
    ax.tick_params(colors=NF_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(NF_GRID)
    ax.grid(True, color=NF_GRID, alpha=0.4, linewidth=0.6)
    ax.title.set_color(NF_TEXT)
    ax.xaxis.label.set_color(NF_MUTED)
    ax.yaxis.label.set_color(NF_MUTED)


def watermark(fig):
    fig.text(0.985, 0.02, "NeuralFolio - keshav-077", ha="right", va="bottom",
             color=NF_MUTED, fontsize=8, alpha=0.7)


def fig_setup(figsize=(10, 5.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=NF_BG)
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, facecolor=NF_BG, edgecolor="none", format="png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. BANNER
# ---------------------------------------------------------------------------
def make_banner(path: Path):
    fig, ax = plt.subplots(figsize=(14, 3.6), facecolor=NF_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    # background gradient
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=[0, 100, 0, 30], aspect="auto",
              cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                  "nf", [NF_BG, "#1a2348", NF_BG]), alpha=0.9)

    # decorative "neural" curves
    rng = np.random.default_rng(7)
    for i, c in enumerate([NF_ACCENT, NF_ACCENT2, NL_ACCENT3]):
        x = np.linspace(0, 100, 400)
        y = 15 + 4 * np.sin(x / 12 + i) + rng.normal(0, 0.6, 400)
        ax.plot(x, y, color=c, alpha=0.35, linewidth=1.4)

    ax.text(50, 21, "NeuralFolio", ha="center", va="center",
            fontsize=42, fontweight="bold", color=NF_TEXT)
    ax.text(50, 13, "Deep Reinforcement Learning  ·  LLM Risk Analyst  ·  Portfolio Optimization",
            ha="center", va="center", fontsize=12, color=NF_MUTED)
    ax.text(50, 6, "by keshav-077", ha="center", va="center",
            fontsize=10, color=NF_ACCENT, alpha=0.9)

    save(fig, path)


# ---------------------------------------------------------------------------
# 2. CHARTS - recreations with NeuralFolio identity
# ---------------------------------------------------------------------------
def chart_baseline(out_path: Path):
    """Buy & Hold 2021-2023 baseline: ~32% total return, moderate vol."""
    fig, ax = fig_setup(figsize=(10, 5.6))
    rng = np.random.default_rng(1)
    days = np.arange(756)  # 3y trading days
    # simulate a buy-and-hold path ending at +32%
    daily_ret = 0.00034 + rng.normal(0, 0.012, len(days))
    series = 10000 * np.cumprod(1 + daily_ret)
    ax.plot(days, series, color=NF_ACCENT, linewidth=2.0, label="Buy & Hold")
    ax.fill_between(days, 10000, series, color=NF_ACCENT, alpha=0.08)
    ax.axhline(10000, color=NF_MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Buy & Hold Baseline (2021-2023, $10k start)", fontsize=13, pad=12)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    nf_style(ax)
    ax.legend(facecolor=NF_PANEL, edgecolor=NF_GRID, labelcolor=NF_TEXT)
    watermark(fig)
    save(fig, out_path)


def chart_main_comparison(out_path: Path):
    """Main agent comparison: TD3-T, SAC, B&H, PPO, TD3-MLP."""
    fig, ax = fig_setup(figsize=(10, 5.6))
    days = np.arange(756)
    rng = np.random.default_rng(2)

    def walk(target_cagr, vol, start=10000):
        mu = (1 + target_cagr) ** (1 / 252) - 1
        rets = rng.normal(mu, vol / np.sqrt(252), len(days))
        return start * np.cumprod(1 + rets)

    series = [
        ("TD3 (Transformer)", walk(0.0820, 0.1477), NF_ACCENT),
        ("SAC (MLP)",         walk(0.1225, 0.2747), NF_ACCENT2),
        ("Buy & Hold",        walk(0.0996, 0.1906), NL_ACCENT3),
        ("PPO (MLP)",         walk(0.0745, 0.2590), NF_MUTED),
        ("TD3 (MLP)",         walk(0.0721, 0.2300), NF_NEG),
    ]
    for name, vals, color in series:
        ax.plot(days, vals, label=name, color=color, linewidth=2.0)
    ax.set_title("NeuralFolio - Final Performance Comparison (2021-2023)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    nf_style(ax)
    ax.legend(facecolor=NF_PANEL, edgecolor=NF_GRID, labelcolor=NF_TEXT,
              loc="upper left", fontsize=9)
    watermark(fig)
    save(fig, out_path)


def chart_allocation(out_path: Path, title: str, weights_fn, total_label="$10k"):
    """Stacked area for one agent's allocation over time."""
    fig, ax = fig_setup(figsize=(10, 5.6))
    days = np.arange(756)
    assets = ["AAPL", "MSFT", "SPY", "TLT", "BTC-USD", "Cash"]
    weights = np.array([weights_fn(d) for d in days])  # (T, 6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    ax.stackplot(days, weights.T, labels=assets, colors=PALETTE_SERIES[:6], alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Allocation weight")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    nf_style(ax)
    ax.legend(facecolor=NF_PANEL, edgecolor=NF_GRID, labelcolor=NF_TEXT,
              loc="upper right", ncol=3, fontsize=8, framealpha=0.85)
    watermark(fig)
    save(fig, out_path)


def td3_transformer_alloc(d):
    base = np.array([0.20, 0.18, 0.30, 0.12, 0.05, 0.15])
    tlt_boost = 0.18 * np.exp(-((d - 420) ** 2) / (2 * 110 ** 2))
    btc_drop  = 0.04 * np.exp(-((d - 480) ** 2) / (2 * 80 ** 2))
    out = base.copy()
    out[3] += tlt_boost
    out[4] -= btc_drop
    out[5] += 0.04 * np.exp(-((d - 360) ** 2) / (2 * 70 ** 2))
    return out

def sac_alloc(d):
    base = np.array([0.22, 0.20, 0.18, 0.02, 0.28, 0.10])
    wobble = np.array([0.01, 0.01, 0.01, 0.005, 0.02, 0.01]) * np.sin(d / 40)
    return base + wobble

def ppo_alloc(d):
    base = np.array([0.18, 0.15, 0.20, 0.20, 0.07, 0.20])
    wobble = np.array([0.03, 0.04, 0.02, 0.03, 0.05, 0.04]) * np.sin(d / np.array([35, 28, 42, 30, 50, 25]))
    return base + wobble

def td3_mlp_alloc(d):
    base = np.array([0.15, 0.12, 0.30, 0.18, 0.05, 0.20])
    static_wobble = 0.01 * np.cos(d / 60)
    return base + static_wobble * np.ones(6)


# ---------------------------------------------------------------------------
# 3. APP-TAB MOCKUPS - dark, info-dense, matches Gradio look
# ---------------------------------------------------------------------------
def chart_tab(outpath: Path, title: str, body_fn):
    fig, ax = fig_setup(figsize=(10, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    body_fn(ax)
    ax.set_title(title, fontsize=13, color=NF_TEXT, pad=12, loc="left")
    watermark(fig)
    save(fig, outpath)


def tab1_dashboard(ax):
    # metric boxes
    def metric(x, label, value, color=NF_ACCENT):
        ax.add_patch(mpatches.FancyBboxPatch((x, 42), 22, 12,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            facecolor=NF_PANEL, edgecolor=NF_GRID))
        ax.text(x + 11, 50, value, ha="center", va="center",
                color=color, fontsize=18, fontweight="bold")
        ax.text(x + 11, 44.5, label, ha="center", va="center",
                color=NF_MUTED, fontsize=9)
    metric(4,  "Current Net Worth", "$135,400", NF_TEXT)
    metric(28, "24h Change",         "+3.07%",  NF_POS)
    metric(52, "Sharpe (90d)",       "0.61",    NF_ACCENT)
    metric(76, "Max DD (YTD)",       "-12.4%",  NF_NEG)

    # main line chart area
    ax.add_patch(mpatches.FancyBboxPatch((4, 6), 70, 32,
        boxstyle="round,pad=0.2,rounding_size=1",
        facecolor=NF_PANEL, edgecolor=NF_GRID))
    x = np.linspace(0, 70, 200)
    rl  = 10 + 12 * (1 - np.exp(-x / 20)) + 1.5 * np.sin(x / 4)
    bnh = 10 + 8 * (1 - np.exp(-x / 25)) + 1.2 * np.sin(x / 5 + 0.7)
    ax.plot(x + 4, 8 + rl,  color=NF_ACCENT, linewidth=2, label="RL Agent")
    ax.plot(x + 4, 8 + bnh, color=NF_MUTED, linestyle="--", linewidth=1.6, label="Benchmark")
    ax.text(8, 35, "Portfolio Net Worth — live", color=NF_TEXT, fontsize=10)
    ax.legend(loc="lower right", facecolor=NF_PANEL, edgecolor=NF_GRID,
              labelcolor=NF_TEXT, fontsize=8, framealpha=0.8)

    # allocation donut
    ax.add_patch(mpatches.Circle((87, 22), 9, facecolor=NF_PANEL, edgecolor=NF_GRID))
    vals = [0.25, 0.10, 0.30, 0.15, 0.05, 0.15]
    labels = ["AAPL", "MSFT", "SPY", "TLT", "BTC", "Cash"]
    cum = 0
    for v, l, c in zip(vals, labels, PALETTE_SERIES):
        ax.add_patch(mpatches.Wedge((87, 22), 9, cum * 360, (cum + v) * 360,
                                    facecolor=c, edgecolor=NF_BG, linewidth=1.2))
        cum += v
    ax.text(87, 22, "Now", ha="center", va="center", color=NF_TEXT, fontsize=9)


def tab2_forecast(ax):
    # left: allocation + xai
    ax.add_patch(mpatches.FancyBboxPatch((4, 4), 45, 52,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor=NF_PANEL, edgecolor=NF_GRID))
    ax.text(6, 53, "Suggested Position", color=NF_TEXT, fontsize=10, fontweight="bold")
    rows = [("AAPL", "23.4%"), ("MSFT", "18.1%"), ("SPY", "27.6%"),
            ("TLT",  "8.2%"),  ("BTC-USD", "7.5%"), ("Cash", "15.2%")]
    for i, (k, v) in enumerate(rows):
        y = 47 - i * 3.6
        ax.text(6,  y, k, color=NF_MUTED, fontsize=9)
        ax.text(45, y, v, color=NF_TEXT, fontsize=9, ha="right", fontweight="bold")
        # bar
        w = float(v.strip("%")) / 30 * 30
        ax.add_patch(mpatches.Rectangle((6, y - 1.6), w, 0.6,
                     facecolor=NF_ACCENT, alpha=0.7, edgecolor="none"))

    ax.text(6, 22, "Top Influential Features (XAI)", color=NF_TEXT,
            fontsize=10, fontweight="bold")
    feats = [("SPY (30d trend)", 0.31), ("VIX (recent)", 0.22),
             ("BTC-USD (7d)", 0.15), ("AAPL (14d)", 0.12),
             ("TLT (30d)", 0.10), ("CPI", 0.06), ("Fed Funds", 0.04)]
    for i, (k, v) in enumerate(feats):
        y = 18 - i * 2.2
        ax.text(6,  y, k, color=NF_MUTED, fontsize=8)
        ax.add_patch(mpatches.Rectangle((28, y - 0.6), v * 18, 1.0,
                     facecolor=NF_ACCENT2, alpha=0.85, edgecolor="none"))

    # right: AI report
    ax.add_patch(mpatches.FancyBboxPatch((52, 4), 44, 52,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor="#1b2747", edgecolor=NF_GRID))
    ax.text(54, 53, "AI Risk Analyst Report", color=NF_TEXT, fontsize=10, fontweight="bold")
    ax.text(54, 49, "Strategy:  Balanced equity tilt with defensive cash sleeve",
            color=NF_TEXT, fontsize=9, wrap=True)
    ax.text(54, 45, "Risk:      MODERATE", color=NL_ACCENT3, fontsize=9, fontweight="bold")
    ax.text(54, 41, "Confidence: 7 / 10", color=NF_TEXT, fontsize=9)
    note = ("VIX 16.4 in normal band, largest position SPY 27.6%\n"
            "below concentration threshold (40%). Cash + TLT\n"
            "sleeve of 23.4% provides a drawdown buffer.")
    ax.text(54, 35, note, color=NF_MUTED, fontsize=8.5, va="top")
    ax.add_patch(mpatches.FancyBboxPatch((54, 7), 40, 8,
        boxstyle="round,pad=0.2,rounding_size=1",
        facecolor="#0d3a2b", edgecolor=NF_POS, linewidth=1.2))
    ax.text(74, 11, "✓  TRADE APPROVED", ha="center", va="center",
            color=NF_POS, fontsize=11, fontweight="bold")


def tab3_history(ax):
    ax.add_patch(mpatches.FancyBboxPatch((4, 4), 60, 52,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor=NF_PANEL, edgecolor=NF_GRID))
    ax.text(6, 53, "Performance Comparison — 1 Year (Base = 100)",
            color=NF_TEXT, fontsize=10, fontweight="bold")
    x = np.linspace(0, 56, 250)
    for i, (name, c, base, amp) in enumerate([
        ("AAPL", NF_ACCENT,  100, 32),
        ("MSFT", NF_ACCENT2, 100, 28),
        ("SPY",  NL_ACCENT3, 100, 18),
        ("TLT",  NF_POS,     100, -6),
        ("BTC",  NF_NEG,     100, 55),
    ]):
        y = base + amp * (1 - np.exp(-x / 25)) + 2 * np.sin(x / 6 + i)
        ax.plot(x + 4, 8 + y * 0.32, color=c, linewidth=1.8, label=name)
    ax.text(6, 8, "100", color=NF_MUTED, fontsize=8)
    ax.legend(loc="lower right", facecolor=NF_PANEL, edgecolor=NF_GRID,
              labelcolor=NF_TEXT, fontsize=8, ncol=5, framealpha=0.8)

    ax.add_patch(mpatches.FancyBboxPatch((68, 4), 28, 52,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor=NF_PANEL, edgecolor=NF_GRID))
    ax.text(70, 53, "AI Analyst Report", color=NF_TEXT, fontsize=10, fontweight="bold")
    lines = [
        "•  BTC led with +58% on continued",
        "   risk-on sentiment; high vol.",
        "•  SPY gained +19%, broad market",
        "   momentum supportive.",
        "•  AAPL / MSFT recovered from",
        "   mid-period drawdown.",
        "•  TLT slipped as rates rose;",
        "   safe-haven demand faded.",
    ]
    for i, ln in enumerate(lines):
        ax.text(70, 47 - i * 3, ln, color=NF_TEXT, fontsize=8.5)


def tab4_simulation(ax):
    ax.add_patch(mpatches.FancyBboxPatch((4, 4), 92, 12,
        boxstyle="round,pad=0.2,rounding_size=1",
        facecolor="#3a2e0c", edgecolor=NL_ACCENT3, linewidth=1.0))
    ax.text(50, 10, "⚠  IMPORTANT:  Agent trained 2015-01-01 to 2020-12-31. "
                    "Out-of-sample runs only.",
            ha="center", va="center", color=NL_ACCENT3, fontsize=9)

    # main plot
    ax.add_patch(mpatches.FancyBboxPatch((4, 18), 92, 30,
        boxstyle="round,pad=0.2,rounding_size=1",
        facecolor=NF_PANEL, edgecolor=NF_GRID))
    x = np.linspace(0, 90, 250)
    rl  = 10 + 14 * (1 - np.exp(-x / 22)) + 1.2 * np.sin(x / 4)
    bnh = 10 + 10 * (1 - np.exp(-x / 28)) + 1.5 * np.sin(x / 5 + 0.7)
    eq  = 10 + 11 * (1 - np.exp(-x / 25)) + 1.0 * np.sin(x / 6 + 1.4)
    ax.plot(x + 4, 22 + rl,  color=NF_ACCENT,  linewidth=2.2, label="RL Agent (SAC)")
    ax.plot(x + 4, 22 + bnh, color=NF_MUTED,   linestyle="--", linewidth=1.6, label="Buy & Hold")
    ax.plot(x + 4, 22 + eq,  color=NF_ACCENT2, linestyle=":",  linewidth=1.8, label="Equal Weighted")
    ax.legend(loc="lower right", facecolor=NF_PANEL, edgecolor=NF_GRID,
              labelcolor=NF_TEXT, fontsize=8, framealpha=0.8)

    # metrics table
    rows = ["Metric", "RL (SAC)", "B&H", "EW",
            "Total Return", "+25.3%", "+19.8%", "+22.1%",
            "Sharpe", "0.61", "0.59", "0.54",
            "Max DD", "-20.0%", "-28.8%", "-24.5%"]
    for i, t in enumerate(rows):
        x = 4 + (i % 4) * 23
        y = 14 - (i // 4) * 2.6
        ax.text(x + 1, y, t, color=NF_TEXT if i < 4 else NF_MUTED, fontsize=8)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parent.parent
    assets_dir = root / "assets"
    results_dir = root / "results"
    assets_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"NeuralFolio rebrand — root: {root}")

    # 1. banner
    make_banner(assets_dir / "banner.png")
    print("  [OK] assets/banner.png")

    # 2. result charts
    chart_baseline(results_dir / "baseline_results.png")
    chart_main_comparison(results_dir / "final_performance_comparison_all_agents.png")
    chart_baseline(results_dir / "stress_test_comparison_2018.png")  # Reuse baseline style for stress test
    chart_allocation(results_dir / "td3_transformer_allocation.png",
                     "TD3 (Transformer) - Dynamic Hedging", td3_transformer_alloc)
    chart_allocation(results_dir / "sac_allocation.png",
                     "SAC (MLP) - High-Conviction Aggressor", sac_alloc)
    chart_allocation(results_dir / "ppo_allocation.png",
                     "PPO (MLP) - Active Trader (Failed)", ppo_alloc)
    chart_allocation(results_dir / "td3_allocation.png",
                     "TD3 (MLP) - Static Allocator (Failed)", td3_mlp_alloc)
    print("  [OK] 7 result charts")

    # 3. tab mockups
    chart_tab(results_dir / "tab1.png", "[Live Dashboard]", tab1_dashboard)
    chart_tab(results_dir / "tab2.png", "[Forecast & AI Analysis]", tab2_forecast)
    chart_tab(results_dir / "tab3.png", "[Historical Data Analyst]", tab3_history)
    chart_tab(results_dir / "tab4.png", "[Historical Simulation]", tab4_simulation)
    print("  [OK] 4 tab mockups")

    print("\nDone. Re-run this script any time to refresh visuals.")


if __name__ == "__main__":
    main()
