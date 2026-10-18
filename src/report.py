"""
report.py
---------
Generates matplotlib charts and prints a clean terminal summary of
cointegration + backtest results.

Charts saved to data/output/charts/:
    spread_zscore_{pair}.png     — top 3 pairs by Sharpe
    pnl_curve_{pair}.png         — top 3 pairs cumulative P&L
    portfolio_pnl.png            — equal-weight portfolio
    cointegration_heatmap.png    — p-value heatmap for all tested pairs
"""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_YF     = BASE_DIR / "data" / "raw" / "yfinance"
PROC_DIR   = BASE_DIR / "data" / "processed"
OUT_DIR    = BASE_DIR / "data" / "output"
CHART_DIR  = OUT_DIR / "charts"

SECTORS = {
    "tech":       ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "financials": ["JPM",  "BAC",  "GS",   "MS"],
    "consumer":   ["KO",   "PEP",  "MCD",  "YUM",  "COST", "WMT"],
    "energy":     ["XOM",  "CVX",  "COP",  "SLB"],
    "healthcare": ["JNJ",  "PFE",  "MRK",  "ABBV"],
    "payments":   ["V",    "MA",   "PYPL"],
    "retail":     ["HD",   "LOW",  "NKE"],
}

EXPLICIT_PAIRS = [("KO","PEP"),("JPM","BAC"),("XOM","CVX"),("JNJ","PFE"),("V","MA"),("HD","LOW")]


def _load_closes() -> pd.DataFrame:
    series = {}
    for p in sorted(RAW_YF.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["close"])
        series[p.stem] = df["close"]
    return pd.DataFrame(series).dropna(how="all")


def plot_spread_zscore(sig_df: pd.DataFrame, pair: str, out_dir: Path) -> None:
    """Spread z-score with entry/exit bands and position shading."""
    data = sig_df[sig_df["pair"] == pair].sort_values("date").copy()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(data["date"], data["zscore"], color="#1f77b4", linewidth=0.8, label="Z-Score")
    ax.axhline(2,  color="#d62728", linestyle="--", linewidth=0.7, label="Entry ±2")
    ax.axhline(-2, color="#d62728", linestyle="--", linewidth=0.7)
    ax.axhline(3,  color="#ff7f0e", linestyle=":",  linewidth=0.7, label="Stop ±3")
    ax.axhline(-3, color="#ff7f0e", linestyle=":",  linewidth=0.7)
    ax.axhline(0,  color="grey",   linestyle="-",  linewidth=0.4)

    # Shade positions
    long_mask  = data["position"] == 1
    short_mask = data["position"] == -1
    ax.fill_between(data["date"], data["zscore"].min()-0.5, data["zscore"].max()+0.5,
                    where=long_mask, alpha=0.12, color="green", label="Long spread")
    ax.fill_between(data["date"], data["zscore"].min()-0.5, data["zscore"].max()+0.5,
                    where=short_mask, alpha=0.12, color="red", label="Short spread")

    ax.set_title(f"Spread Z-Score — {pair}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-Score")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    safe_name = pair.replace("/", "_")
    fig.savefig(out_dir / f"spread_zscore_{safe_name}.png", dpi=150)
    plt.close(fig)
    log.info("  Chart ✓ spread_zscore_%s.png", safe_name)


def plot_pnl_curve(sig_df: pd.DataFrame, pair: str, out_dir: Path) -> None:
    """Cumulative P&L curve for one pair."""
    data = sig_df[sig_df["pair"] == pair].sort_values("date").copy()
    data["spread_return"] = data["spread"].diff()
    data["pnl"] = data["position"].shift(1) * data["spread_return"]
    data["pnl"] = data["pnl"].fillna(0)
    data["cum_pnl"] = data["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data["date"], data["cum_pnl"], color="#2ca02c", linewidth=1)
    ax.fill_between(data["date"], 0, data["cum_pnl"],
                    where=data["cum_pnl"] >= 0, alpha=0.15, color="green")
    ax.fill_between(data["date"], 0, data["cum_pnl"],
                    where=data["cum_pnl"] < 0, alpha=0.15, color="red")
    ax.axhline(0, color="grey", linewidth=0.4)
    ax.set_title(f"Cumulative P&L — {pair}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (spread units)")
    fig.tight_layout()

    safe_name = pair.replace("/", "_")
    fig.savefig(out_dir / f"pnl_curve_{safe_name}.png", dpi=150)
    plt.close(fig)
    log.info("  Chart ✓ pnl_curve_%s.png", safe_name)


def plot_portfolio_pnl(sig_df: pd.DataFrame, out_dir: Path) -> None:
    """Equal-weight portfolio cumulative P&L."""
    pairs = sig_df["pair"].unique()
    pair_pnls = {}
    for pair in pairs:
        data = sig_df[sig_df["pair"] == pair].sort_values("date").copy()
        data["spread_return"] = data["spread"].diff()
        data["pnl"] = data["position"].shift(1) * data["spread_return"]
        data["pnl"] = data["pnl"].fillna(0)
        # Normalize by median absolute spread (matches backtest.py logic)
        median_abs = data["spread"].abs().median()
        if median_abs == 0:
            median_abs = 1.0
        data["daily_return"] = data["pnl"] / median_abs
        pair_pnls[pair] = data.set_index("date")["daily_return"]

    port_df = pd.DataFrame(pair_pnls)
    port_avg = port_df.mean(axis=1)
    cum = (1 + port_avg).cumprod() - 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum.index, cum.values * 100, color="#9467bd", linewidth=1.2)
    ax.fill_between(cum.index, 0, cum.values * 100,
                    where=cum.values >= 0, alpha=0.12, color="purple")
    ax.fill_between(cum.index, 0, cum.values * 100,
                    where=cum.values < 0, alpha=0.12, color="red")
    ax.axhline(0, color="grey", linewidth=0.4)
    ax.set_title("Portfolio Cumulative Return (Equal Weight)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "portfolio_pnl.png", dpi=150)
    plt.close(fig)
    log.info("  Chart ✓ portfolio_pnl.png")


def plot_cointegration_heatmap(out_dir: Path) -> None:
    """Heatmap of p-values for all tested within-sector + explicit pairs."""
    closes = _load_closes()

    candidate_pairs = set()
    for tickers in SECTORS.values():
        for a, b in combinations(tickers, 2):
            candidate_pairs.add((min(a,b), max(a,b)))
    for a, b in EXPLICIT_PAIRS:
        candidate_pairs.add((min(a,b), max(a,b)))

    all_tickers = sorted(set(t for pair in candidate_pairs for t in pair))
    n = len(all_tickers)
    pval_matrix = np.full((n, n), np.nan)
    idx_map = {t: i for i, t in enumerate(all_tickers)}

    for a, b in candidate_pairs:
        if a not in closes.columns or b not in closes.columns:
            continue
        aligned = closes[[a, b]].dropna()
        if len(aligned) < 60:
            continue
        _, p, _ = coint(aligned[a], aligned[b])
        i, j = idx_map[a], idx_map[b]
        pval_matrix[i][j] = p
        pval_matrix[j][i] = p

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.isnan(pval_matrix)
    sns.heatmap(
        pval_matrix,
        mask=mask,
        xticklabels=all_tickers,
        yticklabels=all_tickers,
        cmap="RdYlGn_r",
        vmin=0, vmax=0.1,
        annot=True, fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "p-value"},
        ax=ax,
    )
    ax.set_title("Cointegration P-Values (Engle-Granger)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "cointegration_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("  Chart ✓ cointegration_heatmap.png")


def print_terminal_report() -> None:
    """Print formatted summary tables to the terminal."""
    coint_path = PROC_DIR / "cointegration_results.parquet"
    bt_path    = OUT_DIR  / "backtest_results.csv"

    if coint_path.exists():
        coint_df = pd.read_parquet(coint_path)
        print("\n" + "=" * 72)
        print("  COINTEGRATION RESULTS")
        print("=" * 72)
        print(tabulate(
            coint_df[["pair", "p_raw", "hedge_ratio", "half_life"]].values.tolist(),
            headers=["Pair", "p-value", "Hedge Ratio", "Half-Life (days)"],
            tablefmt="simple_outline",
            floatfmt=(".6f", ".4f", ".1f"),
        ))

    if bt_path.exists():
        bt_df = pd.read_csv(bt_path)
        print("\n" + "=" * 72)
        print("  BACKTEST RESULTS")
        print("=" * 72)
        print(tabulate(
            bt_df[["pair", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate"]].values.tolist(),
            headers=["Pair", "Total Return %", "Sharpe", "Max DD %", "Trades", "Win Rate %"],
            tablefmt="simple_outline",
            floatfmt=(".2f", ".3f", ".2f", ".0f", ".1f"),
        ))


def generate_all_charts() -> None:
    """Generate all charts from processed data."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    sig_df = pd.read_parquet(PROC_DIR / "signals.parquet")
    bt_df  = pd.read_csv(OUT_DIR / "backtest_results.csv")

    # Top 3 pairs by Sharpe (exclude portfolio row)
    pair_metrics = bt_df[bt_df["pair"] != "PORTFOLIO (EW)"].copy()
    top3 = pair_metrics.nlargest(3, "sharpe")["pair"].tolist()
    log.info("Top 3 pairs by Sharpe: %s", top3)

    for pair in top3:
        plot_spread_zscore(sig_df, pair, CHART_DIR)
        plot_pnl_curve(sig_df, pair, CHART_DIR)

    plot_portfolio_pnl(sig_df, CHART_DIR)
    plot_cointegration_heatmap(CHART_DIR)


def main():
    generate_all_charts()
    print_terminal_report()


if __name__ == "__main__":
    main()
