"""
backtest.py
-----------
Simulates trading each cointegrated pair based on z-score signals.
Equal notional per trade, no transaction costs (noted in README).

Metrics per pair: total return, annualised Sharpe, max drawdown, trade count, win rate.
Portfolio-level: equal-weight across all pairs.

Output:
    data/output/backtest_results.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "data" / "output"


def _annualised_sharpe(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def _max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd   = (cum_returns - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def backtest_pair(pair_df: pd.DataFrame) -> dict:
    """
    Backtest a single pair given its signals DataFrame.
    Returns a metrics dict.
    """
    pair_df = pair_df.sort_values("date").copy()
    pair_name = pair_df["pair"].iloc[0]

    # Daily spread returns = diff(spread) * position_yesterday
    pair_df["spread_return"] = pair_df["spread"].diff()
    pair_df["pnl"] = pair_df["position"].shift(1) * pair_df["spread_return"]
    pair_df["pnl"] = pair_df["pnl"].fillna(0)

    # Normalise P&L by initial spread notional for return calculation
    initial_notional = pair_df["spread"].abs().iloc[0]
    if initial_notional == 0:
        initial_notional = 1.0
    pair_df["daily_return"] = pair_df["pnl"] / initial_notional
    pair_df["cum_return"]   = (1 + pair_df["daily_return"]).cumprod()

    total_return = float(pair_df["cum_return"].iloc[-1] - 1)
    sharpe       = _annualised_sharpe(pair_df["daily_return"])
    max_dd       = _max_drawdown(pair_df["cum_return"])

    # Trade counting: a trade starts when position goes from 0 to nonzero
    pos = pair_df["position"]
    trade_starts = ((pos != 0) & (pos.shift(1).fillna(0) == 0))
    trade_ends   = ((pos == 0) & (pos.shift(1).fillna(0) != 0))
    n_trades     = int(trade_starts.sum())

    # Win rate: a trade is a "win" if cumulative P&L during the trade is positive
    wins = 0
    in_trade = False
    trade_pnl = 0.0
    for _, r in pair_df.iterrows():
        if not in_trade and r["position"] != 0:
            in_trade = True
            trade_pnl = 0.0
        if in_trade:
            trade_pnl += r["pnl"]
            if r["position"] == 0:
                if trade_pnl > 0:
                    wins += 1
                in_trade = False

    win_rate = wins / n_trades if n_trades > 0 else 0.0

    return {
        "pair":         pair_name,
        "total_return":  round(total_return * 100, 2),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd * 100, 2),
        "n_trades":      n_trades,
        "win_rate":      round(win_rate * 100, 1),
        "cum_return_series": pair_df[["date", "cum_return"]].copy(),
    }


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest on all pairs with signals.
    Returns (metrics_df, portfolio_cum_return Series).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sig_df = pd.read_parquet(PROC_DIR / "signals.parquet")
    pairs  = sig_df["pair"].unique()
    log.info("Backtesting %d pairs …", len(pairs))

    results    = []
    cum_series = {}

    for pair in pairs:
        pair_data = sig_df[sig_df["pair"] == pair]
        metrics   = backtest_pair(pair_data)
        cum_series[pair] = metrics.pop("cum_return_series").set_index("date")["cum_return"]
        results.append(metrics)
        log.info("  %s  return=%.1f%%  sharpe=%.2f  dd=%.1f%%  trades=%d  win=%.0f%%",
                 pair, metrics["total_return"], metrics["sharpe"],
                 metrics["max_drawdown"], metrics["n_trades"], metrics["win_rate"])

    metrics_df = pd.DataFrame(results)

    # Portfolio-level: equal-weight across pairs
    cum_df = pd.DataFrame(cum_series)
    portfolio_cum = cum_df.mean(axis=1)
    portfolio_daily = portfolio_cum.pct_change().fillna(0)

    port_row = {
        "pair":         "PORTFOLIO (EW)",
        "total_return":  round((portfolio_cum.iloc[-1] - 1) * 100, 2),
        "sharpe":        round(_annualised_sharpe(portfolio_daily), 3),
        "max_drawdown":  round(_max_drawdown(portfolio_cum) * 100, 2),
        "n_trades":      int(metrics_df["n_trades"].sum()),
        "win_rate":      round(metrics_df["win_rate"].mean(), 1),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([port_row])], ignore_index=True)

    out_path = OUT_DIR / "backtest_results.csv"
    metrics_df.to_csv(out_path, index=False)
    log.info("Saved backtest results → %s", out_path)

    return metrics_df, portfolio_cum


def main():
    metrics_df, _ = run_backtest()
    print("\n=== Backtest Results ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
