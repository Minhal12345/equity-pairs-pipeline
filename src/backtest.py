"""
backtest.py
-----------
Simulates trading each cointegrated pair based on z-score signals.
Equal notional per trade, no transaction costs (noted in README).

Metrics per pair: total return, annualised Sharpe, max drawdown, trade count, win rate.
Portfolio-level: equal-weight daily returns across all pairs.

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

    # Normalise P&L by median absolute spread (stable scale).
    # Using initial spread value is fragile because mean-reverting spreads
    # can start near zero, inflating all subsequent returns.
    median_abs_spread = pair_df["spread"].abs().median()
    if median_abs_spread == 0:
        median_abs_spread = 1.0
    pair_df["daily_return"] = pair_df["pnl"] / median_abs_spread
    pair_df["cum_return"]   = (1 + pair_df["daily_return"]).cumprod()

    total_return = float(pair_df["cum_return"].iloc[-1] - 1)
    sharpe       = _annualised_sharpe(pair_df["daily_return"])
    max_dd       = _max_drawdown(pair_df["cum_return"])

    # Trade counting using cumulative-sum trade IDs.
    # A trade starts when position goes from 0 to nonzero.
    pos = pair_df["position"]
    trade_starts = ((pos != 0) & (pos.shift(1).fillna(0) == 0))
    trade_ids = trade_starts.cumsum()
    trade_ids[pos == 0] = 0  # non-trade bars get id 0

    n_trades = int(trade_starts.sum())

    # Win rate via groupby on trade IDs.
    # Open trades at the end of the series are force-closed at the final bar:
    # their realized P&L up to that point is counted.
    if n_trades > 0:
        trade_pnls = pair_df.loc[trade_ids > 0].groupby(trade_ids[trade_ids > 0])["pnl"].sum()
        wins = int((trade_pnls > 0).sum())
        win_rate = wins / n_trades
    else:
        wins = 0
        win_rate = 0.0

    return {
        "pair":         pair_name,
        "total_return":  round(total_return * 100, 2),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd * 100, 2),
        "n_trades":      n_trades,
        "win_rate":      round(win_rate * 100, 1),
        "daily_return_series": pair_df[["date", "daily_return"]].copy(),
        "cum_return_series":   pair_df[["date", "cum_return"]].copy(),
    }


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest on all pairs with signals.
    Returns (metrics_df, portfolio_cum_return Series).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sig_df = pd.read_parquet(PROC_DIR / "signals.parquet")
    log.info("Backtesting %d pairs …", sig_df["pair"].nunique())

    results       = []
    daily_returns = {}

    for pair, pair_data in sig_df.groupby("pair"):
        metrics = backtest_pair(pair_data)
        daily_ret = metrics.pop("daily_return_series").set_index("date")["daily_return"]
        daily_returns[pair] = daily_ret
        metrics.pop("cum_return_series")
        results.append(metrics)
        log.info("  %s  return=%.1f%%  sharpe=%.2f  dd=%.1f%%  trades=%d  win=%.0f%%",
                 pair, metrics["total_return"], metrics["sharpe"],
                 metrics["max_drawdown"], metrics["n_trades"], metrics["win_rate"])

    metrics_df = pd.DataFrame(results)

    # Portfolio-level: equal-weight daily returns across all pairs.
    # Align on the full date index; compute nanmean across pairs each day,
    # then compound. This avoids the bug where averaging cumulative returns
    # and taking pct_change creates spurious returns from composition changes.
    ret_df = pd.DataFrame(daily_returns)
    portfolio_daily = ret_df.mean(axis=1)
    portfolio_cum = (1 + portfolio_daily).cumprod()

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
