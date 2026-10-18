"""
oos_validation.py
-----------------
Out-of-sample validation for the cointegration pairs strategy.

Splits the 3-year data window (Oct 2021 – Oct 2024) into:
    Training: Oct 2021 – Oct 2023  (2 years)
    Test:     Oct 2023 – Oct 2024  (1 year)

Procedure:
    1. Run Engle-Granger cointegration on training period only
    2. Apply BH-FDR correction
    3. For raw-significant pairs (since no FDR survivors expected),
       use training-period hedge ratios to compute test-period spreads
    4. Generate z-score signals on test period
    5. Backtest test-period signals
    6. Compare in-sample vs out-of-sample metrics

Output:
    data/output/oos_validation.csv
    data/output/oos_summary.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_YF   = BASE_DIR / "data" / "raw" / "yfinance"
OUT_DIR  = BASE_DIR / "data" / "output"

# Split date — training ends here, test begins the next trading day
SPLIT_DATE = pd.Timestamp("2023-10-01")

# Signal parameters (same as signals.py)
Z_ENTRY    = 2.0
Z_EXIT     = 0.0
Z_STOPLOSS = 3.0
LOOKBACK   = 60

# Sector groupings (same as cointegration.py)
SECTORS: dict[str, list[str]] = {
    "tech":       ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "financials": ["JPM",  "BAC",  "GS",   "MS"],
    "consumer":   ["KO",   "PEP",  "MCD",  "YUM",  "COST", "WMT"],
    "energy":     ["XOM",  "CVX",  "COP",  "SLB"],
    "healthcare": ["JNJ",  "PFE",  "MRK",  "ABBV"],
    "payments":   ["V",    "MA",   "PYPL"],
    "retail":     ["HD",   "LOW",  "NKE"],
}

EXPLICIT_PAIRS: list[tuple[str, str]] = [
    ("KO",  "PEP"),
    ("JPM", "BAC"),
    ("XOM", "CVX"),
    ("JNJ", "PFE"),
    ("V",   "MA"),
    ("HD",  "LOW"),
]


def load_closes() -> pd.DataFrame:
    """Load all per-ticker close-price series and align on shared dates."""
    series = {}
    for path in sorted(RAW_YF.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["close"])
        series[path.stem] = df["close"]
    return pd.DataFrame(series).dropna(how="all")


def compute_half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life via AR(1) on spread differences."""
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    idx = lag.index.intersection(delta.index)
    lag, delta = lag.loc[idx], delta.loc[idx]
    if len(lag) < 10:
        return np.inf
    model = OLS(delta.values, add_constant(lag.values)).fit()
    phi = 1 + model.params[1]
    if phi >= 1 or phi <= 0:
        return np.inf
    return -np.log(2) / np.log(phi)


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS regression: price_a ~ beta * price_b; return beta."""
    model = OLS(price_a.values, add_constant(price_b.values)).fit()
    return float(model.params[1])


def run_cointegration_on_period(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Run Engle-Granger cointegration on the given price matrix.
    Returns DataFrame with raw p-values and FDR correction.
    """
    from itertools import combinations

    candidate_pairs: set[tuple[str, str]] = set()
    for tickers in SECTORS.values():
        for a, b in combinations(tickers, 2):
            candidate_pairs.add((a, b))
    for a, b in EXPLICIT_PAIRS:
        candidate_pairs.add((a, b))

    results = []
    for a, b in sorted(candidate_pairs):
        if a not in closes.columns or b not in closes.columns:
            continue
        series = closes[[a, b]].dropna()
        if len(series) < 60:
            continue

        price_a = series[a]
        price_b = series[b]
        _, p_val, _ = coint(price_a, price_b)
        hedge = compute_hedge_ratio(price_a, price_b)
        spread = price_a - hedge * price_b
        hl = compute_half_life(spread)

        results.append({
            "pair": f"{a}/{b}",
            "ticker_a": a,
            "ticker_b": b,
            "p_raw": round(p_val, 6),
            "hedge_ratio": round(hedge, 6),
            "half_life": round(hl, 2) if np.isfinite(hl) else None,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    reject, p_fdr, _, _ = multipletests(df["p_raw"], alpha=0.05, method="fdr_bh")
    df["p_fdr"] = np.round(p_fdr, 6)
    df["significant_at_fdr_0.05"] = reject
    return df.sort_values("p_raw").reset_index(drop=True)


def generate_signals_for_pair(
    closes: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
) -> pd.DataFrame:
    """Generate z-score signals for a single pair using a fixed hedge ratio."""
    if ticker_a not in closes.columns or ticker_b not in closes.columns:
        return pd.DataFrame()

    aligned = closes[[ticker_a, ticker_b]].dropna()
    if len(aligned) < LOOKBACK + 1:
        return pd.DataFrame()

    spread = aligned[ticker_a] - hedge_ratio * aligned[ticker_b]
    mean = spread.rolling(LOOKBACK).mean()
    std = spread.rolling(LOOKBACK).std()
    zscore = (spread - mean) / std

    signal = pd.Series(0, index=aligned.index, dtype=int)
    signal[zscore < -Z_ENTRY] = 1
    signal[zscore > Z_ENTRY] = -1
    signal[zscore < -Z_STOPLOSS] = 0
    signal[zscore > Z_STOPLOSS] = 0

    pos = 0
    positions = []
    for z, s in zip(zscore, signal):
        if np.isnan(z):
            positions.append(0)
            continue
        if pos == 0:
            pos = s
        else:
            if abs(z) >= Z_STOPLOSS:
                pos = 0
            elif (pos == 1 and z >= Z_EXIT) or (pos == -1 and z <= Z_EXIT):
                pos = 0
        positions.append(pos)

    return pd.DataFrame({
        "date": aligned.index,
        "pair": f"{ticker_a}/{ticker_b}",
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "spread": spread.values,
        "zscore": zscore.values,
        "signal": signal.values,
        "position": positions,
    })


def backtest_signals(sig_df: pd.DataFrame) -> dict:
    """Backtest a single pair's signals DataFrame. Returns metrics dict."""
    sig_df = sig_df.sort_values("date").copy()
    pair_name = sig_df["pair"].iloc[0]

    sig_df["spread_return"] = sig_df["spread"].diff()
    sig_df["pnl"] = sig_df["position"].shift(1) * sig_df["spread_return"]
    sig_df["pnl"] = sig_df["pnl"].fillna(0)

    initial_notional = sig_df["spread"].abs().iloc[0]
    if initial_notional == 0:
        initial_notional = 1.0
    sig_df["daily_return"] = sig_df["pnl"] / initial_notional
    sig_df["cum_return"] = (1 + sig_df["daily_return"]).cumprod()

    total_return = float(sig_df["cum_return"].iloc[-1] - 1)

    dr = sig_df["daily_return"]
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    peak = sig_df["cum_return"].cummax()
    dd = (sig_df["cum_return"] - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0

    pos = sig_df["position"]
    trade_starts = ((pos != 0) & (pos.shift(1).fillna(0) == 0))
    n_trades = int(trade_starts.sum())

    wins = 0
    in_trade = False
    trade_pnl = 0.0
    for _, r in sig_df.iterrows():
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
        "pair": pair_name,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
    }


def run_oos_validation() -> dict:
    """
    Full out-of-sample validation pipeline.
    Returns summary dict with all results.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    closes = load_closes()

    # Split into training and test
    train_closes = closes[closes.index < SPLIT_DATE]
    test_closes = closes[closes.index >= SPLIT_DATE]

    log.info("Training period: %s to %s (%d trading days)",
             train_closes.index.min().date(), train_closes.index.max().date(),
             len(train_closes))
    log.info("Test period:     %s to %s (%d trading days)",
             test_closes.index.min().date(), test_closes.index.max().date(),
             len(test_closes))

    # Step 1: Cointegration on training period
    log.info("Running cointegration on training period...")
    train_coint = run_cointegration_on_period(train_closes)
    n_total = len(train_coint)
    n_raw_train = int((train_coint["p_raw"] < 0.05).sum())
    n_fdr_train = int(train_coint["significant_at_fdr_0.05"].sum())

    log.info("Training: %d pairs tested, %d raw significant, %d FDR significant",
             n_total, n_raw_train, n_fdr_train)

    # Step 2: Re-test raw-significant pairs on test period
    raw_sig_train = train_coint[train_coint["p_raw"] < 0.05].copy()

    if raw_sig_train.empty:
        log.warning("No raw-significant pairs in training period. Validation cannot proceed.")
        return {"error": "No raw-significant pairs in training period"}

    # Re-run cointegration on test period for the same pairs to check persistence
    test_results = []
    for _, row in raw_sig_train.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        if a not in test_closes.columns or b not in test_closes.columns:
            continue
        series = test_closes[[a, b]].dropna()
        if len(series) < 60:
            test_results.append({
                "pair": row["pair"],
                "ticker_a": a,
                "ticker_b": b,
                "train_p_raw": row["p_raw"],
                "train_hedge": row["hedge_ratio"],
                "train_half_life": row["half_life"],
                "test_p_raw": None,
                "test_hedge": None,
                "still_significant": False,
                "note": "insufficient test data (<60 days)",
            })
            continue

        price_a, price_b = series[a], series[b]
        _, p_val, _ = coint(price_a, price_b)
        test_hedge = compute_hedge_ratio(price_a, price_b)

        test_results.append({
            "pair": row["pair"],
            "ticker_a": a,
            "ticker_b": b,
            "train_p_raw": row["p_raw"],
            "train_hedge": row["hedge_ratio"],
            "train_half_life": row["half_life"],
            "test_p_raw": round(p_val, 6),
            "test_hedge": round(test_hedge, 6),
            "still_significant": p_val < 0.05,
            "hedge_ratio_change_pct": round(
                abs(test_hedge - row["hedge_ratio"]) / abs(row["hedge_ratio"]) * 100, 1
            ) if row["hedge_ratio"] != 0 else None,
        })

    test_df = pd.DataFrame(test_results)
    n_still_sig = int(test_df["still_significant"].sum())
    log.info("Pairs still significant in test period: %d / %d", n_still_sig, len(test_df))

    # Step 3: Backtest on test period using TRAINING hedge ratios
    log.info("Backtesting on test period with training-period hedge ratios...")
    is_metrics = []  # in-sample
    oos_metrics = []  # out-of-sample

    for _, row in raw_sig_train.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        train_hedge = row["hedge_ratio"]

        # In-sample backtest (training period, training hedge ratios)
        is_signals = generate_signals_for_pair(train_closes, a, b, train_hedge)
        if not is_signals.empty and is_signals["position"].abs().sum() > 0:
            is_bt = backtest_signals(is_signals)
            is_bt["period"] = "in_sample"
            is_metrics.append(is_bt)

        # Out-of-sample backtest (test period, training hedge ratios)
        oos_signals = generate_signals_for_pair(test_closes, a, b, train_hedge)
        if not oos_signals.empty and oos_signals["position"].abs().sum() > 0:
            oos_bt = backtest_signals(oos_signals)
            oos_bt["period"] = "out_of_sample"
            oos_metrics.append(oos_bt)

    is_df = pd.DataFrame(is_metrics) if is_metrics else pd.DataFrame()
    oos_df = pd.DataFrame(oos_metrics) if oos_metrics else pd.DataFrame()

    # Step 4: Build comparison table
    comparison = []
    for _, row in raw_sig_train.iterrows():
        pair = row["pair"]
        is_row = is_df[is_df["pair"] == pair].iloc[0] if not is_df.empty and pair in is_df["pair"].values else None
        oos_row = oos_df[oos_df["pair"] == pair].iloc[0] if not oos_df.empty and pair in oos_df["pair"].values else None

        entry = {
            "pair": pair,
            "train_p_raw": row["p_raw"],
        }

        # Find test p-value
        test_match = test_df[test_df["pair"] == pair]
        if not test_match.empty:
            entry["test_p_raw"] = test_match.iloc[0]["test_p_raw"]
            entry["still_cointegrated"] = bool(test_match.iloc[0]["still_significant"])

        if is_row is not None:
            entry["is_sharpe"] = is_row["sharpe"]
            entry["is_return_pct"] = is_row["total_return_pct"]
            entry["is_n_trades"] = is_row["n_trades"]
            entry["is_win_rate_pct"] = is_row["win_rate_pct"]

        if oos_row is not None:
            entry["oos_sharpe"] = oos_row["sharpe"]
            entry["oos_return_pct"] = oos_row["total_return_pct"]
            entry["oos_n_trades"] = oos_row["n_trades"]
            entry["oos_win_rate_pct"] = oos_row["win_rate_pct"]

        if is_row is not None and oos_row is not None:
            entry["sharpe_change"] = round(oos_row["sharpe"] - is_row["sharpe"], 3)
            # Sign flip: positive IS return but negative OOS return, or vice versa
            entry["return_sign_flip"] = (
                (is_row["total_return_pct"] > 0) != (oos_row["total_return_pct"] > 0)
            )

        comparison.append(entry)

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(OUT_DIR / "oos_validation.csv", index=False)

    # Step 5: Build summary
    n_sign_flips = int(comp_df["return_sign_flip"].sum()) if "return_sign_flip" in comp_df.columns else 0
    avg_is_sharpe = round(float(comp_df["is_sharpe"].mean()), 3) if "is_sharpe" in comp_df.columns else None
    avg_oos_sharpe = round(float(comp_df["oos_sharpe"].mean()), 3) if "oos_sharpe" in comp_df.columns else None

    summary = {
        "training_period": {
            "start": str(train_closes.index.min().date()),
            "end": str(train_closes.index.max().date()),
            "trading_days": len(train_closes),
        },
        "test_period": {
            "start": str(test_closes.index.min().date()),
            "end": str(test_closes.index.max().date()),
            "trading_days": len(test_closes),
        },
        "cointegration": {
            "total_pairs_tested": n_total,
            "raw_significant_training": n_raw_train,
            "fdr_significant_training": n_fdr_train,
            "still_significant_in_test": n_still_sig,
            "pairs_tested_oos": len(test_df),
        },
        "backtest_comparison": {
            "avg_in_sample_sharpe": avg_is_sharpe,
            "avg_out_of_sample_sharpe": avg_oos_sharpe,
            "return_sign_flips": n_sign_flips,
            "pairs_with_oos_data": len(oos_df),
        },
        "pair_details": comparison,
    }

    with open(OUT_DIR / "oos_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("Saved OOS validation → %s", OUT_DIR / "oos_validation.csv")
    log.info("Saved OOS summary   → %s", OUT_DIR / "oos_summary.json")

    # Print summary
    print("\n" + "=" * 72)
    print("  OUT-OF-SAMPLE VALIDATION RESULTS")
    print("=" * 72)
    print(f"\nTraining: {summary['training_period']['start']} → {summary['training_period']['end']} "
          f"({summary['training_period']['trading_days']} days)")
    print(f"Test:     {summary['test_period']['start']} → {summary['test_period']['end']} "
          f"({summary['test_period']['trading_days']} days)")
    print(f"\nCointegration persistence:")
    print(f"  Training raw-significant:  {n_raw_train} / {n_total}")
    print(f"  Still significant in test: {n_still_sig} / {n_raw_train}")
    print(f"\nBacktest comparison:")
    print(f"  Avg in-sample Sharpe:     {avg_is_sharpe}")
    print(f"  Avg out-of-sample Sharpe: {avg_oos_sharpe}")
    print(f"  Return sign flips:        {n_sign_flips} / {len(oos_df)}")

    if not comp_df.empty and "is_sharpe" in comp_df.columns and "oos_sharpe" in comp_df.columns:
        print("\nPer-pair breakdown:")
        display_cols = ["pair", "train_p_raw", "still_cointegrated",
                        "is_sharpe", "oos_sharpe", "sharpe_change", "return_sign_flip"]
        display_cols = [c for c in display_cols if c in comp_df.columns]
        print(comp_df[display_cols].to_string(index=False))

    return summary


def main():
    run_oos_validation()


if __name__ == "__main__":
    main()
