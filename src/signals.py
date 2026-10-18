"""
signals.py
----------
For each cointegrated pair, computes the spread, rolling 60-day z-score,
and generates entry/exit/stop-loss signals.

Rules:
    Entry LONG spread  :  z-score crosses below -2
    Entry SHORT spread :  z-score crosses above +2
    Exit               :  z-score reverts to 0
    Stop-loss          :  z-score exceeds ±3

Output:
    data/processed/signals.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_YF   = BASE_DIR / "data" / "raw" / "yfinance"
PROC_DIR = BASE_DIR / "data" / "processed"

Z_ENTRY    = 2.0
Z_EXIT     = 0.0
Z_STOPLOSS = 3.0
LOOKBACK   = 60


def load_closes() -> pd.DataFrame:
    series = {}
    for p in sorted(RAW_YF.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["close"])
        series[p.stem] = df["close"]
    return pd.DataFrame(series).dropna(how="all")


def compute_zscore(spread: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(lookback).mean()
    std  = spread.rolling(lookback).std()
    return (spread - mean) / std


def generate_signals(
    closes: pd.DataFrame,
    coint_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each cointegrated pair, compute spread and z-score signals.
    Returns a long-form DataFrame with columns:
        date, pair, ticker_a, ticker_b, spread, zscore, signal, position
    """
    all_signals = []

    for _, row in coint_df.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        hedge = row["hedge_ratio"]

        if a not in closes.columns or b not in closes.columns:
            log.warning("  Skipping %s/%s — missing price data", a, b)
            continue

        aligned = closes[[a, b]].dropna()
        spread  = aligned[a] - hedge * aligned[b]
        zscore  = compute_zscore(spread)

        # Determine position state using vectorised logic
        signal   = pd.Series(0, index=aligned.index, dtype=int)
        position = pd.Series(0, index=aligned.index, dtype=int)

        # Entry signals
        signal[zscore < -Z_ENTRY] = 1   # long spread
        signal[zscore >  Z_ENTRY] = -1  # short spread

        # Stop-loss overrides
        signal[zscore < -Z_STOPLOSS] = 0
        signal[zscore >  Z_STOPLOSS] = 0

        # Walk forward to compute position (need state machine)
        pos = 0
        positions = []
        for i, (z, s) in enumerate(zip(zscore, signal)):
            if np.isnan(z):
                positions.append(0)
                continue
            if pos == 0:
                # Not in a trade — enter on signal
                pos = s
            else:
                # In a trade — check exit or stop-loss
                if abs(z) >= Z_STOPLOSS:
                    pos = 0
                elif (pos == 1 and z >= Z_EXIT) or (pos == -1 and z <= Z_EXIT):
                    pos = 0
            positions.append(pos)

        df = pd.DataFrame({
            "date":     aligned.index,
            "pair":     f"{a}/{b}",
            "ticker_a": a,
            "ticker_b": b,
            "spread":   spread.values,
            "zscore":   zscore.values,
            "signal":   signal.values,
            "position": positions,
        })
        all_signals.append(df)
        log.info("  Signals ✓ %s/%s  (%d days)", a, b, len(df))

    if not all_signals:
        return pd.DataFrame()

    out_df = pd.concat(all_signals, ignore_index=True)
    out_path = PROC_DIR / "signals.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info("Saved signals → %s  (%d rows)", out_path, len(out_df))
    return out_df


def main():
    closes   = load_closes()
    coint_df = pd.read_parquet(PROC_DIR / "cointegration_results.parquet")
    # Prefer FDR-significant pairs; fall back to raw p < 0.05
    if "significant_at_fdr_0.05" in coint_df.columns:
        fdr_sig = coint_df[coint_df["significant_at_fdr_0.05"]]
        if not fdr_sig.empty:
            coint_df = fdr_sig.copy()
        elif "p_raw" in coint_df.columns:
            coint_df = coint_df[coint_df["p_raw"] < 0.05].copy()
    sig_df   = generate_signals(closes, coint_df)
    print(f"\nGenerated signals for {sig_df['pair'].nunique()} pairs, {len(sig_df)} total rows")


if __name__ == "__main__":
    main()
