"""
reconcile.py
------------
Aligns yfinance and Alpha Vantage close prices for each ticker,
flags divergences >0.5% and missing trading days, and writes
summary statistics to processed/ and output/.

Outputs:
    data/processed/reconciliation.parquet   — per-ticker per-day aligned closes + divergence
    data/output/reconciliation_summary.csv  — summary row per ticker
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_YF    = BASE_DIR / "data" / "raw" / "yfinance"
RAW_AV    = BASE_DIR / "data" / "raw" / "alphavantage"
PROC_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR   = BASE_DIR / "data" / "output"

DIV_THRESHOLD = 0.005   # 0.5 %


def load_close(raw_dir: Path, ticker: str) -> pd.Series | None:
    """Load the 'close' column from a per-ticker Parquet file."""
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["close"])
    return df["close"].rename(ticker)


def reconcile(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Align yfinance and Alpha Vantage close prices, compute divergence,
    flag anomalies, and return the merged DataFrame.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = [p.stem for p in sorted(RAW_YF.glob("*.parquet"))]

    records = []
    all_rows = []

    for ticker in tickers:
        yf_close = load_close(RAW_YF, ticker)
        av_close  = load_close(RAW_AV, ticker)

        if yf_close is None:
            log.warning("  %s: no yfinance data, skipping", ticker)
            continue

        row = {
            "ticker":             ticker,
            "yf_days":            len(yf_close),
            "av_days":            len(av_close) if av_close is not None else 0,
            "pct_divergent_days": None,
            "worst_divergence":   None,
            "missing_yf_days":    0,
            "missing_av_days":    0,
        }

        if av_close is None:
            log.warning("  %s: no Alpha Vantage data — skipping reconciliation", ticker)
            records.append(row)
            # Still store yfinance-only rows so downstream steps have a full processed frame
            tmp = yf_close.to_frame(name="yf_close")
            tmp["av_close"] = np.nan
            tmp["divergence_pct"] = np.nan
            tmp["flag_divergence"] = False
            tmp["ticker"] = ticker
            all_rows.append(tmp)
            continue

        # Align on shared trading days
        merged = pd.DataFrame({"yf_close": yf_close, "av_close": av_close})
        merged["divergence_pct"] = (merged["yf_close"] - merged["av_close"]).abs() / merged["av_close"]
        merged["flag_divergence"] = merged["divergence_pct"] > DIV_THRESHOLD

        # Missing day counts: days present in one source but not the other
        missing_av  = merged["av_close"].isna().sum()
        missing_yf  = merged["yf_close"].isna().sum()
        valid_mask  = merged["yf_close"].notna() & merged["av_close"].notna()
        pct_div     = merged.loc[valid_mask, "flag_divergence"].mean() * 100
        worst_div   = merged.loc[valid_mask, "divergence_pct"].max() * 100

        row.update({
            "pct_divergent_days": round(pct_div, 3),
            "worst_divergence":   round(worst_div, 3),
            "missing_yf_days":    int(missing_yf),
            "missing_av_days":    int(missing_av),
        })
        records.append(row)

        merged["ticker"] = ticker
        all_rows.append(merged)

        log.info(
            "  %s  divergent=%.2f%%  worst=%.3f%%  miss_yf=%d  miss_av=%d",
            ticker, pct_div, worst_div, missing_yf, missing_av
        )

    # Concatenate all tickers
    detail_df = pd.concat(all_rows) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(records)

    detail_path  = PROC_DIR / "reconciliation.parquet"
    summary_path = OUT_DIR  / "reconciliation_summary.csv"
    detail_df.to_parquet(detail_path)
    summary_df.to_csv(summary_path, index=False)

    log.info("Reconciliation complete → %s, %s", detail_path, summary_path)
    return summary_df


def main():
    summary = reconcile()
    print("\n=== Reconciliation Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
