"""
ingest.py
---------
Pulls 3 years of daily OHLCV data from yfinance and Alpha Vantage
for 30 tickers spanning 6 sectors, storing each as a per-ticker Parquet file
in data/raw/{yfinance,alphavantage}/.

Usage:
    python src/ingest.py [--source yfinance|alphavantage|both]
"""

import os
import time
import argparse
import logging
from pathlib import Path

import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",   # Tech
    "JPM",  "BAC",  "GS",   "MS",                        # Financials
    "KO",   "PEP",  "MCD",  "YUM",  "COST", "WMT",      # Consumer
    "XOM",  "CVX",  "COP",  "SLB",                       # Energy
    "JNJ",  "PFE",  "MRK",  "ABBV",                      # Healthcare
    "V",    "MA",   "PYPL",                               # Payments
    "HD",   "LOW",  "NKE",                                # Retail
]

START_DATE = "2021-10-01"
END_DATE   = "2024-10-01"

BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_YF     = BASE_DIR / "data" / "raw" / "yfinance"
RAW_AV     = BASE_DIR / "data" / "raw" / "alphavantage"
AV_API_KEY_ENV = "ALPHAVANTAGE_API_KEY"


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def ingest_yfinance(tickers: list[str] = TICKERS, start: str = START_DATE, end: str = END_DATE) -> None:
    """Download OHLCV data from yfinance and save per-ticker Parquet files."""
    RAW_YF.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %d tickers from yfinance (%s → %s)", len(tickers), start, end)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()

            df = df.dropna(how="all")
            df.index.name = "date"
            df.columns = [c.lower() for c in df.columns]
            out = RAW_YF / f"{ticker}.parquet"
            df.to_parquet(out)
            log.info("  yfinance ✓ %s  (%d rows → %s)", ticker, len(df), out.name)
        except Exception as exc:
            log.warning("  yfinance ✗ %s: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------

AV_BASE = "https://www.alphavantage.co/query"


def _fetch_av_ticker(ticker: str, api_key: str) -> pd.DataFrame:
    """Fetch full daily adjusted OHLCV from Alpha Vantage for one ticker."""
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "json",
    }
    resp = requests.get(AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "Time Series (Daily)" not in payload:
        note = payload.get("Note") or payload.get("Information") or str(payload)
        raise ValueError(f"Alpha Vantage error for {ticker}: {note[:120]}")

    ts = payload["Time Series (Daily)"]
    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "date":   pd.Timestamp(date_str),
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["5. adjusted close"]),
            "volume": int(vals["6. volume"]),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def ingest_alphavantage(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    api_key: str | None = None,
) -> None:
    """Download OHLCV from Alpha Vantage (5 req/min rate limit respected)."""
    RAW_AV.mkdir(parents=True, exist_ok=True)

    if api_key is None:
        api_key = os.environ.get(AV_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Alpha Vantage API key not set. "
            f"Export {AV_API_KEY_ENV}=<your_key> or pass --av-key."
        )

    log.info("Downloading %d tickers from Alpha Vantage (rate-limited 5 req/min)", len(tickers))
    for i, ticker in enumerate(tickers):
        try:
            df = _fetch_av_ticker(ticker, api_key)
            df = df.loc[start:end]
            out = RAW_AV / f"{ticker}.parquet"
            df.to_parquet(out)
            log.info("  alphavantage ✓ %s  (%d rows)", ticker, len(df))
        except Exception as exc:
            log.warning("  alphavantage ✗ %s: %s", ticker, exc)

        # Alpha Vantage free tier: 5 requests per minute → sleep 13 s between calls
        if (i + 1) % 5 == 0:
            log.info("  Rate-limit pause 65 s …")
            time.sleep(65)
        else:
            time.sleep(13)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest equity OHLCV data")
    parser.add_argument("--source", choices=["yfinance", "alphavantage", "both"], default="both")
    parser.add_argument("--av-key", dest="av_key", default=None, help="Alpha Vantage API key")
    args = parser.parse_args()

    if args.source in ("yfinance", "both"):
        ingest_yfinance()
    if args.source in ("alphavantage", "both"):
        ingest_alphavantage(api_key=args.av_key)


if __name__ == "__main__":
    main()
