"""
main.py
-------
Orchestrates the full equity pairs trading pipeline:
    1. Ingest data from yfinance (Alpha Vantage optional)
    2. Reconcile sources
    3. Run cointegration analysis
    4. Generate trading signals
    5. Backtest signals
    6. Generate charts and terminal report

Usage:
    python main.py [--skip-av]  # skip Alpha Vantage ingestion
"""

import argparse
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingest import ingest_yfinance, ingest_alphavantage
from reconcile import reconcile
from cointegration import run_cointegration
from signals import load_closes, generate_signals
from backtest import run_backtest
from report import generate_all_charts, print_terminal_report


def main():
    parser = argparse.ArgumentParser(description="Equity Pairs Trading Pipeline")
    parser.add_argument("--skip-av", action="store_true", help="Skip Alpha Vantage ingestion")
    parser.add_argument("--av-key", default=None, help="Alpha Vantage API key")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Ingest
    print("\n" + "=" * 72)
    print("  STEP 1 — DATA INGESTION")
    print("=" * 72)
    ingest_yfinance()
    if not args.skip_av and args.av_key:
        ingest_alphavantage(api_key=args.av_key)
    elif args.skip_av:
        log.info("Skipping Alpha Vantage ingestion (--skip-av)")
    else:
        log.info("Skipping Alpha Vantage (no --av-key provided)")

    # Step 2: Reconcile
    print("\n" + "=" * 72)
    print("  STEP 2 — RECONCILIATION")
    print("=" * 72)
    reconcile()

    # Step 3: Cointegration
    print("\n" + "=" * 72)
    print("  STEP 3 — COINTEGRATION ANALYSIS")
    print("=" * 72)
    coint_df = run_cointegration()
    if coint_df.empty:
        log.error("No pairs could be tested. Pipeline stopping.")
        sys.exit(1)

    # Use FDR-significant pairs if any survive; otherwise fall back to raw p < 0.05
    sig_pairs = coint_df[coint_df["significant_at_fdr_0.05"]].copy()
    if sig_pairs.empty:
        sig_pairs = coint_df[coint_df["p_raw"] < 0.05].copy()
        if sig_pairs.empty:
            log.error("No pairs significant at p < 0.05. Pipeline stopping.")
            sys.exit(1)
        log.warning("No pairs survive BH-FDR correction. Using %d raw-significant "
                     "pairs (p < 0.05) -- see INVESTIGATION.md for discussion.",
                     len(sig_pairs))

    # Step 4: Signals
    print("\n" + "=" * 72)
    print("  STEP 4 — SIGNAL GENERATION")
    print("=" * 72)
    closes = load_closes()
    generate_signals(closes, sig_pairs)

    # Step 5: Backtest
    print("\n" + "=" * 72)
    print("  STEP 5 — BACKTEST")
    print("=" * 72)
    run_backtest()

    # Step 6: Charts + Report
    print("\n" + "=" * 72)
    print("  STEP 6 — CHARTS & REPORT")
    print("=" * 72)
    generate_all_charts()
    print_terminal_report()

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
