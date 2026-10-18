"""
cointegration.py
----------------
Runs Engle-Granger cointegration tests on within-sector pairs and a set
of manually specified cross-sector pairs.  Applies Benjamini-Hochberg
FDR correction to account for multiple testing.  For each significant pair
(after FDR correction at alpha=0.05), computes the OLS hedge ratio and
Ornstein-Uhlenbeck half-life.

Output:
    data/processed/cointegration_results.parquet
        columns: pair, ticker_a, ticker_b, p_raw, p_fdr,
                 significant_at_fdr_0.05, hedge_ratio, half_life
"""

import logging
from itertools import combinations
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
PROC_DIR = BASE_DIR / "data" / "processed"

# Sector groupings — within-sector pairs will be tested
SECTORS: dict[str, list[str]] = {
    "tech":       ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "financials": ["JPM",  "BAC",  "GS",   "MS"],
    "consumer":   ["KO",   "PEP",  "MCD",  "YUM",  "COST", "WMT"],
    "energy":     ["XOM",  "CVX",  "COP",  "SLB"],
    "healthcare": ["JNJ",  "PFE",  "MRK",  "ABBV"],
    "payments":   ["V",    "MA",   "PYPL"],
    "retail":     ["HD",   "LOW",  "NKE"],
}

# Obvious cross-sector pairs to always test
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
    """
    Ornstein-Uhlenbeck half-life via AR(1) on spread differences.
    half_life = -ln(2) / ln(phi)  where phi is the AR(1) coefficient.
    Returns np.inf if phi >= 1 (non-mean-reverting).
    """
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    # align
    idx = lag.index.intersection(delta.index)
    lag, delta = lag.loc[idx], delta.loc[idx]

    model = OLS(delta.values, add_constant(lag.values)).fit()
    phi = 1 + model.params[1]   # AR(1) coefficient on levels
    if phi >= 1 or phi <= 0:
        return np.inf
    return -np.log(2) / np.log(phi)


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS regression: price_a ~ beta * price_b; return beta."""
    model = OLS(price_a.values, add_constant(price_b.values)).fit()
    return float(model.params[1])


def test_pair(
    ticker_a: str, ticker_b: str, closes: pd.DataFrame
) -> dict | None:
    """
    Run Engle-Granger test on (ticker_a, ticker_b).
    Returns result dict with raw p-value for ALL pairs (not filtered).
    Returns None only if data is insufficient.
    """
    if ticker_a not in closes.columns or ticker_b not in closes.columns:
        return None
    series = closes[[ticker_a, ticker_b]].dropna()
    if len(series) < 60:
        return None

    price_a = series[ticker_a]
    price_b = series[ticker_b]

    _, p_val, _ = coint(price_a, price_b)

    hedge   = compute_hedge_ratio(price_a, price_b)
    spread  = price_a - hedge * price_b
    hl      = compute_half_life(spread)

    return {
        "pair":        f"{ticker_a}/{ticker_b}",
        "ticker_a":    ticker_a,
        "ticker_b":    ticker_b,
        "p_raw":       round(p_val, 6),
        "hedge_ratio": round(hedge, 6),
        "half_life":   round(hl, 2) if np.isfinite(hl) else None,
    }


def run_cointegration() -> pd.DataFrame:
    """
    Test all within-sector pairs + explicit cross-sector pairs.
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    Return DataFrame of ALL tested pairs with raw and FDR-corrected p-values.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    closes = load_closes()
    log.info("Loaded price matrix: %d dates x %d tickers", *closes.shape)

    # Build candidate pair list (deduplicated)
    candidate_pairs: set[tuple[str, str]] = set()
    for tickers in SECTORS.values():
        for a, b in combinations(tickers, 2):
            candidate_pairs.add((a, b))
    for a, b in EXPLICIT_PAIRS:
        candidate_pairs.add((a, b))

    log.info("Testing %d pairs for cointegration ...", len(candidate_pairs))
    results = []
    for a, b in sorted(candidate_pairs):
        res = test_pair(a, b, closes)
        if res:
            results.append(res)

    if not results:
        log.warning("No pairs could be tested -- check data quality.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction
    reject, p_fdr, _, _ = multipletests(df["p_raw"], alpha=0.05, method="fdr_bh")
    df["p_fdr"] = np.round(p_fdr, 6)
    df["significant_at_fdr_0.05"] = reject

    n_raw = (df["p_raw"] < 0.05).sum()
    n_fdr = reject.sum()
    log.info("Raw significant (p < 0.05): %d / %d pairs", n_raw, len(df))
    log.info("After BH-FDR correction:    %d / %d pairs", n_fdr, len(df))

    for _, row in df[df["significant_at_fdr_0.05"]].iterrows():
        log.info("  SIGNIFICANT  %s  p_raw=%.4f  p_fdr=%.4f  hedge=%.4f  "
                 "half_life=%.1f",
                 row["pair"], row["p_raw"], row["p_fdr"], row["hedge_ratio"],
                 row["half_life"] if row["half_life"] else float("nan"))

    df = df.sort_values("p_raw").reset_index(drop=True)
    out = PROC_DIR / "cointegration_results.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved %d tested pairs (%d significant after FDR) -> %s",
             len(df), n_fdr, out)
    return df


def main():
    df = run_cointegration()
    if df.empty:
        print("No pairs tested.")
        return
    sig = df[df["significant_at_fdr_0.05"]]
    print(f"\n=== Cointegration Results ({len(sig)} significant after FDR) ===")
    print(sig[["pair", "p_raw", "p_fdr", "hedge_ratio", "half_life"]].to_string(index=False))
    print(f"\nAll {len(df)} pairs tested. {len(sig)} remain significant after "
          f"Benjamini-Hochberg FDR correction at alpha=0.05.")


if __name__ == "__main__":
    main()
