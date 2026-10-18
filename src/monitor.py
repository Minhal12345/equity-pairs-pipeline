"""
Data quality monitoring module for the Equity Pairs Trading Pipeline.

Runs 7 automated checks against ingested data and produces a structured
JSON report (data/output/quality_report.json) plus a terminal summary table.

Usage:
    python -m src.monitor                    # run all checks
    python -m src.monitor --threshold 0.20   # custom outlier threshold
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── Defaults ────────────────────────────────────────────────────────────────

RAW_YF_DIR = Path("data/raw/yfinance")
RAW_AV_DIR = Path("data/raw/alpha_vantage")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/output")

EXPECTED_SCHEMA = {
    "yfinance": {"open": "float64", "high": "float64", "low": "float64",
                 "close": "float64", "volume": "int64"},
}


# ── Check result dataclass ──────────────────────────────────────────────────


def _result(name: str, status: str, details: str,
            affected_tickers: list[str] | None = None) -> dict[str, Any]:
    """Build a single check result dict."""
    return {
        "check_name": name,
        "status": status,  # pass | warn | fail | error
        "details": details,
        "affected_tickers": affected_tickers or [],
    }


# ── Individual checks ───────────────────────────────────────────────────────


def check_staleness(raw_dir: Path, max_business_days: int = 2) -> dict:
    """Last date in each source should be within *max_business_days* of today."""
    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("staleness", "fail",
                       "No parquet files found in raw directory")

    stale = []
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.tseries.offsets.BDay(max_business_days)

    for pq in parquets:
        df = pd.read_parquet(pq)
        last = pd.to_datetime(df.index).max()
        if last < cutoff:
            stale.append(pq.stem)

    if stale:
        return _result("staleness", "fail",
                       f"{len(stale)} ticker(s) have data older than "
                       f"{max_business_days} business days",
                       stale)
    return _result("staleness", "pass",
                   f"All tickers current within {max_business_days} business days")


def check_coverage(raw_dir: Path, min_pct: float = 0.95) -> dict:
    """Each ticker has >= *min_pct* of expected NYSE trading days."""
    import exchange_calendars as xcals

    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("coverage", "fail",
                       "No parquet files found in raw directory")

    low = []
    details_parts: list[str] = []
    nyse = xcals.get_calendar("XNYS")

    for pq in parquets:
        df = pd.read_parquet(pq)
        idx = pd.to_datetime(df.index)
        start, end = idx.min(), idx.max()
        expected = len(nyse.sessions_in_range(start, end))
        actual = len(df)
        pct = actual / expected if expected else 0
        if pct < min_pct:
            low.append(pq.stem)
            details_parts.append(f"{pq.stem}: {pct:.1%}")

    if low:
        return _result("coverage", "warn",
                       f"{len(low)} ticker(s) below {min_pct:.0%} coverage: "
                       + ", ".join(details_parts), low)
    return _result("coverage", "pass",
                   f"All tickers >= {min_pct:.0%} coverage")


def check_schema(raw_dir: Path,
                 expected: dict[str, str] | None = None) -> dict:
    """Columns and dtypes match the expected schema."""
    if expected is None:
        expected = EXPECTED_SCHEMA["yfinance"]

    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("schema", "fail",
                       "No parquet files found in raw directory")

    bad = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        for col, dtype in expected.items():
            if col not in df.columns:
                bad.append(f"{pq.stem}: missing column '{col}'")
            elif str(df[col].dtype) != dtype:
                bad.append(
                    f"{pq.stem}: '{col}' is {df[col].dtype}, expected {dtype}")

    if bad:
        tickers = list({b.split(":")[0] for b in bad})
        return _result("schema", "fail",
                       f"{len(bad)} schema violation(s): " + "; ".join(bad[:5]),
                       tickers)
    return _result("schema", "pass", "All files match expected schema")


def check_outlier_returns(raw_dir: Path,
                          threshold: float = 0.15) -> dict:
    """Flag ticker-days where |daily return| exceeds *threshold*."""
    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("outlier_returns", "fail",
                       "No parquet files found in raw directory")

    total_count = 0
    sample: list[str] = []
    affected: set[str] = set()

    for pq in parquets:
        df = pd.read_parquet(pq)
        rets = df["close"].pct_change().dropna()
        outliers = rets[rets.abs() > threshold]
        if len(outliers):
            total_count += len(outliers)
            affected.add(pq.stem)
            for date, ret in outliers.items():
                if len(sample) < 5:
                    sample.append(f"{pq.stem} {pd.Timestamp(date).date()}: "
                                  f"{ret:+.1%}")

    if total_count:
        detail = (f"{total_count} day(s) with |return| > "
                  f"{threshold:.0%}")
        if sample:
            detail += " (showing first 5): " + "; ".join(sample)
        return _result("outlier_returns", "warn", detail, sorted(affected))
    return _result("outlier_returns", "pass",
                   f"No returns exceed +/-{threshold:.0%}")


def check_zero_volume(raw_dir: Path) -> dict:
    """Flag ticker-days with volume == 0."""
    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("zero_volume", "fail",
                       "No parquet files found in raw directory")

    total_count = 0
    sample: list[str] = []
    affected: set[str] = set()

    for pq in parquets:
        df = pd.read_parquet(pq)
        if "volume" not in df.columns:
            continue
        zeros = df[df["volume"] == 0]
        if len(zeros):
            total_count += len(zeros)
            affected.add(pq.stem)
            for date in zeros.index[:3]:
                if len(sample) < 5:
                    sample.append(f"{pq.stem} {pd.Timestamp(date).date()}")

    if total_count:
        detail = f"{total_count} zero-volume day(s)"
        if sample:
            detail += " (showing first 5): " + "; ".join(sample)
        return _result("zero_volume", "warn", detail, sorted(affected))
    return _result("zero_volume", "pass", "No zero-volume days found")


def check_divergence_drift(processed_dir: Path) -> dict:
    """Weekly rolling avg divergence; flag if latest week > 2 sigma above baseline."""
    recon_path = processed_dir / "reconciliation.parquet"
    if not recon_path.exists():
        return _result("divergence_drift", "warn",
                       "Reconciliation file not found -- skipping check")

    recon = pd.read_parquet(recon_path)
    if recon["av_close"].isna().all():
        return _result("divergence_drift", "warn",
                       "No Alpha Vantage data available -- "
                       "cannot compute cross-source divergence")

    flagged: list[str] = []
    for ticker, grp in recon.groupby("ticker"):
        grp = grp.sort_index()
        weekly = grp["divergence_pct"].resample("W").mean().dropna()
        # Drop partial first/last weeks
        if len(weekly) >= 2:
            first_full_idx = grp.index.min() + pd.tseries.offsets.Week(weekday=6)
            weekly = weekly[weekly.index >= first_full_idx]
        if len(weekly) < 13:
            continue
        baseline = weekly.iloc[:-1]
        latest = weekly.iloc[-1]
        mu, sigma = baseline.mean(), baseline.std()
        if sigma > 0 and latest > mu + 2 * sigma:
            flagged.append(str(ticker))

    if flagged:
        return _result("divergence_drift", "warn",
                       f"{len(flagged)} ticker(s) show elevated divergence "
                       "in latest week",
                       flagged)
    return _result("divergence_drift", "pass",
                   "Cross-source divergence within baseline range")


def check_price_sanity(raw_dir: Path) -> dict:
    """No negative prices; no price > 10x the ticker's median."""
    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        return _result("price_sanity", "fail",
                       "No parquet files found in raw directory")

    bad: list[str] = []
    affected: set[str] = set()

    for pq in parquets:
        df = pd.read_parquet(pq)
        ticker = pq.stem
        c = df["close"]

        negs = c[c < 0]
        if len(negs):
            bad.append(f"{ticker}: {len(negs)} negative price(s)")
            affected.add(ticker)

        median_price = c.median()
        extreme = c[c > 10 * median_price]
        if len(extreme):
            bad.append(f"{ticker}: {len(extreme)} price(s) > 10x median "
                       f"(median=${median_price:.2f})")
            affected.add(ticker)

    if bad:
        return _result("price_sanity", "fail",
                       "; ".join(bad), sorted(affected))
    return _result("price_sanity", "pass",
                   "All prices non-negative and within 10x median")


# ── Runner ──────────────────────────────────────────────────────────────────


def run_all_checks(raw_dir: Path | None = None,
                   processed_dir: Path | None = None,
                   threshold: float = 0.15) -> dict[str, Any]:
    """Execute all checks and return the full report dict.

    Each check is wrapped in try/except so a single corrupt file or
    unexpected error does not abort the entire monitor run.
    """
    raw = raw_dir or RAW_YF_DIR
    proc = processed_dir or PROCESSED_DIR

    # Fail fast if raw directory is empty or missing
    if not raw.exists() or not list(raw.glob("*.parquet")):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "fail",
            "checks": [_result("data_presence", "fail",
                               "No parquet files found in raw directory")],
        }

    check_fns = [
        ("staleness", lambda: check_staleness(raw)),
        ("coverage", lambda: check_coverage(raw)),
        ("schema", lambda: check_schema(raw)),
        ("outlier_returns", lambda: check_outlier_returns(raw, threshold=threshold)),
        ("zero_volume", lambda: check_zero_volume(raw)),
        ("divergence_drift", lambda: check_divergence_drift(proc)),
        ("price_sanity", lambda: check_price_sanity(raw)),
    ]

    checks = []
    for name, fn in check_fns:
        try:
            checks.append(fn())
        except Exception as e:
            checks.append(_result(name, "error",
                                  f"Check raised {type(e).__name__}: {e}"))

    statuses = [c["status"] for c in checks]
    if "fail" in statuses or "error" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": checks,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a formatted terminal summary table."""
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback without tabulate
        print(f"\n{'='*60}")
        print(f"  Data Quality Report -- {report['timestamp']}")
        print(f"  Overall: {report['overall_status'].upper()}")
        print(f"{'='*60}")
        for c in report["checks"]:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗",
                    "error": "✗"}[c["status"]]
            print(f"  {icon} {c['check_name']:.<30s} {c['status'].upper()}")
            if c["affected_tickers"]:
                print(f"    -> {', '.join(c['affected_tickers'][:5])}")
        print()
        return

    rows = []
    for c in report["checks"]:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗",
                "error": "✗"}[c["status"]]
        tickers = ", ".join(c["affected_tickers"][:5])
        if len(c["affected_tickers"]) > 5:
            tickers += f" (+{len(c['affected_tickers'])-5} more)"
        rows.append([
            f"{icon} {c['status'].upper()}",
            c["check_name"],
            c["details"][:80],
            tickers or "--",
        ])

    print(f"\n{'='*80}")
    print(f"  Data Quality Report -- {report['timestamp']}")
    print(f"  Overall: {report['overall_status'].upper()}")
    print(f"{'='*80}")
    print(tabulate(rows,
                   headers=["Status", "Check", "Details", "Affected"],
                   tablefmt="simple",
                   maxcolwidths=[10, 25, 45, 20]))
    print()


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data quality checks on the equity pipeline.")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Outlier return threshold (default: 0.15)")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Path to raw yfinance parquet directory")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Path to processed parquet directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/output/quality_report.json)")
    args = parser.parse_args()

    raw = Path(args.raw_dir) if args.raw_dir else RAW_YF_DIR
    proc = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR
    out = Path(args.output) if args.output else OUTPUT_DIR / "quality_report.json"

    report = run_all_checks(raw_dir=raw, processed_dir=proc,
                            threshold=args.threshold)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)

    # Exit non-zero on fail (warns are OK)
    if report["overall_status"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
