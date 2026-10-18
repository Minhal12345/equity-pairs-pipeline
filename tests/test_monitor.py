"""Unit tests for src.monitor using synthetic DataFrames."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_raw_dir(tmp_path: Path, tickers: list[str] | None = None,
                  n_days: int = 252, start: str = "2024-01-02",
                  inject: dict | None = None) -> Path:
    """Create a minimal raw/yfinance directory with synthetic parquet files.

    *inject* lets callers modify specific tickers' DataFrames before saving.
    """
    raw_dir = tmp_path / "raw" / "yfinance"
    raw_dir.mkdir(parents=True)
    tickers = tickers or ["AAPL", "MSFT"]
    dates = pd.bdate_range(start, periods=n_days, freq="B")

    for t in tickers:
        rng = np.random.default_rng(hash(t) % 2**32)
        price = 100 + rng.standard_normal(n_days).cumsum()
        price = np.maximum(price, 1)  # keep positive
        df = pd.DataFrame({
            "open": price * (1 + rng.uniform(-0.01, 0.01, n_days)),
            "high": price * (1 + rng.uniform(0, 0.02, n_days)),
            "low": price * (1 - rng.uniform(0, 0.02, n_days)),
            "close": price,
            "volume": rng.integers(1_000_000, 50_000_000, n_days),
        }, index=pd.Index(dates, name="date"))

        if inject and t in inject:
            df = inject[t](df)

        df.to_parquet(raw_dir / f"{t}.parquet")

    return raw_dir


def _make_processed_dir(tmp_path: Path, has_av: bool = False) -> Path:
    """Create a minimal processed directory with reconciliation.parquet."""
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
    rows = []
    for t in ["AAPL", "MSFT"]:
        for d in dates:
            rows.append({
                "date": d,
                "yf_close": 150.0,
                "av_close": 150.0 if has_av else np.nan,
                "divergence_pct": 0.0 if has_av else np.nan,
                "flag_divergence": False,
                "ticker": t,
            })
    recon = pd.DataFrame(rows).set_index("date")
    recon.to_parquet(proc_dir / "reconciliation.parquet")
    return proc_dir


# ── tests ───────────────────────────────────────────────────────────────────

# Import after helpers so fixtures are available.
# We add the project root to sys.path so `src.monitor` resolves.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.monitor import (
    check_staleness,
    check_coverage,
    check_schema,
    check_outlier_returns,
    check_zero_volume,
    check_price_sanity,
    run_all_checks,
)


def test_staleness_detects_old_data(tmp_path):
    """Data ending > 2 business days ago should fail."""
    raw = _make_raw_dir(tmp_path, n_days=50, start="2020-01-02")
    result = check_staleness(raw, max_business_days=2)
    assert result["status"] == "fail"
    assert len(result["affected_tickers"]) == 2


def test_schema_pass_on_valid_data(tmp_path):
    """Correctly-shaped files should pass schema check."""
    raw = _make_raw_dir(tmp_path)
    result = check_schema(raw)
    assert result["status"] == "pass"


def test_outlier_returns_flags_spikes(tmp_path):
    """A 20% daily spike should trigger an outlier warning."""
    def inject_spike(df):
        df.iloc[100, df.columns.get_loc("close")] *= 1.25
        return df

    raw = _make_raw_dir(tmp_path, inject={"AAPL": inject_spike})
    result = check_outlier_returns(raw, threshold=0.15)
    assert result["status"] == "warn"
    assert "AAPL" in result["affected_tickers"]


def test_price_sanity_catches_negative(tmp_path):
    """A negative price should fail the sanity check."""
    def inject_neg(df):
        df.iloc[50, df.columns.get_loc("close")] = -5.0
        return df

    raw = _make_raw_dir(tmp_path, inject={"MSFT": inject_neg})
    result = check_price_sanity(raw)
    assert result["status"] == "fail"
    assert "MSFT" in result["affected_tickers"]


def test_run_all_checks_returns_report(tmp_path):
    """Full run should produce a valid report dict."""
    raw = _make_raw_dir(tmp_path)
    proc = _make_processed_dir(tmp_path)
    report = run_all_checks(raw_dir=raw, processed_dir=proc)
    assert "overall_status" in report
    assert "checks" in report
    assert len(report["checks"]) == 7
    assert all(c["status"] in ("pass", "warn", "fail", "error")
               for c in report["checks"])


def test_empty_raw_dir_fails(tmp_path):
    """An empty raw directory should return an overall fail, not silently pass."""
    raw_dir = tmp_path / "raw" / "yfinance"
    raw_dir.mkdir(parents=True)
    proc = _make_processed_dir(tmp_path)
    report = run_all_checks(raw_dir=raw_dir, processed_dir=proc)
    assert report["overall_status"] == "fail"


def test_corrupt_file_returns_error_not_crash(tmp_path):
    """A corrupt parquet file should produce an error status, not crash the run."""
    raw_dir = tmp_path / "raw" / "yfinance"
    raw_dir.mkdir(parents=True)
    # Write a valid file and a corrupt file
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
    price = 100 + rng.standard_normal(50).cumsum()
    price = np.maximum(price, 1)
    df = pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price,
        "volume": rng.integers(1_000_000, 50_000_000, 50),
    }, index=pd.Index(dates, name="date"))
    df.to_parquet(raw_dir / "AAPL.parquet")

    # Write a corrupt file (not valid parquet)
    (raw_dir / "CORRUPT.parquet").write_bytes(b"this is not parquet data")

    proc = _make_processed_dir(tmp_path)
    report = run_all_checks(raw_dir=raw_dir, processed_dir=proc)
    # Should complete without raising, and at least one check should error
    assert report is not None
    assert "checks" in report
    statuses = [c["status"] for c in report["checks"]]
    assert "error" in statuses, "Corrupt file should produce at least one 'error' status"
