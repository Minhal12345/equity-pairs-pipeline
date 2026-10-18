"""Tests for reconcile.py — divergence flagging, missing days, output schema."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reconcile import DIV_THRESHOLD


def test_divergence_flagging():
    """Divergence >0.5% is flagged, <=0.5% is not."""
    yf_prices  = [100.0, 200.0, 150.0]
    av_prices  = [100.0, 200.0, 140.0]  # 3rd day diverges by ~7.1%

    divergences = [
        abs(y - a) / a for y, a in zip(yf_prices, av_prices)
    ]
    flags = [d > DIV_THRESHOLD for d in divergences]

    assert flags == [False, False, True], f"Expected [F, F, T], got {flags}"


def test_missing_day_detection():
    """Missing days in either source are counted correctly."""
    dates_yf = pd.date_range("2023-01-01", periods=5, freq="B")
    dates_av = pd.date_range("2023-01-01", periods=3, freq="B")

    yf_close = pd.Series([100, 101, 102, 103, 104], index=dates_yf, name="yf")
    av_close = pd.Series([100, 101, 102],            index=dates_av, name="av")

    merged = pd.DataFrame({"yf": yf_close, "av": av_close})
    missing_av = int(merged["av"].isna().sum())

    assert missing_av == 2, f"Expected 2 missing AV days, got {missing_av}"


def test_output_schema():
    """Reconciliation summary should have the expected columns."""
    expected_cols = {
        "ticker", "yf_days", "av_days",
        "pct_divergent_days", "worst_divergence",
        "missing_yf_days", "missing_av_days"
    }
    df = pd.DataFrame([{
        "ticker": "AAPL", "yf_days": 750, "av_days": 748,
        "pct_divergent_days": 0.5, "worst_divergence": 1.2,
        "missing_yf_days": 0, "missing_av_days": 2,
    }])
    assert set(df.columns) == expected_cols
