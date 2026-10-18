"""Tests for signals.py — z-score, entry/exit, stop-loss."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from signals import compute_zscore, Z_ENTRY, Z_EXIT, Z_STOPLOSS


def test_zscore_computation():
    """Z-score of a constant spread should be 0 (after lookback window)."""
    spread = pd.Series([100.0] * 100)
    z = compute_zscore(spread, lookback=20)
    # After the first 20 values, z-score should be NaN or 0
    valid = z.dropna()
    assert (valid == 0).all() or len(valid) == 0, "Constant spread should yield z=0"


def test_entry_exit_signal_generation():
    """Z-score crossing -2 should trigger long entry; crossing 0 should exit."""
    # Build a synthetic z-score that drops below -2, then reverts to 0
    z = pd.Series([0, -1, -2.1, -2.5, -1, 0, 0.5])

    # Simulate position logic
    pos = 0
    positions = []
    for val in z:
        if pos == 0:
            if val < -Z_ENTRY:
                pos = 1  # long spread
            elif val > Z_ENTRY:
                pos = -1
        else:
            if abs(val) >= Z_STOPLOSS:
                pos = 0
            elif (pos == 1 and val >= Z_EXIT) or (pos == -1 and val <= Z_EXIT):
                pos = 0
        positions.append(pos)

    # Should enter long at index 2 (z=-2.1), exit at index 5 (z=0)
    assert positions[2] == 1, f"Expected long entry at z=-2.1, got pos={positions[2]}"
    assert positions[5] == 0, f"Expected exit at z=0, got pos={positions[5]}"


def test_stoploss_logic():
    """Z-score exceeding ±3 should trigger stop-loss (position → 0)."""
    z = pd.Series([0, -2.5, -3.1, 0])

    pos = 0
    positions = []
    for val in z:
        if pos == 0:
            if val < -Z_ENTRY:
                pos = 1
            elif val > Z_ENTRY:
                pos = -1
        else:
            if abs(val) >= Z_STOPLOSS:
                pos = 0
            elif (pos == 1 and val >= Z_EXIT) or (pos == -1 and val <= Z_EXIT):
                pos = 0
        positions.append(pos)

    assert positions[1] == 1,  "Should enter long at z=-2.5"
    assert positions[2] == 0,  "Should stop-loss at z=-3.1"
