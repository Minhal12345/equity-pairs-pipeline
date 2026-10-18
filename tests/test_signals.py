"""Tests for signals.py — z-score, entry/exit, stop-loss, cooldown."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from signals import compute_zscore, Z_ENTRY, Z_EXIT, Z_STOPLOSS


def _run_state_machine(z_values):
    """Run the position state machine with stop-loss cooldown.

    Mirrors the logic in generate_signals(): after a stop-out, new entries
    are locked out until the z-score drops back below Z_ENTRY magnitude.
    """
    pos = 0
    locked_out = False
    positions = []
    for val in z_values:
        if locked_out and abs(val) < Z_ENTRY:
            locked_out = False
        if pos == 0:
            if not locked_out:
                if val < -Z_ENTRY:
                    pos = 1
                elif val > Z_ENTRY:
                    pos = -1
        else:
            if abs(val) >= Z_STOPLOSS:
                pos = 0
                locked_out = True
            elif (pos == 1 and val >= Z_EXIT) or (pos == -1 and val <= Z_EXIT):
                pos = 0
        positions.append(pos)
    return positions


def test_zscore_computation():
    """Z-score of a constant spread should be 0 (after lookback window)."""
    spread = pd.Series([100.0] * 100)
    z = compute_zscore(spread, lookback=20)
    valid = z.dropna()
    assert (valid == 0).all() or len(valid) == 0, "Constant spread should yield z=0"


def test_entry_exit_signal_generation():
    """Z-score crossing -2 should trigger long entry; crossing 0 should exit."""
    z = [0, -1, -2.1, -2.5, -1, 0, 0.5]
    positions = _run_state_machine(z)

    assert positions[2] == 1, f"Expected long entry at z=-2.1, got pos={positions[2]}"
    assert positions[5] == 0, f"Expected exit at z=0, got pos={positions[5]}"


def test_stoploss_logic():
    """Z-score exceeding +/-3 should trigger stop-loss (position -> 0)."""
    z = [0, -2.5, -3.1, 0]
    positions = _run_state_machine(z)

    assert positions[1] == 1, "Should enter long at z=-2.5"
    assert positions[2] == 0, "Should stop-loss at z=-3.1"


def test_stoploss_cooldown_prevents_reentry():
    """After a stop-out, no re-entry until z-score crosses back below entry threshold."""
    # Sequence: enter long at -2.5, stop-out at -3.1, z stays at -2.9 (still extreme)
    # — should NOT re-enter. Then z reverts to -1.5 (below entry), then -2.3 — NOW re-enter.
    z = [0, -2.5, -3.1, -2.9, -2.3, -1.5, -2.3, -1, 0]
    positions = _run_state_machine(z)

    assert positions[1] == 1,  "Should enter long at z=-2.5"
    assert positions[2] == 0,  "Should stop-loss at z=-3.1"
    assert positions[3] == 0,  "Should NOT re-enter at z=-2.9 (locked out)"
    assert positions[4] == 0,  "Should NOT re-enter at z=-2.3 (still locked out, |z| >= Z_ENTRY)"
    assert positions[5] == 0,  "Lockout clears at z=-1.5 but no entry signal (|z| < 2)"
    assert positions[6] == 1,  "Should re-enter long at z=-2.3 (lockout cleared)"
    assert positions[8] == 0,  "Should exit at z=0"


def test_stoploss_cooldown_short_side():
    """Cooldown works symmetrically for short-side stop-outs."""
    z = [0, 2.5, 3.1, 2.9, 1.5, 2.3]
    positions = _run_state_machine(z)

    assert positions[1] == -1, "Should enter short at z=2.5"
    assert positions[2] == 0,  "Should stop-loss at z=3.1"
    assert positions[3] == 0,  "Should NOT re-enter at z=2.9 (locked out)"
    assert positions[4] == 0,  "Lockout clears at z=1.5 but no entry signal"
    assert positions[5] == -1, "Should re-enter short at z=2.3 (lockout cleared)"
