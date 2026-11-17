"""bridge.py, rebuilt on bridge_lib, must still reproduce the BRIDGE.md tables."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge  # noqa: E402


def test_toy1_table():
    rows = {(c, q): (J, D, chi, acc) for c, q, J, D, chi, acc in bridge.toy1()}
    assert rows[(0.42, 0.0)] == pytest.approx((0.420, 0.908, 0.869, 0.730), abs=1e-3)
    assert rows[(0.0, 0.9)][0] == pytest.approx(0.995, abs=1e-3)
    assert rows[(0.9, 0.0)][2] == pytest.approx(0.286, abs=1e-3)


def test_toy2_table():
    out, (I1, I12) = bridge.toy2()
    by_N = {N: (J, D, I, Is) for N, J, D, I, Is in out}
    assert by_N[10] == pytest.approx((0.794, 0.267, 0.488, 0.255), abs=1e-3)
    assert by_N[100][2] == pytest.approx(0.205, abs=1e-3)
    assert (I1, I12) == pytest.approx((0.488, 0.506), abs=1e-3)
