"""Known values and property tests for the shared library."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge_lib as L  # noqa: E402


def random_state(d, rng, rank=None):
    rank = rank or d
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    r = G @ G.conj().T
    return r / np.trace(r).real


def random_channel(d_in, d_out, rng, n_kraus=3):
    """Random CPTP map from a random isometry (Stinespring)."""
    G = rng.normal(size=(d_out * n_kraus, d_in)) + 1j * rng.normal(size=(d_out * n_kraus, d_in))
    V, _ = np.linalg.qr(G)                     # isometry: V^dagger V = I
    K = V.reshape(n_kraus, d_out, d_in)
    return lambda r: sum(k @ r @ k.conj().T for k in K)


def test_known_values():
    zero, one = L.dm(L.ket(1, 0)), L.dm(L.ket(0, 1))
    plus = L.dm(L.ket(1, 1))
    assert L.fidelity(zero, plus) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert L.trace_distance(zero, one) == pytest.approx(1.0, abs=1e-12)


def test_holevo_two_pure_states():
    for c in (0.0, 0.3, 0.7, 0.99):
        th = math.acos(c)
        a, b = L.dm(L.ket(1, 0)), L.dm(L.ket(math.cos(th), math.sin(th)))
        assert L.holevo([0.5, 0.5], [a, b]) == pytest.approx(L.h2((1 + c) / 2), abs=1e-10)


def test_fvdg_and_monotonicity_random():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        d = int(rng.integers(2, 5))
        r, s = random_state(d, rng), random_state(d, rng, rank=int(rng.integers(1, d + 1)))
        J, D = float(L.fidelity(r, s)), float(L.trace_distance(r, s))
        L.check_pair(J, D, tol=1e-8)
        ch = random_channel(d, int(rng.integers(2, 5)), rng)
        L.check_monotone(float(L.fidelity(ch(r), ch(s))), J, tol=1e-8)


def test_mirror_binary_distributions_saturate():
    for p in (0.1, 0.3, 0.45):
        a, b = [p, 1 - p], [1 - p, p]
        assert L.bhattacharyya(a, b) ** 2 + L.total_variation(a, b) ** 2 == pytest.approx(1.0, abs=1e-12)


def test_partial_trace():
    rng = np.random.default_rng(2)
    A, B = random_state(2, rng), random_state(3, rng)
    AB = np.kron(A, B)
    assert np.allclose(L.partial_trace(AB, [2, 3], [0]), A)
    assert np.allclose(L.partial_trace(AB, [2, 3], [1]), B)


def test_checks_fail_loudly():
    with pytest.raises(L.CheckFailed):
        L.check_pair(0.9, 0.9)              # violates the upper bound
    with pytest.raises(L.CheckFailed):
        L.check_dpi([0.5, 0.6], 1.0)        # information rose along the chain
    with pytest.raises(L.CheckFailed):
        L.check_monotone(0.4, 0.5)
