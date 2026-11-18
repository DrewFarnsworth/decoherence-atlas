"""One test per toy running its checks (fast parts only), and a deliberate break that must fail loudly."""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge_lib as L  # noqa: E402


class Quiet(L.Output):
    def __init__(self):
        self.flags, self.summary = [], []

    def say(self, line):
        self.summary.append(line)


def test_toy1_checks():
    import toy1_markers as T
    for N in (1, 2, 5):
        for c in (0.95, 0.42, 0.0):
            for k in range(N + 1):
                T.explicit_check([c] * N, k)
    J, D, chi, _ = T.noisy_read(0.9, 0.5, 4)
    Jd, Dd, chid = T.noisy_dense(0.9, 0.5, 4)
    assert (J, D, chi) == pytest.approx((Jd, Dd, chid), abs=1e-9)


def test_toy2_checks():
    import toy2_hom as T
    assert T.c_closed(1.3, 0.7, 1.0, 2.0) == pytest.approx(T.c_quad(1.3, 0.7, 1.0, 2.0), abs=1e-8)
    rho, _, _ = T.dephased_matrix(0.5)
    assert L.purity(rho) == pytest.approx(0.5, abs=1e-3)
    assert T.wander_visibility(1e-9, 2.0) == pytest.approx(1.0, abs=1e-6)


def test_toy3_checks():
    import toy3_superradiance as T
    T.checks(Quiet())


def test_toy4_checks():
    import toy4_graph_domains as T
    T.run_checks(Quiet())


def test_toy5_checks():
    import toy5_qed as T
    T.checks(Quiet())


def test_toy6_checks():
    import toy6_radical_pair as T
    T.checks(Quiet())


def test_toy7_site_basis_reproduces_toy4():
    import toy4_graph_domains as T4
    import toy7_parts as T7
    V, blocks = T4.graph("G1", 0.2)
    P = T7.propagator(V.copy(), T7.coupling_ops("local", 8), 1.0, 3.0)
    I8 = np.eye(8, dtype=complex)
    mine = T7.domains_for(T7.in_basis(T7.evolve_basis(P, I8), I8))
    ref = T4.analyse(T4.records(V, np.zeros(8), 1.0, 3.0), blocks, ["trace"])["trace"]
    assert set(mine["A"]) == set(ref["A"]) and set(mine["B"]) == set(ref["B"])


def test_checks_fail_loudly_when_broken(monkeypatch):
    import toy1_markers as T
    real = L.fidelity
    monkeypatch.setattr(L, "fidelity", lambda r, s: 1.01 * real(r, s))
    with pytest.raises(L.CheckFailed):
        T.explicit_check([0.9, 0.9], 1)


def test_toy7_couplings():
    import toy7_parts as T7
    for kind in ("local", "collective", "random"):
        ops = T7.coupling_ops(kind, 8)
        assert all(np.allclose(A, A.conj().T) for A in ops)
    # trace preserved under every coupling
    import toy4_graph_domains as T4
    V, _ = T4.graph("G1", 0.2)
    for kind in ("local", "collective", "random"):
        P = T7.propagator(V.copy(), T7.coupling_ops(kind, 8), 1.0, 2.0)
        r = T7.evolve_basis(P, np.eye(8, dtype=complex))
        assert np.allclose(np.trace(r, axis1=1, axis2=2), 1.0, atol=1e-10)


def test_exchange_join_is_overlap_squared_for_pure_states():
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.normal(size=4) + 1j * rng.normal(size=4); a /= np.linalg.norm(a)
        b = rng.normal(size=4) + 1j * rng.normal(size=4); b /= np.linalg.norm(b)
        assert L.exchange_join(L.dm(a), L.dm(b)) == pytest.approx(abs(np.vdot(a, b)) ** 2, abs=1e-12)


def test_toy1b_chain_cases():
    import toy1b_chain as T
    kws = [("A", {}), ("B", {"cs": [0.9, 0.7, 0.5]}), ("C", {"c": 0.6}), ("D", {"phi": 1.0}),
           ("E", {"cP": 0.6, "measure": True}), ("F", {"cP": 0.6, "measure": True}), ("G", {"R": 0.3})]
    for case, kw in kws:
        ch, vecs, nq, Je, Jphys, Jout, V, Vmax = T.analyse(case, **kw)
        assert 0 <= Jphys <= Je + 1e-10 <= 1 + 1e-9          # more records never join more
        assert V <= Vmax * Je + 1e-9                          # fringes never exceed what the records allow
    assert T.port_bhattacharyya(0.5) == pytest.approx(1.0)


def test_toy8_checks():
    import toy8_web as T
    T.run_checks(Quiet())


def test_toy8b_checks():
    import toy8b_structured as T
    T.run_checks(Quiet())
