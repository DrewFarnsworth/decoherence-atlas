"""Fast checks on the shared code and the pieces of each model. Run: python -m pytest -q"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas.constants import hbar, meV, nm  # noqa: E402
from atlas.report import Check  # noqa: E402
from atlas.scattering import most_probable_speed, sigma_eff_vdw  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pass_rule():
    assert Check("x", "u", 1.29, measured=1.0).evaluate().status == "pass"
    assert Check("x", "u", 1.31, measured=1.0).evaluate().status == "fail"
    assert Check("x", "u", 0.65, measured_low=0.9).evaluate().status == "pass"     # 0.9 * 0.7 = 0.63
    assert Check("x", "u", 0.60, measured_low=0.9).evaluate().status == "fail"
    assert Check("x", "u", 600, measured_low=200, measured_high=500).evaluate().status == "pass"
    assert Check("x", "u", None, measured=1).evaluate().status == "not computed"


def test_vdw_prefactor_matches_hornberger():
    # PRL 90:160401 Eq. 6: sigma = (C6/hbar)^(2/5) v_g^(3/5)/v_m * (8.4946 + 1.6989 u^2)
    C6, v_g, v_m = 2.1 * meV * nm**6, 420.0, 117.0
    u = v_m / v_g
    expected = (C6 / hbar) ** 0.4 * v_g**0.6 / v_m * (8.4946 + 1.6989 * u**2)
    assert sigma_eff_vdw(C6, v_m, v_g) == pytest.approx(expected, rel=1e-3)


def test_b1_reproduces_the_papers_theory_within_30_percent():
    b1 = load("b1")
    from atlas.constants import amu
    from atlas.scattering import decoherence_pressure
    for gas, (m, C6, _, _, p0_theo) in b1.GASES.items():
        sig = sigma_eff_vdw(C6 * meV * nm**6, b1.V_M, most_probable_speed(m * amu, b1.T_GAS))
        p0 = decoherence_pressure(sig, 2 * b1.L, b1.T_GAS) / 100 / 1e-7
        assert p0 == pytest.approx(p0_theo, rel=0.30), gas


def test_b4_quaternion_echo_matches_matrices():
    b4 = load("b4")
    rng = np.random.default_rng(0)
    Delta = rng.normal(0, 2e4, 5)
    b = rng.normal(0, 3e2, 5)
    taus = np.array([1e-4, 4e-4, 1e-3])
    tz = np.diag([1.0, -1.0]); tx = np.array([[0, 1.0], [1.0, 0]])
    for k in range(5):
        for t in taus:
            U0 = expm(-1j * t * (-b[k] / 4) * tx)
            U1 = expm(-1j * t * (Delta[k] / 2 * tz - b[k] / 4 * tx))
            V0, V1 = U1 @ U0, U0 @ U1
            L_direct = abs((2 + np.trace(V0.conj().T @ V1)) / 4)
            L_quat = b4.echo(Delta[k:k + 1], b[k:k + 1], np.array([t]))[0]
            assert L_quat == pytest.approx(L_direct, rel=1e-9, abs=1e-12)


def test_b5_debye_limits():
    b5 = load("b5")
    # at T -> 0, -ln f -> 3 E_R / (2 kB theta)
    from atlas.constants import kB
    assert b5.minus_ln_f(1.0) == pytest.approx(3 * b5.E_R / (2 * kB * b5.THETA_D), rel=1e-3)
    assert b5.f_LM(295) > b5.f_LM(600) > b5.f_LM(1000)


def test_b6_harmonic_kinetic_energy_limits():
    b6 = load("b6")
    from atlas.constants import kB, wavenumber_to_J
    # high temperature -> kB T / 2 per degree of freedom; zero temperature -> hbar w / 4
    assert b6.kinetic(10.0, 3000) == pytest.approx(kB * 3000 / 2, rel=1e-3)
    assert b6.kinetic(3000.0, 1.0) == pytest.approx(wavenumber_to_J(3000.0) / 4, rel=1e-6)


def test_b7_transition_dipole_is_physical():
    b7 = load("b7")
    assert 0.15 < b7.MU_D < 0.35          # debye; liquid-water O-H stretch literature range


def test_b8_optical_dephasing_high_temperature_limit():
    b8 = load("b8")
    # slow-modulation high-T limit: |exp(-g)| ~ exp(-lam kT t^2) (Gaussian), with lam, kT in rad/fs
    T = 300.0
    lam_w, kT_w = b8.LAM * b8.W, b8.K_CM * T * b8.W
    t_gauss = 1 / math.sqrt(lam_w * kT_w)
    # exact result is longer than the static-limit Gaussian (motional narrowing)
    assert b8.optical_dephasing_time(T) > 0.9 * t_gauss


def test_b9_overlap_and_dip():
    b9 = load("b9")
    # emitter overlap reduces to T2 / 2T1
    assert b9.emitter_overlap(1e-9, 1.2e-9) == pytest.approx(0.6, rel=2e-3)
    assert b9.emitter_overlap(1e-9, 2e-9) == pytest.approx(1.0, rel=2e-3)
    # rectangular filter: first zero of the exchange term at tau = 1 / (2 dnu)
    P, dnu = b9.spdc_dip(np.array([0.0, 1 / (2 * (3e8 * b9.DLAM / b9.LAM0**2))]))
    assert P[0] == pytest.approx(0.0, abs=1e-6)
    assert P[1] == pytest.approx(0.5, abs=2e-3)


def test_b10_duality_is_an_equality_for_pure_states():
    b10 = load("b10")
    for R in (0.0, 0.1, 0.3, 0.5):
        V, D = b10.visibility(R), b10.distinguishability(R)
        assert V == pytest.approx(2 * math.sqrt(R * (1 - R)), abs=1e-3)
        assert V**2 + D**2 == pytest.approx(1.0, abs=2e-3)
