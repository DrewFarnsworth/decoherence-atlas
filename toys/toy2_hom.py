"""Toy 2: Hong-Ou-Mandel interference with partially mismatched photons.

Two single photons meet at a 50:50 beam splitter. The parts are the two photons; the join
is that nothing records which photon is which. Coincidence probability
P_c = (1 - Tr(rho_1 rho_2)) / 2, dip visibility M = Tr(rho_1 rho_2).

Case a: pure Gaussian photons (closed form checked against quadrature).
Case b: one emitter with Markovian pure dephasing; both photons share the same mixed state.
        The key table puts the candidates for the physical join of mixed records side by side.
Case c: slow spectral wandering (Ornstein-Uhlenbeck), photons emitted dt apart (the B9 case).
Registered joins: a spectrometer with frequency bins, and a time-tagging detector with jitter.

Units: case a in units of sigma_1 (frequency) and 1/sigma_1 (time); cases b and c in units of
Gamma = 1/T1.

Run: python toys/toy2_hom.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

SEED = 202611
HEADER = {"event": "two photons meet at a 50:50 beam splitter",
          "alternatives": "which photon left by which port (exchange)",
          "prior": "uniform"}


# ---------------------------------------------------------------- case a
def c_closed(dw, tau, s1, s2):
    S = s1 * s1 + s2 * s2
    return math.sqrt(2 * s1 * s2 / S) * math.exp(-dw * dw / (4 * S)) * math.exp(-tau * tau * s1 * s1 * s2 * s2 / S)


def c_quad(dw, tau, s1, s2):
    """|<phi_1|phi_2>| with delay tau, amplitudes exp(-(w - w_k)^2 / (4 s_k^2)), normalized."""
    w1, w2 = dw / 2, -dw / 2
    n1 = (2 * math.pi * s1 * s1) ** -0.25
    n2 = (2 * math.pi * s2 * s2) ** -0.25
    f = lambda w, part: (n1 * n2 * math.exp(-(w - w1) ** 2 / (4 * s1 * s1) - (w - w2) ** 2 / (4 * s2 * s2))
                         * (math.cos(w * tau) if part == 0 else math.sin(w * tau)))
    lim = 12 * max(s1, s2) + abs(dw)
    re = quad(f, -lim, lim, args=(0,), epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    im = quad(f, -lim, lim, args=(1,), epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return math.hypot(re, im)


def spectrometer_join(dw, s1, s2, bin_width):
    """Bhattacharyya and total variation of the binned spectra |phi_k(w)|^2 (std s_k)."""
    lo = min(dw / 2 - 8 * s1, -dw / 2 - 8 * s2)
    hi = max(dw / 2 + 8 * s1, -dw / 2 + 8 * s2)
    lo -= bin_width; hi += bin_width
    edges = np.arange(math.floor(lo / bin_width) * bin_width, hi + bin_width, bin_width)
    def binned(mu, s):
        cdf = 0.5 * (1 + erf((edges - mu) / (s * math.sqrt(2))))
        return np.diff(cdf)
    p, q = binned(dw / 2, s1), binned(-dw / 2, s2)
    return L.bhattacharyya(p, q), L.total_variation(p, q)


def timetag_join(tau, s1, s2, jitter):
    """Bhattacharyya and total variation of arrival-time distributions (std 1/(2 s_k)), with
    Gaussian jitter added, offset by tau."""
    a = math.sqrt((1 / (2 * s1)) ** 2 + jitter ** 2)
    b = math.sqrt((1 / (2 * s2)) ** 2 + jitter ** 2)
    BC = math.sqrt(2 * a * b / (a * a + b * b)) * math.exp(-tau * tau / (4 * (a * a + b * b)))
    t = np.linspace(-abs(tau) - 12 * max(a, b), abs(tau) + 12 * max(a, b), 20001)
    p = np.exp(-(t - tau / 2) ** 2 / (2 * a * a)) / (a * math.sqrt(2 * math.pi))
    q = np.exp(-(t + tau / 2) ** 2 / (2 * b * b)) / (b * math.sqrt(2 * math.pi))
    TV = 0.5 * np.trapezoid(np.abs(p - q), t)
    return BC, float(TV)


# ---------------------------------------------------------------- case b
def dephased_matrix(gstar, n=700, tmax=14.0, gamma=1.0, detuning=0.0):
    """rho(t, t') = Gamma e^{-Gamma (t + t')/2} e^{-gamma* |t - t'|} e^{-i det (t - t')} on a grid (Gamma = 1)."""
    t = (np.arange(n) + 0.5) * tmax / n
    dt = tmax / n
    T, Tp = np.meshgrid(t, t, indexing="ij")
    rho = gamma * np.exp(-gamma * (T + Tp) / 2) * np.exp(-gstar * np.abs(T - Tp)) * np.exp(-1j * detuning * (T - Tp)) * dt
    return rho / np.trace(rho).real, t, dt


def explicit_environment_join(gstar, n_traj=4000, n=500, tmax=12.0, rng=None):
    """Each photon's phase follows its own random walk phi(t) = sqrt(2 gamma*) W(t); the trajectory is the
    environment's record. Returns sqrt(E |<psi_1|psi_2>|^2) over independent trajectories, and E|<.>|."""
    rng = rng or np.random.default_rng(SEED)
    t = (np.arange(n) + 0.5) * tmax / n
    dt = tmax / n
    amp = np.sqrt(np.exp(-t) * dt)
    ov2, ov1 = [], []
    for _ in range(n_traj // 500):
        dW = rng.normal(0, math.sqrt(dt), (2, 500, n))
        phi = np.cumsum(math.sqrt(2 * gstar) * dW, axis=2)
        psi1 = amp * np.exp(1j * phi[0]); psi2 = amp * np.exp(1j * phi[1])
        o = np.sum(psi1.conj() * psi2, axis=1) / np.sqrt(np.sum(np.abs(psi1) ** 2, 1) * np.sum(np.abs(psi2) ** 2, 1))
        ov2.append(np.abs(o) ** 2); ov1.append(np.abs(o))
    ov2, ov1 = np.concatenate(ov2), np.concatenate(ov1)
    return math.sqrt(ov2.mean()), ov2.std() / math.sqrt(len(ov2)), ov1.mean()


# ---------------------------------------------------------------- case c
def wander_visibility(x, sigma_ratio):
    """M(dt) for detuning variance 2 Sigma^2 (1 - e^{-x}), x = dt / tau_c, |c|^2 = 1/(1 + Delta^2) (Gamma = 1)."""
    var = 2 * sigma_ratio ** 2 * (1 - math.exp(-x))
    if var < 1e-18:
        return 1.0
    return gauss_lorentz_average(math.sqrt(var))


def gauss_lorentz_average(s):
    """E[1 / (1 + Delta^2)] for Delta ~ N(0, s^2), integrated in units of s so narrow peaks are resolved."""
    f = lambda u: math.exp(-u * u / 2) / math.sqrt(2 * math.pi) / (1 + (s * u) ** 2)
    return 2 * quad(f, 0, 40, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


def averaged_linewidth(sigma_ratio):
    """FWHM of the time-averaged line: Lorentzian (FWHM Gamma) convolved with a Gaussian of std Sigma."""
    w = np.linspace(-60 * (sigma_ratio + 1), 60 * (sigma_ratio + 1), 40001)
    lor = (0.5 / math.pi) / (w ** 2 + 0.25)
    g = np.exp(-w ** 2 / (2 * sigma_ratio ** 2)) if sigma_ratio > 0 else None
    prof = np.convolve(lor, g / g.sum(), mode="same") if g is not None else lor
    half = prof.max() / 2
    above = w[prof >= half]
    return float(above.max() - above.min())


def main():
    out = L.Output("toy2", "Toy 2: Hong-Ou-Mandel with partially mismatched photons")
    out.flag("case a units", "frequencies in units of sigma_1, times in 1/sigma_1; detuning split symmetrically (+dw/2, -dw/2)",
             "the spec gives the sweep in ratios only")
    out.flag("case a registered channels", "spectrometer bins aligned to a grid starting at a multiple of the bin width; "
             "time tagger: arrival-time intensity |phi(t)|^2 (std 1/(2 sigma)) convolved with Gaussian jitter",
             "bin placement and jitter shape are not fixed by the spec")
    out.flag("case b grid", "700 x 700 time grid to t = 14/Gamma, midpoint rule, normalized to unit trace",
             "spec asks for a time grid; this size meets the 1e-3 purity check")
    out.flag("case b explicit environment", "independent Wiener phase walks phi = sqrt(2 gamma*) W(t), 4,000 trajectory pairs; "
             "J_phys := sqrt(E |<psi_1|psi_2>|^2); also report E |<psi_1|psi_2>|",
             "the spec does not define J_phys for independent environments; with this definition "
             "J_phys^2 = Tr(rho_1 rho_2) holds by construction, so the check is a test of the code, not of physics")
    out.flag("exchange join (adopted 17 Nov)", "J_e for 'which photon is which' = Tr(rho1 rho2) (|c|^2 for pure photons); "
             "which-path questions keep |c|", "the overlap of the pair and the swapped pair, with environments explicit")
    out.flag("case b separate fidelities", "photons alone: F of the swapped two-photon states = F(rho, rho)^2, since both are rho x rho "
             "and fidelity multiplies over tensor products; environments alone: 1 by construction (same trajectory "
             "distribution in both alternatives)", "the spec asks for each separately")
    out.flag("case d recoil geometries", "one-sided: one recorded kick dp = q = 2 hbar k (the spec's factor); symmetric splitter: "
             "the two reflections push opposite faces, net dp = 0; no angle-of-incidence factors", "adopted 17 Nov: both, flagged")
    out.flag("case d loss model", "equal loss l at both output ports; coincidences counted only when both photons arrive",
             "the spec says absorption at both ports, recorded in a bath")
    out.flag("case c Sigma and tau_c", "swept as ratios only; not taken from B9",
             "B9 holds Gamma (T1) and the measured visibilities but no independent spectral-diffusion data; "
             "fitting Sigma to the visibility is forbidden by the spec")

    # ---- case a
    for (dw, tau, r) in [(0, 0, 1), (1.3, 0.7, 2), (4.0, 2.5, 5), (0.0, 3.0, 1), (6.0, 0.0, 2)]:
        a, b = c_closed(dw, tau, 1.0, float(r)), c_quad(dw, tau, 1.0, float(r))
        L.check(abs(a - b) < 1e-8, f"case a closed form vs quadrature: {a} vs {b} at dw={dw} tau={tau} s2/s1={r}")
    out.say("Case a: closed form for |c| agrees with numerical quadrature to 1e-8 at five test points.")
    L.check(abs((1 - c_closed(0, 0, 1, 1) ** 2) / 2) < 1e-15, "P_c = 0 for identical pure photons")
    L.check(abs((1 - c_closed(60, 0, 1, 1) ** 2) / 2 - 0.5) < 1e-12, "P_c = 1/2 for orthogonal photons")
    rows = []
    for r in (1, 2, 5):
        for dw in np.linspace(0, 6, 7):
            for st in np.linspace(0, 4, 5):
                c = c_closed(dw, st, 1.0, float(r))
                Jx, Dx = spectrometer_join(dw, 1.0, float(r), 0.1)
                Jt, Dt = timetag_join(st, 1.0, float(r), 0.05)
                L.check_pair(c, math.sqrt(1 - c * c), pure=True)
                L.check_monotone(Jx, c, label="spectrometer join >= J_phys")
                L.check_monotone(Jt, c, label="time-tag join >= J_phys")
                rows.append([r, dw, st, c, c * c, (1 - c * c) / 2, Jx, Dx, Jt, Dt])
    out.table("a_pure_gaussian", {**HEADER, "photons": "pure Gaussian amplitudes", "time": "delay tau = st / sigma_1",
                                  "channels": "J_phys = |c|; spectrometer bin 0.1 sigma_1; time tagger jitter 0.05 / sigma_1"},
              ["sigma2/sigma1", "dw/sigma1", "sigma1 tau", "|c| (which-path join)", "|c|^2 (exchange join = M)", "P_c",
               "J^X spectrometer", "D^X spectrometer",
               "J^X time tag", "D^X time tag"], rows)

    rows = []
    for bw in np.logspace(-2, 2, 9):
        for dw, st in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            c = c_closed(dw, st, 1.0, 1.0)
            Jx, Dx = spectrometer_join(dw, 1.0, 1.0, float(bw))
            Jt, Dt = timetag_join(st, 1.0, 1.0, float(bw))
            L.check_monotone(Jx, c); L.check_monotone(Jt, c)
            rows.append([bw, dw, st, c, Jx, Dx, Jt, Dt])
    out.table("a_registered_vs_resolution", {**HEADER, "photons": "pure Gaussian, sigma2 = sigma1",
                                             "channels": "spectrometer bin width or time-tag jitter = resolution (units of sigma_1 or 1/sigma_1)"},
              ["resolution", "dw/sigma1", "sigma1 tau", "J_phys", "J^X spectrometer", "D^X spectrometer",
               "J^X time tag (jitter)", "D^X time tag"], rows)

    # ---- case b: the key table
    rows = []
    rng = np.random.default_rng(SEED)
    for g in (0.0, 0.1, 0.5, 1.0, 5.0):
        rho, t, dt = dephased_matrix(g)
        F = float(L.fidelity(rho, rho))
        P = L.purity(rho)
        M_formula = 1 / (1 + 2 * g)
        L.check(abs(F - 1) < 1e-6, f"case b: F(rho, rho) = 1, got {F}")
        L.check(abs(P - M_formula) < 1e-3, f"case b: grid purity {P} vs Gamma/(Gamma + 2 gamma*) = {M_formula}")
        Jenv, err, mean_abs = explicit_environment_join(g, rng=rng)
        L.check(abs(Jenv ** 2 - P) < max(5 * err, 2e-3) + 2e-3,
                f"case b: J_phys^2 from the explicit environment {Jenv**2} vs Tr(rho^2) {P}")
        Je_joint = Jenv ** 2                        # exchange join over photons and their environments, jointly
        L.check(abs(Je_joint - M_formula) < max(5 * err, 2e-3) + 2e-3, "case b: J_e (exchange) = M = Tr(rho1 rho2)")
        F_photons = F * F                          # swapped pair is rho x rho again; fidelity multiplies over tensor products
        L.check(abs(F_photons - 1) < 1e-6, "case b: photons alone (swapped) are identical")
        rows.append([g, F, math.sqrt(P), Je_joint, 1.0, 1.0, mean_abs, M_formula, P])
    L.check(abs(rows[0][-1] - 1) < 1e-3, "case b: M = 1 at gamma* = 0")
    out.table("b_key_table", {**HEADER, "photons": "both from one emitter with pure dephasing gamma*; identical mixed states",
                              "time": "whole wave packets", "seed": SEED},
              ["gamma*/Gamma", "F(rho1, rho2)", "sqrt Tr(rho1 rho2)", "J_e exchange: pair vs swapped pair, environments explicit",
               "J_e photons alone", "J_e environments alone", "E|overlap| (explicit env)", "M = T2/2T1", "Tr(rho^2) on grid"], rows,
              note="Candidates side by side. The question is which photon is which (exchange). Taken jointly over the photons "
                   "and their environments, J_e = Tr(rho1 rho2) = M; the photons alone and the environments alone each give 1. "
                   "Adopted 17 Nov: the exchange join is Tr(rho1 rho2); which-path questions keep |c|.")
    for g, F, sq, Je, Jp, Jenvs, ma, M, P in rows:
        out.say(f"Case b, gamma*/Gamma = {g}: F = {F:.4f}, joint exchange J_e = {Je:.4f}, photons alone 1, environments alone 1, "
                f"M = {M:.4f}")
    # Tr(rho1 rho2) <= F^2 for unequal mixed states
    for g1, g2, det in [(0.1, 1.0, 0.0), (0.5, 0.5, 0.8), (0.0, 2.0, 1.5)]:
        r1, _, _ = dephased_matrix(g1, n=400)
        r2, _, _ = dephased_matrix(g2, n=400, detuning=det)
        L.check(L.overlap_trace(r1, r2) <= float(L.fidelity(r1, r2)) ** 2 + 1e-9, "Tr(rho1 rho2) <= F^2")
    out.say("Tr(rho1 rho2) <= F(rho1, rho2)^2 checked on three unequal mixed pairs.")

    # ---- case d: an imperfect splitter
    rows = []
    for ell in (0.0, 0.05, 0.2):
        # equal loss at both ports, recorded in a bath ("lost or not"); the record does not depend on which photon
        M = 1.0
        pc_dist = 0.5 * (1 - ell) ** 2
        pc = pc_dist * (1 - M)
        Je = M * 1.0                                   # loss records identical for both exchange alternatives
        Vcoinc = (pc_dist - pc) / pc_dist
        L.check(abs(Vcoinc - Je) < 1e-12, "case d, loss: coincidence visibility unchanged")
        rows.append(["loss and heating", f"loss {ell}", 1.0, Je, Vcoinc, "lost-or-not record, independent of which photon"])
    lam = 1.0
    for geom in ("one-sided (one recorded kick)", "symmetric splitter (opposite faces)"):
        for xr in np.logspace(-6, -1, 6):
            dp_over_hbar = (2 * 2 * math.pi / lam) if geom.startswith("one") else 0.0     # q = 2 hbar k, or net 0
            overlap = math.exp(-(dp_over_hbar * xr * lam) ** 2 / 2)
            Je = 1.0 * overlap
            rows.append(["mirror recoil", f"{geom}; dx/lambda = {xr:.0e}", 1.0, Je, Je,
                         "mirror states for both-reflected vs both-transmitted overlap exp(-dp^2 dx^2 / 2 hbar^2)"])
    for R in (0.3, 0.4, 0.5, 0.6, 0.7):
        T = 1 - R
        M = 1.0
        pc_dist = R * R + T * T
        pc = R * R + T * T - 2 * R * T * M
        Vmax = 2 * R * T / (R * R + T * T)
        V = (pc_dist - pc) / pc_dist
        L.check(abs(V - Vmax * M) < 1e-12, "case d, unequal split: V = V_max J_e")
        rows.append(["unequal split", f"R = {R}", Vmax, M, V, "no record; V_max = 2RT/(R^2 + T^2)"])
    out.table("d_imperfect_splitter", {**HEADER, "photons": "identical pure photons (M = 1) unless stated",
                                       "question": "which photon is which (exchange)"},
              ["effect", "setting", "V_max", "J_e", "V (coincidences only)", "note"], rows,
              note="Recoil geometry is flagged: with a symmetric splitter the two reflections push opposite faces, so "
                   "'both reflected' and 'both transmitted' leave the same mirror momentum and no record.")
    out.say("Case d: loss leaves J_e and the coincidence visibility unchanged; recoil enters only in the one-sided geometry "
            f"(factor {math.exp(-(4 * math.pi * 0.1) ** 2 / 2):.3f} at dx/lambda = 0.1, 1 to 1e-10 at 1e-6); "
            "an unequal split lowers V_max to 2RT/(R^2 + T^2) with J_e unchanged.")

    # ---- case c
    rows = []
    for sr in (0.5, 2.0, 10.0):
        lw = averaged_linewidth(sr)
        for x in np.logspace(-3, 3, 25):
            M = wander_visibility(float(x), sr)
            rows.append([sr, x, M, lw])
        M0 = wander_visibility(1e-9, sr)
        Minf = wander_visibility(1e6, sr)
        # static-disorder average: detuning variance 2 Sigma^2
        Mstatic = gauss_lorentz_average(math.sqrt(2) * sr)
        L.check(abs(M0 - 1) < 1e-6, "case c: M -> 1 as dt/tau_c -> 0")
        L.check(abs(Minf - Mstatic) < 1e-6, "case c: M -> static-disorder average")
        out.say(f"Case c, Sigma/Gamma = {sr}: averaged linewidth {lw:.2f} Gamma; M = {wander_visibility(0.001, sr):.4f} at dt/tau_c = 0.001, "
                f"{wander_visibility(1.0, sr):.4f} at 1, {Mstatic:.4f} at the static limit")
    out.table("c_spectral_wandering", {**HEADER, "photons": "exponential wave packets (Gamma = 1), Ornstein-Uhlenbeck detuning",
                                       "time": "photons emitted dt apart"},
              ["Sigma/Gamma", "dt/tau_c", "M", "time-averaged FWHM / Gamma"], rows,
              note="A broad averaged line (width set by Sigma) with high visibility for photons close in time. "
                   "B9 (QD1-QD3, dt = 3 ns, T1 = 0.67-1.1 ns) has no independent Sigma or tau_c, so no B9 prediction is made here.")

    # registered join for cases b and c: spectrometer
    rows = []
    for bw in np.logspace(-2, 2, 9):
        rows.append([bw, "b (any gamma*)", 1.0, "identical spectra: a spectrometer cannot see pure dephasing"])
    for sr in (2.0,):
        for x in (0.01, 1.0, 100.0):
            var = 2 * sr ** 2 * (1 - math.exp(-x))
            ds = np.random.default_rng(SEED).normal(0, math.sqrt(var), 400)
            for bw in np.logspace(-2, 2, 5):
                bcs = []
                for d in ds:
                    edges = np.arange(-80, 80 + bw, bw)
                    cdf = lambda mu: 0.5 + np.arctan(2 * (edges - mu)) / math.pi
                    p, q = np.diff(cdf(d / 2)), np.diff(cdf(-d / 2))
                    bcs.append(L.bhattacharyya(p / p.sum(), q / q.sum()))
                rows.append([bw, f"c (Sigma/Gamma = 2, dt/tau_c = {x})", float(np.mean(bcs)), "mean over 400 detuning draws"])
    out.table("bc_registered", {**HEADER, "channel": "spectrometer with bin width = resolution (units of Gamma)"},
              ["resolution / Gamma", "case", "J^X", "note"], rows)
    out.finish()


if __name__ == "__main__":
    main()
