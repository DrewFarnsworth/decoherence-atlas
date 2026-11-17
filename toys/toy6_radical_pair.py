"""Toy 6: radical pair.

Two electron spins S1, S2 and one nucleus I coupled to S1 only:
    H = omega (S1z + S2z) + a S1 . I,      omega = gamma_e B,  a = gamma_e * (1 mT)
Start: electrons in the singlet, nucleus in |m>. Recombination at the same rate k from singlet
and triplet, so the singlet yield is Phi_S = k Tr[P_S (k - L)^-1 rho_0], with L the Liouvillian
(unitary, plus optional electron dephasing at rate r: L_i = sqrt(r/2) sigma_z^(i), so each
electron spin's coherence decays at rate r).

Alternatives: the nuclear projection m in {-I..I}, uniform prior.
Records: the electron pair at time t (nucleus traced out); the same weighted by k e^{-kt};
and the chemical product, a classical bit (singlet or triplet product).

Run: python toys/toy6_radical_pair.py
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

GAMMA_E = 1.76085963023e11          # rad s^-1 T^-1
A_FIELD = 1e-3                       # T: a = 1 mT in field units
B_LIST = [0.0, 0.05e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3]
I_LIST = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)]
K_LIST = np.logspace(4, 9, 21)
R_LIST = [0.0, 1e5, 1e6, 1e7]
# gyromagnetic ratios (rad s^-1 T^-1), standard tables, from memory: flagged
GAMMA_129XE = -7.452e7
GAMMA_131XE = 2.209e7
HEADER_BASE = {"event": "the radical pair reacts (singlet or triplet product)",
               "alternatives": "nuclear spin projection m in {-I, ..., I}",
               "prior": "uniform over m"}

sx = np.array([[0, 1], [1, 0]], complex) / 2
sy = np.array([[0, -1j], [1j, 0]], complex) / 2
sz = np.array([[1, 0], [0, -1]], complex) / 2


def spin_ops(I):
    d = int(2 * I + 1)
    if d == 1:
        z = np.zeros((1, 1), complex)
        return z, z, z
    m = np.array([float(I) - k for k in range(d)])
    Iz = np.diag(m).astype(complex)
    Ip = np.zeros((d, d), complex)
    for k in range(1, d):
        Ip[k - 1, k] = math.sqrt(float(I) * (float(I) + 1) - m[k] * (m[k] + 1))
    Ix = (Ip + Ip.conj().T) / 2
    Iy = (Ip - Ip.conj().T) / (2j)
    return Ix, Iy, Iz


def system(I, B, a=GAMMA_E * A_FIELD):
    d = int(2 * I + 1)
    e2, eI = np.eye(2), np.eye(d)
    op1 = lambda s: np.kron(np.kron(s, e2), eI)
    op2 = lambda s: np.kron(np.kron(e2, s), eI)
    opI = lambda s: np.kron(np.kron(e2, e2), s)
    Ix, Iy, Iz = spin_ops(I)
    w = GAMMA_E * B
    H = w * (op1(sz) + op2(sz)) + a * (op1(sx) @ opI(Ix) + op1(sy) @ opI(Iy) + op1(sz) @ opI(Iz))
    singlet = np.array([0, 1, -1, 0], complex) / math.sqrt(2)
    PS = np.kron(np.outer(singlet, singlet.conj()), eI)
    deph = [op1(2 * sz), op2(2 * sz)]          # sigma_z of each electron
    return H, PS, deph, d, singlet


def liouvillian(H, deph, r):
    n = H.shape[0]
    I = np.eye(n)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for Z in deph:
        if r > 0:
            c = math.sqrt(r / 2) * Z
            Lsup += (np.kron(c.conj(), c) - 0.5 * np.kron(I, c.conj().T @ c) - 0.5 * np.kron((c.conj().T @ c).T, I))
    return Lsup


def vec(m):
    return m.flatten(order="F")


def unvec(v, n):
    return v.reshape(n, n, order="F")


def initial(I, m_index, singlet):
    d = int(2 * I + 1)
    nuc = np.zeros(d, complex); nuc[m_index] = 1
    psi = np.kron(singlet, nuc)
    return np.outer(psi, psi.conj())


def singlet_yield(H, PS, deph, r, rho0, k):
    n = H.shape[0]
    Lsup = liouvillian(H, deph, r)
    x = solve(k * np.eye(n * n) - Lsup, vec(rho0))
    return float(np.real(k * np.trace(PS @ unvec(x, n))))


def electron_records(I, B, r, times, a=GAMMA_E * A_FIELD):
    """Electron-pair states (nucleus traced out) at each time, for each alternative m."""
    H, PS, deph, d, singlet = system(I, B, a)
    n = H.shape[0]
    dt = times[1] - times[0]
    if r == 0:
        U = expm(-1j * H * dt)
        step = lambda rho: U @ rho @ U.conj().T
    else:
        P = expm(liouvillian(H, deph, r) * dt)
        step = lambda rho: unvec(P @ vec(rho), n)
    recs = []
    for mi in range(d):
        rho = initial(I, mi, singlet)
        series = []
        for _ in times:
            series.append(L.partial_trace(rho, [2, 2, d], [0, 1]))
            rho = step(rho)
        recs.append(series)
    return recs, H, PS


def record_measures(recs, t_index):
    d = len(recs)
    states = [recs[m][t_index] for m in range(d)]
    if d == 1:
        return 1.0, 1.0, 0.0, 0.0
    Js, Ds = [], []
    for i in range(d):
        for j in range(i + 1, d):
            J = float(L.fidelity(states[i], states[j])); D = float(L.trace_distance(states[i], states[j]))
            L.check_pair(J, D, tol=1e-7)
            Js.append(J); Ds.append(D)
    chi = L.holevo(np.full(d, 1 / d), states)
    return float(np.mean(Js)), float(np.min(Js)), float(np.mean(Ds)), chi


def product_info(phis):
    d = len(phis)
    return L.mutual_information(np.full(d, 1 / d), [[p, 1 - p] for p in phis])


def yields_for(I, B, k, r, a=GAMMA_E * A_FIELD):
    H, PS, deph, d, singlet = system(I, B, a)
    return [singlet_yield(H, PS, deph, r, initial(I, mi, singlet), k) for mi in range(d)]


def checks(out):
    # Larmor frequency in 50 uT
    f = GAMMA_E * 50e-6 / (2 * math.pi)
    L.check(abs(f - 1.4e6) / 1.4e6 < 0.01, f"Larmor frequency at 50 uT: {f}")
    # I = 0, r = 0: singlet never changes
    for B in B_LIST:
        for k in (1e4, 1e7):
            phis = yields_for(Fraction(0), B, k, 0.0)
            L.check(abs(phis[0] - 1) < 1e-10, "I = 0: Phi_S = 1")
    recs, H, PS = electron_records(Fraction(0), 1e-3, 0.0, np.linspace(0, 1e-7, 5))
    singlet = np.array([0, 1, -1, 0], complex) / math.sqrt(2)
    for st in recs[0]:
        L.check(abs(np.real(singlet.conj() @ st @ singlet) - 1) < 1e-10, "I = 0: the pair stays singlet")
    # trace preserved (r = 0)
    recs, H, PS = electron_records(Fraction(3, 2), 2e-3, 0.0, np.linspace(0, 5e-7, 200))
    for series in recs:
        for st in series:
            L.check(abs(np.trace(st).real - 1) < 1e-10, "trace preserved")
    # k -> infinity: nothing has time to be written
    for I in (Fraction(1, 2), Fraction(3, 2)):
        phis = yields_for(I, 1e-3, 1e14, 0.0)
        L.check(max(abs(p - 1) for p in phis) < 1e-6 and product_info(phis) < 1e-9, "k -> infinity: no effect, no information")
    # high-field, slow-reaction limit for one spin-1/2 nucleus: S-T0 mixing only, Phi_S -> 1/2
    phis = yields_for(Fraction(1, 2), 1.0, 1e3, 0.0)
    L.check(abs(np.mean(phis) - 0.5) < 2e-3, f"high-field slow-reaction limit Phi_S = 1/2, got {np.mean(phis)}")
    # low-field effect: Phi_S(B) is not monotonic below about 2a
    Bs = np.linspace(0, 3e-3, 61)
    ys = [np.mean(yields_for(Fraction(1, 2), B, 1e6, 0.0)) for B in Bs]
    dif = np.diff(ys)
    L.check(np.any(dif > 1e-6) and np.any(dif < -1e-6), "low-field effect: Phi_S(B) changes direction at low field")
    out.say(f"One spin-1/2 nucleus, k = 1e6 /s: Phi_S = {ys[0]:.4f} at B = 0, extreme {min(ys):.4f} at B = {Bs[int(np.argmin(ys))]*1e3:.2f} mT, "
            f"{ys[-1]:.4f} at 3 mT (low-field effect: the yield first moves one way, then the other)")
    # QuTiP cross-check with relaxation
    import qutip as qt
    I, B, r = Fraction(1, 2), 1e-3, 1e6
    H, PS, deph, d, singlet = system(I, B)
    rho0 = initial(I, 0, singlet)
    times = np.linspace(0, 2e-7, 21)
    c_ops = [qt.Qobj(math.sqrt(r / 2) * Z) for Z in deph]
    res = qt.mesolve(qt.Qobj(H), qt.Qobj(rho0), times, c_ops, options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 100000})
    Lsup = liouvillian(H, deph, r)
    err = max(float(np.max(np.abs(unvec(expm(Lsup * t) @ vec(rho0), 8) - s.full()))) for t, s in zip(times, res.states))
    L.check(err < 1e-6, f"QuTiP vs hand-built Lindblad: {err}")
    out.say(f"QuTiP mesolve and the hand-built Lindblad superoperator agree to {err:.1e} (I = 1/2, B = 1 mT, r = 1e6 /s).")
    out.say("Checks passed: I = 0 keeps the singlet (Phi_S = 1, J = 1, E = 0); trace preserved; 50 uT gives 1.4 MHz; "
            "k -> infinity removes the effect; high-field slow limit Phi_S = 1/2; low-field effect present.")


def main():
    out = L.Output("toy6", "Toy 6: radical pair")
    out.flag("electron relaxation", "L_i = sqrt(r/2) sigma_z for each electron, so each electron's coherence decays at rate r",
             "the spec says 'dephasing at rate r' without fixing the operator")
    out.flag("anisotropic hyperfine", "not run", "optional in the spec")
    out.flag("pairwise join for I >= 1", "J reported as the mean and the minimum over pairs of m values",
             "with more than two alternatives there is no single pairwise J")
    out.flag("xenon hyperfine constants", f"a(129Xe) = 1 mT taken as the base; a(131Xe) = 1 mT x |gamma(131Xe)/gamma(129Xe)| = "
             f"{abs(GAMMA_131XE / GAMMA_129XE):.3f} mT; gammas {GAMMA_129XE:.4g} and {GAMMA_131XE:.4g} rad/s/T from memory",
             "the spec asks the coder to scale with the nuclear moments and flag; signs dropped (isotropic coupling to one nucleus)")
    out.flag("record time grid", "0 to 5 us, 2,001 points (2.5 ns): resolves the hyperfine period (36 ns at 1 mT)", "not fixed by the spec")
    out.flag("reaction-weighted records", "J and chi averaged with weight k e^{-kt} on the 5 us grid, renormalized on the grid",
             "for k below about 1e6 /s part of the weight lies beyond 5 us")
    out.flag("P3 threshold grid", "k extended to 1e4..1e13 /s (46 points) for the threshold only",
             "on the spec's grid (up to 1e9 /s, k/a = 5.7) the 1% threshold is not reached for most I and B")
    checks(out)
    a = GAMMA_E * A_FIELD

    # 1. records over time
    times = np.linspace(0, 5e-6, 2001)
    rows = []
    for I in (Fraction(1, 2), Fraction(3, 2)):
        for B in B_LIST:
            recs, _, _ = electron_records(I, B, 0.0, times)
            idx = list(range(0, len(times), 2))
            series = [record_measures(recs, ti) for ti in idx]
            Js = np.array([m[0] for m in series])
            # rejoinings: local minima of J followed by a rise of at least 0.05
            minima = [idx[i] for i in range(1, len(Js) - 1) if Js[i] < Js[i - 1] and Js[i] <= Js[i + 1]
                      and Js[i + 1:i + 40].max(initial=Js[i]) - Js[i] >= 0.05]
            for n, ti in enumerate(idx):
                if ti % 40 == 0:
                    Jm, Jmin, Dm, chi = series[n]
                    rows.append([str(I), B * 1e3, times[ti] * 1e9, Jm, Jmin, Dm, chi])
            first = ", ".join(f"{times[t]*1e9:.0f}" for t in minima[:3])
            out.say(f"I = {I}, B = {B*1e3:g} mT: J min {Js.min():.3f}; records come undone (J rises by >= 0.05 after a minimum) "
                    f"{len(minima)} times in 5 us, first at {first} ns")
    out.table("1_records_over_time", {**HEADER_BASE, "channel": "electron pair at time t (nucleus traced out)", "r": 0,
                                      "a": "1 mT", "time": "0 to 5 us"},
              ["I", "B (mT)", "t (ns)", "mean J over m pairs", "min J", "mean D", "chi bits"], rows)

    # 2-4. inheritance into chemistry, isotope effect, P3 threshold
    rows2, rows3, rows4 = [], [], []
    for I in I_LIST:
        for B in B_LIST:
            dphis, Es = [], []
            for k in K_LIST:
                phis = yields_for(I, B, k, 0.0)
                E = product_info(phis)
                L.check(E <= math.log2(len(phis)) + 1e-12, "information into chemistry exceeds H(prior)")
                dPhi = float(np.mean(phis)) - 1.0
                rows2.append([str(I), B * 1e3, k, k / a, E])
                rows3.append([str(I), B * 1e3, k / a, float(np.mean(phis)), dPhi])
                dphis.append(abs(dPhi))
            # threshold on an extended grid (the spec's grid stops at 1e9 /s)
            kx = np.logspace(4, 13, 46)
            dx = [abs(np.mean(yields_for(I, B, k, 0.0)) - 1.0) for k in kx]
            thr = next((kx[i] / a for i in range(len(kx)) if all(x < 0.01 for x in dx[i:])), float("nan"))
            rows4.append([str(I), B * 1e3, thr])
    out.table("2_inheritance_into_chemistry", {**HEADER_BASE, "channel": "chemical product, one classical bit", "r": 0},
              ["I", "B (mT)", "k (1/s)", "k/a", "E(m -> product) bits"], rows2)
    out.table("3_isotope_effect", {**HEADER_BASE, "channel": "singlet yield averaged over m", "r": 0,
                                   "reference": "Phi_S(I = 0) = 1 at every B"},
              ["I", "B (mT)", "k/a", "Phi_S(I)", "Delta Phi_S = Phi_S(I) - Phi_S(0)"], rows3)
    out.table("4_P3_threshold", {**HEADER_BASE, "criterion": "|Delta Phi_S| < 1% for all larger k on the grid", "r": 0},
              ["I", "B (mT)", "threshold k/a"], rows4,
              note="I = 0 has no isotope effect by construction (threshold at the bottom of the grid).")
    for I, Bm, thr in rows4:
        if Bm in (0.0, 0.05, 5.0) and I != "0":
            out.say(f"P3 threshold, I = {I}, B = {Bm:g} mT: isotope effect below 1% for k/a above {thr:.3g} (k above {thr*a:.3g} /s)")
    # field shift of the isotope difference at k = 1e6 /s
    for I in (Fraction(1, 2), Fraction(3, 2)):
        d0 = np.mean(yields_for(I, 0.0, 1e6, 0.0)) - 1
        d2 = np.mean(yields_for(I, 2e-3, 1e6, 0.0)) - 1
        out.say(f"Field shift of the isotope effect, I = {I}, k = 1e6 /s: Delta Phi_S = {d0:+.4f} at 0 mT, {d2:+.4f} at 2 mT")

    # xenon variant
    rows = []
    ratio = abs(GAMMA_131XE / GAMMA_129XE)
    for name, I, aval in (("132Xe/134Xe", Fraction(0), 0.0), ("129Xe", Fraction(1, 2), 1.0), ("131Xe", Fraction(3, 2), ratio)):
        for B in (0.0, 0.05e-3, 2e-3):
            for k in (1e5, 1e6, 1e7, 1e8):
                phis = yields_for(I, B, k, 0.0, a=GAMMA_E * 1e-3 * aval) if aval > 0 else [1.0]
                rows.append([name, str(I), aval, B * 1e3, k, float(np.mean(phis)), float(np.mean(phis)) - 1.0, product_info(phis) if len(phis) > 1 else 0.0])
    out.table("4b_xenon", {**HEADER_BASE, "channel": "singlet yield", "a": "scaled by |gamma_n|; base a(129Xe) = 1 mT (flagged)", "r": 0},
              ["isotope", "I", "a (mT)", "B (mT)", "k (1/s)", "Phi_S", "Delta Phi_S", "E bits"], rows,
              note="Compare with the radical-pair model of xenon anaesthesia, arXiv:2204.09147, which this toy does not reproduce.")

    # 5. relaxation
    rows = []
    for r in R_LIST:
        for I in (Fraction(1, 2), Fraction(3, 2)):
            for B in (0.05e-3, 2e-3):
                for k in (1e5, 1e6, 1e7):
                    phis = yields_for(I, B, k, r)
                    rows.append([r, str(I), B * 1e3, k, float(np.mean(phis)), float(np.mean(phis)) - 1.0, product_info(phis)])
    out.table("5_relaxation", {**HEADER_BASE, "channel": "chemical product", "relaxation": "L_i = sqrt(r/2) sigma_z (flagged)"},
              ["r (1/s)", "I", "B (mT)", "k (1/s)", "Phi_S", "Delta Phi_S", "E bits"], rows,
              note="With r > 0 the I = 0 reference is no longer 1, so Delta Phi_S here is against 1, not against Phi_S(I = 0).")
    for r in R_LIST:
        sel = [x for x in rows if x[0] == r and x[1] == "1/2" and x[2] == 0.05 and x[3] == 1e6]
        out.say(f"r = {r:g} /s, I = 1/2, B = 0.05 mT, k = 1e6 /s: E(m -> product) = {sel[0][6]:.4f} bits")

    # reaction-weighted records for one setting
    rows = []
    for I in (Fraction(1, 2), Fraction(3, 2)):
        recs, _, _ = electron_records(I, 0.05e-3, 0.0, times)
        idx = list(range(0, len(times), 10))
        meas = [record_measures(recs, i) for i in idx]
        tt = times[idx]
        for k in (1e6, 1e7, 1e8):
            w = k * np.exp(-k * tt); w = w / w.sum()
            rows.append([str(I), k, float(np.sum(w * [m[0] for m in meas])), float(np.sum(w * [m[3] for m in meas]))])
    out.table("1b_reaction_weighted", {**HEADER_BASE, "channel": "electron pair, weighted by k e^{-kt}", "B": "0.05 mT", "r": 0},
              ["I", "k (1/s)", "weighted mean J", "weighted chi bits"], rows)
    out.finish()


if __name__ == "__main__":
    main()
