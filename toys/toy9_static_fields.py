"""Toy 9: records left by static fields (companion revision 4.6, the event rule).

Revision 4.6: "Whether an interaction is an event depends on the record it leaves, not on whether its quantum
is real or virtual. A static field records where it differs between the alternatives and something registers
the difference: the Coulomb force entangles two charges held in superposition. Where the field is the same
across the alternatives, as inside a bound state, it joins."

No photon is emitted in any case here; every record is written by a static potential acting for a time tau.

A. Two bodies, each in a two-arm superposition, coupled only by a static potential C / r (Coulomb or
   gravity). Phases phi_ab = -C tau / (hbar r_ab). The question is body 1's arm; the record is body 2.
     phase reader (body 2 also in superposition): J = |cos(dphi / 2)|, dphi = phi_00 - phi_01 - phi_10 + phi_11
     kick reader (body 2 localised, spread sigma_x): J = exp(-dp^2 / (8 sigma_p^2)), sigma_p = hbar / (2 sigma_x)
B. A bound state read from outside. A probe electron in two arms (R +- dx/2 along a fixed direction) reads the
   alternatives' potentials for a time tau. Alternatives:
     identical constituents swapped (the field is identical)
     hydrogen 1s or 2s (both spherical, both neutral: the fields differ only where the charge reaches)
     a neutral, dipole-free pair along z or along x (differ first in the quadrupole)
     a hydrogen-like pair, electron above or below the proton (differ in the dipole)
Every record is tested against the record threshold (D >= D_min in one copy).

Run: python toys/toy9_static_fields.py           (seconds)
     python toys/toy9_static_fields.py --quick
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

HBAR = 1.054571817e-34
KE = 8.9875517923e9
QE = 1.602176634e-19
G_N = 6.67430e-11
A0 = 5.29177210903e-11


# ---------------------------------------------------------------- A: two bodies in superposition
def two_body(C, d1, d2, D, tau):
    """Statevector of two two-arm bodies on a line (arms of body 1 at -+d1/2, of body 2 at D -+ d2/2), after the
    static potential C / r acts for tau. Returns (dphi, J of body 2 as the record of body 1's arm, concurrence)."""
    x1 = np.array([-d1 / 2, d1 / 2])
    x2 = D + np.array([-d2 / 2, d2 / 2])
    phi = -C * tau / (HBAR * np.abs(x2[None, :] - x1[:, None]))
    psi = 0.5 * np.exp(1j * phi)
    rec = [psi[a] / np.linalg.norm(psi[a]) for a in (0, 1)]
    J = L.fidelity(np.outer(rec[0], rec[0].conj()), np.outer(rec[1], rec[1].conj()))
    conc = 2 * abs(psi[0, 0] * psi[1, 1] - psi[0, 1] * psi[1, 0])
    dphi = phi[0, 0] - phi[0, 1] - phi[1, 0] + phi[1, 1]
    return float(dphi), float(J), float(conc)


def kick_reader(C, d1, D, tau, sigma_x):
    """Body 2 localised at D (not in superposition): it records body 1's arm in its momentum kick."""
    F = lambda r: C / r ** 2
    dp = tau * abs(F(D + d1 / 2) - F(D - d1 / 2))
    sp = HBAR / (2 * sigma_x)
    return dp, math.exp(-dp ** 2 / (8 * sp ** 2))


A_CASES = [
    # name, C (J m), d1, d2, D, tau, reader spread (m)
    ("two electrons, D = 10 um", KE * QE * QE, 1e-6, 1e-6, 10e-6, 1e-9, 1e-6),
    ("two electrons, D = 100 um", KE * QE * QE, 1e-6, 1e-6, 100e-6, 1e-9, 1e-6),
    ("two electrons, D = 1 mm", KE * QE * QE, 1e-6, 1e-6, 1e-3, 1e-9, 1e-6),
    ("two electrons, D = 1 mm, tau = 1 us", KE * QE * QE, 1e-6, 1e-6, 1e-3, 1e-6, 1e-6),
    ("electron and positron (attractive), D = 100 um", -KE * QE * QE, 1e-6, 1e-6, 100e-6, 1e-9, 1e-6),
    ("gravity: Bose et al. masses (1e-14 kg, dx = 250 um, d = 450 um, 2.5 s)", -G_N * 1e-28, 250e-6, 250e-6, 450e-6, 2.5, 1e-9),
    ("gravity: same masses, d = 1 mm", -G_N * 1e-28, 250e-6, 250e-6, 1e-3, 2.5, 1e-9),
]


# ---------------------------------------------------------------- B: bound states read from outside
def _sph_potential(n_of_x, x):
    """Potential (units k e / a0) of a proton plus a spherical electron density n(x) (units a0^-3), at x = r / a0:
    Q_out / x - int_x^inf 4 pi x' n dx'. Both terms computed from the outside, so nothing cancels."""
    q_out = quad(lambda y: 4 * math.pi * y * y * n_of_x(y), x, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]
    tail = quad(lambda y: 4 * math.pi * y * n_of_x(y), x, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]
    return q_out / x - tail


N1S = lambda x: math.exp(-2 * x) / math.pi
N2S = lambda x: (2 - x) ** 2 * math.exp(-x) / (32 * math.pi)


def point_potential(charges, r):
    """Potential (units k e / a0) at r (units a0) of point charges [(q in e, position in a0)]."""
    return math.fsum(q / float(np.linalg.norm(r - np.asarray(p, float))) for q, p in charges)   # exact rounding: order-free


Z, X = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
B_CASES = {
    "identical constituents swapped (He-like)": ("points", [(2, (0, 0, 0)), (-1, Z), (-1, -X)], [(2, (0, 0, 0)), (-1, -X), (-1, Z)]),
    "hydrogen 1s or 2s (spherical, neutral)": ("spherical", N1S, N2S),
    "neutral pair along z or along x (quadrupole)": ("points", [(2, (0, 0, 0)), (-1, Z), (-1, -Z)], [(2, (0, 0, 0)), (-1, X), (-1, -X)]),
    "electron above or below the proton (dipole)": ("points", [(1, (0, 0, 0)), (-1, Z)], [(1, (0, 0, 0)), (-1, -Z)]),
}
U_DIR = np.array([1.0, 2.0, 3.0]) / math.sqrt(14)        # avoids the nodes of the dipole and quadrupole differences
DX_A0 = 1.0                                              # probe arm separation, in a0
TAU_B = 1e-11                                            # read time, s


def bound_read(case, R):
    """dphi between the probe's two arms, and J, for the two alternatives of a bound-state case at distance R (a0)."""
    kind, a0_alt, a1_alt = B_CASES[case]
    arms = [(R - DX_A0 / 2) * U_DIR, (R + DX_A0 / 2) * U_DIR]
    if kind == "points":
        dV = [point_potential(a1_alt, r) - point_potential(a0_alt, r) for r in arms]
    else:
        dV = [_sph_potential(a1_alt, float(np.linalg.norm(r))) - _sph_potential(a0_alt, float(np.linalg.norm(r))) for r in arms]
    scale = KE * QE * QE / A0 * TAU_B / HBAR              # probe charge -e times potential (k e / a0), over hbar, times tau
    dphi = abs(dV[1] - dV[0]) * scale
    return dphi, abs(math.cos(dphi / 2))


def slope(case, R1, R2):
    return math.log(bound_read(case, R2)[0] / bound_read(case, R1)[0]) / math.log(R2 / R1)


# ---------------------------------------------------------------- checks
def run_checks(out):
    for C, taus in ((KE * QE * QE, (1e-12, 1e-9)), (-KE * QE * QE, (1e-12, 1e-9)), (-G_N * 1e-28, (1.0, 2.5))):
        for tau in taus:                                  # phases stay below ~1e3 rad, so double precision holds
            dphi, J, conc = two_body(C, 1e-6, 1e-6, 1e-5, tau)
            L.check(abs(J - abs(math.cos(dphi / 2))) < 1e-9, "two bodies: J = |cos(dphi/2)| from the state vector")
            L.check(abs(J * J + conc * conc - 1) < 1e-9, "two bodies: J^2 + C^2 = 1 for pure records")
    L.check(abs(two_body(KE * QE * QE, 1e-6, 1e-6, 1e-4, 1e-9)[1] - two_body(-KE * QE * QE, 1e-6, 1e-6, 1e-4, 1e-9)[1]) < 1e-12,
            "attraction and repulsion leave the same join")
    js = [kick_reader(KE * QE * QE, 1e-6, 1e-4, 1e-9, s)[1] for s in (1e-9, 1e-7, 1e-5)]
    L.check(js[0] >= js[1] >= js[2], "kick reader: a sharper momentum reader (wider in position) cuts more")
    for x in (0.5, 1.0, 3.0, 10.0):
        L.check(abs(_sph_potential(N1S, x) / ((1 / x + 1) * math.exp(-2 * x)) - 1) < 1e-9, "hydrogen 1s potential matches (1/r + 1/a0) e^(-2r/a0)")
    for R in (2.0, 50.0, 1e4):
        L.check(bound_read("identical constituents swapped (He-like)", R)[0] == 0.0, "identical constituents: the field is identical, J = 1")
    L.check(abs(slope("electron above or below the proton (dipole)", 300, 3000) + 3) < 0.05, "dipole difference: dphi falls as R^-3")
    L.check(abs(slope("neutral pair along z or along x (quadrupole)", 300, 3000) + 4) < 0.05, "quadrupole difference: dphi falls as R^-4")
    L.check(bound_read("hydrogen 1s or 2s (spherical, neutral)", 60.0)[0] < 1e-12, "spherical rearrangement: the outside field joins (exponentially)")
    out.say("Checks: J = |cos(dphi/2)| and J^2 + C^2 = 1 from the state vector; attraction and repulsion join alike; the 1s potential "
            "matches its closed form; identical constituents leave the field unchanged; dipole and quadrupole differences fall as "
            "R^-3 and R^-4 at the probe; the 1s/2s difference vanishes exponentially outside the atom.")


# ---------------------------------------------------------------- main
def main(quick=False):
    out = L.Output("toy9", "Toy 9: records left by static fields")
    out.flag("static potential only", "phases phi = -C tau / (hbar r) from a static C / r for a time tau; no radiation, no retardation, "
             "no motion of the arms during tau", "tests revision 4.6's rule that an interaction counts by its record, real or virtual quanta")
    out.flag("phase reader geometry", "both bodies on one line, arms separated d1 and d2, centres D apart", "the simplest geometry; "
             "Bose et al. used a similar collinear arrangement")
    out.flag("gravity parameters", "m = 1e-14 kg, arm separation 250 um, centre distance 450 um, 2.5 s", "from memory of Bose et al. "
             "(2017); check against the paper before quoting")
    out.flag("kick reader", "body 2 localised as a minimum-uncertainty Gaussian of spread sigma_x; impulse approximation (it does not "
             "move during tau); J = exp(-dp^2 / (8 sigma_p^2))", "the same field read by a different reader leaves a different join")
    out.flag("bound-state probe", f"an electron in two arms at R -+ {DX_A0 / 2:g} a0 along (1, 2, 3)/sqrt14, read time {TAU_B:g} s; "
             "point-charge models except hydrogen 1s/2s (exact densities)", "the direction avoids the nodes of the dipole and "
             "quadrupole differences; tau sets only the scale of dphi")
    out.flag("record threshold", f"D = sqrt(1 - J^2) (pure records) against D_min = {L.D_MIN}, one copy (R_min = 1)",
             "line (0) of revision 4.6; the probe is the only reader here")
    out.flag("B11 hook", "atlas benchmark B11 (image-charge decoherence of electrons over a surface, Sonnentag & Hasselbach 2007, "
             "from memory) would calibrate these readers against a measured static-field cut", "waiting for the paper's data")
    run_checks(out)

    rows = []
    for name, C, d1, d2, D, tau, sx in A_CASES:
        dphi, J, conc = two_body(C, d1, d2, D, tau)
        dp, Jk = kick_reader(C, d1, D, tau, sx)
        Dd = math.sqrt(max(0.0, 1 - J * J))
        rows.append([name, D, tau, dphi, J, conc, Dd, "yes" if L.is_event(Dd, 1, L.D_MIN, 1) else "no", sx, Jk])
    out.table("1_two_bodies", {"event": "a static potential acting for tau between two bodies",
                               "question": "which arm body 1 is in", "prior": "uniform",
                               "records": "body 2 (phase reader in superposition; kick reader localised)"},
              ["case", "D (m)", "tau (s)", "dphi (rad)", "J, phase reader", "concurrence", "D = sqrt(1 - J^2)",
               "event (D >= D_min, 1 copy)", "kick reader sigma_x (m)", "J, kick reader"], rows,
              note="No photon is emitted; the record is written by the static field alone.")
    for r in rows:
        out.say(f"{r[0]}: dphi = {r[3]:.3g} rad, J = {r[4]:.4f} (phase reader), {r[9]:.4f} (kick reader); event: {r[7]}")

    Rs = [1.5, 2, 3, 5, 10, 20, 50, 100, 300, 1000] if quick else [1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 300, 1000, 3000, 1e4]
    rows = []
    for case in B_CASES:
        for R in Rs:
            dphi, J = bound_read(case, R)
            Dd = math.sqrt(max(0.0, 1 - J * J))
            rows.append([case, R, dphi, J, Dd, "yes" if L.is_event(Dd, 1, L.D_MIN, 1) else "no"])
    out.table("2_bound_states", {"event": "a probe electron reads a bound state's static field for tau",
                                 "question": "which internal arrangement", "prior": "uniform",
                                 "records": f"the probe's two arms (dx = {DX_A0:g} a0)", "tau (s)": TAU_B},
              ["alternatives", "R (a0)", "dphi (rad)", "J", "D", "event (D >= D_min, 1 copy)"], rows,
              note="Inside the bound state the field differs and the probe cuts; outside, the difference falls as a power set by the "
                   "first multipole that differs, or exponentially when only the spherical arrangement differs.")
    for case in B_CASES:
        sel = [r for r in rows if r[0] == case]
        last = next((r[1] for r in reversed(sel) if r[5] == "yes"), None)
        out.say(f"{case}: J = {', '.join(f'{r[3]:.3f}' for r in sel)} at R = {', '.join(f'{r[1]:g}' for r in sel)} a0; "
                f"an event (at tau = {TAU_B:g} s) out to R = {last if last is not None else 'nowhere'} a0")
    out.say(L.SATURATION_NOTE)
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
