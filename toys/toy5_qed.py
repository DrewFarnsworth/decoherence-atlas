"""Toy 5: Compton and electron-electron scattering (tree-level QED, natural units, m_e = 1).

Part a: helicity amplitudes built numerically from Dirac spinors (Weyl representation) and photon
        helicity vectors, for Moller, Bhabha, e+e- -> mu+mu- and Compton. Concurrence of the
        outgoing helicity state; Moller's t/u interference fraction X(theta); Compton's
        concurrence in the centre-of-mass and lab frames.
Part b: the soft-photon cut. J_phys(lambda) = (lambda/Lambda)^A with
        A = alpha/(8 pi^2) * Integral dOmega [-(J.J)],  J^mu = p1^mu/(p1.n) - p2^mu/(p2.n),  n = (1, n_hat).
        Slow limit A = alpha |v1 - v2|^2 / (3 pi).
Part c: frame dependence of the electron's energy change in Compton scattering.

Run: python toys/toy5_qed.py           (full)
     python toys/toy5_qed.py --quick   (checks and a few points)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

ALPHA = 1 / 137.035999
E2 = 4 * math.pi * ALPHA
M_E = 1.0
M_MU = 206.768283
HBAR_EV_S = 6.582119569e-16
ME_EV = 0.51099895e6
METRIC = np.diag([1.0, -1, -1, -1])

s0 = np.eye(2, dtype=complex)
sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex)]
Z2 = np.zeros((2, 2), complex)
G0 = np.block([[Z2, s0], [s0, Z2]])
GAMMA = [G0] + [np.block([[Z2, s], [-s, Z2]]) for s in sig]


def slash(p):
    return p[0] * GAMMA[0] - p[1] * GAMMA[1] - p[2] * GAMMA[2] - p[3] * GAMMA[3]


def dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def four(m, pvec):
    pvec = np.asarray(pvec, float)
    return np.concatenate([[math.sqrt(m * m + pvec @ pvec)], pvec])


def angles(pvec):
    pn = np.linalg.norm(pvec)
    if pn == 0:
        return 0.0, 0.0
    th = math.acos(max(-1, min(1, pvec[2] / pn)))
    ph = math.atan2(pvec[1], pvec[0])
    return th, ph


def xi(pvec, lam):
    th, ph = angles(pvec)
    if lam > 0:
        return np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)])
    return np.array([-np.exp(-1j * ph) * math.sin(th / 2), math.cos(th / 2)])


def u_spinor(p, lam):
    E, pv = p[0], p[1:]
    pn = np.linalg.norm(pv)
    x = xi(pv if pn > 0 else np.array([0, 0, 1.0]), lam)
    return np.concatenate([math.sqrt(max(E - lam * pn, 0)) * x, math.sqrt(E + lam * pn) * x])


def v_spinor(p, lam):
    E, pv = p[0], p[1:]
    pn = np.linalg.norm(pv)
    x = xi(pv if pn > 0 else np.array([0, 0, 1.0]), -lam)
    return np.concatenate([math.sqrt(E + lam * pn) * x, -math.sqrt(max(E - lam * pn, 0)) * x])


def bar(s):
    return s.conj() @ G0


def eps(k, lam):
    """Photon helicity polarization vector (contravariant)."""
    th, ph = angles(k[1:])
    t = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
    f = np.array([-math.sin(ph), math.cos(ph), 0.0])
    e = -lam * (t + 1j * lam * f) / math.sqrt(2)
    return np.concatenate([[0], e])


def current(a, b):
    """a-bar gamma^mu b, contravariant components."""
    return np.array([bar(a) @ GAMMA[mu] @ b for mu in range(4)])


def cdot(j1, j2):
    return j1[0] * j2[0] - j1[1] * j2[1] - j1[2] * j2[2] - j1[3] * j2[3]


H = (1, -1)


# ---------------------------------------------------------------- processes in the CM frame
def cm_momenta(m_in, m_out, sqrt_s, theta):
    E = sqrt_s / 2
    pin = math.sqrt(E * E - m_in * m_in)
    pout = math.sqrt(max(E * E - m_out * m_out, 0))
    p1 = four(m_in, [0, 0, pin]); p2 = four(m_in, [0, 0, -pin])
    n = np.array([math.sin(theta), 0, math.cos(theta)])
    p3 = four(m_out, pout * n); p4 = four(m_out, -pout * n)
    return p1, p2, p3, p4


def moller(p1, p2, p3, p4, l1, l2, l3, l4, parts=False):
    u1, u2, u3, u4 = u_spinor(p1, l1), u_spinor(p2, l2), u_spinor(p3, l3), u_spinor(p4, l4)
    t, u = dot(p1 - p3, p1 - p3), dot(p1 - p4, p1 - p4)
    Mt = E2 * cdot(current(u3, u1), current(u4, u2)) / t
    Mu = E2 * cdot(current(u4, u1), current(u3, u2)) / u
    return (Mt, Mu) if parts else Mt - Mu


def bhabha(p1, p2, p3, p4, l1, l2, l3, l4):
    """e-(p1) e+(p2) -> e-(p3) e+(p4)."""
    u1, v2, u3, v4 = u_spinor(p1, l1), v_spinor(p2, l2), u_spinor(p3, l3), v_spinor(p4, l4)
    s, t = dot(p1 + p2, p1 + p2), dot(p1 - p3, p1 - p3)
    return E2 * (-cdot(current(u3, u1), current(v2, v4)) / t + cdot(current(u3, v4), current(v2, u1)) / s)


def eemumu(p1, p2, p3, p4, l1, l2, l3, l4):
    u1, v2, u3, v4 = u_spinor(p1, l1), v_spinor(p2, l2), u_spinor(p3, l3), v_spinor(p4, l4)
    s = dot(p1 + p2, p1 + p2)
    return E2 * cdot(current(u3, v4), current(v2, u1)) / s


def compton(p, k, pp, kp, le, lg, lep, lgp):
    """gamma(k, lg) e(p, le) -> gamma(kp, lgp) e(pp, lep)."""
    u, ub = u_spinor(p, le), bar(u_spinor(pp, lep))
    e1, e2 = slash(eps(k, lg)), slash(eps(kp, lgp).conj())
    I4 = np.eye(4)
    A = e2 @ (slash(p + k) + M_E * I4) @ e1 / (2 * dot(p, k))
    B = e1 @ (slash(p - kp) + M_E * I4) @ e2 / (-2 * dot(p, kp))
    return E2 * (ub @ (A + B) @ u)


def concurrence(M):
    """M[a, b] amplitudes for outgoing helicities (+, -) x (+, -)."""
    n = np.sum(np.abs(M) ** 2)
    if n == 0:
        return float("nan")
    return float(2 * abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) / n)


def out_matrix(fn, moms, lin):
    return np.array([[fn(*moms, *lin, a, b) for b in H] for a in H])


def spin_sum(fn, moms):
    return sum(abs(fn(*moms, a, b, c, d)) ** 2 for a in H for b in H for c in H for d in H)


# ---------------------------------------------------------------- Compton kinematics
def compton_lab(Eg, theta):
    Ep = Eg / (1 + Eg / M_E * (1 - math.cos(theta)))
    k = np.array([Eg, 0, 0, Eg]); p = np.array([M_E, 0, 0, 0.0])
    kp = Ep * np.array([1, math.sin(theta), 0, math.cos(theta)])
    pp = k + p - kp
    return p, k, pp, kp


def boost_z(v, beta):
    g = 1 / math.sqrt(1 - beta * beta)
    return np.array([g * (v[0] - beta * v[3]), v[1], v[2], g * (v[3] - beta * v[0])])


def compton_cm(Eg, theta_lab):
    p, k, pp, kp = compton_lab(Eg, theta_lab)
    beta = (k[3] + p[3]) / (k[0] + p[0])
    return [boost_z(x, beta) for x in (p, k, pp, kp)], beta


# ---------------------------------------------------------------- part b
def soft_factor(v1, v2, n_theta=400, n_phi=200):
    """A = alpha/(8 pi^2) * Integral dOmega [-(J.J)], J = p1/(p1.n) - p2/(p2.n), per unit mass (velocities)."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    g1, g2 = 1 / math.sqrt(1 - v1 @ v1), 1 / math.sqrt(1 - v2 @ v2)
    P1, P2 = g1 * np.concatenate([[1], v1]), g2 * np.concatenate([[1], v2])
    th = (np.arange(n_theta) + 0.5) * math.pi / n_theta
    ph = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
    T, F = np.meshgrid(th, ph, indexing="ij")
    nx, ny, nz = np.sin(T) * np.cos(F), np.sin(T) * np.sin(F), np.cos(T)
    def comp(P):
        pn = P[0] - (P[1] * nx + P[2] * ny + P[3] * nz)
        return [P[i] / pn for i in range(4)]
    a, b = comp(P1), comp(P2)
    J = [a[i] - b[i] for i in range(4)]
    JJ = J[0] ** 2 - J[1] ** 2 - J[2] ** 2 - J[3] ** 2
    integral = np.sum(-JJ * np.sin(T)) * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return ALPHA / (8 * math.pi ** 2) * integral


# ---------------------------------------------------------------- checks
def checks(out):
    # massless-limit cross sections at sqrt(s) = 2000 m_e, theta = 1 rad
    rs, th = 2000.0, 1.0
    for name, fn, m_out, formula in (
        ("Moller", moller, M_E, lambda s, t, u: 2 * E2 ** 2 * ((s * s + u * u) / t ** 2 + (s * s + t * t) / u ** 2 + 2 * s * s / (t * u))),
        ("Bhabha", bhabha, M_E, lambda s, t, u: 2 * E2 ** 2 * ((s * s + u * u) / t ** 2 + (t * t + u * u) / s ** 2 + 2 * u * u / (s * t))),
        ("ee->mumu", eemumu, M_MU, lambda s, t, u: 2 * E2 ** 2 * (t * t + u * u) / s ** 2),
    ):
        moms = cm_momenta(M_E, m_out, rs if name != "ee->mumu" else 2e5, th)
        s = dot(moms[0] + moms[1], moms[0] + moms[1]); t = dot(moms[0] - moms[2], moms[0] - moms[2]); u = dot(moms[0] - moms[3], moms[0] - moms[3])
        got = spin_sum(fn, moms) / 4
        exp = formula(s, t, u)
        L.check(abs(got / exp - 1) < 2e-3, f"{name}: spin-averaged |M|^2 {got} vs massless formula {exp}")
    # Klein-Nishina (Peskin 5.87), exact with mass
    for Eg, th in ((0.01, 0.7), (1.0, 2.0), (20.0, 0.3)):
        p, k, pp, kp = compton_lab(Eg, th)
        L.check(abs(kp[0] - Eg / (1 + Eg * (1 - math.cos(th)))) < 1e-12, "Compton kinematics")
        L.check(abs(dot(pp, pp) - 1) < 1e-9, "outgoing electron on shell")
        got = sum(abs(compton(p, k, pp, kp, a, b, c, d)) ** 2 for a in H for b in H for c in H for d in H) / 4
        pk, pkp = dot(p, k), dot(p, kp)
        exp = 2 * E2 ** 2 * (pkp / pk + pk / pkp + 2 * (1 / pk - 1 / pkp) + (1 / pk - 1 / pkp) ** 2)
        L.check(abs(got / exp - 1) < 1e-9, f"Klein-Nishina: {got} vs {exp}")
    # Moller from |RL> at 90 degrees, high energy: t- and u-channel outcomes equal in size -> maximal entanglement
    moms = cm_momenta(M_E, M_E, 2000.0, math.pi / 2)
    C = concurrence(out_matrix(moller, moms, (1, -1)))
    L.check(abs(C - 1) < 1e-3, f"Moller RL at 90 degrees: concurrence {C}, expected 1 (indistinguishable t and u channels)")
    # e+e- -> mu+mu-, incoming |RL> at high energy: the s-channel photon makes RL and LR with amplitudes
    # proportional to (1 + cos) and (1 - cos), so C = (1 - cos^2)/(1 + cos^2), maximal at 90 degrees
    for th in (1.0, math.pi / 2):
        moms = cm_momenta(M_E, M_MU, 2e5, th)
        C = concurrence(out_matrix(eemumu, moms, (1, -1)))
        c = math.cos(th)
        L.check(abs(C - (1 - c * c) / (1 + c * c)) < 2e-3, f"ee->mumu: concurrence {C} vs (1 - cos^2)/(1 + cos^2)")
    # soft factor: slow limit and frame-independence of the angular integral
    for dv in (1e-3, 1e-2):
        A = soft_factor([0, 0, 0], [0, 0, dv])
        L.check(abs(A / (ALPHA * dv * dv / (3 * math.pi)) - 1) < 1e-3, f"soft factor slow limit at dv = {dv}: {A}")
    out.say("Checks passed: Moller, Bhabha and e+e- -> mu+mu- reproduce the massless spin-averaged formulas; Compton "
            "reproduces Klein-Nishina exactly; Compton kinematics; Moller |RL> at 90 deg is maximally entangled; "
            "e+e- -> mu+mu- gives C = (1 - cos^2)/(1 + cos^2), the s-channel mechanism; the soft factor matches alpha dv^2/(3 pi).")


def main(quick=False):
    out = L.Output("toy5", "Toy 5: Compton and electron-electron scattering")
    out.flag("units and couplings", "m_e = 1, e^2 = 4 pi alpha; amplitudes at tree level only", "natural units as the spec asks")
    out.flag("incoming helicities", "Moller, Bhabha, e+e- -> mu+mu-: e- right-handed, second particle left-handed (|RL>); "
             "Compton: photon +, electron + along its own momentum", "the spec asks for definite incoming helicities without naming them")
    out.flag("Cervera-Lierta comparison", "checked against two closed forms (Moller |RL> maximal at 90 deg; e+e- -> mu+mu- "
             "C = (1 - cos^2)/(1 + cos^2) at high energy), not against their published curves",
             "the published curves were not retrieved; compare 1_concurrence.csv with their figures")
    out.flag("lab-frame helicity", "helicity along each particle's momentum in the frame in use", "standard; the spec asks for both frames")
    out.flag("soft factor normalization", "derived from coherent-state overlap exp(-1/2 |alpha - beta|^2); agrees with the spec's slow limit",
             "Carney et al. 2017 was not re-read for the prefactor")
    out.flag("soft cutoff Lambda", "kinetic energy of the faster outcome, m (gamma - 1), with the slower at rest", "the spec says energy transferred, flagged")
    out.flag("T_half", "reported as log10 of seconds, since 2^(1/A) overflows", "A is tiny at small velocity differences")
    checks(out)

    # part a: concurrence
    energies = [2.5, 10.0] if quick else list(np.logspace(math.log10(2.05), math.log10(100e6 / ME_EV), 12))
    thetas = [0.5, math.pi / 2] if quick else list(np.linspace(0.05, math.pi - 0.05, 30))
    rows, xrows = [], []
    for rs in energies:
        for th in thetas:
            for name, fn, mo in (("Moller", moller, M_E), ("Bhabha", bhabha, M_E), ("ee->mumu", eemumu, M_MU)):
                if rs <= 2 * mo + 1e-9:
                    continue
                moms = cm_momenta(M_E, mo, rs, th)
                rows.append([name, rs * ME_EV / 1e6, th, concurrence(out_matrix(fn, moms, (1, -1))), concurrence(out_matrix(fn, moms, (1, 1)))])
            moms = cm_momenta(M_E, M_E, rs, th)
            num = den = dist = 0.0
            for a in H:
                for b in H:
                    for c in H:
                        for d in H:
                            Mt, Mu = moller(*moms, a, b, c, d, parts=True)
                            num += 2 * (Mt * np.conj(Mu)).real
                            den += abs(Mt) ** 2 + abs(Mu) ** 2
                            dist += abs(Mt - Mu) ** 2
            xrows.append([rs * ME_EV / 1e6, th, -num / den, dist / den])
    out.table("1_concurrence", {"event": "two-particle scattering", "alternatives": "outgoing helicity configurations",
                                "prior": "none (definite incoming helicities)", "frame": "centre of mass"},
              ["process", "sqrt(s) (MeV)", "theta (rad)", "concurrence, incoming |RL>", "concurrence, incoming |RR>"], rows)
    out.table("2_moller_interference", {"event": "Moller scattering", "alternatives": "which electron went where (t vs u channel)",
                                        "frame": "centre of mass", "sum": "over all helicities"},
              ["sqrt(s) (MeV)", "theta (rad)", "X = -2 Re(Mt Mu*)/(|Mt|^2+|Mu|^2) (identical, M = Mt - Mu)",
               "|Mt - Mu|^2 / (|Mt|^2 + |Mu|^2) (1 for distinguishable)"], xrows)
    sel = [r for r in xrows if abs(r[1] - math.pi / 2) < 0.06]
    if sel:
        out.say(f"Moller at ~90 deg: interference fraction X = {sel[0][2]:.3f} at {sel[0][0]:.2f} MeV, {sel[-1][2]:.3f} at {sel[-1][0]:.3g} MeV")

    # Compton in two frames
    rows = []
    for Eg_keV in ((100.0, 1000.0) if quick else (1.0, 100.0, 1000.0, 10000.0)):
        Eg = Eg_keV * 1e3 / ME_EV
        for th in ((1.0,) if quick else np.linspace(0.1, math.pi - 0.1, 15)):
            lab = compton_lab(Eg, th)
            (cm, beta) = compton_cm(Eg, th)
            Ml = np.array([[compton(*lab, 1, 1, a, b) for b in H] for a in H])
            Mc = np.array([[compton(*cm, 1, 1, a, b) for b in H] for a in H])
            rows.append([Eg_keV, th, concurrence(Ml), concurrence(Mc), beta])
    out.table("3_compton_frames", {"event": "Compton scattering", "alternatives": "outgoing electron and photon helicities",
                                   "incoming": "photon +, electron + (along its own momentum)", "theta": "lab-frame photon angle"},
              ["E_gamma (keV)", "theta lab (rad)", "concurrence, lab frame", "concurrence, CM frame", "CM boost beta"], rows,
              note="Helicity of a massive particle depends on the frame, so the join itself can differ between frames.")
    diffs = [abs(r[2] - r[3]) for r in rows]
    out.say(f"Compton: largest difference between lab and CM concurrence = {max(diffs):.3f}")

    # part b: soft-photon cut
    rows = []
    for dv in (1e-6, 1e-3, 0.1, 0.9):
        A = soft_factor([0, 0, 0], [0, 0, dv]) if dv < 0.95 else float("nan")
        g = 1 / math.sqrt(1 - dv * dv)
        Lam = ME_EV * (g - 1)                       # eV
        log10_Thalf = math.log10(HBAR_EV_S / Lam) + math.log10(2) / A
        for T in (1e-15, 1e-9, 1.0, 1e9, 1e17):
            lam = HBAR_EV_S / T
            logJ = A * math.log10(min(lam / Lam, 1.0))
            for Eth in (1e-6, 1e-3, 1.0, 1e3):
                logJX = A * math.log10(min(Eth / Lam, 1.0))
                rows.append([dv, A, Lam, T, logJ, Eth, logJX, log10_Thalf])
        out.say(f"Soft photons, |dv|/c = {dv:g}: A = {A:.3e}, Lambda = {Lam:.3g} eV, log10 T_half (s) = {log10_Thalf:.4g}")
    out.table("4_soft_photon_cut", {"event": "an electron leaves with velocity v1 = 0 or v2", "alternatives": "the two velocities",
                                    "prior": "uniform", "channel": "J_phys: photons above hbar/T; J^X: photons above E_th"},
              ["|v1 - v2|/c", "A", "Lambda (eV)", "T (s)", "log10 J_phys(T)", "E_th (eV)", "log10 J^X", "log10 T_half (s)"], rows)

    # part c: frame table
    rows = []
    for Eg_keV in (1.0, 100.0, 1000.0, 10000.0):
        Eg = Eg_keV * 1e3 / ME_EV
        for th in ((math.pi / 2,) if quick else np.linspace(0.0, math.pi, 7)):
            p, k, pp, kp = compton_lab(Eg, th)
            (cm, bcm) = compton_cm(Eg, th)
            dK_lab = (pp[0] - p[0]) * ME_EV
            dK_cm = (cm[2][0] - cm[0][0]) * ME_EV
            prev = None
            for beta in np.linspace(-0.99, 0.99, 23):
                dK = (boost_z(pp, beta)[0] - boost_z(p, beta)[0]) * ME_EV
                flip = prev is not None and np.sign(dK) != np.sign(prev) and abs(dK) > 1e-9
                rows.append([Eg_keV, th, beta, dK, dK_lab, dK_cm, 0.0, "sign flip" if flip else ""])
                prev = dK
    out.table("5_frame_table", {"event": "Compton scattering on an electron at rest (lab)", "quantity": "electron energy change"},
              ["E_gamma (keV)", "theta lab (rad)", "boost beta along beam", "dE electron in boosted frame (eV)",
               "dE lab (eV)", "dE CM (eV)", "change in invariant mass", "mark"], rows,
              note="Reported, not resolved: which change of tension is valence, and in which frame.")
    out.say(f"Frame table: {sum(1 for r in rows if r[-1])} sign flips of the electron's energy change across boosts; "
            "the CM change is 0 for elastic scattering.")
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
