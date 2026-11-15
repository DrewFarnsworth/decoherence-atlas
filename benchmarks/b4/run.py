"""B4: one electron spin (an NV centre in diamond) in a bath of 13C nuclear spins.

Event: two 13C nuclei flip-flop through their magnetic dipole coupling
(virtual photons). Because the NV electron couples to each nucleus with a
different strength, the flip-flop happens differently depending on the
electron's state, so it leaves a which-state record in the bath.

Model: pair-correlation (second-order cluster-correlation) theory for the
Hahn echo, as in Maze et al. 2008 and Zhao, Ho & Liu 2012:
  - 13C nuclei placed at random on the diamond lattice at the given abundance;
  - secular hyperfine coupling A_i (dipolar) between the NV electron and each
    nucleus, with the field along the NV axis [111];
  - secular dipolar coupling b_ij between nuclei;
  - each pair is a pseudo-spin with Hamiltonian (m * Delta_ij / 2) tau_z - (b_ij / 4) tau_x
    where m = 0 or 1 is the NV state and Delta_ij = A_i - A_j;
  - the echo is the product over pairs. Single-nucleus terms are pure phases
    that the echo removes in this secular model (they produce the collapse and
    revival pattern, not the envelope).
T2 is the 1/e time of the echo envelope, averaged over random baths.

Target: T2 = 0.65 ms at natural abundance (1.1%) and 1.8 ms at 0.3%
(Mizuochi et al. 2009, single NV centres, Hahn echo).

Sources:
  Mizuochi N et al. (2009) PRB 80:041201, arXiv:0811.4731
  Maze JR, Taylor JM, Lukin MD (2008) PRB 78:094303
  Zhao N, Ho S-W, Liu R-B (2012) PRB 85:115303, arXiv:1108.2343
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import c, gamma_13C, gamma_e, hbar, mu0
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent
A_LAT = 3.567e-10       # m, diamond lattice constant
R_BATH_11 = 6.0e-9      # m, bath radius at 1.1% abundance; scaled as abundance^(-1/3) (converged: see tests)
R_MIN = 0.5e-9          # m, nuclei closer than this have contact hyperfine terms; excluded
R_PAIR_11 = 2.0e-9      # m, largest flip-flop pair separation at 1.1%; scaled the same way
N_BATHS = 16            # random bath configurations averaged
B_FIELD = 0.03          # T (300 G), along the NV axis
SEED = 20261115

NV_AXIS = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)


def diamond_sites(radius):
    """Positions of carbon sites within radius of a vacancy at the origin."""
    n = int(radius / A_LAT) + 2
    rng = np.arange(-n, n + 1)
    cells = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([fcc, fcc + 0.25])
    pts = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * A_LAT
    r = np.linalg.norm(pts, axis=1)
    keep = (r > 1e-12) & (r <= radius)   # remove the vacancy site itself
    return pts[keep]


def radii(abundance, scale=1.0):
    f = (0.011 / abundance) ** (1 / 3) * scale
    return R_BATH_11 * f, R_PAIR_11 * f


def bath_couplings(abundance, rng, scale=1.0):
    """Pick a random bath; return pair quantities Delta_ij and b_ij (rad/s)."""
    R_BATH, R_PAIR = radii(abundance, scale)
    sites = diamond_sites(R_BATH)
    occ = rng.random(len(sites)) < abundance
    pos = sites[occ]
    r = np.linalg.norm(pos, axis=1)
    pos = pos[r >= R_MIN]
    r = np.linalg.norm(pos, axis=1)
    cos_t = pos @ NV_AXIS / r
    # secular dipolar hyperfine, NV electron (S=1) with 13C: A = (mu0/4pi) ge gC hbar (1-3cos^2)/r^3
    A = (mu0 / (4 * math.pi)) * gamma_e * gamma_13C * hbar * (1 - 3 * cos_t**2) / r**3
    i, j = np.triu_indices(len(pos), k=1)
    d = pos[j] - pos[i]
    rij = np.linalg.norm(d, axis=1)
    near = rij <= R_PAIR
    i, j, d, rij = i[near], j[near], d[near], rij[near]
    cos_ij = d @ NV_AXIS / rij
    b = (mu0 / (4 * math.pi)) * gamma_13C**2 * hbar * (1 - 3 * cos_ij**2) / rij**3
    return A[i] - A[j], b, len(pos)


def _quat(hz, hx, t):
    """exp(-i t (hz tau_z + hx tau_x)) as a real quaternion (a0, ax, ay, az),
    meaning a0 - i (ax tau_x + ay tau_y + az tau_z)."""
    w = np.sqrt(hz**2 + hx**2)
    w = np.where(w == 0, 1e-30, w)
    s = np.sin(w * t) / w
    return np.cos(w * t), s * hx, np.zeros_like(s * hx), s * hz


def _qmul(p, q):
    """Product of SU(2) elements in the quaternion form above."""
    a0, ax, ay, az = p
    b0, bx, by, bz = q
    return (a0 * b0 - ax * bx - ay * by - az * bz,
            a0 * bx + b0 * ax + (ay * bz - az * by),
            a0 * by + b0 * ay + (az * bx - ax * bz),
            a0 * bz + b0 * az + (ax * by - ay * bx))


def _qconj(p):
    return (p[0], -p[1], -p[2], -p[3])


def echo(Delta, b, taus, chunk=20000):
    """Hahn echo |L(2 tau)| for one bath, product over pairs (high temperature).

    For each pair, the pseudo-spin space {|ud>, |du>} evolves under
    h_m = (m Delta / 2) tau_z - (b / 4) tau_x for NV state m. The echo overlap
    is Tr(U0^+ U1^+ U0 U1) / 2 = s, the scalar part of that product; the frozen
    states |uu>, |dd> contribute 1 each, so L_pair = (1 + s) / 2.
    """
    T = taus[:, None]
    logL = np.zeros(len(taus))
    for k in range(0, len(Delta), chunk):
        D, bb = Delta[None, k:k + chunk], b[None, k:k + chunk]
        q0 = _quat(0 * D, -bb / 4, T)
        q1 = _quat(D / 2, -bb / 4, T)
        prod = _qmul(_qmul(_qconj(q0), _qconj(q1)), _qmul(q0, q1))
        Lpair = (1 + prod[0]) / 2
        logL += np.sum(np.log(np.abs(Lpair)), axis=1)
    return np.exp(logL)


def T2_for(abundance, n_baths=N_BATHS, seed=SEED, scale=1.0, n_tau=300):
    rng = np.random.default_rng(seed)
    taus = np.linspace(1e-6, 4e-3, n_tau)     # half echo times, s
    t2s, n_spins = [], []
    for _ in range(n_baths):
        Delta, b, n = bath_couplings(abundance, rng, scale)
        Lt = echo(Delta, b, taus)
        below = np.nonzero(Lt < math.exp(-1))[0]
        t2s.append(2 * taus[below[0]] if len(below) else np.nan)
        n_spins.append(n)
    t2s = np.array(t2s)
    return float(np.nanmedian(t2s)), float(np.nanpercentile(t2s, 16)), float(np.nanpercentile(t2s, 84)), int(np.mean(n_spins))


def main():
    rows = []
    for ab, target in [(0.011, 0.65e-3), (0.003, 1.8e-3)]:
        med, lo, hi, n = T2_for(ab)
        rows.append((ab, med, lo, hi, n, target))
        print(f"abundance {ab * 100:.1f}%: T2 median {med * 1e3:.2f} ms "
              f"(16-84%: {lo * 1e3:.2f}-{hi * 1e3:.2f} ms), ~{n} nuclei; measured {target * 1e3:.2f} ms")
    checks = [Check(f"Hahn-echo T2 at {ab * 100:.1f}% 13C", "ms", round(med * 1e3, 3),
                    measured=target * 1e3, source="Mizuochi 2009",
                    note=f"median over {N_BATHS} random baths; spread {lo * 1e3:.2f}-{hi * 1e3:.2f} ms")
              for ab, med, lo, hi, n, target in rows]
    # Atlas point: the record is carried by nuclear flip-flops; the boson is a
    # virtual photon of the nuclear Larmor frequency.
    f_L = gamma_13C * B_FIELD / (2 * math.pi)
    points = [AtlasPoint(f"NV electron spin, {ab * 100:.1f}% 13C", boson="virtual photons (magnetic dipole)",
                         probe_wavelength_m=c / f_L, domain_size_m=radii(ab)[1],
                         temperature_K=300.0, time_s=med, record_overlap=math.exp(-1))
              for ab, med, lo, hi, n, target in rows]
    notes = ("The record is not carried away by any particle; it is written into the bath's "
             "own flip-flops. The domain is the electron plus the nuclei within a few nm, "
             "and T2 is how long their records stay alike. The model has no fitted parameter. "
             "It counts only 13C pairs; real crystals also hold nitrogen and other "
             "electron spins, which can only shorten T2, and at 0.3% 13C those other spins "
             "are likely to dominate. A single NV's T2 also varies with its own bath: the "
             "computed 16-84% spread is reported with each value.")
    write_result(HERE, "B4", "NV centre spin in a 13C bath", checks, points, notes)


if __name__ == "__main__":
    main()
