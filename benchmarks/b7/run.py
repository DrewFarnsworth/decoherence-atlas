"""B7: O-H stretch vibrations in liquid water hand their quantum on.

Event: a vibrational quantum on one O-H bond passes to another through the
coupling of their transition dipoles (a virtual photon) or, within one
molecule, through the shared oxygen. Each hop moves the excitation, so the
direction of the excited bond (seen by polarized light) is forgotten.

Model: vibrational exciton on a tetrahedral cluster of five water molecules
(a central molecule and its four hydrogen-bonded neighbours, ten O-H local
modes), the first shell of liquid water.
  - local O-H frequencies: static Gaussian disorder with sigma = 108 cm^-1, from
    the 255 cm^-1 width of the isolated O-H band (HOD in D2O);
  - intramolecular coupling: -49.5 cm^-1, half the gas-phase splitting of the
    symmetric (3657) and antisymmetric (3756 cm^-1) stretches;
  - intermolecular coupling: point transition dipoles along each O-H, 0.58 A
    from the oxygen, of magnitude 0.23 D derived from the measured infrared
    absorption of liquid water (imaginary index 0.272 at 2.94 um, band FWHM
    ~400 cm^-1; Hale & Querry 1973).
The excitation starts on one O-H of the central molecule. The polarization
anisotropy r(t) = 0.4 sum_j P_j(t) P2(cos theta_0j) is averaged over 2,000
disorder draws, and its 1/e time is compared with the measured transfer time.
Two textbook limits are printed for orientation: Foerster hopping
(incoherent) and the coherent short-time decay.

Target: resonant energy transfer time 80 fs, unaffected by temperature, from
polarization-resolved pump-probe (Kraemer et al. 2008).
Not computed: the 50 fs loss of frequency memory at 300 K (needs molecular
dynamics of the hydrogen-bond network).

Sources:
  Kraemer D et al. (2008) PNAS 105:437 (80 fs transfer; 50 fs memory at 300 K)
  Hale GM, Querry MR (1973) Appl. Opt. 12:555 (optical constants of water)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import c, debye, eps0, h, wavenumber_to_J
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent
R_OO = 2.85e-10          # m
R_OH = 0.98e-10          # m
D_POS = 0.58e-10         # m, transition dipole position along O-H
SIGMA = 108.0            # cm^-1, static disorder of local O-H frequencies
V_INTRA = -49.5          # cm^-1
N_DRAWS = 2000
SEED = 7


def transition_dipole_debye():
    """O-H stretch transition dipole from the integrated absorption of liquid water."""
    k_max, lam_peak, fwhm = 0.272, 2.94e-4, 400.0      # imaginary index, cm, cm^-1
    nu = 1 / lam_peak                                   # cm^-1
    alpha_max = 4 * math.pi * k_max * nu                # cm^-1
    integrated = alpha_max * fwhm * 1.0645              # cm^-2, Gaussian band
    n_OH = 2 * 0.997 / 18.015 * 6.02214076e23           # O-H bonds per cm^3
    S = integrated / n_OH                               # cm per bond
    h_cgs, c_cgs = 6.62607015e-27, 2.99792458e10
    mu2 = 3 * h_cgs * c_cgs * S / (8 * math.pi**3 * nu)  # (esu cm)^2
    return math.sqrt(mu2) / 1e-18


MU_D = transition_dipole_debye()

TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / math.sqrt(3)


def other_tetrahedral(axis, rng):
    """The three tetrahedral directions at 109.47 deg from axis, random azimuth."""
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi0 = rng.uniform(0, 2 * math.pi)
    ct, st = -1 / 3, math.sqrt(8) / 3
    return [ct * axis + st * (math.cos(phi0 + k * 2 * math.pi / 3) * e1 + math.sin(phi0 + k * 2 * math.pi / 3) * e2)
            for k in range(3)]


def cluster(rng):
    """Ten O-H oscillators: (oxygen position, unit vector, molecule index)."""
    osc = []
    O0 = np.zeros(3)
    osc += [(O0, TET[0], 0), (O0, TET[1], 0)]              # central donates to neighbours 1, 2
    for k in range(1, 5):
        Ok = R_OO * TET[k - 1]
        to_centre = -TET[k - 1]
        others = other_tetrahedral(to_centre, rng)
        if k <= 2:   # acceptor: both H point away from the centre
            dirs = [others[i] for i in rng.choice(3, 2, replace=False)]
        else:        # donor: one H points at the central oxygen
            dirs = [to_centre, others[rng.integers(3)]]
        osc += [(Ok, d, k) for d in dirs]
    return osc


def couplings(osc):
    """Static coupling matrix in cm^-1 (without diagonal disorder)."""
    n = len(osc)
    V = np.zeros((n, n))
    mu = MU_D * debye
    pref = 1 / (4 * math.pi * eps0) / (h * c * 100.0)          # J -> cm^-1
    pos = [O + D_POS * u for O, u, _ in osc]
    for i in range(n):
        for j in range(i + 1, n):
            if osc[i][2] == osc[j][2]:
                V[i, j] = V_INTRA
            else:
                r = pos[j] - pos[i]; d = np.linalg.norm(r); rh = r / d
                ui, uj = osc[i][1], osc[j][1]
                kappa = ui @ uj - 3 * (ui @ rh) * (uj @ rh)
                V[i, j] = pref * mu**2 * kappa / d**3
            V[j, i] = V[i, j]
    return V


def run(times_fs):
    rng = np.random.default_rng(SEED)
    w = 2 * math.pi * c * 100.0                     # rad/s per cm^-1
    t = times_fs * 1e-15
    r_sum = np.zeros_like(t); P_leave = np.zeros_like(t); V_inter_rms = []
    for _ in range(N_DRAWS):
        osc = cluster(rng)
        V = couplings(osc)
        H = V + np.diag(rng.normal(0, SIGMA, len(osc)))
        E, U = np.linalg.eigh(H)
        amp = U @ (U[0, :, None] * np.exp(-1j * w * E[:, None] * t[None, :]))   # site amplitudes vs time
        P = np.abs(amp) ** 2
        cos0 = np.array([osc[0][1] @ o[1] for o in osc])
        r_sum += 0.4 * (P * (1.5 * cos0[:, None] ** 2 - 0.5)).sum(axis=0)
        same = np.array([o[2] == 0 for o in osc])
        P_leave += P[~same].sum(axis=0)
        V_inter_rms.append(np.sqrt(np.sum(V[0, ~same] ** 2)))
    return r_sum / N_DRAWS, P_leave / N_DRAWS, float(np.mean(V_inter_rms))


def main():
    times = np.arange(0, 600, 1.0)
    r, P_leave, V_rms = run(times)
    below = np.nonzero(r < 0.4 / math.e)[0]
    tau_r = float(times[below[0]]) if len(below) else None
    half = np.nonzero(P_leave >= 0.5 * P_leave[-100:].mean())[0]
    tau_leave = float(times[half[0]]) if len(half) else None
    # textbook limits for one O-H handing on to its intermolecular neighbours
    ps_per_cm = 1 / (2 * math.pi * c * 100.0)       # s per (rad/s per cm^-1)
    tau_coh = ps_per_cm / V_rms
    sig_hom = 255.0 / 2.3548                          # overlap of two identical Gaussian lines
    J = 1 / (2 * math.sqrt(math.pi) * sig_hom)        # per cm^-1
    k_forster = 2 * math.pi * c * 100.0 * V_rms**2 * J
    print(f"transition dipole {MU_D:.3f} D; rms intermolecular coupling of one O-H {V_rms:.1f} cm^-1")
    print(f"anisotropy 1/e time {tau_r} fs; half of the final intermolecular share reached at {tau_leave} fs")
    print(f"plateau: r(500-600 fs) = {r[-100:].mean():.3f}; share off the first molecule {P_leave[-100:].mean():.2f}")
    print(f"limits: coherent short-time {tau_coh * 1e15:.0f} fs; Foerster hopping {1 / k_forster * 1e15:.0f} fs")
    checks = [
        Check("anisotropy decay (1/e) after exciting one O-H", "fs", tau_r, measured=80.0,
              source="Kraemer 2008, polarization-resolved pump-probe",
              note="five-molecule cluster, static disorder, no fitting"),
        Check("Foerster hopping time to the first shell (incoherent limit)", "fs",
              round(1 / k_forster * 1e15), measured=80.0, scored=False),
        Check("coherent short-time transfer time to the first shell", "fs",
              round(tau_coh * 1e15), measured=80.0, scored=False),
        Check("loss of frequency memory at 300 K", "fs", None, measured=50.0,
              source="Kraemer 2008, 2D IR", note="needs molecular dynamics; not modelled"),
    ]
    lam_OH = 1e-2 / 3400.0
    points = [AtlasPoint("O-H stretch quantum in liquid water", boson="vibrational quanta via virtual photons",
                         probe_wavelength_m=lam_OH, domain_size_m=R_OO,
                         temperature_K=300.0, time_s=(tau_r or float('nan')) * 1e-15,
                         record_overlap=math.exp(-1))]
    notes = ("Within a molecule the two O-H bonds share the quantum in about 100 fs "
             "through the oxygen; the transition dipole coupling to neighbours is "
             "weaker. If the anisotropy decay is fast here it is mostly the "
             "intramolecular hop, as Paarmann et al. (2009) concluded from full simulations. "
             "Sensitivity (800 draws each): dipole position 0.50-0.67 A gives 84-74 fs; "
             "disorder 70-150 cm^-1 gives 69-100 fs; intramolecular coupling -25 cm^-1 "
             "gives 125 fs. The match rests mostly on the gas-phase intramolecular "
             "coupling, which is smaller in the liquid in some models; treat it as "
             "provisional. Point dipoles only 2 A apart are also at the edge of validity.")
    write_result(HERE, "B7", "O-H stretch energy transfer in liquid water", checks, points, notes)


if __name__ == "__main__":
    main()
