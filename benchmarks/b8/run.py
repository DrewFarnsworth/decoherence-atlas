"""B8: electronic excitations in the seven-pigment FMO photosynthetic complex.

Event: an electronic excitation is shared by pigments through dipole coupling
(virtual photons), while the protein's vibrations (phonons) keep writing
records of where it is. The question is how long the records stay alike.

Two quantities are computed, both with the field's standard model
(Ishizaki & Fleming 2009: Adolphs-Renger Hamiltonian, Drude-Lorentz bath with
reorganization energy 35 cm^-1 and bath response time 50 fs on every pigment):

  1. optical dephasing: the overlap of the protein's record with and without
     the excitation, |exp(-g(t))|, exact for one pigment linearly coupled to a
     harmonic bath (second-order cumulant). This is what a homogeneous line
     width measures, and Duan et al. derived their 60 fs from such a width
     (175 cm^-1, tau = 1/(pi c width)).
  2. inter-exciton coherence: the coherence between exciton states after
     exciting pigment 1, computed with the hierarchical equations of motion
     (HEOM, QuTiP), numerically exact for this model.

Targets:
  room temperature: electronic coherence ~60 fs (Duan et al. 2017);
  20 K: 200-500 fs (arXiv:2104.01462).
Sensitivity (not scored): reorganization energy 190 cm^-1, the value of Duan's
fitted spectral density, in place of 35.

Sources:
  Duan H-G et al. (2017) PNAS 114:8493 (60 fs; homogeneous width 175 cm^-1)
  Ishizaki A, Fleming GR (2009) PNAS 106:17255 (model and parameters)
  Low-temperature study: arXiv:2104.01462
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent

# Adolphs-Renger / Ishizaki-Fleming FMO Hamiltonian, cm^-1 (site energies relative to 12,210 cm^-1)
H_CM = np.array([
    [410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
    [-87.7, 530, 30.8, 8.2, 0.7, 11.8, 4.3],
    [5.5, 30.8, 210, -53.5, -2.2, -9.6, 6.0],
    [-5.9, 8.2, -53.5, 320, -70.7, -17.0, -63.3],
    [6.7, 0.7, -2.2, -70.7, 480, 81.1, -1.3],
    [-13.7, 11.8, -9.6, -17.0, 81.1, 630, 39.7],
    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 440]])
LAM = 35.0                    # cm^-1, reorganization energy
TAU_C = 50.0                  # fs, bath response time (gamma^-1)
W = 2 * math.pi * 2.99792458e-5   # rad/fs per cm^-1
K_CM = 0.69503476             # cm^-1 per K
GAMMA_CM = 1 / (TAU_C * W)    # cm^-1


def optical_dephasing_time(T, lam=LAM, gamma_cm=GAMMA_CM):
    """1/e time (fs) of |exp(-g(t))| for a Drude-Lorentz bath at temperature T.

    Re g(t) = (1/pi) Integral J(w)/w^2 coth(w / 2kT) (1 - cos w t) dw,
    J(w) = 2 lam w gamma / (w^2 + gamma^2), all in cm^-1 (t converted).
    """
    kT = K_CM * T

    def re_g(t_fs):
        x = W * t_fs   # rad per cm^-1
        f = lambda w: (2 * lam * gamma_cm / (w**2 + gamma_cm**2)) / w / math.tanh(w / (2 * kT)) * (1 - math.cos(w * x))
        return quad(f, 1e-6, 50 * gamma_cm + 20 * kT, limit=400)[0] / math.pi

    lo, hi = 0.0, 2000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if re_g(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exciton_coherence_time(T, lam=LAM, Nk=0, depth=4, pade=False, t_max=1000.0):
    """1/e time (fs) of the summed exciton-basis coherence after exciting pigment 1 (HEOM),
    measured as its decay toward the late-time value."""
    import qutip as qt
    from qutip.solver.heom import DrudeLorentzBath, DrudeLorentzPadeBath, HEOMSolver
    Bath = DrudeLorentzPadeBath if pade else DrudeLorentzBath
    Hq = qt.Qobj(H_CM * W)
    baths = [Bath(qt.projection(7, i, i), lam * W, 1 / TAU_C, K_CM * T * W, Nk=Nk) for i in range(7)]
    solver = HEOMSolver(Hq, baths, max_depth=depth,
                        options={"nsteps": 50000, "store_ados": False, "progress_bar": False})
    times = np.arange(0, t_max, 2.0)
    res = solver.run(qt.projection(7, 0, 0), times)
    _, V = np.linalg.eigh(H_CM)
    C = []
    for st in res.states:
        rho_ex = V.T @ st.full() @ V
        C.append(np.sum(np.abs(rho_ex)) - np.sum(np.abs(np.diag(rho_ex))))
    C = np.array(C)
    # With strong coupling the protein holds a steady mix of exciton states
    # (polaron dressing); that standing part is not a decaying coherence.
    # Measure the decay of C toward its late-time value.
    C_late = float(np.mean(C[times >= 0.8 * t_max]))
    X = C - C_late
    above = np.nonzero(X >= X[0] / math.e)[0]
    return float(times[above[-1]]) if len(above) else 0.0, times, C


def main():
    rows = {}
    for T in (294.0, 77.0, 20.0):
        rows[("opt", T, LAM)] = optical_dephasing_time(T)
    rows[("opt", 294.0, 190.0)] = optical_dephasing_time(294.0, lam=190.0)
    for k, v in rows.items():
        print(f"optical dephasing, T = {k[1]:.0f} K, lambda = {k[2]:.0f} cm^-1: {v:.0f} fs")
    hx = {}
    hx[294.0], _, _ = exciton_coherence_time(294.0, Nk=1, depth=4)
    print(f"exciton coherence (HEOM), 294 K: {hx[294.0]:.0f} fs")
    hx[77.0], _, _ = exciton_coherence_time(77.0, Nk=2, depth=4, pade=True)
    print(f"exciton coherence (HEOM), 77 K: {hx[77.0]:.0f} fs")
    hx[20.0], _, _ = exciton_coherence_time(20.0, Nk=3, depth=3, pade=True, t_max=1500.0)
    print(f"exciton coherence (HEOM), 20 K: {hx[20.0]:.0f} fs")
    hx_190, _, _ = exciton_coherence_time(294.0, lam=190.0, Nk=1, depth=5)
    print(f"exciton coherence (HEOM), 294 K, lambda 190: {hx_190:.0f} fs")

    checks = [
        Check("optical dephasing time at room temperature", "fs", round(rows[("opt", 294.0, LAM)]),
              measured=60.0, source="Duan 2017 (homogeneous width 175 cm^-1)",
              note="standard model, lambda = 35 cm^-1"),
        Check("inter-exciton coherence time at room temperature (HEOM)", "fs", round(hx[294.0]),
              measured=60.0, source="Duan 2017", note="standard model, lambda = 35 cm^-1"),
        Check("inter-exciton coherence time at 20 K (HEOM)", "fs", round(hx[20.0]),
              measured_low=200.0, measured_high=500.0, source="arXiv:2104.01462"),
        Check("optical dephasing time at 20 K", "fs", round(rows[("opt", 20.0, LAM)]),
              measured_low=200.0, measured_high=500.0, scored=False, source="arXiv:2104.01462"),
        Check("optical dephasing, room temperature, lambda = 190 cm^-1", "fs",
              round(rows[("opt", 294.0, 190.0)]), measured=60.0, scored=False,
              note="Duan's fitted reorganization energy, Drude shape assumed"),
        Check("inter-exciton coherence, room temperature, lambda = 190 cm^-1 (HEOM)", "fs",
              round(hx_190), measured=60.0, scored=False,
              note="Duan's fitted reorganization energy, Drude shape assumed"),
    ]
    lam_exc = 1e-2 / 12400.0     # ~806 nm, the Qy excitation
    points = [AtlasPoint(f"FMO excitation, {T:.0f} K", boson="virtual photons (excitonic coupling); phonons",
                         probe_wavelength_m=lam_exc, domain_size_m=2.5e-9,
                         temperature_K=T, time_s=hx[T] * 1e-15, record_overlap=math.exp(-1))
              for T in (294.0, 77.0, 20.0)]
    notes = ("With the standard weak coupling (35 cm^-1) the coherence between exciton "
             "states outlives the measurement by about 2.5 times; with Duan's stronger "
             "coupling (190 cm^-1, Drude shape assumed) it falls to about the measured "
             "60 fs. The coherence time is measured as decay toward the late-time value, "
             "because strong coupling leaves a standing mix of exciton states. "
             "domain_size is a placeholder of ~2 pigment spacings (the excitation is "
             "shared over a few neighbouring pigments); computing it from the exciton "
             "states is the next step.")
    write_result(HERE, "B8", "FMO electronic coherence", checks, points, notes)


if __name__ == "__main__":
    main()
