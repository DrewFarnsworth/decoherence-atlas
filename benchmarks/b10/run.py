"""B10: the double slit, with a dial for which-path information.

The double slit is a two-path interferometer. Here it is done in its clean
form: single photons from one NV centre in diamond pass a 50:50 beam
splitter (the two slits), travel two paths, and meet at a second beam
splitter whose reflection R can be set between 0 and 0.5 by a voltage.

  R = 0:    the paths are never mixed, so each detector says which path the
            photon took. Full cut: D = 1, no fringes (V = 0).
  R = 0.5:  the paths are fully mixed and no record says which. Full join:
            D = 0, full fringes (V = 1).
  between:  partial records, partial fringes.

The core's line (1), I^2 + D^2 <= 1 (the wave-particle duality relation),
is tested directly, with I read as the fringe visibility V. The choice of R
was made after each photon had entered (Wheeler's delayed choice), which
changes nothing in the arithmetic.

Model: the photon's two-path state is propagated through 2 x 2 beam-splitter
matrices and a phase shift. V comes from the detection probabilities as the
phase is scanned; D from the same probabilities with one path blocked, as
the paper measured it. R is computed from the paper's independently
measured electro-optic settings, R = sin^2(2 beta) sin^2(pi V_EOM / 2 V_pi),
with beta = 24 deg and V_pi = 217 V. No fitting.

Targets (Jacques et al. 2008): V = 93 +/- 2% at 150 V, 42 +/- 2% at 40 V,
0 at 0 V; and V^2 + D^2 = 0.97 +/- 0.03 over all settings.

Sources:
  Jacques V et al. (2008) PRL 100:220402, arXiv:0801.0979
  Englert B-G (1996) PRL 77:2154 (the duality relation)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent
BETA = math.radians(24.0)
V_PI = 217.0
LAMBDA = 670e-9


def reflection(v_eom):
    return math.sin(2 * BETA) ** 2 * math.sin(math.pi / 2 * v_eom / V_PI) ** 2


def splitter(R):
    r, t = math.sqrt(R), math.sqrt(1 - R)
    return np.array([[t, 1j * r], [1j * r, t]])


def detect(R, phase, block=None):
    """Detection probabilities at the two outputs for one photon entering port 0."""
    psi = splitter(0.5) @ np.array([1.0, 0.0])            # the two "slits"
    psi = np.array([psi[0], psi[1] * np.exp(1j * phase)])  # path difference
    if block is not None:
        psi[block] = 0.0                                   # block one path, as for measuring D
    out = splitter(R) @ psi
    return np.abs(out) ** 2


def visibility(R):
    phases = np.linspace(0, 2 * np.pi, 721)
    P1 = np.array([detect(R, p)[0] for p in phases])
    return (P1.max() - P1.min()) / (P1.max() + P1.min())


def distinguishability(R):
    D = 0.0
    for blocked in (1, 0):
        N = detect(R, 0.0, block=blocked)
        D += abs(N[0] - N[1]) / (N[0] + N[1]) / 2       # each blocked run sees half the photons' paths
    return D


def main():
    rows = []
    for v_eom, V_meas in [(150.0, 0.93), (40.0, 0.42), (0.0, 0.0)]:
        R = reflection(v_eom)
        V = visibility(R)
        D = distinguishability(R)
        rows.append((v_eom, R, V, D, V_meas))
        print(f"V_EOM {v_eom:5.0f} V: R = {R:.3f}, V = {V:.3f} (measured {V_meas}), D = {D:.3f}, V^2 + D^2 = {V**2 + D**2:.3f}")
    checks = [Check(f"fringe visibility at V_EOM = {v:.0f} V (R = {R:.3f})", "fraction", round(V, 3),
                    measured=Vm, source="Jacques et al. 2008, Fig. 3",
                    note="R from the measured electro-optic settings, ideal optics")
              for v, R, V, D, Vm in rows]
    s = [V**2 + D**2 for _, _, V, D, _ in rows]
    checks.append(Check("V^2 + D^2 (the duality relation, line (1) of the core)", "dimensionless",
                        round(float(np.mean(s)), 3), measured=0.97, source="Jacques et al. 2008 (+/- 0.03)",
                        note="equality for pure states; the measured shortfall is imperfect optics"))
    points = [AtlasPoint(f"single photon, two paths, R = {R:.2f}", boson="real photon",
                         probe_wavelength_m=LAMBDA, domain_size_m=None, temperature_K=300.0,
                         time_s=None, record_overlap=V)
              for _, R, V, _, _ in rows]
    notes = ("The output beam splitter decides how much which-path record survives: it is a "
             "dial between cut and join. At every setting the fringes and the record trade off "
             "exactly as I^2 + D^2 = 1 says for pure states. The model overshoots the best "
             "visibility (0.99 against 0.93) because it assumes perfect optics.")
    write_result(HERE, "B10", "The double slit, with a which-path dial", checks, points, notes)


if __name__ == "__main__":
    main()
