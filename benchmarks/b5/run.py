"""B5: iron-57 nuclei absorbing gamma rays without recoil (the Mossbauer effect).

Event: a 14.41 keV gamma photon is absorbed by one iron-57 nucleus. Whether
the whole crystal recoils as one (no phonon made: the event leaves no record
of which atom took it) or a phonon is made (the lattice records it) is set by
the recoil-free fraction

    f = exp(-k^2 <x^2>),

the overlap between the lattice's state before and after the photon's
momentum kick. In the atlas's terms f is the record overlap I directly.

Model: Debye model of the iron lattice, with the calorimetric Debye
temperature (470 K) taken from an independent source rather than fitted:

    -ln f = (3 E_R / (2 kB theta)) [1 + 4 (T/theta)^2 Integral_0^{theta/T} x/(e^x - 1) dx]

with recoil energy E_R = E_gamma^2 / (2 M c^2).

Target: f = 0.799 +/- 0.003 for alpha-iron at room temperature.

Sources:
  Lamb-Moessbauer factor of alpha-Fe, 0.799 +/- 0.003 (resonant-detector measurement)
  Debye temperature of iron, 470 K: Kittel, Introduction to Solid State Physics, Table 5.1
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import amu, c, eV, hbar, kB
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent
E_GAMMA = 14.4125e3 * eV        # J
M_FE57 = 56.935 * amu           # kg
THETA_D = 470.0                 # K, calorimetric Debye temperature of Fe
T_ROOM = 295.0                  # K

E_R = E_GAMMA**2 / (2 * M_FE57 * c**2)
k = E_GAMMA / (hbar * c)


def debye_integral(y):
    return quad(lambda x: x / math.expm1(x) if x > 0 else 1.0, 0, y)[0]


def minus_ln_f(T, theta=THETA_D):
    return (3 * E_R / (2 * kB * theta)) * (1 + 4 * (T / theta) ** 2 * debye_integral(theta / T))


def f_LM(T, theta=THETA_D):
    return math.exp(-minus_ln_f(T, theta))


def main():
    f_room = f_LM(T_ROOM)
    x_rms = math.sqrt(minus_ln_f(T_ROOM)) / k      # rms displacement along the beam
    print(f"recoil energy E_R = {E_R / eV * 1e3:.3f} meV; gamma wavelength {2 * math.pi / k * 1e10:.3f} A")
    print(f"f(295 K) = {f_room:.4f}; rms displacement {x_rms * 1e10:.4f} A")
    for T in (4, 77, 200, 295, 400, 600, 800, 1000):
        print(f"  T = {T:5d} K   f = {f_LM(T):.4f}")
    # sensitivity: Debye temperatures quoted for Fe range ~420-470 K
    f_420 = f_LM(T_ROOM, 420.0)
    checks = [
        Check("recoil-free fraction of alpha-Fe at room temperature", "fraction",
              round(f_room, 4), measured=0.799, source="Lamb-Moessbauer factor 0.799 +/- 0.003",
              note="Debye model, theta = 470 K (calorimetric), no fitting"),
        Check("same, with theta = 420 K (sensitivity)", "fraction", round(f_420, 4),
              measured=0.799, scored=False, note="lower end of quoted Debye temperatures"),
    ]
    points = [AtlasPoint("Fe-57 nucleus in alpha-iron, 14.4 keV gamma", boson="real photon (gamma) and phonons",
                         probe_wavelength_m=2 * math.pi / k, domain_size_m=x_rms,
                         temperature_K=T_ROOM, time_s=141e-9, record_overlap=f_room)]
    notes = ("The domain is the whole crystal: when no phonon is made, the event is shared "
             "by every atom and no record says which nucleus took it. The overlap is "
             "set by how far each nucleus wanders (0.06 A) against the photon's wavelength "
             "(0.86 A). time_s is the Fe-57 excited-state lifetime (141 ns).")
    write_result(HERE, "B5", "Moessbauer recoil-free fraction of alpha-iron", checks, points, notes)


if __name__ == "__main__":
    main()
