"""B2: hot C70 fullerenes decohered by their own thermal emission.

Event: the molecule emits a real photon. The photon carries a which-path
record only if its wavelength is short enough to resolve the path separation
(up to ~1 um here), so visible photons decohere and infrared ones do not.

Model: the paper's own Eq. 1 (spectral emission rate of a hot molecule of
finite heat capacity) and Eq. 2 (visibility in the Talbot-Lau geometry, with
the path separation growing to one grating period at the middle grating and
shrinking again). The molecule cools by its own emission along the way.

Input not available here: the paper's spectral absorption cross-section of hot
C70 (Coheur et al. 1996) and the distribution of molecular temperatures for
each laser power (fitted in Hornberger et al. 2005). So this script uses a
grey-body stand-in: the measured absolute cross-section at 532 nm
(1.97e-21 m^2, Eibenberger et al. 2014) held flat across the visible band and
cut off below the S1 absorption onset. That makes the emission rate the weak
link, and it is scored against the paper's statement that a molecule at
2500 K emits about three visible photons in 4 ms.

What is scored:
  1. visible photons emitted in 4 ms at 2500 K (paper: "about three");
  2. visibility at entry temperature 3000 K (paper: complete loss near 3000 K);
  3. visibility at 2000 K (paper: emission negligible below about 2000 K).
The power-by-power visibilities (0.47, 0.29, 0.07, 0 at 0, 3, 6, 10.5 W) need
the temperature distribution and are listed as not computed.

Sources:
  Hackermueller L et al. (2004) Nature 427:711, arXiv:quant-ph/0402146
  Eibenberger S et al. (2014) PRL 112:250402, arXiv:1402.5307 (sigma at 532 nm)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import c, eV, h, hbar, kB
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent

D = 991e-9           # m, grating period
L = 0.38             # m, grating separation (= Talbot length)
V = 190.0            # m/s
GAP = 0.072          # m, heating stage to first grating
CV = 202 * kB        # J/K, heat capacity used by the paper
SIGMA_532 = 1.97e-21 # m^2, measured C70 absorption cross-section at 532 nm
E_ONSET = 1.65       # eV, red edge of the visible absorption (~750 nm) for the grey-body stand-in
E_MAX = 4.0          # eV, upper end of the integration
V0 = 0.47            # vacuum visibility at 190 m/s

E_grid = np.linspace(0.05, E_MAX, 4000) * eV           # photon energies, J
omega = E_grid / hbar
lam = 2 * np.pi * c / omega


def sigma_abs(E):
    """Grey-body stand-in for hot C70: flat at the 532 nm value above the onset."""
    return np.where(E >= E_ONSET * eV, SIGMA_532, 0.0)


def emission_rate_per_omega(T):
    """Paper's Eq. 1: R(omega, T) in photons per second per unit angular frequency."""
    x = hbar * omega / (kB * T)
    return omega**2 / (math.pi**2 * c**2) * sigma_abs(E_grid) * np.exp(-x - (kB / (2 * CV)) * x**2)


def photons_and_power(T):
    R = emission_rate_per_omega(T)
    rate = np.trapezoid(R, omega)
    power = np.trapezoid(R * hbar * omega, omega)
    return rate, power, R


def visible_photons(T, duration, with_cooling=True):
    """Photons emitted in the visible (400-750 nm) over a duration, letting the molecule cool."""
    n, t, dt = 0.0, 0.0, duration / 400
    vis = (lam >= 400e-9) & (lam <= 750e-9)
    while t < duration:
        rate, power, R = photons_and_power(T)
        n += np.trapezoid(R[vis], omega[vis]) * dt
        if with_cooling:
            T -= power / CV * dt
        t += dt
    return n


def visibility_ratio(T_entry_heater):
    """V/V0 for a molecule leaving the heater at the given temperature.

    It cools over the 7.2 cm gap, then crosses both interferometer stages.
    The decoherence exponent per photon is 1 - sinc(2 pi dx(t) / lambda).
    """
    T = T_entry_heater
    # gap: cooling only
    t_gap, steps = GAP / V, 200
    for _ in range(steps):
        _, power, _ = photons_and_power(T)
        T -= power / CV * (t_gap / steps)
    T_at_G1 = T
    # interferometer: 0 .. 2L/v
    t_tot, steps = 2 * L / V, 400
    dt = t_tot / steps
    exponent = 0.0
    for i in range(steps):
        t = (i + 0.5) * dt
        dx = D * (L - abs(V * t - L)) / L          # path separation, peaks at d at G2
        _, power, R = photons_and_power(T)
        arg = 2 * np.pi * dx / lam
        exponent += np.trapezoid(R * (1 - np.sinc(arg / np.pi)), omega) * dt
        T -= power / CV * dt
    return math.exp(-exponent), T_at_G1


def main():
    n_2500 = visible_photons(2500.0, 4e-3)
    print(f"visible photons in 4 ms from 2500 K: {n_2500:.2f}")
    curve = []
    for T in range(1000, 5001, 250):
        r, T1 = visibility_ratio(float(T))
        curve.append((T, T1, r))
        print(f"heater exit {T:5d} K -> at G1 {T1:7.0f} K -> V/V0 = {r:.3f}")

    def ratio_at_G1(target):
        # find heater temperature whose G1 temperature is target, by interpolation
        Ts = np.array([x[1] for x in curve]); rs = np.array([x[2] for x in curve])
        return float(np.interp(target, Ts, rs))

    r3000 = ratio_at_G1(3000.0)
    r2000 = ratio_at_G1(2000.0)
    checks = [
        Check("visible photons emitted in 4 ms at 2500 K", "photons", round(n_2500, 2),
              measured=3.0, source="Hackermueller 2004, text (model statement)",
              note="grey-body stand-in for the hot C70 absorption spectrum"),
        Check("V/V0 for a molecule entering at 3000 K", "ratio", round(r3000, 3),
              measured_high=0.1, source="Hackermueller 2004: complete loss near 3000 K",
              note="scored as at most 0.1"),
        Check("V/V0 for a molecule entering at 2000 K", "ratio", round(r2000, 3),
              measured_low=0.9, source="Hackermueller 2004: emission negligible below ~2000 K",
              note="scored as at least 0.9"),
        Check("temperature at the first grating for a molecule leaving the heater at 5000 K",
              "K", round(curve[-1][1]), measured_high=3000.0,
              source="Hackermueller 2004: even the hottest molecules are below ~3000 K on entry",
              note="tests radiative cooling over the 7.2 cm gap"),
    ]
    for P, Vm in [(0, 0.47), (3, 0.29), (6, 0.07), (10.5, 0.0)]:
        checks.append(Check(f"visibility at {P} W heating", "fraction", None, measured=Vm,
                            source="Hackermueller 2004, Fig. 2",
                            note="needs the temperature distribution per power (Hornberger 2005)"))
    # Atlas point: the domain is the separation below which emitted visible
    # photons leave the records alike; photon wavelength sets it.
    _, _, R = photons_and_power(2500.0)
    mean_lam = float(np.trapezoid(R * lam, omega) / np.trapezoid(R, omega))
    points = [AtlasPoint("hot C70 emitting at 2500 K", boson="real photons (thermal emission)",
                         probe_wavelength_m=mean_lam, domain_size_m=mean_lam / (2 * math.pi),
                         temperature_K=2500.0, time_s=2 * L / V,
                         record_overlap=visibility_ratio(2500.0)[0])]
    notes = ("Emission model is the weak link: replace the grey-body stand-in with the "
             "hot C70 spectrum (Coheur 1996; Hornberger 2005 Fig. 3) to score the "
             "power-by-power visibilities.")
    write_result(HERE, "B2", "C70 decoherence by thermal emission (2004)", checks, points, notes)


if __name__ == "__main__":
    main()
