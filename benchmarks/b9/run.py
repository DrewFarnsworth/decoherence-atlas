"""B9: two photons meeting at a beam splitter (Hong-Ou-Mandel interference).

This benchmark measures distinguishability directly. Two photons enter a
50:50 beam splitter from opposite sides. If nothing records which photon is
which, the two ways of both reaching different detectors cancel, and the
photons always leave together. The dip in coincidences has depth

    V = |<psi_1|psi_2>|^2,

the squared overlap of the two photons' states: the core's I^2, read off a
detector. Any difference between them (arrival time, colour, polarization,
or a record left in the emitter) fills the dip in proportion.

Part a: photon pairs from a crystal (down-conversion), filtered by a
rectangular 6.25 nm interference filter at 1550 nm. Delaying one photon
makes them distinguishable by arrival time. The dip is computed numerically
from the filter's transmission: coincidences P(tau) = 1/2 [1 - Re C(tau)],
with C(tau) the normalized Fourier transform of the pair's joint spectrum at
twice the delay (the two photons' frequencies are anticorrelated).
Target: dip width 0.38 mm of delay, visibility 99.74 +/- 2.03% (Kim et al.).
The paper does not define "width"; it matches the zero-to-zero base, which
is reported here.

Part b: successive photons from one quantum dot, 3 ns apart. The textbook
prediction takes the overlap from the emitter's lifetime T1 and its
spectroscopic coherence time T2: V = T2 / 2T1. Target: the measured
visibilities 0.79, 0.83, 0.83 (Reigue et al. 2017). The spectroscopic T2 is
an average over seconds, and it includes slow drifts of the line (charges
moving nearby) that cannot tell apart two photons emitted 3 ns apart. So
this prediction is expected to come out low; how low is the measure of how
much of the record is written too slowly to matter.

Sources:
  Kim Y-S et al. (2016), Two-photon interference of temporally separated photons,
      arXiv:1607.03678 (Fig. 2a; 6.25 nm rectangular filter; 0.38 mm; 99.74%)
  Reigue A et al. (2017) PRL 118:233602, arXiv:1612.07121 (Table I)
  Hong CK, Ou ZY, Mandel L (1987) PRL 59:2044 (the original effect)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import c
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent

LAM0 = 1550e-9        # m, degenerate photons from a 775 nm pump
DLAM = 6.25e-9        # m, rectangular filter bandwidth

# Quantum dots, Reigue et al. 2017 Table I: T1 (ps), T2/2T1 from Fourier spectroscopy, measured V
DOTS = {"QD1": (1100.0, 0.35, 0.79), "QD2": (750.0, 0.23, 0.83), "QD3": (670.0, 0.71, 0.83)}


def spdc_dip(delays_s):
    """Coincidence probability vs delay for a frequency-anticorrelated pair through a rectangular filter."""
    dnu = c * DLAM / LAM0**2                       # Hz, filter width in frequency
    nu = np.linspace(-dnu, dnu, 4001)              # detuning of one photon from degeneracy
    phi2 = (np.abs(nu) <= dnu / 2).astype(float)   # joint spectrum after both filters
    norm = np.trapezoid(phi2, nu)
    # exchange term: the two photons' frequencies swap, phase difference 2*(2 pi nu)*tau
    C = np.array([np.trapezoid(phi2 * np.cos(4 * np.pi * nu * t), nu) / norm for t in delays_s])
    return 0.5 * (1 - C), dnu


def width_measures(delays, P):
    """Zero-to-zero base width and FWHM of the dip, in delay (s)."""
    i0 = int(np.argmin(np.abs(delays)))
    right = delays[i0:]; Pr = P[i0:]
    # base: first return of P to its far-delay value 0.5 (the sinc's first zero)
    up = np.nonzero(np.diff(np.sign(Pr - 0.5)))[0]
    base = 2 * right[up[0]] if len(up) else float("nan")
    half = np.nonzero(Pr >= 0.25)[0]
    fwhm = 2 * right[half[0]] if len(half) else float("nan")
    return base, fwhm


def emitter_overlap(T1, T2):
    """Overlap |<psi1|psi2>|^2 of two photons from an emitter with lifetime T1 and coherence time T2.

    Pure dephasing at rate gamma* = 1/T2 - 1/2T1 randomizes the phase during
    emission; the overlap integral gives Gamma / (Gamma + 2 gamma*) = T2 / 2T1.
    Computed numerically to keep the steps visible.
    """
    g = 1 / T1
    gstar = 1 / T2 - 1 / (2 * T1)
    # V = integral_0^inf integral_0^inf G^2 e^{-G(t1+t2)} e^{-2 g*|t1-t2|} dt1 dt2
    s = np.linspace(0, 25 * T1, 2001)
    T1g, T2g = np.meshgrid(s, s, indexing="ij")
    f = g**2 * np.exp(-g * (T1g + T2g) - 2 * gstar * np.abs(T1g - T2g))
    return float(np.trapezoid(np.trapezoid(f, s, axis=1), s))


def main():
    delays = np.linspace(-3e-12, 3e-12, 3001)
    P, dnu = spdc_dip(delays)
    base, fwhm = width_measures(delays, P)
    vis = (0.5 - P.min()) / 0.5
    base_mm = base * c * 1e3
    print(f"filter width {dnu / 1e12:.3f} THz; dip base {base * 1e12:.3f} ps = {base_mm:.3f} mm of delay; "
          f"FWHM {fwhm * c * 1e3:.3f} mm; visibility {vis:.4f}")
    checks = [
        Check("HOM dip width (zero-to-zero), 6.25 nm rectangular filter", "mm", round(base_mm, 3),
              measured=0.38, source="Kim et al. 2016, Fig. 2a",
              note="width definition not stated in the paper; FWHM would be "
                   f"{fwhm * c * 1e3:.2f} mm"),
        Check("HOM dip visibility, identical filtered photons", "fraction", round(vis, 4),
              measured=0.9974, source="Kim et al. 2016 (+/- 0.0203)"),
    ]
    points = [AtlasPoint("down-converted photon pair, 6.25 nm filter", boson="real photons",
                         probe_wavelength_m=LAM0, domain_size_m=base * c / 2,
                         temperature_K=None, time_s=base / 2, record_overlap=vis)]
    for name, (T1ps, ratio, V_meas) in DOTS.items():
        T1 = T1ps * 1e-12
        T2 = 2 * T1 * ratio
        V = emitter_overlap(T1, T2)
        print(f"{name}: T1 {T1ps:.0f} ps, T2 {T2 * 1e12:.0f} ps -> predicted V {V:.3f}; measured {V_meas}")
        checks.append(Check(f"two-photon visibility, quantum dot {name}, from T2/2T1", "fraction",
                            round(V, 3), measured=V_meas, source="Reigue et al. 2017, Table I",
                            note="spectroscopic T2 includes slow line drifts that cannot "
                                 "distinguish photons 3 ns apart"))
        points.append(AtlasPoint(f"quantum-dot photon, {name}", boson="real photons; phonons",
                                 probe_wavelength_m=None, domain_size_m=c * T2,
                                 temperature_K=4.0, time_s=3e-9, record_overlap=V_meas ** 0.5))
    notes = ("Part a passes because nothing but arrival time distinguishes the photons, and the "
             "model includes it. Part b's textbook prediction fails for two of three dots: the "
             "emitter's slow drifts are a real record, but one written too slowly to tell apart "
             "photons 3 ns apart. Overlap must be taken between the two records actually "
             "compared, over the time that separates them. QD3, where slow drift is small, passes.")
    write_result(HERE, "B9", "Two-photon interference (Hong-Ou-Mandel)", checks, points, notes)


if __name__ == "__main__":
    main()
