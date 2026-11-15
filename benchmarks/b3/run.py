"""B3: the largest superpositions measured, and why they survive.

Two experiments at the current limit of matter-wave interferometry:
  a. oligoporphyrins of ~26.8 kDa, ~2,000 atoms (Fein et al. 2019, LUMI, 2 m baseline);
  b. sodium nanoclusters of ~172 kDa, ~7,500 atoms, ~8 nm across, made at 77 K
     (Pedalino et al. 2026, MUSCLE; the record since January 2026).

Test: every decoherence channel stays below threshold, so the computed
record overlap after the full flight must be consistent with the observed
fringes. Channels computed:
  - collisions with residual gas (saturated regime; van der Waals C6 from the
    London formula with each paper's polarizability and ionization energy);
  - scattering of 300 K thermal photons from the chamber walls (Joos-Zeh,
    long-wavelength limit).
Not computed: emission and absorption of thermal photons (needs the hot
molecule's infrared spectrum); for the 77 K clusters emission is negligible.

Residual gas composition is not reported. N2 and H2 bracket the usual vacuum.

Observed bounds:
  a. fringes at 0.93 +/- 0.06 of the expected visibility, so all decoherence
     together leaves at least 0.87 (Fein 2019);
  b. one global factor of 0.78 covers every loss (misalignment, vibration,
     decoherence), so decoherence alone leaves at least 0.78 (Pedalino 2026).

Sources:
  Fein YY et al. (2019) Nat. Phys. 15:1242 (v = 261 m/s, L = 0.98 m, d = 266 nm,
      2e-8 mbar, alpha = 4 pi eps0 x 1321 A^3, coherence time 7.5 ms)
  Pedalino S et al. (2026) Nature 649:866; arXiv:2507.21211 (v ~ 160 m/s,
      L = 0.983 m, d = 133 nm, ~9e-9 mbar, 172 kDa, W = 2.4 eV, E_i = 2.53 eV)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import amu, angstrom, h
from atlas.report import AtlasPoint, Check, write_result
from atlas.scattering import (blackbody_scattering_Lambda, decoherence_pressure,
                              london_C6, most_probable_speed, sigma_eff_vdw,
                              thermal_photon_wavelength)

HERE = Path(__file__).parent
T_GAS = 300.0
MBAR = 100.0

GASES = {  # mass amu, polarizability volume (A^3), ionization energy (eV)
    "N2": (28.01, 1.75, 15.58),
    "H2": (2.016, 0.80, 15.43),
}

EXPERIMENTS = {
    "porphyrin 26.8 kDa (Fein 2019)": dict(
        mass=26777.0, alpha=1321.0, I=7.0, v=261.0, L=0.98, d=266e-9,
        p_mbar=2e-8, observed_floor=0.87,
        I_note="ionization energy not reported; 7 eV assumed (typical large organic)"),
    "Na cluster 172 kDa (Pedalino 2026)": dict(
        mass=172000.0, alpha=None, I=2.53, v=160.0, L=0.983, d=133e-9,
        p_mbar=9e-9, observed_floor=0.78,
        I_note="first ionization threshold 2.53 eV from the paper"),
}


def sodium_cluster_polarizability(mass_amu):
    """Classical metal sphere: alpha = R^3, R = r_s N^(1/3), r_s(Na) = 2.08 A.

    Electron spill-out makes real clusters somewhat more polarizable; this is
    the lower estimate.
    """
    N = mass_amu / 22.99
    R = 2.08 * angstrom * N ** (1 / 3)
    return R**3, R


def main():
    checks, points, lines = [], [], []
    for name, ex in EXPERIMENTS.items():
        if ex["alpha"] is None:
            alpha, R = sodium_cluster_polarizability(ex["mass"])
            lines.append(f"{name}: radius {R / 1e-9:.2f} nm, alpha {alpha / angstrom**3:.3g} A^3")
        else:
            alpha = ex["alpha"] * angstrom**3
        t_flight = 2 * ex["L"] / ex["v"]
        # thermal photon scattering, separation ~ grating period
        Lam = blackbody_scattering_Lambda(alpha, 300.0)
        I_bb = math.exp(-Lam * ex["d"] ** 2 * t_flight)
        worst = 1.0
        for gas, (m_g, a_g, I_g) in GASES.items():
            C6 = london_C6(alpha, a_g * angstrom**3, ex["I"], I_g)
            v_g = most_probable_speed(m_g * amu, T_GAS)
            sig = sigma_eff_vdw(C6, ex["v"], v_g)
            p0 = decoherence_pressure(sig, 2 * ex["L"], T_GAS) / MBAR
            I_coll = math.exp(-ex["p_mbar"] / p0)
            total = I_coll * I_bb
            worst = min(worst, total)
            lines.append(f"{name} in {gas}: sigma_eff {sig / 1e-18:.0f} nm^2, p0 {p0:.2e} mbar, "
                         f"collisions leave {I_coll:.3f}, thermal-photon scattering leaves {I_bb:.6f}")
        checks.append(Check(
            name=f"record overlap left after the flight, {name} (worst of N2/H2)", unit="fraction",
            computed=round(worst, 3), measured_low=ex["observed_floor"],
            source=name, note=ex["I_note"]))
        checks.append(Check(
            name=f"thermal-photon scattering exponent, {name}", unit="dimensionless",
            computed=Lam * ex["d"] ** 2 * t_flight, measured_high=0.01, scored=False,
            note="far below threshold: 300 K photons (~10 um) cannot resolve ~100 nm"))
        lam_dB = h / (ex["mass"] * amu * ex["v"])
        points.append(AtlasPoint(
            label=name, boson="virtual photons (gas collisions); thermal photons",
            probe_wavelength_m=thermal_photon_wavelength(300.0), domain_size_m=ex["d"],
            temperature_K=300.0 if "porphyrin" in name else 77.0, time_s=t_flight,
            record_overlap=worst))
        lines.append(f"{name}: flight time {t_flight * 1e3:.1f} ms, de Broglie wavelength {lam_dB / 1e-15:.0f} fm")
    print("\n".join(lines))
    notes = ("Both superpositions survive because the vacuum makes collisions rare and "
             "room-temperature photons are far too long to resolve the path separation. "
             "The domain here is the full path separation, held for the whole flight.")
    write_result(HERE, "B3", "Largest superpositions: porphyrins (2019) and Na clusters (2026)",
                 checks, points, notes)


if __name__ == "__main__":
    main()
