"""B1: C70 fullerenes decohered by collisions with background gas.

Event: a gas particle scatters off the fullerene through the van der Waals
interaction (virtual photons). Each collision leaves a complete which-path
record, because the gas particle's de Broglie wavelength (~0.1 Angstrom) is
ten thousand times shorter than the ~1 um path separation.

Model: Hornberger et al. 2003, the paper's own theory, with the paper's own
inputs (C6 values from PRL Table I; L = 0.22 m; v = 117 m/s; gas at 300 K).
Target: the measured decoherence pressures p0 (Appl. Phys. B Table 1).

Sources:
  Hornberger K et al. (2003) PRL 90:160401, arXiv:quant-ph/0303093 (C6, Eq. 5-6)
  Hornberger K et al. (2003) Appl. Phys. B 77:781, arXiv:quant-ph/0307238 (Eq. 9-11, Table 1)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import amu, h, meV, nm
from atlas.report import AtlasPoint, Check, write_result
from atlas.scattering import decoherence_pressure, most_probable_speed, sigma_eff_vdw

HERE = Path(__file__).parent

T_GAS = 300.0            # K, room temperature (assumed; not stated in the papers)
L = 0.22                 # m, grating separation for Table 1
V_M = 117.0              # m/s, mean molecular velocity for Table 1
MBAR = 100.0             # Pa per mbar
SEPARATION = 991e-9      # m, path separation at the middle grating ~ grating period

# gas: (mass in amu, C6 in meV nm^6 from PRL Table I, measured p0 and error in 1e-7 mbar
#       from Appl. Phys. B Table 1, the paper's own theory p0 in 1e-7 mbar)
GASES = {
    "H2":  (2.016, 0.80, 4.6, 0.7, 7.3),
    "D2":  (4.028, 0.77, 8.0, 1.2, 9.2),
    "He":  (4.003, 0.31, 10.7, 1.6, 13.8),
    "CH4": (16.04, 3.3, 8.1, 1.2, 7.9),
    "Ne":  (20.18, 0.71, 13.2, 2.0, 16.0),
    "N2":  (28.01, 2.1, 11.5, 1.7, 11.3),
    "Ar":  (39.95, 2.3, 10.8, 1.6, 11.8),
    "Kr":  (83.80, 3.4, 12.9, 1.9, 12.4),
    "Xe":  (131.3, 5.1, 10.6, 1.6, 11.5),
}


def main():
    checks, points = [], []
    lines = ["gas   sigma_eff(nm^2)  p0 computed  p0 paper-theory  p0 measured  (1e-7 mbar)"]
    for gas, (m, C6_meVnm6, p0_exp, dp0, p0_theo) in GASES.items():
        C6 = C6_meVnm6 * meV * nm**6
        v_g = most_probable_speed(m * amu, T_GAS)
        sig = sigma_eff_vdw(C6, V_M, v_g)
        p0 = decoherence_pressure(sig, 2 * L, T_GAS) / MBAR / 1e-7
        lines.append(f"{gas:4s} {sig / nm**2:10.0f} {p0:14.1f} {p0_theo:14.1f} {p0_exp:12.1f} ± {dp0}")
        checks.append(Check(
            name=f"decoherence pressure p0, {gas}", unit="1e-7 mbar",
            computed=round(p0, 2), measured=p0_exp,
            source="Hornberger 2003 Appl. Phys. B, Table 1",
            note=f"paper's own theory gave {p0_theo}; measured ± {dp0}"))
        # Sanity check against the paper's own theory column (not scored)
        checks.append(Check(
            name=f"p0 against the paper's theory, {gas}", unit="1e-7 mbar",
            computed=round(p0, 2), measured=p0_theo, scored=False,
            note="checks that this code reproduces the published model"))
        # Atlas point: the gas particle's wavelength is the probe; the domain is
        # the distance over which one collision leaves records alike.
        lam_gas = h / (m * amu * v_g)
        points.append(AtlasPoint(
            label=f"C70 + {gas} collision", boson="virtual photons (van der Waals)",
            probe_wavelength_m=lam_gas, domain_size_m=lam_gas / (2 * math.pi),
            temperature_K=T_GAS, time_s=2 * L / V_M,
            record_overlap=math.exp(-1.0)))
    print("\n".join(lines))
    notes = ("Saturated regime: separation ~1 um is far above every gas wavelength "
             "(0.1-0.5 Angstrom), so each collision decoheres completely and "
             "the overlap falls as exp(-collision count). H2 is the known "
             "outlier in the paper itself (theory 7.3, measured 4.6).")
    write_result(HERE, "B1", "C70 collisional decoherence (Talbot-Lau, 2003)", checks, points, notes)


if __name__ == "__main__":
    main()
