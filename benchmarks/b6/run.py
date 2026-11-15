"""B6: how far a proton in ice is spread out.

Event: a neutron scatters off a proton in ice and the proton's momentum is
read from the recoil (deep inelastic neutron scattering). The neutron couples
through the strong nuclear force, which the atlas does not model; it is the
probe, entered as a known measuring device. What the atlas computes is the
proton's own domain: the spread of positions over which it stays
undistinguished, set by the vibrational quanta (phonons and molecular
vibrations, all electromagnetic) of the ice it sits in.

Model: quasi-harmonic proton. Each H2O molecule has six degrees of freedom
that move mainly its two protons: two O-H stretches, one bend, three
librations (hindered rotations). Per proton that is one stretch, half a bend
and one and a half librations, plus a small share of the molecule's
translation. Each harmonic degree of freedom carries kinetic energy

    K(omega) = (hbar omega / 4) coth(hbar omega / 2 kB T),

and the proton carries the fraction m_O/(m_O+m_H) of the stretch's kinetic
energy. Frequencies are the measured infrared band centres of ice Ih,
not fitted.

Target: proton mean kinetic energy 156 +/- 2 meV at 271 K (DINS, Flammini et al.
2012). The paper's own path-integral simulation gave 143 +/- 2 meV.

Sources:
  Flammini D et al. (2012) J. Chem. Phys. 136:024504 (measured 156 +/- 2 meV; PICPMD 143)
  Ice Ih band centres: O-H stretch ~3220 cm^-1, bend ~1650 cm^-1, libration ~840 cm^-1
  (Bertie JE, Whalley E (1964, 1967) J. Chem. Phys. 40:1637 and 46:1271)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.constants import amu, h, hbar, kB, meV, wavenumber_to_J, wavenumber_to_rad_per_s
from atlas.report import AtlasPoint, Check, write_result

HERE = Path(__file__).parent
T = 271.0
NU_STRETCH, NU_BEND, NU_LIB = 3220.0, 1650.0, 840.0     # cm^-1, ice Ih band centres
M_H, M_O = 1.00783 * amu, 15.9949 * amu


def kinetic(nu, T):
    E = wavenumber_to_J(nu)
    return (E / 4) / math.tanh(E / (2 * kB * T))


def spread(nu, mass, T):
    """rms position spread of a harmonic degree of freedom."""
    w = wavenumber_to_rad_per_s(nu)
    E = hbar * w
    return math.sqrt(hbar / (2 * mass * w) / math.tanh(E / (2 * kB * T)))


def main():
    frac = M_O / (M_O + M_H)
    K_stretch = frac * kinetic(NU_STRETCH, T)
    K_bend = 0.5 * kinetic(NU_BEND, T)
    K_lib = 1.5 * kinetic(NU_LIB, T)
    K_trans = 1.5 * kB * T * M_H / (2 * M_H + M_O)
    K_total = K_stretch + K_bend + K_lib + K_trans
    print(f"proton kinetic energy at {T} K:")
    for name, K in [("stretch", K_stretch), ("half bend", K_bend), ("1.5 librations", K_lib),
                    ("translation share", K_trans)]:
        print(f"  {name:18s} {K / meV:6.1f} meV")
    print(f"  total              {K_total / meV:6.1f} meV   (measured 156 +/- 2; PICPMD 143 +/- 2)")
    mu_OH = M_H * M_O / (M_H + M_O)
    s_par = spread(NU_STRETCH, mu_OH, T)
    s_perp = spread(NU_LIB, M_H, T)
    print(f"position spread: along O-H {s_par * 1e10:.3f} A, across {s_perp * 1e10:.3f} A")
    checks = [
        Check("proton mean kinetic energy in ice Ih", "meV", round(K_total / meV, 1),
              measured=156.0, source="Flammini 2012, DINS at 271 K",
              note="quasi-harmonic, ice infrared band centres, no fitting"),
        Check("against the path-integral simulation", "meV", round(K_total / meV, 1),
              measured=143.0, scored=False, source="Flammini 2012, PICPMD at 269 K"),
    ]
    # The probe for the proton's domain is the vibrational quantum that would record it;
    # its wavelength is the stretch's infrared wavelength in vacuum.
    lam_stretch = 1e-2 / NU_STRETCH
    points = [AtlasPoint("proton in ice Ih (along O-H)", boson="vibrational quanta (stretch)",
                         probe_wavelength_m=lam_stretch, domain_size_m=s_par, temperature_K=T,
                         time_s=None, record_overlap=None),
              AtlasPoint("proton in ice Ih (across O-H)", boson="vibrational quanta (libration)",
                         probe_wavelength_m=1e-2 / NU_LIB, domain_size_m=s_perp, temperature_K=T,
                         time_s=None, record_overlap=None)]
    notes = ("The proton is undistinguished over about 0.07 A along its bond and 0.1-0.15 A "
             "across it: a domain set entirely by its own zero-point motion, since at 271 K "
             "the stretch and bend are barely thermally excited.")
    write_result(HERE, "B6", "Proton momentum in hexagonal ice", checks, points, notes)


if __name__ == "__main__":
    main()
