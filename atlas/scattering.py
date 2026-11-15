"""Scattering decoherence: the organising theory of the atlas.

For an environment that probes a particle by scattering, the record overlap
between two positions a distance dx apart decays as

    I(dx, t) = exp(-Lambda dx^2 t)      dx well below the probe wavelength
    I(dx, t) = exp(-Gamma t)            dx well beyond it (saturated)

Gas collisions in molecule interferometers sit in the saturated regime: the
gas particles' de Broglie wavelengths (~0.1 Angstrom) are tiny compared with
path separations (~100 nm to 1 um), so every collision leaves a complete
which-path record. Thermal photons sit in the other regime: their wavelengths
(~10 um at room temperature) dwarf the separations.
"""

import math

from scipy.special import gamma as Gamma
from scipy.special import zeta

from .constants import c, eV, hbar, kB


def most_probable_speed(mass_kg, T):
    """Most probable speed of a Maxwell-Boltzmann gas, sqrt(2 kB T / m)."""
    return math.sqrt(2 * kB * T / mass_kg)


def london_C6(alpha1_m3, alpha2_m3, I1_eV, I2_eV):
    """London estimate of the van der Waals C6 coefficient (J m^6).

    alpha are polarizability volumes alpha/(4 pi eps0) in m^3; I are
    ionization energies in eV. U(r) = -C6 / r^6.
    """
    I1, I2 = I1_eV * eV, I2_eV * eV
    return 1.5 * (I1 * I2 / (I1 + I2)) * alpha1_m3 * alpha2_m3


def sigma_eff_vdw(C6, v_m, v_gas):
    """Effective decoherence cross-section for a -C6/r^6 interaction.

    Hornberger et al. 2003 (Appl. Phys. B 77:781, Eq. 9-10): averaged over a
    thermal gas with most probable speed v_gas, for a beam at speed v_m.
    Valid for v_m small compared with v_gas (expansion in u = v_m/v_gas).
    """
    u = v_m / v_gas
    pref = 2 * (3 * math.pi**6 * C6 / (8 * hbar)) ** 0.4 / (
        math.sqrt(math.pi) * Gamma(0.4) * math.sin(math.pi / 5))
    G = Gamma(9 / 5) * (1 - u**2) + (2 / 3) * Gamma(14 / 5) * u**2
    return pref * v_gas**0.6 / v_m * G


def decoherence_pressure(sigma_eff, path_length, T_gas):
    """Pressure p0 (Pa) at which collisions reduce visibility to 1/e.

    V(p) = V(0) exp(-p/p0),   p0 = kB T / (path_length * sigma_eff).
    In a Talbot-Lau interferometer the decohering path is 2L (both stages).
    """
    return kB * T_gas / (path_length * sigma_eff)


def blackbody_scattering_Lambda(alpha_m3, T):
    """Localization rate (m^-2 s^-1) from scattering of thermal photons.

    Joos-Zeh long-wavelength result (Schlosshauer 2007, Eq. 3.73), with
    a^6 Re[(eps-1)/(eps+2)]^2 written as alpha^2 (polarizability volume):
        Lambda = 8! * 8 zeta(9) c alpha^2 / (9 pi) * (kB T / hbar c)^9
    Valid for separations far below the thermal wavelength.
    """
    return math.factorial(8) * 8 * zeta(9) * c * alpha_m3**2 / (9 * math.pi) * (kB * T / (hbar * c)) ** 9


def thermal_photon_wavelength(T):
    """Wavelength at the peak of the blackbody spectrum (Wien), m."""
    return 2.897771955e-3 / T
