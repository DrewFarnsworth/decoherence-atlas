"""Physical constants (CODATA 2018 exact or recommended values), SI units."""

import math

h = 6.62607015e-34          # Planck constant, J s
hbar = h / (2 * math.pi)    # J s
c = 299792458.0             # speed of light, m/s
kB = 1.380649e-23           # Boltzmann constant, J/K
e = 1.602176634e-19         # elementary charge, C (also J per eV)
eps0 = 8.8541878128e-12     # vacuum permittivity, F/m
mu0 = 1.25663706212e-6      # vacuum permeability, N/A^2
amu = 1.66053906660e-27     # atomic mass unit, kg
m_p = 1.67262192369e-27     # proton mass, kg
NA = 6.02214076e23          # Avogadro constant, 1/mol

eV = e                      # J
meV = 1e-3 * eV
debye = 3.33564e-30         # C m
angstrom = 1e-10            # m
nm = 1e-9                   # m
fs = 1e-15                  # s

# Spectroscopic wavenumber (cm^-1) conversions
def wavenumber_to_J(nu_cm):
    """Energy of a quantum with wavenumber nu_cm (in cm^-1)."""
    return h * c * 100.0 * nu_cm


def wavenumber_to_rad_per_s(nu_cm):
    """Angular frequency for a wavenumber in cm^-1."""
    return 2 * math.pi * c * 100.0 * nu_cm


# Gyromagnetic ratios, rad s^-1 T^-1
gamma_e = 1.76085963023e11   # electron (magnitude)
gamma_13C = 6.728284e7       # carbon-13 nucleus
