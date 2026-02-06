"""Physical constants — single source of truth for the entire codebase.

All values sourced from ``scipy.constants`` (CODATA 2018).
Import from here instead of defining local constants.
"""

import scipy.constants as _sc

# Electromagnetic
e = _sc.e                     # Elementary charge [C]
epsilon_0 = _sc.epsilon_0     # Vacuum permittivity [F/m]
mu_0 = _sc.mu_0               # Vacuum permeability [H/m]
c = _sc.c                     # Speed of light [m/s]

# Masses
m_e = _sc.m_e                 # Electron mass [kg]
m_p = _sc.m_p                 # Proton mass [kg]
m_n = _sc.m_n                 # Neutron mass [kg]
m_d = 3.34358377e-27           # Deuterium mass [kg]

# Thermodynamic
k_B = _sc.k                   # Boltzmann constant [J/K]
h = _sc.h                     # Planck constant [J*s]
hbar = _sc.hbar               # Reduced Planck constant [J*s]

# Mathematical
pi = _sc.pi

# Derived
eV = _sc.eV                   # 1 eV in Joules
