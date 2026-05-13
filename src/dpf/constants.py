"""Physical constants used by implementation code.

These are standards-scoped implementation constants, not KR-scoped
experimental inputs or validation targets. Values should come from
``scipy.constants`` unless SciPy does not expose the required derived value.
Import from here instead of defining local constants.
"""

import scipy.constants as _sc

CONSTANTS_SCOPE = "standards_scoped_implementation_constants"
CONSTANTS_AUTHORITY = "scipy.constants"

# Electromagnetic
e = _sc.e                     # Elementary charge [C]
epsilon_0 = _sc.epsilon_0     # Vacuum permittivity [F/m]
mu_0 = _sc.mu_0               # Vacuum permeability [H/m]
c = _sc.c                     # Speed of light [m/s]

# Masses
m_e = _sc.m_e                 # Electron mass [kg]
m_p = _sc.m_p                 # Proton mass [kg]
m_n = _sc.m_n                 # Neutron mass [kg]
m_d = _sc.physical_constants["deuteron mass"][0]  # Deuteron mass [kg]
m_D2 = 2 * m_d                 # D2 molecular mass approximation [kg]

# Thermodynamic
k_B = _sc.k                   # Boltzmann constant [J/K]
h = _sc.h                     # Planck constant [J*s]
hbar = _sc.hbar               # Reduced Planck constant [J*s]

# Mathematical
pi = _sc.pi

# Derived
eV = _sc.eV                   # 1 eV in Joules
