"""Physical constants and unit conversion factors.

pyvtst works internally in atomic units: lengths in bohr, energies in hartree,
masses in amu. Every value here is derived from the CODATA constants shipped
with the installed SciPy (``scipy.constants``); no physical value is typed in
by hand. The CODATA edition therefore follows the SciPy version.

Factors are named ``X_TO_Y``: multiply a value in X by ``X_TO_Y`` to get Y.
"""

from scipy import constants as _sc
from scipy.constants import physical_constants as _pc

# SI constants
BOLTZMANN = _sc.k  # J/K
PLANCK = _sc.h  # J s
HBAR = _sc.hbar  # J s
SPEED_OF_LIGHT = _sc.c  # m/s
AVOGADRO = _sc.N_A  # 1/mol
ELECTRON_MASS = _sc.m_e  # kg
ATOMIC_MASS_CONSTANT = _pc["atomic mass constant"][0]  # kg

# Length
BOHR_TO_M = _pc["Bohr radius"][0]
BOHR_TO_ANGSTROM = BOHR_TO_M / _sc.angstrom
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Energy
HARTREE_TO_J = _pc["Hartree energy"][0]
HARTREE_TO_EV = HARTREE_TO_J / _sc.eV
HARTREE_TO_KJ_PER_MOL = HARTREE_TO_J * _sc.N_A / 1e3
HARTREE_TO_KCAL_PER_MOL = HARTREE_TO_J * _sc.N_A / _sc.calorie / 1e3
HARTREE_TO_CM1 = HARTREE_TO_J / (_sc.h * _sc.c * 100.0)

J_TO_HARTREE = 1.0 / HARTREE_TO_J
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
KJ_PER_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KJ_PER_MOL
KCAL_PER_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_PER_MOL
CM1_TO_HARTREE = 1.0 / HARTREE_TO_CM1

# Mass
AMU_TO_ME = ATOMIC_MASS_CONSTANT / _sc.m_e
ME_TO_AMU = 1.0 / AMU_TO_ME
AMU_TO_KG = ATOMIC_MASS_CONSTANT

# Constants in atomic units
BOLTZMANN_AU = _sc.k / HARTREE_TO_J  # hartree/K
