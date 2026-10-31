from scipy import constants
from scipy.constants import physical_constants

from pyvtst.core import units


def test_bohr_matches_codata():
    expected = physical_constants["Bohr radius"][0] / constants.angstrom
    assert expected == units.BOHR_TO_ANGSTROM


def test_hartree_matches_codata():
    assert physical_constants["Hartree energy"][0] == units.HARTREE_TO_J


def test_hartree_kcal_round_trip():
    back = 1.0 * units.HARTREE_TO_KCAL_PER_MOL * units.KCAL_PER_MOL_TO_HARTREE
    assert abs(back - 1.0) < 1e-12


def test_hartree_to_kcal_reference():
    assert abs(units.HARTREE_TO_KCAL_PER_MOL - 627.5095) < 1e-3


def test_hartree_to_wavenumber_reference():
    assert abs(units.HARTREE_TO_CM1 - 219474.63) < 1e-1


def test_other_reference_values():
    assert abs(units.HARTREE_TO_EV - 27.211386) < 1e-5
    assert abs(units.HARTREE_TO_KJ_PER_MOL - 2625.4996) < 1e-3
    assert abs(units.AMU_TO_ME - 1822.888486) < 1e-5
    assert abs(units.BOHR_TO_ANGSTROM * units.ANGSTROM_TO_BOHR - 1.0) < 1e-15


def test_public_import_path():
    from pyvtst.units import BOHR_TO_ANGSTROM  # noqa: PLC0415

    assert isinstance(BOHR_TO_ANGSTROM, float)
    assert BOHR_TO_ANGSTROM > 0
