import dataclasses

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.elements import element
from pyvtst.core.units import BOHR_TO_ANGSTROM

WATER_COORDS = np.array(
    [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]
)
CH4_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.19, 1.19, 1.19],
        [-1.19, -1.19, 1.19],
        [-1.19, 1.19, -1.19],
        [1.19, -1.19, -1.19],
    ]
)
CH3_COORDS = np.array(
    [[0.0, 0.0, 0.0], [2.04, 0.0, 0.0], [-1.02, 1.77, 0.0], [-1.02, -1.77, 0.0]]
)


def water(**kwargs):
    return Molecule(("O", "H", "H"), WATER_COORDS, **kwargs)


# Molecule fields and unit convention


def test_explicit_construction():
    mol = Molecule(("O", "H", "H"), WATER_COORDS, charge=0, mult=1)
    assert mol.symbols == ("O", "H", "H")
    assert mol.coords.shape == (3, 3)
    np.testing.assert_array_equal(mol.coords, WATER_COORDS)
    assert mol.coords.dtype == np.float64
    assert mol.charge == 0
    assert mol.mult == 1
    assert mol.symmetry_number == 1
    assert mol.name == ""


def test_symbols_canonicalized():
    mol = Molecule(["c", "CL", "h"], np.zeros((3, 3)) + np.arange(3)[:, None])
    assert mol.symbols == ("C", "Cl", "H")


# Default masses


def test_default_masses_most_abundant_isotope():
    mol = Molecule(("C", "H", "H", "H", "H"), CH4_COORDS)
    assert mol.masses.shape == (5,)
    assert mol.masses[0] == 12.0
    np.testing.assert_allclose(mol.masses[1:], 1.00782503, atol=1e-6)


def test_explicit_masses_isotopologue():
    masses = [12.0, 1.00782503, 1.00782503, 1.00782503, 2.01410178]
    mol = Molecule(("C", "H", "H", "H", "H"), CH4_COORDS, masses=masses)
    assert mol.masses[4] == 2.01410178
    np.testing.assert_array_equal(mol.masses, masses)


# Immutability


def test_field_assignment_fails():
    mol = water()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mol.charge = 1
    assert mol.charge == 0


def test_array_write_fails():
    mol = water()
    with pytest.raises(ValueError, match="read-only"):
        mol.coords[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        mol.masses[0] = 1.0
    np.testing.assert_array_equal(mol.coords, WATER_COORDS)


def test_caller_array_decoupled():
    coords = WATER_COORDS.copy()
    masses = np.array([15.99491461957, 1.00782503223, 1.00782503223])
    mol = Molecule(("O", "H", "H"), coords, masses=masses)
    coords[0, 0] = 99.0
    masses[0] = 99.0
    np.testing.assert_array_equal(mol.coords, WATER_COORDS)
    assert mol.masses[0] == 15.99491461957


# Basic derived properties


def test_derived_properties_water():
    mol = water()
    assert mol.n_atoms == 3
    assert mol.atomic_numbers == (8, 1, 1)
    assert mol.total_mass == float(np.sum(mol.masses))
    np.testing.assert_array_equal(mol.coords_angstrom, mol.coords * BOHR_TO_ANGSTROM)


# Input validation


def test_empty_symbols_rejected():
    with pytest.raises(ValueError, match="symbols"):
        Molecule((), np.zeros((0, 3)))


def test_unknown_element_rejected():
    with pytest.raises(ValueError, match="Xx"):
        Molecule(("Xx",), np.zeros((1, 3)))


def test_coordinate_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="coords"):
        Molecule(("O", "H", "H"), np.zeros((2, 3)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_coordinate_rejected(bad):
    coords = WATER_COORDS.copy()
    coords[1, 2] = bad
    with pytest.raises(ValueError, match="coords"):
        Molecule(("O", "H", "H"), coords)


@pytest.mark.parametrize("masses", [[16.0, 1.0], [16.0, 1.0, 0.0], [16.0, -1.0, 1.0]])
def test_bad_masses_rejected(masses):
    with pytest.raises(ValueError, match="masses"):
        water(masses=masses)


def test_multiplicity_below_one_rejected():
    with pytest.raises(ValueError, match="mult"):
        water(mult=0)


def test_symmetry_number_below_one_rejected():
    with pytest.raises(ValueError, match="symmetry_number"):
        water(symmetry_number=0)


def test_neutral_h_singlet_rejected():
    with pytest.raises(ValueError, match="multiplicity") as excinfo:
        Molecule(("H",), np.zeros((1, 3)), mult=1)
    assert "electron" in str(excinfo.value)


def test_too_many_unpaired_rejected():
    with pytest.raises(ValueError, match="multiplicity"):
        Molecule(("H",), np.zeros((1, 3)), mult=4)


def test_ch3_doublet_accepted():
    mol = Molecule(("C", "H", "H", "H"), CH3_COORDS, mult=2)
    assert mol.mult == 2


def test_charged_species_parity():
    Molecule(("H",), np.zeros((1, 3)), charge=1, mult=1)  # bare proton
    with pytest.raises(ValueError, match="multiplicity"):
        Molecule(("H",), np.zeros((1, 3)), charge=1, mult=2)


# Value equality


def test_equal_molecules():
    assert water() == water()


def test_different_coordinates_not_equal():
    coords = WATER_COORDS.copy()
    coords[2, 1] += 1e-9
    assert water() != Molecule(("O", "H", "H"), coords)


def test_other_fields_affect_equality():
    assert water() != water(name="w")
    assert water() != water(symmetry_number=2)


def test_unhashable():
    with pytest.raises(TypeError):
        hash(water())


# Derived modification


def test_replace_coordinates():
    mol = water(name="w")
    new_coords = WATER_COORDS * 1.01
    moved = mol.replace(coords=new_coords)
    np.testing.assert_array_equal(moved.coords, new_coords)
    assert moved.replace(coords=WATER_COORDS) == mol
    assert moved.symbols == mol.symbols
    assert moved.name == "w"
    np.testing.assert_array_equal(mol.coords, WATER_COORDS)


def test_replace_symbols_resets_default_masses():
    co = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.13]])
    no = co.replace(symbols=("N", "O"), mult=2)
    np.testing.assert_array_equal(no.masses, [element("N").mass, element("O").mass])


def test_replace_keeps_explicit_masses_when_symbols_unchanged():
    masses = [15.99491461957, 2.01410178, 1.00782503223]
    hdo = water(masses=masses)
    np.testing.assert_array_equal(
        hdo.replace(coords=WATER_COORDS * 1.01).masses, masses
    )
    np.testing.assert_array_equal(hdo.replace(symbols=["o", "h", "h"]).masses, masses)


def test_replace_symbols_with_explicit_masses():
    co = Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.13]])
    c13o = co.replace(symbols=("C", "O"), masses=[13.00335484, 15.99491462])
    assert c13o.masses[0] == 13.00335484


def test_replace_invalid_rejected():
    with pytest.raises(ValueError, match="coords"):
        water().replace(coords=np.zeros((2, 3)))
