import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.build import molecule as ase_molecule  # noqa: E402

from pyvtst import Molecule  # noqa: E402
from pyvtst.core.elements import element  # noqa: E402
from pyvtst.core.units import BOHR_TO_ANGSTROM  # noqa: E402


def test_convert_ase_water():
    atoms = ase_molecule("H2O")
    mol = Molecule.from_ase(atoms)
    assert mol.symbols == ("O", "H", "H")
    assert mol.charge == 0
    assert mol.mult == 1
    np.testing.assert_allclose(
        mol.coords, atoms.get_positions() / BOHR_TO_ANGSTROM, rtol=0, atol=1e-14
    )


def test_ase_masses_ignored():
    atoms = ase_molecule("H2O")
    mol = Molecule.from_ase(atoms)
    assert mol.masses[0] == element("O").mass
    assert mol.masses[0] != atoms.get_masses()[0]


def test_charge_and_multiplicity_from_info():
    atoms = ase.Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], info={"charge": 0, "multiplicity": 2}
    )
    mol = Molecule.from_ase(atoms)
    assert (mol.charge, mol.mult) == (0, 2)


def test_spin_key_taken_as_multiplicity():
    atoms = ase.Atoms("H", positions=[[0.0, 0.0, 0.0]], info={"spin": 2})
    assert Molecule.from_ase(atoms).mult == 2


def test_explicit_arguments_override_info():
    atoms = ase_molecule("O2")
    atoms.info["multiplicity"] = 3
    assert Molecule.from_ase(atoms, mult=1).mult == 1


def test_to_ase_output():
    mol = Molecule.from_ase(ase_molecule("H2O"), name="water")
    atoms = mol.to_ase()
    np.testing.assert_array_equal(atoms.get_masses(), mol.masses)
    assert atoms.info == {"charge": 0, "multiplicity": 1, "name": "water"}
    assert not atoms.pbc.any()
    assert atoms.calc is None


def test_round_trip():
    original = Molecule.from_ase(ase_molecule("CH3"), mult=2, name="methyl")
    back = Molecule.from_ase(original.to_ase())
    np.testing.assert_allclose(back.coords, original.coords, rtol=0, atol=1e-10)
    assert back.replace(coords=original.coords) == original
