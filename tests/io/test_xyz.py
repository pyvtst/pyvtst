from pathlib import Path

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.units import BOHR_TO_ANGSTROM

DATA = Path(__file__).parent / "data"

WATER_ANGSTROM = np.array(
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
)


def test_read_water():
    mol = Molecule.from_xyz(DATA / "water.xyz")
    assert mol.symbols == ("O", "H", "H")
    np.testing.assert_allclose(
        mol.coords, WATER_ANGSTROM / BOHR_TO_ANGSTROM, rtol=0, atol=1e-12
    )
    assert mol.name == "water"


def test_accepts_str_path_and_name_override():
    mol = Molecule.from_xyz(str(DATA / "water.xyz"), name="H2O")
    assert mol.name == "H2O"


def test_charge_and_multiplicity_passed_through():
    mol = Molecule.from_xyz(DATA / "h.xyz", mult=2)
    assert mol.mult == 2
    assert mol.symbols == ("H",)


def test_atom_count_mismatch():
    with pytest.raises(ValueError, match=r"declared 3 atoms, found 2"):
        Molecule.from_xyz(DATA / "count_mismatch.xyz")


def test_malformed_atom_line():
    with pytest.raises(ValueError, match=r"line 4"):
        Molecule.from_xyz(DATA / "bad_coordinate.xyz")


def test_multi_frame_reads_first():
    mol = Molecule.from_xyz(DATA / "multi_frame.xyz")
    assert mol.name == "frame 1"
    assert mol.coords[1, 2] * BOHR_TO_ANGSTROM == pytest.approx(0.74, abs=1e-12)


def test_round_trip(tmp_path):
    masses = [15.99491461957, 2.01410178, 1.00782503223]
    original = Molecule(
        ("O", "H", "H"),
        WATER_ANGSTROM / BOHR_TO_ANGSTROM,
        charge=1,
        mult=2,
        masses=masses,
        symmetry_number=1,
        name="HDO+",
    )
    path = tmp_path / "out.xyz"
    original.to_xyz(path)
    back = Molecule.from_xyz(path, charge=1, mult=2, masses=masses, symmetry_number=1)
    np.testing.assert_allclose(back.coords, original.coords, rtol=0, atol=1e-8)
    assert back.replace(coords=original.coords) == original
