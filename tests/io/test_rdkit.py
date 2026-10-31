import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit.Chem import AllChem

from pyvtst import Molecule


def test_methane():
    mol = Molecule.from_smiles("C")
    assert mol.n_atoms == 5
    assert sorted(mol.symbols) == ["C", "H", "H", "H", "H"]
    assert (mol.charge, mol.mult) == (0, 1)
    xyz = mol.coords_angstrom
    c = mol.symbols.index("C")
    distances = [np.linalg.norm(xyz[i] - xyz[c]) for i in range(5) if i != c]
    assert all(1.0 < d < 1.2 for d in distances)
    assert mol.name == "C"


def test_methyl_radical():
    mol = Molecule.from_smiles("[CH3]")
    assert mol.n_atoms == 4
    assert mol.mult == 2


def test_hydrogen_atom():
    mol = Molecule.from_smiles("[H]")
    assert mol.symbols == ("H",)
    assert mol.mult == 2
    np.testing.assert_array_equal(mol.coords, np.zeros((1, 3)))


def test_hydroxide_charge():
    mol = Molecule.from_smiles("[OH-]")
    assert (mol.charge, mol.mult) == (-1, 1)


def test_override_multiplicity():
    assert Molecule.from_smiles("O=O", mult=3).mult == 3


def test_deterministic():
    assert Molecule.from_smiles("CCO") == Molecule.from_smiles("CCO")


def test_invalid_smiles():
    with pytest.raises(ValueError, match="C1CC"):
        Molecule.from_smiles("C1CC")


def test_embedding_failure(monkeypatch):
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="CCO") as excinfo:
        Molecule.from_smiles("CCO")
    message = str(excinfo.value)
    assert "n_conformers" in message
    assert "seed" in message
