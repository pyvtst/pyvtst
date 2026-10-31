import sys

import numpy as np
import pytest

from pyvtst import Molecule


@pytest.fixture
def water():
    return Molecule(
        ("O", "H", "H"), [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]
    )


def test_missing_ase_to_ase(monkeypatch, water):
    monkeypatch.setitem(sys.modules, "ase", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[ase\]"'):
        water.to_ase()


def test_missing_ase_from_ase(monkeypatch):
    monkeypatch.setitem(sys.modules, "ase", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[ase\]"'):
        Molecule.from_ase(object())  # type: ignore[arg-type]


def test_missing_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[rdkit\]"') as excinfo:
        Molecule.from_smiles("C")
    assert excinfo.value.__cause__ is not None


def test_xyz_works_without_optional_packages(monkeypatch, water, tmp_path):
    monkeypatch.setitem(sys.modules, "ase", None)
    monkeypatch.setitem(sys.modules, "rdkit", None)
    path = tmp_path / "w.xyz"
    water.to_xyz(path)
    np.testing.assert_allclose(Molecule.from_xyz(path).coords, water.coords, atol=1e-8)
