"""H + H2 minimum-energy path at UHF/6-31G with PySCF (slow)."""

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("geometric")
from pyscf import gto
from pyscf.hessian.thermo import harmonic_analysis

from pyvtst import Molecule, Reaction
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from pyvtst.engines import PySCFEngine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rxn():
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], symmetry_number=2)
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    guess = Molecule(
        ["H", "H", "H"],
        [[-1.7, 0.0, 0.0], [0.0, 0.0, 0.0], [1.8, 0.05, 0.0]],
        mult=2,
        symmetry_number=2,
    )
    # HF on a doublet is unrestricted: UHF/6-31G.
    return Reaction([h2, h], guess, PySCFEngine("HF", "6-31G"))


def test_mep_methods_agree_at_uhf(rxn):
    lqa = rxn.compute_mep(-0.1, 0.1, ds=0.02)
    euler = rxn.compute_mep(-0.1, 0.1, ds=0.002, method="euler")
    assert lqa.oriented
    for s in (-0.1, 0.1):
        a = lqa.v_mep[np.argmin(np.abs(lqa.s - s))]
        b = euler.v_mep[np.argmin(np.abs(euler.s - s))]
        assert abs(a - b) * HARTREE_TO_KCAL_PER_MOL < 0.1


def test_mep_saddle_frequencies_match_pyscf(rxn):
    path = rxn.compute_mep(-0.1, 0.1, ds=0.02)
    ts = rxn.optimize().ts
    mol = ts.molecule
    pyscf_mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(mol.symbols, mol.coords, strict=True)],
        basis="6-31G",
        spin=1,
        unit="Bohr",
    )
    n = mol.n_atoms
    hessian = rxn.engine.hessian(mol).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    result = harmonic_analysis(
        pyscf_mol, hessian, imaginary_freq=False, mass=np.array(mol.masses)
    )
    reference = np.sort(np.asarray(result["freq_wavenumber"]))
    generalized = path.frequencies[path.ts_index]
    assert generalized is not None
    np.testing.assert_allclose(generalized, reference[reference > 0], atol=1.0)
