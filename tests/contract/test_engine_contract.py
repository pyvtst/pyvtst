"""Conformance checks every engine must pass. Add new engines to ENGINES."""

import importlib.util

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.finite_difference import fd_hessian
from pyvtst.ports import Engine, LevelOfTheory
from tests.fakes.leps import LEPSEngine

STEP = 1e-4


def _leps():
    mol = Molecule(
        ["H", "H", "H"],
        [[-1.6, 0.1, 0.0], [0.0, 0.0, 0.05], [1.9, 0.2, 0.0]],
        mult=2,
    )
    return LEPSEngine(), mol


def _pyscf_hf():
    from pyvtst.engines import PySCFEngine  # noqa: PLC0415

    mol = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.22], [0.0, 1.45, -0.9], [0.0, -1.4, -0.88]],
        name="water",
    )
    # A 1e-4 bohr step amplifies SCF noise 10^4-fold; the default conv_tol of
    # 1e-10 leaves up to ~3e-3 hartree/bohr² of noise in the numeric Hessian.
    return PySCFEngine("HF", "sto-3g", conv_tol=1e-12), mol


needs_pyscf = pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None, reason="PySCF not installed"
)

ENGINES = [
    pytest.param(_leps, id="leps"),
    pytest.param(_pyscf_hf, id="pyscf-hf-sto3g", marks=needs_pyscf),
]


@pytest.fixture(params=ENGINES, scope="module")
def case(request):
    return request.param()


def test_satisfies_protocol(case):
    engine, _ = case
    assert isinstance(engine, Engine)
    assert isinstance(engine.level, LevelOfTheory)


def test_shapes_and_types(case):
    engine, mol = case
    n = mol.n_atoms
    assert type(engine.energy(mol)) is float
    grad = engine.gradient(mol)
    assert grad.shape == (n, 3)
    assert grad.dtype == np.float64
    hess = engine.hessian(mol)
    assert hess.shape == (3 * n, 3 * n)
    np.testing.assert_allclose(hess, hess.T, rtol=0, atol=1e-10)


def test_gradient_is_derivative_of_energy(case):
    engine, mol = case
    x0 = mol.coords.ravel()
    numeric = np.empty_like(x0)
    for j in range(x0.size):
        x = x0.copy()
        x[j] += STEP
        e_plus = engine.energy(mol.replace(coords=x.reshape(-1, 3)))
        x[j] -= 2 * STEP
        e_minus = engine.energy(mol.replace(coords=x.reshape(-1, 3)))
        numeric[j] = (e_plus - e_minus) / (2 * STEP)
    np.testing.assert_allclose(engine.gradient(mol).ravel(), numeric, rtol=0, atol=1e-5)


def test_hessian_is_derivative_of_gradient(case):
    engine, mol = case
    numeric = fd_hessian(
        lambda x: engine.gradient(mol.replace(coords=x)), mol.coords, STEP
    )
    np.testing.assert_allclose(engine.hessian(mol), numeric, rtol=0, atol=1e-4)


def test_input_unchanged_and_outputs_owned(case):
    engine, mol = case
    before = mol.replace()
    grad = engine.gradient(mol)
    hess = engine.hessian(mol)
    grad[...] = 0.0
    hess[...] = 0.0
    assert mol == before
    assert np.any(engine.gradient(mol) != 0.0)
