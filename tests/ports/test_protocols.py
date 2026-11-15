import dataclasses

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.ports import (
    ConvergenceError,
    Engine,
    EngineError,
    EngineResult,
    LevelOfTheory,
    OptimizationResult,
    Optimizer,
)


class StubEngine:
    level = LevelOfTheory("stub", "none")

    def compute(self, mol, *, hessian):
        n = 3 * mol.n_atoms
        return EngineResult(0.0, np.zeros((mol.n_atoms, 3)), np.zeros((n, n)))

    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return np.zeros((mol.n_atoms, 3))

    def hessian(self, mol):
        return np.zeros((3 * mol.n_atoms, 3 * mol.n_atoms))


class NoHessian:
    level = LevelOfTheory("stub", "none")

    def compute(self, mol, *, hessian):
        return EngineResult(0.0, np.zeros((mol.n_atoms, 3)))

    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return np.zeros((mol.n_atoms, 3))


class NoCompute:
    level = LevelOfTheory("stub", "none")

    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return np.zeros((mol.n_atoms, 3))

    def hessian(self, mol):
        return np.zeros((3 * mol.n_atoms, 3 * mol.n_atoms))


class Harmonic(Engine):
    """Implements only compute and inherits the other methods."""

    level = LevelOfTheory("harmonic", "none")

    def __init__(self, hessian=True):
        self.with_hessian = hessian
        self.runs: list[bool] = []

    def compute(self, mol, *, hessian):
        self.runs.append(hessian)
        n = 3 * mol.n_atoms
        full = np.eye(n) if self.with_hessian and hessian else None
        return EngineResult(float(np.sum(mol.coords**2)), 2 * mol.coords, full)


class StubOptimizer:
    def minimize(self, mol, engine):
        return OptimizationResult(mol, engine.energy(mol), 0)

    def saddle(self, mol, engine):
        return OptimizationResult(mol, engine.energy(mol), 0)

    def key(self):
        return "stub"


def test_stub_engine_satisfies_protocol():
    assert isinstance(StubEngine(), Engine)


def test_missing_hessian_fails_protocol():
    assert not isinstance(NoHessian(), Engine)


def test_missing_compute_fails_protocol():
    assert not isinstance(NoCompute(), Engine)


def test_subclass_inherits_wrappers_over_compute():
    engine = Harmonic()
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert isinstance(engine, Engine)
    assert engine.energy(mol) == 1.0
    np.testing.assert_array_equal(engine.gradient(mol), 2 * mol.coords)
    np.testing.assert_array_equal(engine.hessian(mol), np.eye(6))
    assert engine.runs == [False, False, True]


def test_inherited_hessian_needs_a_hessian():
    mol = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    with pytest.raises(ValueError, match="no Hessian"):
        Harmonic(hessian=False).hessian(mol)


def test_engine_result_is_immutable():
    result = EngineResult(0.0, np.zeros((1, 3)))
    assert result.hessian is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.energy = 1.0  # type: ignore[misc]


def test_stub_optimizer_satisfies_protocol():
    assert isinstance(StubOptimizer(), Optimizer)
    assert not isinstance(StubEngine(), Optimizer)


def test_optimization_result_is_immutable():
    mol = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    result = OptimizationResult(mol, -0.5, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.energy = 0.0  # type: ignore[misc]


def test_errors_are_runtime_errors():
    assert issubclass(EngineError, RuntimeError)
    assert issubclass(ConvergenceError, RuntimeError)


def test_convergence_error_carries_last_molecule():
    mol = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    err = ConvergenceError("no luck", mol)
    assert err.last_molecule is mol
    assert str(err) == "no luck"
    assert ConvergenceError("no luck").last_molecule is None
