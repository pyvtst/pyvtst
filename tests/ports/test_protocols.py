import dataclasses

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.ports import (
    ConvergenceError,
    Engine,
    EngineError,
    LevelOfTheory,
    OptimizationResult,
    Optimizer,
)


class StubEngine:
    level = LevelOfTheory("stub", "none")

    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return np.zeros((mol.n_atoms, 3))

    def hessian(self, mol):
        return np.zeros((3 * mol.n_atoms, 3 * mol.n_atoms))


class NoHessian:
    level = LevelOfTheory("stub", "none")

    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return np.zeros((mol.n_atoms, 3))


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
