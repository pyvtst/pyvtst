"""Input-file fixtures: a registry with the LEPS engine and an exact optimizer."""

import pytest

from pyvtst.inputfile import DEFAULT_REGISTRY, AdapterEntry, Field
from pyvtst.ports import ConvergenceError, OptimizationResult
from tests.fakes.leps import R0, LEPSEngine, saddle_distance

# H2 + H on the LEPS surface. The "exact" optimizer replaces the guesses with
# the exact H2 minimum and saddle point, so nothing is optimized.
H3_INPUT = '''\
title = "H2 + H on LEPS"

[engine]
type = "leps"

[optimizer]
type = "exact"

[reactants.r1]
name = "H2"
units = "bohr"
xyz = """
H 0.0 0.0 0.0
H 1.8 0.0 0.0
"""

[reactants.r2]
name = "H"
mult = 2
xyz = "H 0 0 0"

[ts]
name = "H3"
mult = 2
units = "bohr"
xyz = """
H -1.6 0.0 0.0
H  0.0 0.0 0.0
H  1.9 0.0 0.0
"""

[[rates]]
T = [300, 1000]
tunneling = "eckart"
'''

# The same reaction with the built-in PySCF engine, for load-only tests.
PYSCF_INPUT = '''\
[engine]
type = "pyscf"
method = "B3LYP"
basis = "def2-SVP"

[reactants.r1]
symmetry_number = 2
xyz = """
H 0.0  0.0 0.0
H 0.74 0.0 0.0
"""

[reactants.r2]
mult = 2
xyz = "H 0 0 0"

[ts]
mult = 2
symmetry_number = 2
xyz = """
H -0.85 0.0 0.0
H  0.0  0.0 0.0
H  1.0  0.0 0.0
"""

[[rates]]
T = 300
'''


class CountingLEPS:
    """The LEPS engine, counting calls; raises ``error`` on the first call if set."""

    def __init__(self, error=None):
        self.inner = LEPSEngine()
        self.calls = 0
        self.error = error

    @property
    def level(self):
        return self.inner.level

    def _count(self):
        if self.error is not None:
            raise self.error
        self.calls += 1

    def energy(self, mol):
        self._count()
        return self.inner.energy(mol)

    def gradient(self, mol):
        self._count()
        return self.inner.gradient(mol)

    def hessian(self, mol):
        self._count()
        return self.inner.hessian(mol)


class ExactOptimizer:
    """Returns the exact LEPS saddle point and H2 minimum without optimizing."""

    def __init__(self, fail_saddle=False):
        self.fail_saddle = fail_saddle

    def key(self):
        return "exact"

    def minimize(self, mol, engine):
        return self._result(mol, engine, [[0.0, 0.0, 0.0], [R0, 0.0, 0.0]])

    def saddle(self, mol, engine):
        if self.fail_saddle:
            raise ConvergenceError("did not converge in 300 steps")
        r = saddle_distance()
        return self._result(
            mol, engine, [[-r, 0.0, 0.0], [0.0, 0.0, 0.0], [r, 0.0, 0.0]]
        )

    def _result(self, mol, engine, coords):
        out = mol.replace(coords=coords)
        return OptimizationResult(out, engine.energy(out), 0)


def leps_registry(engine):
    """The default registry plus engine ``"leps"`` (``engine``) and optimizer
    ``"exact"`` (option ``fail_saddle``)."""
    return DEFAULT_REGISTRY.with_engine(
        "leps", AdapterEntry(fields=(), build=lambda options: engine)
    ).with_optimizer(
        "exact",
        AdapterEntry(
            fields=(Field("fail_saddle", "bool", default=False),),
            build=lambda options: ExactOptimizer(**options),
        ),
    )


@pytest.fixture
def leps():
    return CountingLEPS()


@pytest.fixture
def registry(leps):
    return leps_registry(leps)


@pytest.fixture
def write(tmp_path):
    """Write ``text`` to ``tmp_path / name`` and return the path."""

    def _write(text, name="h3.toml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
