"""Shared fixtures: the analytic LEPS engine and H / H2 / H3 structures."""

import pytest

from pyvtst import Molecule
from tests.fakes.leps import LEPSEngine


@pytest.fixture
def leps_engine():
    return LEPSEngine()


@pytest.fixture
def h_atom():
    return Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, name="H")


@pytest.fixture
def h2_stretched():
    """H2 at 1.8 bohr, beyond the 1.401 bohr equilibrium."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], name="H2")


@pytest.fixture
def h3_ts_guess():
    """Slightly bent H3 with unequal distances (1.6 and 1.9 bohr)."""
    return Molecule(
        ["H", "H", "H"],
        [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]],
        mult=2,
        name="H3",
    )
