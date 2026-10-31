"""The engine port: energies, gradients and Hessians in atomic units."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule

FloatArray = npt.NDArray[np.float64]

__all__ = ["Engine", "LevelOfTheory"]


@runtime_checkable
class Engine(Protocol):
    """A quantum-chemistry engine.

    Every method uses the molecule's charge and multiplicity, leaves the
    molecule unchanged and returns arrays the caller may modify.
    """

    @property
    def level(self) -> LevelOfTheory:
        """The level of theory every result is computed at."""
        ...

    def energy(self, mol: Molecule) -> float:
        """Total electronic energy in hartree."""
        ...

    def gradient(self, mol: Molecule) -> FloatArray:
        """Cartesian gradient dE/dx in hartree/bohr, shape (N, 3)."""
        ...

    def hessian(self, mol: Molecule) -> FloatArray:
        """Cartesian Hessian in hartree/bohr², not mass-weighted, shape (3N, 3N).

        Row and column index ``3*i + k`` is atom ``i``, component ``k``.
        """
        ...
