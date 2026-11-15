"""The engine port: energies, gradients and Hessians in atomic units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule

FloatArray = npt.NDArray[np.float64]

__all__ = ["Engine", "EngineResult", "LevelOfTheory"]


@dataclass(frozen=True)
class EngineResult:
    """Energy, gradient and optionally Hessian of one molecule.

    Attributes:
        energy: Total electronic energy in hartree.
        gradient: Cartesian gradient dE/dx in hartree/bohr, shape (N, 3).
        hessian: Cartesian Hessian in hartree/bohr², not mass-weighted, shape
            (3N, 3N), or ``None``. Row and column index ``3*i + k`` is atom
            ``i``, component ``k``.
    """

    energy: float
    gradient: FloatArray
    hessian: FloatArray | None = None


@runtime_checkable
class Engine(Protocol):
    """A quantum-chemistry engine.

    :meth:`compute` returns everything one calculation gives, so a program
    that writes the energy, gradient and Hessian in a single run is run once.
    :meth:`energy`, :meth:`gradient` and :meth:`hessian` ask for one of them.
    A class that subclasses ``Engine`` inherits those three as wrappers over
    :meth:`compute`, and may override them when one quantity is cheaper on
    its own.

    Every method uses the molecule's charge and multiplicity, leaves the
    molecule unchanged and returns arrays the caller may modify.
    """

    @property
    def level(self) -> LevelOfTheory:
        """The level of theory every result is computed at."""
        ...

    def compute(self, mol: Molecule, *, hessian: bool) -> EngineResult:
        """Energy and gradient, and the Hessian when ``hessian`` is true.

        With ``hessian=False`` the result may still hold a Hessian, when the
        calculation gives one anyway, or ``None``.
        """
        ...

    def energy(self, mol: Molecule) -> float:
        """Total electronic energy in hartree."""
        return self.compute(mol, hessian=False).energy

    def gradient(self, mol: Molecule) -> FloatArray:
        """Cartesian gradient dE/dx in hartree/bohr, shape (N, 3)."""
        return self.compute(mol, hessian=False).gradient

    def hessian(self, mol: Molecule) -> FloatArray:
        """Cartesian Hessian in hartree/bohr², not mass-weighted, shape (3N, 3N).

        Row and column index ``3*i + k`` is atom ``i``, component ``k``.
        """
        result = self.compute(mol, hessian=True)
        if result.hessian is None:
            raise ValueError("hessian: compute(hessian=True) returned no Hessian")
        return result.hessian
