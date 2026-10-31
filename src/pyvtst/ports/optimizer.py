"""The optimizer port: minima and first-order saddle points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of a converged optimization.

    Attributes:
        molecule: The optimized structure. Only its coordinates differ from
            the input.
        energy: Energy in hartree at ``molecule``.
        n_iterations: Number of optimization steps taken.
    """

    molecule: Molecule
    energy: float
    n_iterations: int


@runtime_checkable
class Optimizer(Protocol):
    """A geometry optimizer that drives any :class:`Engine`.

    Both methods raise ``ConvergenceError`` when they do not converge, and
    let ``EngineError`` from the engine propagate unchanged.
    """

    def minimize(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        """Locate the minimum nearest to ``mol``."""
        ...

    def saddle(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        """Locate a first-order saddle point starting from the guess ``mol``.

        Checking that the result has exactly one imaginary mode is the
        caller's job.
        """
        ...

    def key(self) -> str:
        """Identify the optimizer and every setting that can change a result."""
        ...
