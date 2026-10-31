"""Exceptions raised by engines and optimizers."""

from __future__ import annotations

from pyvtst.core.molecule import Molecule


class EngineError(RuntimeError):
    """A calculation could not produce a trustworthy result (e.g. SCF failed)."""


class ConvergenceError(RuntimeError):
    """An optimizer did not converge.

    Attributes:
        last_molecule: The last geometry evaluated, if any. It can be a
            rejected trial step rather than the best point found, so check it
            before restarting from it.
    """

    def __init__(self, message: str, last_molecule: Molecule | None = None) -> None:
        super().__init__(message)
        self.last_molecule = last_molecule
