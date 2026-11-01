"""pyvtst: thermal rate constants from conventional and variational TST."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from pyvtst import units
from pyvtst.core.mep import ReactionPath
from pyvtst.core.molecule import Molecule

if TYPE_CHECKING:
    from pyvtst.plot import plot_free_energy, plot_path, plot_paths
    from pyvtst.reaction import RateResult, Reaction

__version__ = "0.1.0"

__all__ = [
    "Molecule",
    "RateResult",
    "Reaction",
    "ReactionPath",
    "__version__",
    "plot_free_energy",
    "plot_path",
    "plot_paths",
    "units",
]

# Reaction pulls in the port and adapter packages, and the plot functions
# matplotlib when called, so they are loaded on first access; importing pyvtst
# (or pyvtst.core) alone stays light.
_LAZY = {
    "RateResult": "pyvtst.reaction",
    "Reaction": "pyvtst.reaction",
    "plot_free_energy": "pyvtst.plot",
    "plot_path": "pyvtst.plot",
    "plot_paths": "pyvtst.plot",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pyvtst' has no attribute {name!r}")
