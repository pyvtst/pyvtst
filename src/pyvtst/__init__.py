"""pyvtst: thermal rate constants from conventional and variational TST."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from pyvtst import units
from pyvtst.core.molecule import Molecule

if TYPE_CHECKING:
    from pyvtst.reaction import RateResult, Reaction

__version__ = "0.1.0.dev0"

__all__ = ["Molecule", "RateResult", "Reaction", "__version__", "units"]

# Reaction pulls in the port and adapter packages, so it is loaded on first
# access; importing pyvtst (or pyvtst.core) alone stays light.
_LAZY = {"RateResult": "pyvtst.reaction", "Reaction": "pyvtst.reaction"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pyvtst' has no attribute {name!r}")
