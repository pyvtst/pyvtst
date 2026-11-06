"""TOML input files: describe a calculation in a file and run it.

Example::

    from pyvtst import inputfile

    spec = inputfile.load("h3.toml")        # validate only
    results = inputfile.run("h3.toml", output="h3.out")
    print(results[0].table())

The same files run from the shell with ``pyvtst h3.toml``.
"""

from __future__ import annotations

import os
import warnings

from pyvtst.inputfile.build import execute
from pyvtst.inputfile.registry import (
    DEFAULT_REGISTRY,
    AdapterEntry,
    Field,
    Registry,
)
from pyvtst.inputfile.report import write_report
from pyvtst.inputfile.schema import InputError, InputSpec, load
from pyvtst.reaction import RateResult

__all__ = [
    "DEFAULT_REGISTRY",
    "AdapterEntry",
    "Field",
    "InputError",
    "InputSpec",
    "Registry",
    "load",
    "run",
]


def run(
    path: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    registry: Registry | None = None,
) -> tuple[RateResult, ...]:
    """Load an input file, run every rate job and return the results in order.

    Args:
        path: The TOML input file.
        output: Where to write the report, or ``None`` for no report. When a
            calculation fails, the report is written before the error is
            raised, with everything that finished.
        registry: The engines and optimizers ``type`` may select. ``None``
            means :data:`DEFAULT_REGISTRY`.

    Warnings issued during the run are listed in the report and issued again
    to the caller as :class:`UserWarning`.

    Raises:
        InputError: The file could not be loaded; nothing was computed.
    """
    reg = DEFAULT_REGISTRY if registry is None else registry
    spec = load(path, reg)
    log = execute(spec, reg)
    if output is not None:
        write_report(log, output)
    for message in log.warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
    if log.exception is not None:
        raise log.exception
    return tuple(log.results)
