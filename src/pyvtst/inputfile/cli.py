"""The ``pyvtst`` command: run an input file and write its report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pyvtst.inputfile.build import execute
from pyvtst.inputfile.registry import DEFAULT_REGISTRY, Registry
from pyvtst.inputfile.report import write_report
from pyvtst.inputfile.schema import InputError, load
from pyvtst.ports.errors import ConvergenceError, EngineError

EXIT_OK = 0
EXIT_CALCULATION = 1
EXIT_INPUT = 2
EXIT_INTERRUPTED = 130

# Errors a calculation can stop with that are not bugs in pyvtst: the user
# sees their message, not a traceback.
CALCULATION_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    ValueError,
    OSError,
    ConvergenceError,
    EngineError,
)


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    """Run ``pyvtst INPUT [-o OUTPUT]`` and return the exit code.

    Exit codes: 0 when every rate job finished, 2 when the input could not be
    loaded (nothing is computed and no report is written), 1 when a
    calculation failed (the report ends with the error), 130 on Ctrl-C.
    """
    from pyvtst import __version__  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="pyvtst",
        description=(
            "Compute rate constants from a pyvtst TOML input file and write a "
            "report (by default INPUT with the suffix .out)."
        ),
    )
    parser.add_argument("input", metavar="INPUT", help="the TOML input file")
    parser.add_argument(
        "-o", "--output", metavar="PATH", help="write the report to PATH"
    )
    parser.add_argument("--version", action="version", version=f"pyvtst {__version__}")
    args = parser.parse_args(argv)
    reg = DEFAULT_REGISTRY if registry is None else registry

    try:
        spec = load(args.input, reg)
    except InputError as exc:
        for error in exc.errors:
            print(error, file=sys.stderr)
        return EXIT_INPUT

    output = Path(args.output) if args.output else Path(args.input).with_suffix(".out")
    log = execute(spec, reg)
    for warning in log.warnings:
        print(f"pyvtst: warning: {warning}", file=sys.stderr)
    write_report(log, output)

    failure = log.exception
    if failure is None:
        print(output)
        return EXIT_OK
    if isinstance(failure, KeyboardInterrupt):
        print(f"pyvtst: interrupted; partial report in {output}", file=sys.stderr)
        return EXIT_INTERRUPTED
    if isinstance(failure, CALCULATION_ERRORS):
        print(f"pyvtst: error: {failure}", file=sys.stderr)
        print(f"pyvtst: partial report in {output}", file=sys.stderr)
        return EXIT_CALCULATION
    raise failure
