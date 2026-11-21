"""ORCA 6: executable lookup, keywords and blocks, result files and running jobs.

Shared by :class:`pyvtst.engines.OrcaEngine` and
:class:`pyvtst.optimizers.OrcaOptimizer`. ORCA behaviours relied on (observed
with ORCA 6.0.1):

- ``orca job.inp`` run in a directory writes its log to stdout, and the
  result files ``job.engrad``, ``job.hess`` and ``job.xyz`` there. ORCA must be
  called with the full path of its real executable, because it finds its
  helper programs next to it.
- The exit status does not show every failure: a non-converged SCF and an
  optimization that stops at its cycle limit both exit with status 0. A
  successful log ends with ``****ORCA TERMINATED NORMALLY****``, and a failed
  one prints ``ORCA finished by error termination``.
- ``job.engrad`` lists the number of atoms, the energy, the gradient dE/dx
  (hartree/bohr) and the atomic numbers with the coordinates (bohr, 7
  decimals), in the input's frame.
- ``job.hess`` holds the full Cartesian Hessian (hartree/bohr², not
  mass-weighted) in ``$hessian``, in blocks of up to 5 columns, and the atoms
  in ``$atoms`` (bohr), moved to the centre of mass but not rotated.
- ``job.xyz`` is the last geometry of an optimization, in angstrom.

``subprocess`` only ever runs the user's own ORCA, from a list of arguments and
without a shell.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess  # nosec B404
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import element, formula
from pyvtst.core.molecule import Molecule
from pyvtst.core.units import ANGSTROM_TO_BOHR
from pyvtst.programs import process
from pyvtst.programs.process import job_name as job_name
from pyvtst.programs.process import label as label

FloatArray = npt.NDArray[np.float64]

REFERENCES = ("auto", "restricted", "unrestricted")
HESSIANS = ("analytic", "numerical")

# Coordinates in the result files must match the input this closely (bohr);
# anything else means ORCA moved the molecule. ``job.engrad`` has 7 decimals.
COORDINATE_TOLERANCE = 1e-6

# Lines of the ORCA log quoted in error messages.
LOG_TAIL_LINES = 20

INPUT_NAME = "job.inp"
LOG_NAME = "job.out"
ERR_NAME = "job.err"
ENGRAD_NAME = "job.engrad"
HESS_NAME = "job.hess"
XYZ_NAME = "job.xyz"

# Result files read after a run.
RESULT_NAMES = (ENGRAD_NAME, HESS_NAME, XYZ_NAME)

# Bulky scratch files that are removed from a kept job directory.
SCRATCH_PATTERNS = (
    "*.tmp",
    "*.tmp.*",
    "*.gbw",
    "*.densities",
    "*.bas[0-9]*",
    "*.hostnames",
    "*.interp",
)

NORMAL_MARKER = "ORCA TERMINATED NORMALLY"
ERROR_MARKER = "error termination"
HEADER_MARKER = "Program Version"
NO_ANALYTIC_HESSIAN = "please use NumFreq"


# --- Executable lookup -------------------------------------------------------


def find_orca(executable: str | os.PathLike[str] | None) -> str:
    """Locate ORCA: ``executable``, else ``$PYVTST_ORCA``, else ``orca`` on PATH.

    See :func:`pyvtst.programs.process.find_executable`. Symbolic links are
    resolved, because ORCA finds its helper programs next to the path it is
    called with.
    """
    return process.find_executable(
        "orca", "PYVTST_ORCA", "executable", executable, resolve_links=True
    )


def check_run_settings(nprocs: object, maxcore: object, timeout: object) -> None:
    """Validate the run settings shared by the ORCA engine and optimizer."""
    process.check_run_settings(nprocs, timeout)
    if isinstance(maxcore, bool) or not isinstance(maxcore, int) or maxcore < 1:
        raise ValueError(
            f"maxcore: must be an integer of at least 1 (MB per process), "
            f"got {maxcore!r}"
        )


# --- Keywords and blocks -----------------------------------------------------

# Keywords the decks set themselves; user keywords may not name them. A keyword
# is also reserved when it ends in ``Opt`` (TightOpt, COpt, ...), starts with
# ``NEB-`` or ``GOAT-``, or is ``PAL`` followed by digits.
RESERVED_KEYWORDS = (
    "SP",
    "EnGrad",
    "NumGrad",
    "Opt",
    "OptTS",
    "L-OptTS",
    "ScanTS",
    "Freq",
    "AnFreq",
    "NumFreq",
    "IRC",
    "MD",
    "NEB",
    "GOAT",
    "Bohrs",
    "Angs",
    "RHF",
    "UHF",
    "ROHF",
    "RKS",
    "UKS",
    "ROKS",
    "MORead",
    "UseSym",
)
_RESERVED_LOWER = frozenset(k.lower() for k in RESERVED_KEYWORDS)

# Blocks the decks write themselves, and what to use instead.
RESERVED_BLOCKS = {
    "pal": "use nprocs",
    "maxcore": "use maxcore",
    "geom": "the optimizer sets it",
    "freq": "pyvtst sets it",
    "coords": "pyvtst sets it",
    "irc": "not supported",
    "neb": "not supported",
    "md": "not supported",
}

_BLOCK_NAME = re.compile(r"(?<![\w%])%\s*([A-Za-z]\w*)")


def _is_reserved(token: str) -> bool:
    low = token.lower()
    return (
        low in _RESERVED_LOWER
        or low.endswith("opt")
        or low.startswith(("neb-", "goat-"))
        or re.fullmatch(r"pal\d+", low) is not None
    )


def normalize_keywords(keywords: str) -> str:
    """Fold case and collapse whitespace; reject reserved keywords and ``#``.

    ORCA keywords are case-insensitive and whitespace-separated, so equivalent
    spellings normalize to the same string.

    Raises:
        ValueError: ``keywords`` is not a string, contains ``#`` (ORCA reads
            the rest of the line as a comment), or names a keyword the decks
            set (:data:`RESERVED_KEYWORDS`).
    """
    if not isinstance(keywords, str):
        raise ValueError(f"keywords: expected a string, got {keywords!r}")
    if "#" in keywords:
        raise ValueError(
            "keywords: '#' starts a comment in ORCA input and would hide the "
            "keywords pyvtst adds after it"
        )
    collapsed = " ".join(keywords.split())
    for token in collapsed.split():
        if _is_reserved(token):
            raise ValueError(
                f"keywords: {token!r} is set by pyvtst and cannot be given "
                f"(job types, units, parallel settings and references are "
                f"reserved)"
            )
    return collapsed.lower()


def normalize_blocks(blocks: str) -> str:
    """Keep line breaks and case; strip each line, collapse its spaces and drop
    empty lines. Reject blocks that conflict with the decks.

    Raises:
        ValueError: ``blocks`` is not a string, contains ``#``, does not start
            with ``%``, names a reserved block (:data:`RESERVED_BLOCKS`), or
            contains ``RunTyp``, a line that starts with ``!`` or ``*``, or
            ``$new_job``.
    """
    if not isinstance(blocks, str):
        raise ValueError(f"blocks: expected a string, got {blocks!r}")
    if "#" in blocks:
        raise ValueError(
            "blocks: '#' starts a comment in ORCA input and would hide the "
            "rest of its line, including a closing 'end'"
        )
    lines = [" ".join(line.split()) for line in blocks.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    if not lines[0].startswith("%"):
        raise ValueError(f"blocks: must start with '%', got {lines[0]!r}")
    text = "\n".join(lines)
    for line in lines:
        if line[0] in "!*":
            raise ValueError(f"blocks: a line cannot start with {line[0]!r}: {line!r}")
    if "$new_job" in text.lower():
        raise ValueError("blocks: '$new_job' is not supported")
    if re.search(r"\bruntyp\b", text, re.IGNORECASE):
        raise ValueError("blocks: 'RunTyp' is set by pyvtst and cannot be given")
    for name in _BLOCK_NAME.findall(text):
        if name.lower() in RESERVED_BLOCKS:
            raise ValueError(
                f"blocks: '%{name}' is set by pyvtst and cannot be given "
                f"({RESERVED_BLOCKS[name.lower()]})"
            )
    return text


# --- Decks -------------------------------------------------------------------

# The level's basis for a method that sets its own basis (e.g. r2SCAN-3c).
NO_BASIS = "-"

CONVERGENCE = ("normal", "tight", "verytight")

# Minimum-search keyword, and the keyword added to OptTS, for each convergence.
_OPT_KEYWORD = {"normal": "Opt", "tight": "TightOpt", "verytight": "VeryTightOpt"}
_TS_KEYWORD = {"normal": "", "tight": "TightOpt", "verytight": "VeryTightOpt"}


def reference_keyword(reference: str, mult: int) -> str:
    """The reference keyword for ``reference``: ``""``, ``"RHF"`` or ``"UHF"``.

    ``RHF`` and ``UHF`` also select RKS and UKS for DFT methods.

    Raises:
        ValueError: ``reference`` is ``"restricted"`` and ``mult`` is above 1.
    """
    if reference == "auto":
        return ""
    if reference == "unrestricted":
        return "UHF"
    if mult != 1:
        raise ValueError(
            f"reference: a restricted reference needs multiplicity 1, got {mult}"
        )
    return "RHF"


def _deck(
    mol: Molecule,
    *,
    method: str,
    basis: str,
    reference: str,
    keywords: str,
    blocks: str,
    job: str,
    own_blocks: str,
    title: str,
    nprocs: int,
    maxcore: int,
) -> str:
    prefix = reference_keyword(reference, mol.mult)
    line = " ".join(
        part
        for part in (
            prefix,
            method,
            "" if basis == NO_BASIS else basis,
            keywords,
            job,
            "Bohrs",
        )
        if part
    )
    pal = f"%pal nprocs {nprocs} end\n" if nprocs > 1 else ""
    user = f"{blocks}\n" if blocks else ""
    atoms = "".join(
        f"  {symbol} {x:.12f} {y:.12f} {z:.12f}\n"
        for symbol, (x, y, z) in zip(mol.symbols, mol.coords.tolist(), strict=True)
    )
    return (
        f"# pyvtst {title} {formula(mol.symbols)}\n"
        f"! {line}\n"
        f"{pal}"
        f"%maxcore {maxcore}\n"
        f"{own_blocks}"
        f"{user}"
        "\n"
        f"* xyz {mol.charge} {mol.mult}\n"
        f"{atoms}"
        "*\n"
    )


def engine_deck(
    mol: Molecule,
    *,
    method: str,
    basis: str,
    reference: str,
    keywords: str,
    blocks: str,
    hessian: bool,
    numerical: bool,
    nprocs: int,
    maxcore: int,
) -> str:
    """The ORCA input for an energy and gradient (``EnGrad``) on ``mol``, and the
    Hessian too when ``hessian`` (``Freq``, or ``NumFreq`` when ``numerical``).

    Every deck is a gradient job, so all runs use the same SCF convergence.
    Hessian decks fix ``TransInvar true`` (ORCA's default), so a later change
    of the default cannot change results. ``basis`` is :data:`NO_BASIS` for a
    method that sets its own basis.

    Raises:
        ValueError: see :func:`reference_keyword`.
    """
    job = "EnGrad"
    own = ""
    if hessian:
        job += " NumFreq" if numerical else " Freq"
        own = "%freq\n  TransInvar true\nend\n"
    return _deck(
        mol,
        method=method,
        basis=basis,
        reference=reference,
        keywords=keywords,
        blocks=blocks,
        job=job,
        own_blocks=own,
        title="freq" if hessian else "engrad",
        nprocs=nprocs,
        maxcore=maxcore,
    )


def opt_deck(
    mol: Molecule,
    *,
    method: str,
    basis: str,
    reference: str,
    keywords: str,
    blocks: str,
    saddle: bool,
    convergence: str,
    max_cycles: int,
    numerical: bool,
    nprocs: int,
    maxcore: int,
) -> str:
    """The ORCA input for an optimization of ``mol`` to a minimum (``Opt``,
    ``TightOpt`` or ``VeryTightOpt``) or a saddle point (``OptTS`` with the same
    criteria and an initial Hessian, numerical when ``numerical``).

    Raises:
        ValueError: see :func:`reference_keyword`.
    """
    if saddle:
        job = " ".join(k for k in ("OptTS", _TS_KEYWORD[convergence]) if k)
        extra = "  Calc_Hess true\n" + ("  NumHess true\n" if numerical else "")
    else:
        job = _OPT_KEYWORD[convergence]
        extra = ""
    own = f"%geom\n  MaxIter {max_cycles}\n{extra}end\n"
    return _deck(
        mol,
        method=method,
        basis=basis,
        reference=reference,
        keywords=keywords,
        blocks=blocks,
        job=job,
        own_blocks=own,
        title="ts" if saddle else "opt",
        nprocs=nprocs,
        maxcore=maxcore,
    )


# --- Result files ------------------------------------------------------------


def _data_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _number(token: str, what: str) -> float:
    try:
        return float(token.replace("D", "E"))
    except ValueError:
        raise ValueError(f"{what}: expected a number, got {token!r}") from None


@dataclass(frozen=True)
class Engrad:
    """The contents of a ``job.engrad`` file, in atomic units."""

    energy: float
    gradient: FloatArray
    numbers: tuple[int, ...]
    coords: FloatArray


def parse_engrad(text: str) -> Engrad:
    """Read a ``job.engrad`` file.

    Raises:
        ValueError: The text is not in the ``.engrad`` layout.
    """
    lines = _data_lines(text)
    if len(lines) < 2:
        raise ValueError(f"{ENGRAD_NAME}: too short")
    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError(
            f"{ENGRAD_NAME}: expected the number of atoms, got {lines[0]!r}"
        ) from None
    if n < 1 or len(lines) != 2 + 3 * n + n:
        raise ValueError(
            f"{ENGRAD_NAME}: expected {2 + 4 * n} data lines for {n} atoms, "
            f"got {len(lines)}"
        )
    energy = _number(lines[1], ENGRAD_NAME)
    gradient = np.array(
        [_number(t, ENGRAD_NAME) for t in lines[2 : 2 + 3 * n]], dtype=np.float64
    ).reshape(n, 3)
    numbers: list[int] = []
    coords = np.zeros((n, 3))
    for i, line in enumerate(lines[2 + 3 * n :]):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{ENGRAD_NAME}: bad atom line {line!r}")
        try:
            numbers.append(int(fields[0]))
        except ValueError:
            raise ValueError(f"{ENGRAD_NAME}: bad atom line {line!r}") from None
        coords[i] = [_number(t, ENGRAD_NAME) for t in fields[1:]]
    return Engrad(energy, gradient, tuple(numbers), coords)


def _check_atoms(
    numbers: tuple[int, ...] | list[int], mol: Molecule, what: str
) -> None:
    expected = [element(s).atomic_number for s in mol.symbols]
    if list(numbers) != expected:
        raise ValueError(
            f"{what}: atomic numbers {list(numbers)} differ from the input's {expected}"
        )


def engrad_results(engrad: Engrad, mol: Molecule) -> tuple[float, FloatArray]:
    """Energy and gradient (N, 3), after checking atoms and geometry.

    Raises:
        ValueError: The atoms or the geometry differ from ``mol``'s (to
            :data:`COORDINATE_TOLERANCE`).
    """
    _check_atoms(engrad.numbers, mol, ENGRAD_NAME)
    deviation = float(np.max(np.abs(engrad.coords - mol.coords)))
    if deviation > COORDINATE_TOLERANCE:
        raise ValueError(
            f"{ENGRAD_NAME}: the geometry differs from the input by up to "
            f"{deviation:.3g} bohr (was the molecule reoriented?)"
        )
    return engrad.energy, engrad.gradient.copy()


@dataclass(frozen=True)
class Hess:
    """The Hessian and the atoms of a ``job.hess`` file, in atomic units."""

    hessian: FloatArray
    symbols: tuple[str, ...]
    coords: FloatArray


def _block(lines: list[str], name: str) -> int:
    """Index of the line after ``$name``."""
    for i, line in enumerate(lines):
        if line.strip().lower() == f"${name}":
            return i + 1
    raise ValueError(f"{HESS_NAME}: no ${name} block")


def parse_hess(text: str) -> Hess:
    """Read the ``$hessian`` and ``$atoms`` blocks of a ``job.hess`` file.

    Raises:
        ValueError: A block is missing, truncated or malformed.
    """
    lines = text.splitlines()
    i = _block(lines, "hessian")
    try:
        dim = int(lines[i].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{HESS_NAME}: bad $hessian dimension line") from None
    i += 1
    hessian = np.zeros((dim, dim))
    seen = 0
    while seen < dim:
        try:
            columns = [int(t) for t in lines[i].split()]
        except (IndexError, ValueError):
            raise ValueError(f"{HESS_NAME}: bad column header in $hessian") from None
        if not columns or max(columns) >= dim:
            raise ValueError(f"{HESS_NAME}: bad column header in $hessian")
        i += 1
        for row in range(dim):
            try:
                fields = lines[i].split()
            except IndexError:
                raise ValueError(f"{HESS_NAME}: $hessian is truncated") from None
            i += 1
            if len(fields) != 1 + len(columns) or fields[0] != str(row):
                raise ValueError(f"{HESS_NAME}: bad $hessian row {lines[i - 1]!r}")
            hessian[row, columns] = [_number(t, HESS_NAME) for t in fields[1:]]
        seen += len(columns)
    j = _block(lines, "atoms")
    try:
        n = int(lines[j].split()[0])
        rows = [lines[j + 1 + k].split() for k in range(n)]
    except (IndexError, ValueError):
        raise ValueError(f"{HESS_NAME}: $atoms is truncated") from None
    if any(len(r) != 5 for r in rows):
        raise ValueError(f"{HESS_NAME}: bad $atoms line")
    symbols = tuple(r[0] for r in rows)
    coords = np.array([[_number(t, HESS_NAME) for t in r[2:]] for r in rows])
    return Hess(hessian, symbols, coords)


def hess_result(hess: Hess, mol: Molecule) -> FloatArray:
    """The full symmetric Cartesian Hessian (3N, 3N), after checking the atoms.

    ORCA moves the geometry of ``job.hess`` to the centre of mass, so the
    geometry may differ from ``mol``'s by one common translation, but not by
    anything else.

    Raises:
        ValueError: The atoms differ, the size is wrong, or the geometry is
            rotated or distorted relative to ``mol``'s.
    """
    _check_atoms([element(s).atomic_number for s in hess.symbols], mol, HESS_NAME)
    n = 3 * mol.n_atoms
    if hess.hessian.shape != (n, n):
        raise ValueError(
            f"{HESS_NAME}: expected a {n}x{n} Hessian, got {hess.hessian.shape}"
        )
    shift = hess.coords - mol.coords
    deviation = float(np.max(np.abs(shift - shift[0])))
    if deviation > COORDINATE_TOLERANCE:
        raise ValueError(
            f"{HESS_NAME}: the geometry differs from the input by more than a "
            f"translation (up to {deviation:.3g} bohr; was the molecule rotated?)"
        )
    return 0.5 * (hess.hessian + hess.hessian.T)


def parse_xyz(text: str) -> tuple[tuple[str, ...], FloatArray]:
    """Read a ``job.xyz`` file: symbols and coordinates in bohr.

    Raises:
        ValueError: The text is not in the xyz layout.
    """
    lines = text.splitlines()
    try:
        n = int(lines[0].split()[0])
        rows = [lines[2 + k].split() for k in range(n)]
    except (IndexError, ValueError):
        raise ValueError(f"{XYZ_NAME}: not an xyz file") from None
    if n < 1 or any(len(r) < 4 for r in rows):
        raise ValueError(f"{XYZ_NAME}: not an xyz file")
    coords = np.array([[_number(t, XYZ_NAME) for t in r[1:4]] for r in rows])
    return tuple(r[0] for r in rows), coords * ANGSTROM_TO_BOHR


def xyz_coordinates(text: str, mol: Molecule) -> FloatArray:
    """The geometry of a ``job.xyz`` file (bohr, shape (N, 3)), after checking
    the atoms.

    Raises:
        ValueError: The text is not an xyz file or its atoms differ from
            ``mol``'s.
    """
    symbols, coords = parse_xyz(text)
    _check_atoms([element(s).atomic_number for s in symbols], mol, XYZ_NAME)
    return coords


# --- Running jobs ------------------------------------------------------------


@dataclass(frozen=True)
class RunSettings:
    """How to run ORCA; none of it changes a result."""

    orca: str
    nprocs: int = 1
    maxcore: int = 2000
    workdir: str | os.PathLike[str] | None = None
    timeout: float | None = None
    verbose: bool = False


@dataclass(frozen=True)
class JobResult:
    """The outcome of one ORCA run.

    Attributes:
        log: The ORCA log (stdout), or ``""`` when none was written.
        err: What ORCA printed to stderr.
        failure: Why the run failed, or ``None`` after a normal termination.
        files: The text of each result file ORCA wrote (:data:`RESULT_NAMES`).
    """

    log: str
    err: str
    failure: str | None
    files: Mapping[str, str]

    def describe(self, reason: str) -> str:
        """``reason`` followed by the end of the ORCA log (or stderr)."""
        text = self.log or self.err
        if not text.strip():
            return reason
        tail = "\n".join(text.rstrip().splitlines()[-LOG_TAIL_LINES:])
        return f"{reason}\n--- end of the ORCA log ---\n{tail}"


def run_job(settings: RunSettings, deck: str, name: str) -> JobResult:
    """Run ORCA on ``deck`` in a new job directory and read the result files.

    Without ``settings.workdir`` the directory is temporary and removed
    afterwards; with it, a new subdirectory ``<number>-<name>`` keeps the input,
    the log, stderr and the result files (bulky scratch files are removed). A
    run that takes longer than ``settings.timeout`` is stopped, with every
    process it started, and reported as a failure.
    """
    with process.job_directory(settings.workdir, name, prefix="pyvtst-orca-") as jobdir:
        (jobdir / INPUT_NAME).write_text(deck, encoding="utf-8")
        if settings.verbose:
            print(
                f"pyvtst: running ORCA ({name}) in {jobdir}; log: {jobdir / LOG_NAME}",
                file=sys.stderr,
            )
        try:
            status, message = process.execute(
                [settings.orca, INPUT_NAME],
                jobdir,
                timeout=settings.timeout,
                verbose=False,
                stdout_path=jobdir / LOG_NAME,
                stderr_path=jobdir / ERR_NAME,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            status, message, timed_out = None, "", True
        log = _read(jobdir / LOG_NAME)
        err = _read(jobdir / ERR_NAME) or message
        files = {n: _read(jobdir / n) for n in RESULT_NAMES if (jobdir / n).is_file()}
        if settings.workdir is not None:
            _remove_scratch(jobdir)
        if timed_out:
            failure: str | None = (
                f"ORCA did not finish within {settings.timeout:g} s and was stopped"
            )
        else:
            failure = _failure(status, log, err)
        return JobResult(log, err, failure, files)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _remove_scratch(jobdir: Path) -> None:
    for pattern in SCRATCH_PATTERNS:
        for path in jobdir.glob(pattern):
            with contextlib.suppress(OSError):
                path.unlink()


def _failure(status: int | None, log: str, err: str) -> str | None:
    if status is None:
        return f"ORCA could not be started: {err}".rstrip(": ")
    if HEADER_MARKER not in log:
        return (
            "the log does not look like ORCA output: the executable may not "
            "be ORCA (set PYVTST_ORCA or pass executable=)"
        )
    if ERROR_MARKER in log:
        reason = "ORCA reported an error termination"
        if NO_ANALYTIC_HESSIAN in log:
            reason += ': this method has no analytic Hessian; use hessian="numerical"'
        return reason
    if status != 0:
        reason = f"ORCA exited with status {status}"
        if NO_ANALYTIC_HESSIAN in log:
            reason += ': this method has no analytic Hessian; use hessian="numerical"'
        return reason
    if NORMAL_MARKER not in log:
        return "the ORCA log does not report a normal termination"
    return None


def cycles(log: str) -> list[int]:
    """The optimization cycle numbers in an ORCA log."""
    return [int(n) for n in re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", log)]
