"""Gaussian 16: executable lookup, input decks, running jobs and ``.fchk`` files.

Shared by :class:`pyvtst.engines.GaussianEngine` and
:class:`pyvtst.optimizers.GaussianOptimizer`. Gaussian behaviours relied on:

- ``g16 job.gjf`` run in a directory writes ``job.log`` there, and the
  checkpoint named by ``%Chk``.
- A successful run ends its log with ``Normal termination``; a failed one
  prints ``Error termination``.
- ``formchk job.chk job.fchk`` writes the formatted checkpoint, whose layout
  is a header line ``<name, 40 columns> <type> [N=] <value or count>``
  followed, for arrays, by the values in rows (5 reals or 6 integers per row,
  5 strings of 12 characters, 72 logicals).
- ``NoSymm`` keeps the input orientation, so ``.fchk`` arrays are in the
  input's atom order and frame; ``Units=AU`` reads coordinates in bohr.
- ``Opt`` logs ``Step number <n>`` for each geometry and ``Stationary point
  found`` on convergence, and ``Number of steps exceeded`` or ``Optimization
  stopped`` when it gives up.

``subprocess`` only ever runs the user's own Gaussian and formchk, from a list
of arguments and without a shell.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import element, formula
from pyvtst.core.molecule import Molecule

FloatArray = npt.NDArray[np.float64]

REFERENCES = ("auto", "restricted", "unrestricted")

# Job types and keywords the decks set themselves; user keywords may not name
# them, since they would conflict with the deck.
RESERVED_KEYWORDS = (
    "Opt",
    "Freq",
    "Force",
    "SP",
    "IRC",
    "Scan",
    "Geom",
    "Guess",
    "Units",
    "NoSymm",
    "Symmetry",
)

FORCE_JOB = "Force"
FREQ_JOB = "Freq=NoRaman"

# Coordinates in the formatted checkpoint must match the input this closely
# (bohr); anything else means Gaussian moved the molecule.
COORDINATE_TOLERANCE = 1e-6

# Lines of the Gaussian log quoted in error messages.
LOG_TAIL_LINES = 20

INPUT_NAME = "job.gjf"
LOG_NAME = "job.log"
CHK_NAME = "job.chk"
FCHK_NAME = "job.fchk"


# --- Executable lookup -------------------------------------------------------


def find_executable(
    program: str, variable: str, argument: str, value: str | os.PathLike[str] | None
) -> str:
    """Locate ``program``: the ``argument`` value, else ``$variable``, else PATH.

    An argument or a non-empty variable is taken as is; if it does not name an
    executable file, ``FileNotFoundError`` is raised rather than falling back
    to PATH, so a different install is never run silently.

    Returns:
        The absolute path of the executable.
    """
    if value is not None:
        return _executable(Path(value), argument)
    from_env = os.environ.get(variable, "")
    if from_env:
        return _executable(Path(from_env), variable)
    found = shutil.which(program)
    if found is None:
        raise FileNotFoundError(
            f"cannot find {program}: pass {argument}=..., set {variable}, "
            f"or put {program} on PATH"
        )
    return str(Path(found).absolute())


def _executable(path: Path, source: str) -> str:
    path = path.expanduser()
    if not (path.is_file() and os.access(path, os.X_OK)):
        raise FileNotFoundError(f"{source}: {str(path)!r} is not an executable file")
    return str(path.absolute())


def find_programs(
    executable: str | os.PathLike[str] | None,
    formchk: str | os.PathLike[str] | None,
) -> tuple[str, str]:
    """Locate Gaussian and formchk; see :func:`find_executable`."""
    return (
        find_executable("g16", "PYVTST_G16", "executable", executable),
        find_executable("formchk", "PYVTST_FORMCHK", "formchk", formchk),
    )


def check_run_settings(nprocs: object, memory: object, timeout: object) -> None:
    """Validate the run settings shared by the Gaussian engine and optimizer."""
    if isinstance(nprocs, bool) or not isinstance(nprocs, int) or nprocs < 1:
        raise ValueError(f"nprocs: must be an integer of at least 1, got {nprocs!r}")
    if not isinstance(memory, str) or not memory.strip() or " " in memory.strip():
        raise ValueError(f"memory: expected a size such as '4GB', got {memory!r}")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not timeout > 0
    ):
        raise ValueError(f"timeout: must be a positive number, got {timeout!r}")


def label(mol: Molecule) -> str:
    """The molecule's name, or its formula when it has none."""
    return mol.name or formula(mol.symbols)


def job_name(job: str, mol: Molecule) -> str:
    """``<job>-<label>``, the readable part of a kept job directory's name.

    Characters other than letters, digits, ``.``, ``_``, ``+`` and ``-`` in
    the label become ``_``, so the name is a safe directory name.
    """
    return f"{job}-" + re.sub(r"[^A-Za-z0-9._+-]", "_", label(mol))


# --- Route and deck ----------------------------------------------------------


def normalize_keywords(keywords: str) -> str:
    """Fold case and collapse whitespace; reject reserved keywords.

    Gaussian keywords are case-insensitive and whitespace-separated, so
    equivalent spellings normalize to the same string.

    Raises:
        ValueError: ``keywords`` names a job type or a keyword the deck sets
            (:data:`RESERVED_KEYWORDS`), alone, as ``key=value`` or as
            ``key(...)``.
    """
    if not isinstance(keywords, str):
        raise ValueError(f"keywords: expected a string, got {keywords!r}")
    collapsed = " ".join(keywords.split())
    reserved = {k.lower(): k for k in RESERVED_KEYWORDS}
    for token in _route_tokens(collapsed):
        name = re.split(r"[=(]", token, maxsplit=1)[0]
        if name.lower() in reserved:
            raise ValueError(
                f"keywords: {name!r} is set by pyvtst and cannot be given "
                f"(reserved: {', '.join(RESERVED_KEYWORDS)})"
            )
    return collapsed.lower()


def _route_tokens(text: str) -> list[str]:
    """Split a route string into keywords at spaces, commas and slashes that
    are outside parentheses."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if depth == 0 and char in " ,/":
            if current:
                tokens.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def reference_prefix(reference: str, mult: int) -> str:
    """The method prefix for ``reference``: ``""``, ``"R"`` or ``"U"``.

    Raises:
        ValueError: ``reference`` is ``"restricted"`` and ``mult`` is above 1.
    """
    if reference == "auto":
        return ""
    if reference == "unrestricted":
        return "U"
    if mult != 1:
        raise ValueError(
            f"reference: a restricted reference needs multiplicity 1, got {mult}"
        )
    return "R"


def opt_job(*, saddle: bool, max_cycles: int, convergence: str) -> str:
    """The ``Opt`` job keyword for a minimum or a saddle point.

    ``convergence`` is ``"default"``, ``"tight"`` or ``"verytight"``.
    """
    options = ["TS", "CalcFC", "NoEigenTest"] if saddle else []
    options.append(f"MaxCycles={max_cycles}")
    if convergence == "tight":
        options.append("Tight")
    elif convergence == "verytight":
        options.append("VeryTight")
    return "Opt=(" + ",".join(options) + ")"


def build_deck(
    mol: Molecule,
    *,
    method: str,
    basis: str,
    reference: str,
    keywords: str,
    job: str,
    title: str,
    nprocs: int,
    memory: str,
) -> str:
    """The Gaussian input for one job on ``mol``, coordinates in bohr.

    Raises:
        ValueError: see :func:`reference_prefix`.
    """
    prefix = reference_prefix(reference, mol.mult)
    route = " ".join(
        part
        for part in (f"#P {prefix}{method}/{basis}", job, "NoSymm Units=AU", keywords)
        if part
    )
    atoms = "".join(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}\n"
        for symbol, (x, y, z) in zip(mol.symbols, mol.coords.tolist(), strict=True)
    )
    return (
        f"%NProcShared={nprocs}\n"
        f"%Mem={memory}\n"
        f"%Chk={CHK_NAME}\n"
        f"{route}\n"
        "\n"
        f"pyvtst {title} {formula(mol.symbols)}\n"
        "\n"
        f"{mol.charge} {mol.mult}\n"
        f"{atoms}"
        "\n"
    )


# --- Formatted checkpoint ----------------------------------------------------

# Values per row of each array type in a formatted checkpoint.
_PER_ROW = {"I": 6, "R": 5, "C": 5, "H": 5, "L": 72}


@dataclass(frozen=True)
class Fchk:
    """The integer and real entries of a formatted checkpoint, by name.

    String and logical entries are skipped.
    """

    scalars: Mapping[str, int | float]
    arrays: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]]

    def scalar(self, name: str) -> float:
        if name not in self.scalars:
            raise ValueError(f"no {name!r} entry")
        return float(self.scalars[name])

    def array(self, name: str) -> npt.NDArray[np.float64]:
        if name not in self.arrays:
            raise ValueError(f"no {name!r} entry")
        return np.asarray(self.arrays[name], dtype=np.float64)


def parse_fchk(text: str) -> Fchk:
    """Read a formatted checkpoint.

    Raises:
        ValueError: The text is not in the formatted-checkpoint layout.
    """
    lines = text.splitlines()
    scalars: dict[str, int | float] = {}
    arrays: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] = {}
    i = 2  # title and job-type lines
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        name = line[:40].strip()
        fields = line[40:].split()
        if not fields or fields[0] not in _PER_ROW:
            raise ValueError(f"line {i}: not an entry header: {line!r}")
        kind = fields[0]
        if len(fields) >= 3 and fields[1] == "N=":
            count = _int(fields[2], i)
            if kind in "IR":
                values, i = _read_values(lines, i, count, kind, name)
                arrays[name] = values
            else:
                i += math.ceil(count / _PER_ROW[kind])
        elif len(fields) >= 2 and kind == "I":
            scalars[name] = _int(fields[1], i)
        elif len(fields) >= 2 and kind == "R":
            scalars[name] = _float(fields[1], i)
    return Fchk(scalars, arrays)


def _read_values(
    lines: list[str], i: int, count: int, kind: str, name: str
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.int64], int]:
    tokens: list[str] = []
    while len(tokens) < count:
        if i >= len(lines):
            raise ValueError(f"{name!r}: expected {count} values, file ends")
        tokens.extend(lines[i].split())
        i += 1
    if len(tokens) != count:
        raise ValueError(f"{name!r}: expected {count} values, got {len(tokens)}")
    if kind == "I":
        return np.array([_int(t, i) for t in tokens], dtype=np.int64), i
    return np.array([_float(t, i) for t in tokens], dtype=np.float64), i


def _int(token: str, line: int) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f"line {line}: expected an integer, got {token!r}") from None


def _float(token: str, line: int) -> float:
    try:
        return float(token.replace("D", "E"))
    except ValueError:
        raise ValueError(f"line {line}: expected a number, got {token!r}") from None


def fchk_coordinates(fchk: Fchk, mol: Molecule) -> FloatArray:
    """The current geometry (bohr, shape (N, 3)), after checking the atoms.

    Raises:
        ValueError: An entry is missing or the atoms differ from ``mol``'s.
    """
    numbers = fchk.array("Atomic numbers")
    expected = [element(s).atomic_number for s in mol.symbols]
    if numbers.tolist() != expected:
        raise ValueError(
            f"atomic numbers {[int(z) for z in numbers]} differ from the "
            f"input's {expected}"
        )
    return _shaped(fchk.array("Current cartesian coordinates"), (mol.n_atoms, 3))


def fchk_results(
    fchk: Fchk, mol: Molecule, *, hessian: bool
) -> tuple[float, FloatArray, FloatArray | None]:
    """Energy, gradient and (when ``hessian``) the full Cartesian Hessian.

    The geometry must match ``mol``'s to :data:`COORDINATE_TOLERANCE`.

    Raises:
        ValueError: An entry is missing or has the wrong size, or the atoms or
            the geometry differ from ``mol``'s.
    """
    coords = fchk_coordinates(fchk, mol)
    deviation = float(np.max(np.abs(coords - mol.coords)))
    if deviation > COORDINATE_TOLERANCE:
        raise ValueError(
            f"the geometry differs from the input by up to {deviation:.3g} bohr "
            "(was the molecule reoriented?)"
        )
    energy = fchk.scalar("Total Energy")
    gradient = _shaped(fchk.array("Cartesian Gradient"), (mol.n_atoms, 3))
    if not hessian:
        return energy, gradient, None
    n = 3 * mol.n_atoms
    lower = _shaped(fchk.array("Cartesian Force Constants"), (n * (n + 1) // 2,))
    full = np.zeros((n, n))
    rows, cols = np.tril_indices(n)
    full[rows, cols] = lower
    full[cols, rows] = lower
    return energy, gradient, full


def _shaped(values: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    if values.size != math.prod(shape):
        raise ValueError(f"expected {math.prod(shape)} values, got {values.size}")
    return values.reshape(shape)


# --- Running jobs ------------------------------------------------------------


@dataclass(frozen=True)
class RunSettings:
    """How to run Gaussian; none of it changes a result."""

    g16: str
    formchk: str
    nprocs: int = 1
    memory: str = "2GB"
    workdir: str | os.PathLike[str] | None = None
    timeout: float | None = None
    verbose: bool = False


@dataclass(frozen=True)
class JobResult:
    """The outcome of one Gaussian run.

    Attributes:
        log: The Gaussian log, or ``""`` when none was written.
        output: What Gaussian printed to stdout and stderr (not captured when
            verbose).
        failure: Why the run failed, or ``None`` after a normal termination.
        fchk: The formatted checkpoint, when ``formchk`` succeeded.
        fchk_failure: Why there is no ``fchk``, or ``None``.
    """

    log: str
    output: str
    failure: str | None
    fchk: Fchk | None
    fchk_failure: str | None

    def describe(self, reason: str) -> str:
        """``reason`` followed by the end of the Gaussian log (or output)."""
        text = self.log or self.output
        if not text.strip():
            return reason
        tail = "\n".join(text.rstrip().splitlines()[-LOG_TAIL_LINES:])
        return f"{reason}\n--- end of the Gaussian log ---\n{tail}"


def run_job(
    settings: RunSettings,
    deck: str,
    name: str,
    *,
    formchk_on_failure: bool = False,
) -> JobResult:
    """Run Gaussian on ``deck`` in a new job directory, then ``formchk``.

    Without ``settings.workdir`` the directory is temporary and removed
    afterwards; with it, a new subdirectory ``<number>-<name>`` keeps the
    input, the log and the formatted checkpoint (the binary checkpoint is
    removed once converted). Numbers count up from ``0001`` in run order.
    ``formchk`` runs after a normal termination, or after any failure but a
    timeout when ``formchk_on_failure`` is set. A run that takes longer than
    ``settings.timeout`` is stopped and reported as a failure.
    """
    with _job_directory(settings.workdir, name) as jobdir:
        (jobdir / INPUT_NAME).write_text(deck, encoding="utf-8")
        if settings.verbose:
            print(f"pyvtst: running Gaussian ({name}) in {jobdir}", file=sys.stderr)
        try:
            status, output = _execute([settings.g16, INPUT_NAME], jobdir, settings)
            timed_out = False
        except subprocess.TimeoutExpired:
            status, output, timed_out = None, "", True
        log_path = jobdir / LOG_NAME
        log = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        if timed_out:
            failure: str | None = (
                f"Gaussian did not finish within {settings.timeout:g} s and was stopped"
            )
        else:
            failure = _failure(status, log)
        fchk: Fchk | None = None
        fchk_failure: str | None = "Gaussian failed"
        if failure is None or (formchk_on_failure and not timed_out):
            fchk, fchk_failure = _formchk(settings, jobdir)
        return JobResult(log, output, failure, fchk, fchk_failure)


def _failure(status: int | None, log: str) -> str | None:
    if status is None:
        return "Gaussian could not be started"
    if "Error termination" in log:
        return "Gaussian reported an error termination"
    if status != 0:
        return f"Gaussian exited with status {status}"
    if not log:
        return f"Gaussian wrote no {LOG_NAME}"
    if "Normal termination" not in log:
        return "the Gaussian log does not report a normal termination"
    return None


def _formchk(settings: RunSettings, jobdir: Path) -> tuple[Fchk | None, str | None]:
    chk = jobdir / CHK_NAME
    if not chk.is_file():
        return None, f"Gaussian wrote no {CHK_NAME}"
    status, output = _execute([settings.formchk, CHK_NAME, FCHK_NAME], jobdir, settings)
    fchk_path = jobdir / FCHK_NAME
    if status != 0 or not fchk_path.is_file():
        detail = output.strip().splitlines()[-1:] if output.strip() else []
        return None, "formchk failed" + (f": {detail[0]}" if detail else "")
    if settings.workdir is not None:
        chk.unlink()
    try:
        return parse_fchk(fchk_path.read_text(encoding="utf-8", errors="replace")), None
    except ValueError as exc:
        return None, f"cannot read {FCHK_NAME}: {exc}"


def _execute(
    args: list[str], cwd: Path, settings: RunSettings
) -> tuple[int | None, str]:
    """Run ``args`` in ``cwd``; return the exit status (``None`` if it could
    not start) and the captured output."""
    stream = None if settings.verbose else subprocess.PIPE
    try:
        # A list of arguments and no shell; the executables are the user's own
        # Gaussian and formchk, located by find_programs.
        proc = subprocess.Popen(  # nosec B603
            args,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=None if settings.verbose else subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return None, str(exc)
    try:
        output, _ = proc.communicate(timeout=settings.timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise
    return proc.returncode, output or ""


def _kill(proc: subprocess.Popen[str]) -> None:
    """Stop ``proc`` and, where possible, every process it started."""
    with contextlib.suppress(ProcessLookupError):
        if sys.platform == "win32":
            proc.kill()
        else:
            # Gaussian runs each link as a child process; the session started
            # for the run holds them all.
            os.killpg(proc.pid, signal.SIGKILL)


@contextlib.contextmanager
def _job_directory(workdir: str | os.PathLike[str] | None, name: str) -> Iterator[Path]:
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="pyvtst-gaussian-") as tmp:
            yield Path(tmp)
        return
    root = Path(workdir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    yield _numbered_directory(root, name)


_NUMBERED = re.compile(r"(\d+)-")


def _last_number(root: Path) -> int:
    """The largest ``<number>-`` prefix among the entries of ``root``, or 0."""
    numbers = [
        int(m.group(1)) for p in root.iterdir() if (m := _NUMBERED.match(p.name))
    ]
    return max(numbers, default=0)


def _numbered_directory(root: Path, name: str) -> Path:
    """Create ``root/<number>-<name>`` with the next free number.

    ``mkdir`` fails if the directory exists, so a name taken in the meantime
    (by another run sharing ``root``) moves on to the next number.
    """
    number = _last_number(root) + 1
    while True:
        path = root / f"{number:04d}-{name}"
        try:
            path.mkdir()
        except FileExistsError:
            number += 1
            continue
        return path
