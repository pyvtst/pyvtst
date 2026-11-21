"""Running external quantum-chemistry programs: lookup, job directories, processes.

Shared by the Gaussian and ORCA modules. ``subprocess`` only ever runs the
user's own program, from a list of arguments and without a shell.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from pyvtst.core.elements import formula
from pyvtst.core.molecule import Molecule

# --- Executable lookup -------------------------------------------------------


def find_executable(
    program: str,
    variable: str,
    argument: str,
    value: str | os.PathLike[str] | None,
    *,
    resolve_links: bool = False,
) -> str:
    """Locate ``program``: the ``argument`` value, else ``$variable``, else PATH.

    An argument or a non-empty variable is taken as is; if it does not name an
    executable file, ``FileNotFoundError`` is raised rather than falling back
    to PATH, so a different install is never run silently.

    Args:
        program: The program name searched on PATH.
        variable: The environment variable that may name the executable.
        argument: The constructor argument name, for error messages.
        value: The value given for ``argument``, or ``None``.
        resolve_links: Follow symbolic links, so the real executable is
            returned. A program that finds its helpers next to the path it was
            called with needs this.

    Returns:
        The absolute path of the executable.
    """
    if value is not None:
        return _executable(Path(value), argument, resolve_links)
    from_env = os.environ.get(variable, "")
    if from_env:
        return _executable(Path(from_env), variable, resolve_links)
    found = shutil.which(program)
    if found is None:
        raise FileNotFoundError(
            f"cannot find {program}: pass {argument}=..., set {variable}, "
            f"or put {program} on PATH"
        )
    return _absolute(Path(found), resolve_links)


def _absolute(path: Path, resolve_links: bool) -> str:
    return str(path.resolve() if resolve_links else path.absolute())


def _executable(path: Path, source: str, resolve_links: bool) -> str:
    path = path.expanduser()
    if not (path.is_file() and os.access(path, os.X_OK)):
        raise FileNotFoundError(f"{source}: {str(path)!r} is not an executable file")
    return _absolute(path, resolve_links)


def check_run_settings(nprocs: object, timeout: object) -> None:
    """Validate the run settings every program shares."""
    if isinstance(nprocs, bool) or not isinstance(nprocs, int) or nprocs < 1:
        raise ValueError(f"nprocs: must be an integer of at least 1, got {nprocs!r}")
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


# --- Processes ---------------------------------------------------------------


def execute(
    args: list[str],
    cwd: Path,
    *,
    timeout: float | None,
    verbose: bool,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> tuple[int | None, str]:
    """Run ``args`` in ``cwd``; return the exit status (``None`` if it could
    not start) and the captured output.

    Without file paths the output is captured, with stderr merged into stdout;
    when ``verbose`` it is passed through instead and ``""`` is returned. With
    ``stdout_path`` (and ``stderr_path``) the streams go to those files and
    ``""`` is returned. A stream with no path then follows the same rule as
    above.

    Raises:
        subprocess.TimeoutExpired: The process ran longer than ``timeout``; it
            and the processes it started have been killed.
    """
    with contextlib.ExitStack() as stack:
        out_file = stack.enter_context(open(stdout_path, "wb")) if stdout_path else None
        err_file = stack.enter_context(open(stderr_path, "wb")) if stderr_path else None
        stdout: int | IO[bytes] | None
        stderr: int | IO[bytes] | None
        stdout = out_file if out_file else (None if verbose else subprocess.PIPE)
        if err_file:
            stderr = err_file
        elif out_file is None:
            stderr = None if verbose else subprocess.STDOUT
        else:
            stderr = None if verbose else subprocess.DEVNULL
        try:
            # A list of arguments and no shell; the executables are the user's
            # own programs, located by find_executable.
            proc = subprocess.Popen(  # nosec B603
                args,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=out_file is None,
                start_new_session=True,
            )
        except OSError as exc:
            return None, str(exc)
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.communicate()
            raise
        return proc.returncode, output or ""


def _kill(proc: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
    """Stop ``proc`` and, where possible, every process it started."""
    with contextlib.suppress(ProcessLookupError):
        if sys.platform == "win32":
            proc.kill()
        else:
            # A program may start helper processes; the session started for
            # the run holds them all.
            os.killpg(proc.pid, signal.SIGKILL)


# --- Job directories ---------------------------------------------------------


@contextlib.contextmanager
def job_directory(
    workdir: str | os.PathLike[str] | None, name: str, *, prefix: str
) -> Iterator[Path]:
    """A directory for one job.

    Without ``workdir``: a temporary directory named with ``prefix``, removed
    afterwards. With it: a new ``<number>-<name>`` subdirectory, kept.
    """
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            yield Path(tmp)
        return
    root = Path(workdir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    yield numbered_directory(root, name)


_NUMBERED = re.compile(r"(\d+)-")


def last_number(root: Path) -> int:
    """The largest ``<number>-`` prefix among the entries of ``root``, or 0."""
    numbers = [
        int(m.group(1)) for p in root.iterdir() if (m := _NUMBERED.match(p.name))
    ]
    return max(numbers, default=0)


def numbered_directory(root: Path, name: str) -> Path:
    """Create ``root/<number>-<name>`` with the next free number.

    ``mkdir`` fails if the directory exists, so a name taken in the meantime
    (by another run sharing ``root``) moves on to the next number.
    """
    number = last_number(root) + 1
    while True:
        path = root / f"{number:04d}-{name}"
        try:
            path.mkdir()
        except FileExistsError:
            number += 1
            continue
        return path
