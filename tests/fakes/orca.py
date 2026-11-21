"""A stand-in for ORCA 6, backed by the analytic LEPS surface.

:func:`install` writes an executable script, ``orca``, that runs
:func:`orca_main` with the current interpreter.

Fake ``orca job.inp`` reads the deck and evaluates the LEPS surface at its
geometry (``EnGrad``: ``job.engrad``; ``Freq`` or ``NumFreq``: also
``job.hess``) or optimizes on it (``Opt``/``TightOpt``/``VeryTightOpt``: Newton
steps to a minimum; ``OptTS``: to a first-order saddle point), and writes
``job.xyz``. It prints its log to stdout with the marker lines ORCA 6 prints,
and MPI-like noise to stderr, as ORCA does on a parallel machine. The files
follow the layouts ORCA 6.0.1 writes: ``job.hess`` has the geometry moved to
the centre of mass. Every call appends its deck to a calls file.

``FAKE_ORCA_MODE`` selects a failure, each as ORCA 6.0.1 shows it:

- ``scf_error``: an error termination after a non-converged SCF, exit 0.
- ``input_error``: an input error, exit 4.
- ``no_analytic_hessian``: ``Freq`` stops with "please use NumFreq", exit 25.
- ``maxiter``: the optimization stops at ``MaxIter`` with normal termination
  and exit 0; ``job.xyz`` has the last cycle's geometry, but ``job.engrad``
  has the coordinates of a further step that was not evaluated.
- ``rotated``: the result files have x and y swapped.
- ``hang``: the run sleeps for an hour.
- ``not_orca``: a different program, with no ORCA output.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyvtst.core.units import BOHR_TO_ANGSTROM
from tests.fakes.gaussian import GRADIENT_TOLERANCE, MAX_STEP, _internal_projector

ROOT = Path(__file__).resolve().parents[2]

BANNER = (
    "\n                                 *****************\n"
    "                                 * O   R   C   A *\n"
    "                                 *****************\n\n"
    "                         Program Version 6.0.1  -  RELEASE   -\n\n"
)
NORMAL = "\n                             ****ORCA TERMINATED NORMALLY****\n"
NOISE = "[node:1] osc_ucx_component.c:375  Error: OSC UCX component priority\n" * 40


@dataclass(frozen=True)
class FakeOrca:
    """The installed fake program and the record of its calls."""

    orca: Path
    calls_file: Path

    def decks(self) -> list[str]:
        """The input deck of every fake ``orca`` call, in order."""
        if not self.calls_file.is_file():
            return []
        lines = self.calls_file.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def keyword_lines(self) -> list[str]:
        """The ``!`` line of every call."""
        return [_keyword_line(deck) for deck in self.decks()]

    @property
    def count(self) -> int:
        return len(self.decks())

    def reset(self) -> None:
        self.calls_file.unlink(missing_ok=True)


def install(directory: Path) -> FakeOrca:
    """Write an executable fake ``orca`` script into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    calls = directory / "calls.jsonl"
    path = directory / "orca"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tests.fakes.orca import orca_main\n"
        f"sys.exit(orca_main(sys.argv[1:], {str(calls)!r}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return FakeOrca(path, calls)


# --- deck --------------------------------------------------------------------


def _keyword_line(deck: str) -> str:
    return next(line for line in deck.splitlines() if line.startswith("!"))


@dataclass(frozen=True)
class _Deck:
    keywords: list[str]
    charge: int
    mult: int
    symbols: list[str]
    coords: list[list[float]]
    max_iter: int
    calc_hess: bool
    num_hess: bool


def _parse(deck: str) -> _Deck:
    keywords = _keyword_line(deck)[1:].lower().split()
    lines = deck.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("* xyz"))
    _, _, charge, mult = lines[start].split()
    symbols, coords = [], []
    for line in lines[start + 1 :]:
        if line.strip() == "*":
            break
        symbol, *xyz = line.split()
        symbols.append(symbol)
        coords.append([float(v) for v in xyz])
    geom = re.search(r"%geom\n(.*?)\nend", deck, re.DOTALL | re.IGNORECASE)
    text = geom.group(1).lower() if geom else ""
    max_iter = re.search(r"maxiter\s+(\d+)", text)
    return _Deck(
        keywords=keywords,
        charge=int(charge),
        mult=int(mult),
        symbols=symbols,
        coords=coords,
        max_iter=int(max_iter.group(1)) if max_iter else 50,
        calc_hess="calc_hess true" in text,
        num_hess="numhess true" in text,
    )


# --- files -------------------------------------------------------------------


def _engrad(energy: float, gradient: Any, numbers: list[int], coords: Any) -> str:
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f" {len(numbers)}",
        "#",
        "# The current total energy in Eh",
        "#",
        f"{energy:20.12f}",
        "#",
        "# The current gradient in Eh/bohr",
        "#",
        *(f"{g:21.12f}" for g in gradient.ravel()),
        "#",
        "# The atomic numbers and current coordinates in Bohr",
        "#",
        *(
            f"{z:4d} {x:12.7f} {y:12.7f} {w:12.7f} "
            for z, (x, y, w) in zip(numbers, coords.tolist(), strict=True)
        ),
    ]
    return "\n".join(lines) + "\n"


def _hess(hessian: Any, symbols: list[str], masses: Any, coords: Any) -> str:
    n = hessian.shape[0]
    lines = [
        "",
        "$orca_hessian_file",
        "",
        "$act_atom",
        "  0",
        "",
        "$act_energy",
        "        0.000000",
        "",
        "$hessian",
        str(n),
    ]
    for start in range(0, n, 5):
        cols = range(start, min(start + 5, n))
        lines.append(" " * 9 + "".join(f"{c:<19d}" for c in cols))
        for row in range(n):
            lines.append(
                f"{row:5d}     " + "".join(f"{hessian[row, c]:19.10E}" for c in cols)
            )
    lines += [
        "",
        "$vibrational_frequencies",
        str(n),
        *(f"{i:5d}        0.0000000000000000" for i in range(n)),
        "",
        "#",
        "# The atoms: label  mass x y z (in bohrs)",
        "#",
        "$atoms",
        str(len(symbols)),
        *(
            f" {s:2s} {m:12.5f} {x:18.12f} {y:18.12f} {z:18.12f}"
            for s, m, (x, y, z) in zip(symbols, masses, coords.tolist(), strict=True)
        ),
        "",
        "$end",
    ]
    return "\n".join(lines) + "\n"


def _xyz(symbols: list[str], coords: Any, energy: float) -> str:
    angstrom = coords * BOHR_TO_ANGSTROM
    rows = "".join(
        f"  {s:2s} {x:20.14f} {y:20.14f} {z:20.14f}\n"
        for s, (x, y, z) in zip(symbols, angstrom.tolist(), strict=True)
    )
    return f"{len(symbols)}\nCoordinates from ORCA-job job E {energy:.12f}\n{rows}"


# --- fake orca ---------------------------------------------------------------


def orca_main(argv: list[str], calls: str) -> int:
    deck_path = Path(argv[0])
    deck = deck_path.read_text(encoding="utf-8")
    with open(calls, "a", encoding="utf-8") as f:
        f.write(json.dumps(deck) + "\n")
    mode = os.environ.get("FAKE_ORCA_MODE", "")
    out = sys.stdout
    if mode == "not_orca":
        out.write("Orca screen reader: no display\n")
        return 0
    out.write(BANNER)
    if mode == "hang":
        out.flush()
        time.sleep(3600)
    if mode == "input_error":
        out.write(
            "                                 INPUT ERROR\n"
            "            UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE\n"
        )
        return 4
    sys.stderr.write(NOISE)
    parsed = _parse(deck)
    assert "bohrs" in parsed.keywords
    if mode == "scf_error":
        out.write(
            "               *        SCF NOT CONVERGED AFTER   3 CYCLES         *\n"
            "ORCA finished by error termination in LEANSCF\n"
            "  .... aborting the run\n"
        )
        return 0
    if mode == "no_analytic_hessian" and "freq" in parsed.keywords:
        out.write(
            "ERROR: MP2 analytic Hessian calculations are not implemented - "
            "please use NumFreq\n ==> : Skipping actual calculation\n"
        )
        return 25

    import numpy as np  # noqa: PLC0415

    from pyvtst import Molecule  # noqa: PLC0415
    from tests.fakes.leps import LEPSEngine  # noqa: PLC0415

    engine = LEPSEngine()
    mol = Molecule(
        parsed.symbols, parsed.coords, charge=parsed.charge, mult=parsed.mult
    )
    numbers = [1] * mol.n_atoms
    jobdir = deck_path.parent

    def frame(coords: Any) -> Any:
        return coords[:, [1, 0, 2]] if mode == "rotated" else coords

    opt = next((k for k in parsed.keywords if k.endswith("opt") or k == "optts"), None)
    if opt is not None:
        saddle = "optts" in parsed.keywords
        if saddle:
            assert parsed.calc_hess
        geometries, converged = _optimize(
            engine, mol, saddle=saddle, max_cycles=parsed.max_iter, mode=mode
        )
        for k in range(1, len(geometries) + 1):
            out.write(
                f"         *                GEOMETRY OPTIMIZATION CYCLE {k:3d}"
                "            *\n"
            )
        last = geometries[-1]
        energy, gradient = engine.energy(last), np.asarray(engine.gradient(last))
        engrad_coords = last.coords
        if converged:
            out.write(
                "                    ***********************HURRAY***********"
                "*********\n"
                "                    ***        THE OPTIMIZATION HAS CONVERGED"
                "     ***\n"
                "                 *** FINAL ENERGY EVALUATION AT THE STATIONARY "
                "POINT ***\n"
                "                                *** OPTIMIZATION RUN DONE ***\n"
            )
        else:
            out.write(
                "       The optimization did not converge but reached the "
                "maximum number of\n       optimization cycles.\n"
                "       Please check your results very carefully.\n"
            )
            # ORCA 6.0.1 leaves the coordinates of the next, unevaluated step.
            engrad_coords = last.coords + 0.01
        (jobdir / "job.engrad").write_text(
            _engrad(energy, gradient, numbers, frame(engrad_coords)), "utf-8"
        )
        (jobdir / "job.xyz").write_text(
            _xyz(parsed.symbols, frame(last.coords), energy), "utf-8"
        )
        out.write(NORMAL)
        return 0

    assert "engrad" in parsed.keywords
    energy, gradient = engine.energy(mol), np.asarray(engine.gradient(mol))
    (jobdir / "job.engrad").write_text(
        _engrad(energy, gradient, numbers, frame(mol.coords)), "utf-8"
    )
    if "freq" in parsed.keywords or "numfreq" in parsed.keywords:
        assert re.search(r"%freq\n\s*TransInvar true\nend", deck)
        masses = np.asarray(mol.masses)
        centre = masses @ mol.coords / masses.sum()
        (jobdir / "job.hess").write_text(
            _hess(
                engine.hessian(mol),
                parsed.symbols,
                masses,
                frame(mol.coords - centre),
            ),
            "utf-8",
        )
    out.write(NORMAL)
    return 0


def _optimize(
    engine: Any, mol: Any, *, saddle: bool, max_cycles: int, mode: str
) -> tuple[list[Any], bool]:
    """Newton steps with translations and rotations projected out; uphill
    along the lowest mode for a saddle point, downhill along every mode
    otherwise.

    Returns every geometry evaluated, one per cycle, and whether the gradient
    converged.
    """
    import numpy as np  # noqa: PLC0415

    geometries = []
    for _ in range(max_cycles):
        geometries.append(mol)
        g = np.ravel(engine.gradient(mol))
        if mode != "maxiter" and np.max(np.abs(g)) < GRADIENT_TOLERANCE:
            return geometries, True
        p = _internal_projector(mol.coords)
        w, v = np.linalg.eigh(p @ engine.hessian(mol) @ p)
        keep = np.flatnonzero(np.abs(w) > 1e-6)
        gq = v.T @ g
        dq = np.zeros_like(gq)
        dq[keep] = -gq[keep] / np.abs(w[keep])
        if saddle and keep.size:
            dq[keep[0]] *= -1.0
        dx = v @ dq
        norm = float(np.linalg.norm(dx))
        if norm > MAX_STEP:
            dx *= MAX_STEP / norm
        mol = mol.replace(coords=mol.coords + dx.reshape(-1, 3))
    return geometries, False
