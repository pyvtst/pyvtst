"""Stand-ins for Gaussian 16 and formchk, backed by the analytic LEPS surface.

:func:`install` writes two executable scripts, ``g16`` and ``formchk``, that
run :func:`g16_main` and :func:`formchk_main` with the current interpreter.

Fake ``g16 job.gjf`` reads the deck, evaluates the LEPS surface at its
geometry (``Force``: energy and gradient; ``Freq``: also the Hessian) or
optimizes on it (``Opt``: Newton steps to a minimum, or to a first-order
saddle point with ``TS``), and writes ``job.log`` with the marker lines
Gaussian prints and ``job.chk`` as JSON. Every call appends its deck to a
calls file. Fake ``formchk job.chk job.fchk`` writes a real fixed-format
formatted checkpoint from that JSON.

``FAKE_G16_MODE`` selects a failure:

- ``error``: an error termination (as after a non-converged SCF).
- ``steps_exceeded``: ``Opt`` stops at ``MaxCycles`` without converging.
- ``hang``: the run sleeps for an hour.
- ``reoriented``: the checkpoint geometry has x and y swapped.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Newton steps stop when the largest gradient component is below this.
GRADIENT_TOLERANCE = 1e-9
MAX_STEP = 0.1  # bohr
NORMAL = " Normal termination of Gaussian 16 at Sat Nov  7 12:00:00 2026.\n"
ERROR = " Error termination request processed by link 9999.\n"


@dataclass(frozen=True)
class FakeGaussian:
    """Installed fake programs and the record of their calls."""

    g16: Path
    formchk: Path
    calls_file: Path

    def decks(self) -> list[str]:
        """The input deck of every fake ``g16`` call, in order."""
        if not self.calls_file.is_file():
            return []
        lines = self.calls_file.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def routes(self) -> list[str]:
        """The route line of every call."""
        return [_route(deck) for deck in self.decks()]

    @property
    def count(self) -> int:
        return len(self.decks())

    def reset(self) -> None:
        self.calls_file.unlink(missing_ok=True)


def install(directory: Path) -> FakeGaussian:
    """Write executable fake ``g16`` and ``formchk`` scripts into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    calls = directory / "calls.jsonl"
    paths = []
    for name, func in (("g16", "g16_main"), ("formchk", "formchk_main")):
        path = directory / name
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"from tests.fakes.gaussian import {func}\n"
            f"sys.exit({func}(sys.argv[1:], {str(calls)!r}))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        paths.append(path)
    return FakeGaussian(paths[0], paths[1], calls)


def tilted_h3() -> Any:
    """A bent H3 (doublet, named ``H3``) that is not aligned with any axis."""
    import numpy as np  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    from pyvtst import Molecule  # noqa: PLC0415

    base = np.array([[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]])
    coords = Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).apply(base) + 0.25
    return Molecule(["H"] * 3, coords, mult=2, name="H3")


# --- fake g16 ----------------------------------------------------------------


def _route(deck: str) -> str:
    return next(line for line in deck.splitlines() if line.startswith("#"))


def _parse_deck(deck: str) -> tuple[str, int, int, list[str], list[list[float]]]:
    lines = deck.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#"))
    # route, blank, title, blank, charge and multiplicity, atoms, blank
    charge, mult = (int(v) for v in lines[start + 4].split())
    symbols, coords = [], []
    for line in lines[start + 5 :]:
        if not line.strip():
            break
        symbol, *xyz = line.split()
        symbols.append(symbol)
        coords.append([float(v) for v in xyz])
    return lines[start], charge, mult, symbols, coords


def g16_main(argv: list[str], calls: str) -> int:
    deck_path = Path(argv[0])
    deck = deck_path.read_text(encoding="utf-8")
    with open(calls, "a", encoding="utf-8") as f:
        f.write(json.dumps(deck) + "\n")
    log_path = deck_path.with_suffix(".log")
    mode = os.environ.get("FAKE_G16_MODE", "")
    route, charge, mult, symbols, coords = _parse_deck(deck)
    chk_name = re.search(r"%Chk=(\S+)", deck, re.IGNORECASE)
    assert chk_name is not None
    assert "units=au" in route.lower()
    assert "nosymm" in route.lower()
    log = [" Entering Gaussian System, Link 0=g16\n", f" {route}\n"]

    if mode == "hang":
        time.sleep(3600)
    if mode == "error":
        log += [" Convergence failure -- run terminated.\n", ERROR]
        log_path.write_text("".join(log), encoding="utf-8")
        return 1

    import numpy as np  # noqa: PLC0415

    from pyvtst import Molecule  # noqa: PLC0415
    from tests.fakes.leps import LEPSEngine  # noqa: PLC0415

    engine = LEPSEngine()
    mol = Molecule(symbols, coords, charge=charge, mult=mult)
    job = route.lower()
    status = 0
    hessian = None
    if " opt" in job:
        mol, n_steps, converged = _optimize(engine, mol, job, mode)
        log += [
            f" Step number {k:3d} out of a maximum of {_max_cycles(job):3d}\n"
            for k in range(1, n_steps + 1)
        ]
        if converged:
            log += ["    -- Stationary point found.\n", NORMAL]
        else:
            log += [
                " Optimization stopped.\n",
                f"    -- Number of steps exceeded,  NStep= {n_steps:3d}\n",
                ERROR,
            ]
            status = 1
    else:
        if " freq" in job:
            hessian = engine.hessian(mol)
        log.append(NORMAL)
    energy, gradient = engine.energy(mol), engine.gradient(mol)

    out = mol.coords[:, [1, 0, 2]] if mode == "reoriented" else mol.coords
    chk: dict[str, Any] = {
        "route": route,
        "numbers": [1] * mol.n_atoms,
        "coords": out.ravel().tolist(),
        "energy": energy,
        "gradient": np.ravel(gradient).tolist(),
        "hessian": None
        if hessian is None
        else hessian[np.tril_indices(3 * mol.n_atoms)].tolist(),
    }
    (deck_path.parent / chk_name.group(1)).write_text(json.dumps(chk), "utf-8")
    log_path.write_text("".join(log), encoding="utf-8")
    return status


def _max_cycles(job: str) -> int:
    match = re.search(r"maxcycles=(\d+)", job)
    return int(match.group(1)) if match else 100


def _optimize(engine: Any, mol: Any, job: str, mode: str) -> tuple[Any, int, bool]:
    """Newton steps with translations and rotations projected out; uphill
    along the lowest mode for ``TS``, downhill along every mode otherwise.

    Returns the last geometry, the number of geometries evaluated and whether
    the gradient converged.
    """
    import numpy as np  # noqa: PLC0415

    saddle = "ts" in re.split(r"[(),= ]+", job)
    max_cycles = _max_cycles(job)
    for step in range(1, max_cycles + 1):
        g = np.ravel(engine.gradient(mol))
        if mode != "steps_exceeded" and np.max(np.abs(g)) < GRADIENT_TOLERANCE:
            return mol, step, True
        if step == max_cycles:
            break
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
    return mol, max_cycles, False


def _internal_projector(coords: Any) -> Any:
    """Projector onto the complement of rigid translations and rotations."""
    import numpy as np  # noqa: PLC0415

    centered = coords - coords.mean(axis=0)
    vectors = []
    for k in range(3):
        shift = np.zeros_like(coords)
        shift[:, k] = 1.0
        vectors.append(shift.ravel())
        vectors.append(np.cross(np.eye(3)[k], centered).ravel())
    u, s, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    q = u[:, s > 1e-6 * s.max()]
    return np.eye(coords.size) - q @ q.T


# --- fake formchk ------------------------------------------------------------


def _header(name: str, kind: str, count: int) -> str:
    return f"{name:<40}   {kind}   N={count:12d}\n"


def _reals(name: str, values: list[float]) -> str:
    rows = [
        "".join(f"{v:16.8E}" for v in values[i : i + 5]) + "\n"
        for i in range(0, len(values), 5)
    ]
    return _header(name, "R", len(values)) + "".join(rows)


def _ints(name: str, values: list[int]) -> str:
    rows = [
        "".join(f"{v:12d}" for v in values[i : i + 6]) + "\n"
        for i in range(0, len(values), 6)
    ]
    return _header(name, "I", len(values)) + "".join(rows)


def _strings(name: str, text: str) -> str:
    chunks = [text[i : i + 12] for i in range(0, len(text), 12)] or [""]
    rows = [
        "".join(f"{c:<12}" for c in chunks[i : i + 5]) + "\n"
        for i in range(0, len(chunks), 5)
    ]
    return _header(name, "C", len(chunks)) + "".join(rows)


def formchk_main(argv: list[str], calls: str) -> int:
    chk_path, fchk_path = Path(argv[0]), Path(argv[1])
    data = json.loads(chk_path.read_text(encoding="utf-8"))
    n = len(data["numbers"])
    parts = [
        "pyvtst fake checkpoint\n",
        "Force     RHF                                                         LEPS\n",
        f"{'Number of atoms':<40}   I     {n:12d}\n",
        _strings("Route", data["route"]),
        f"{'Charge':<40}   I     {0:12d}\n",
        _ints("Atomic numbers", data["numbers"]),
        _reals("Current cartesian coordinates", data["coords"]),
        f"{'Virial Ratio':<40}   R     {2.0:22.15E}\n",
        f"{'Total Energy':<40}   R     {data['energy']:22.15E}\n",
        f"{'Unrestricted':<40}   L     {'F':>12}\n",
        _reals("Cartesian Gradient", data["gradient"]),
    ]
    if data["hessian"] is not None:
        parts.append(_reals("Cartesian Force Constants", data["hessian"]))
    parts.append(_header("Fake logical flags", "L", 3) + "  T F T\n")
    assert math.isfinite(data["energy"])
    fchk_path.write_text("".join(parts), encoding="utf-8")
    return 0
