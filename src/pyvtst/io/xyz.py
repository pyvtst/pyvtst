"""Native XYZ reader and writer (coordinates in ångström on disk)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import BOHR_TO_ANGSTROM


def read_xyz(
    path: str | os.PathLike[str],
    charge: int = 0,
    mult: int = 1,
    masses: npt.ArrayLike | None = None,
    symmetry_number: int = 1,
    name: str | None = None,
) -> Molecule:
    """Read the first frame of an XYZ file.

    The comment line becomes the molecule name unless ``name`` is given.
    Extra columns after ``symbol x y z`` are ignored.
    """
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f"{path}: line 1: expected the atom count, file is empty")
    try:
        n_atoms = int(lines[0].split()[0])
    except ValueError:
        raise ValueError(
            f"{path}: line 1: expected the atom count, got {lines[0]!r}"
        ) from None
    if n_atoms < 1:
        raise ValueError(f"{path}: line 1: atom count must be positive, got {n_atoms}")
    comment = lines[1].strip() if len(lines) > 1 else ""

    symbols = []
    coords = np.empty((n_atoms, 3))
    for i in range(n_atoms):
        lineno = i + 3
        if lineno > len(lines) or not lines[lineno - 1].strip():
            raise ValueError(f"{path}: declared {n_atoms} atoms, found {i}")
        line = lines[lineno - 1]
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(
                f"{path}: line {lineno}: expected 'symbol x y z', got {line!r}"
            )
        try:
            coords[i] = [float(v) for v in fields[1:4]]
        except ValueError:
            raise ValueError(
                f"{path}: line {lineno}: non-numeric coordinate in {line!r}"
            ) from None
        symbols.append(fields[0])

    return Molecule(
        symbols=tuple(symbols),
        coords=coords / BOHR_TO_ANGSTROM,
        charge=charge,
        mult=mult,
        masses=masses,
        symmetry_number=symmetry_number,
        name=comment if name is None else name,
    )


def write_xyz(molecule: Molecule, path: str | os.PathLike[str]) -> None:
    """Write ``molecule`` as a single-frame XYZ file with its name as the comment."""
    comment = " ".join(molecule.name.splitlines())
    lines = [str(molecule.n_atoms), comment]
    for symbol, (x, y, z) in zip(
        molecule.symbols, molecule.coords_angstrom, strict=True
    ):
        lines.append(f"{symbol:<2s} {x:16.10f} {y:16.10f} {z:16.10f}")
    Path(path).write_text("\n".join(lines) + "\n")
