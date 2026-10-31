"""Conversion between ``Molecule`` and ``ase.Atoms``. ASE is imported lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy.typing as npt

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import BOHR_TO_ANGSTROM

if TYPE_CHECKING:
    import ase


def _require_ase() -> None:
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Molecule.from_ase / to_ase needs ASE: pip install "pyvtst[ase]"'
        ) from exc


def from_ase(
    atoms: ase.Atoms,
    charge: int | None = None,
    mult: int | None = None,
    masses: npt.ArrayLike | None = None,
    symmetry_number: int = 1,
    name: str | None = None,
) -> Molecule:
    """Build a ``Molecule`` from ``atoms``.

    Charge and multiplicity default to ``atoms.info["charge"]`` and
    ``atoms.info["multiplicity"]`` (or ``"spin"``), then to 0 and 1. ASE's
    masses (average atomic weights) are ignored in favour of pyvtst's isotope
    masses unless ``masses`` is given. Cells and calculators are ignored.
    """
    _require_ase()
    info = atoms.info
    if mult is None:
        mult = info["multiplicity"] if "multiplicity" in info else info.get("spin", 1)
    return Molecule(
        symbols=atoms.get_chemical_symbols(),
        coords=atoms.get_positions() / BOHR_TO_ANGSTROM,
        charge=info.get("charge", 0) if charge is None else charge,
        mult=mult,
        masses=masses,
        symmetry_number=symmetry_number,
        name=info.get("name", "") if name is None else name,
    )


def to_ase(molecule: Molecule) -> ase.Atoms:
    """Return a new non-periodic ``ase.Atoms`` with the molecule's masses and info."""
    _require_ase()
    from ase import Atoms

    return Atoms(
        symbols=list(molecule.symbols),
        positions=molecule.coords_angstrom,
        masses=molecule.masses.copy(),
        pbc=False,
        info={
            "charge": molecule.charge,
            "multiplicity": molecule.mult,
            "name": molecule.name,
        },
    )
