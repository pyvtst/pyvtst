"""Conversion between ``Molecule`` and ``ase.Atoms``. ASE is imported lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy.typing as npt

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import BOHR_TO_ANGSTROM

if TYPE_CHECKING:
    import ase


def _ase_atoms() -> type[ase.Atoms]:
    # ASE is optional, so it is imported on first use rather than at load time.
    try:
        from ase import Atoms  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'pyvtst.io.from_ase / to_ase needs ASE: pip install "pyvtst[ase]"'
        ) from exc
    return Atoms


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
    _ase_atoms()  # an object from a missing ASE still gets the install hint
    info: dict[str, Any] = atoms.info
    if charge is None:
        charge = info.get("charge", 0)
    if mult is None:
        mult = info.get("multiplicity", info.get("spin", 1))
    if name is None:
        name = info.get("name", "")
    return Molecule(
        symbols=atoms.get_chemical_symbols(),
        coords=atoms.get_positions() / BOHR_TO_ANGSTROM,
        charge=charge,
        mult=mult,
        masses=masses,
        symmetry_number=symmetry_number,
        name=name,
    )


def to_ase(molecule: Molecule) -> ase.Atoms:
    """Return a new non-periodic ``ase.Atoms`` with the molecule's masses and info."""
    Atoms = _ase_atoms()
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
