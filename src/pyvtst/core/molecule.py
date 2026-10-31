"""The immutable molecular structure type used by every pyvtst computation."""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import canonical_symbol, element
from pyvtst.core.units import BOHR_TO_ANGSTROM

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False, init=False)
class Molecule:
    """A molecular structure in atomic units.

    Attributes:
        symbols: Canonical element symbols, one per atom.
        coords: Cartesian coordinates in bohr, read-only array of shape (N, 3).
        charge: Total charge.
        mult: Spin multiplicity, 2S + 1.
        masses: Atomic masses in amu, read-only array of shape (N,). Defaults
            to the most abundant isotope of each element.
        symmetry_number: Rotational symmetry number.
        name: Optional label.

    Instances are immutable and compare by value; use :meth:`replace` to get a
    modified copy. They are deliberately unhashable. Readers and writers for
    XYZ files, ASE and SMILES live in :mod:`pyvtst.io`.
    """

    # The constructor accepts array-like input and normalizes it, so it is
    # written by hand; the fields below are the stored, validated types.
    symbols: tuple[str, ...]
    coords: FloatArray
    charge: int
    mult: int
    masses: FloatArray
    symmetry_number: int
    name: str

    def __init__(
        self,
        symbols: Iterable[str],
        coords: npt.ArrayLike,
        charge: SupportsIndex = 0,
        mult: SupportsIndex = 1,
        masses: npt.ArrayLike | None = None,
        symmetry_number: SupportsIndex = 1,
        name: str = "",
    ) -> None:
        syms = tuple(canonical_symbol(s) for s in symbols)
        if not syms:
            raise ValueError("symbols: a molecule needs at least one atom")
        numbers = [element(s).atomic_number for s in syms]
        n = len(syms)

        try:
            xyz = np.array(coords, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"coords: not a numeric array ({exc})") from None
        if xyz.shape != (n, 3):
            raise ValueError(
                f"coords: expected shape ({n}, 3) for {n} atoms, got {xyz.shape}"
            )
        if not np.all(np.isfinite(xyz)):
            raise ValueError("coords: all coordinates must be finite")

        if masses is None:
            m = np.array([element(s).mass for s in syms], dtype=np.float64)
        else:
            try:
                m = np.array(masses, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"masses: not a numeric array ({exc})") from None
            if m.shape != (n,):
                raise ValueError(f"masses: expected {n} values, got shape {m.shape}")
            if not np.all(np.isfinite(m) & (m > 0)):
                raise ValueError("masses: all masses must be finite and positive")

        q = operator.index(charge)
        spin = operator.index(mult)
        sigma = operator.index(symmetry_number)
        if spin < 1:
            raise ValueError(f"mult: multiplicity must be >= 1, got {spin}")
        if sigma < 1:
            raise ValueError(f"symmetry_number: must be >= 1, got {sigma}")
        n_electrons = sum(numbers) - q
        n_unpaired = spin - 1
        if n_unpaired > n_electrons or n_unpaired % 2 != n_electrons % 2:
            raise ValueError(
                f"mult: multiplicity {spin} is inconsistent with "
                f"{n_electrons} electrons (charge {q})"
            )

        xyz.flags.writeable = False
        m.flags.writeable = False
        object.__setattr__(self, "symbols", syms)
        object.__setattr__(self, "coords", xyz)
        object.__setattr__(self, "masses", m)
        object.__setattr__(self, "charge", q)
        object.__setattr__(self, "mult", spin)
        object.__setattr__(self, "symmetry_number", sigma)
        object.__setattr__(self, "name", str(name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        return (
            self.symbols == other.symbols
            and np.array_equal(self.coords, other.coords)
            and self.charge == other.charge
            and self.mult == other.mult
            and np.array_equal(self.masses, other.masses)
            and self.symmetry_number == other.symmetry_number
            and self.name == other.name
        )

    __hash__ = None  # type: ignore[assignment]

    def replace(self, **changes: Any) -> Molecule:
        """Return a validated copy with the given fields replaced.

        If ``symbols`` change and ``masses`` is not given, the new molecule gets
        default masses for its new elements rather than the old masses.
        """
        if "symbols" in changes and "masses" not in changes:
            symbols = tuple(changes["symbols"])
            changes["symbols"] = symbols
            if tuple(canonical_symbol(s) for s in symbols) != self.symbols:
                changes["masses"] = None
        return dataclasses.replace(self, **changes)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return tuple(element(s).atomic_number for s in self.symbols)

    @property
    def total_mass(self) -> float:
        """Total mass in amu."""
        return float(np.sum(self.masses))

    @property
    def coords_angstrom(self) -> FloatArray:
        """Coordinates in ångström, as a new array."""
        return self.coords * BOHR_TO_ANGSTROM
