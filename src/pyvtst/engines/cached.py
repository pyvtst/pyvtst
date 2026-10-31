"""An engine that serves results from a store and computes only on a miss."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from pyvtst.core.keys import cache_key, molecule_key
from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine
from pyvtst.ports.store import Record, Store, Value

FloatArray = npt.NDArray[np.float64]
T = TypeVar("T", bound=Value)


class CachedEngine:
    """Wrap an engine so its energies, gradients and Hessians are stored.

    Args:
        engine: The engine that computes results on a miss.
        store: Where results are kept.
        engine_id: Identity of the wrapped engine in cache keys. Defaults to
            its fully qualified class name; pin it to keep a cache valid
            across a class rename.

    A result is reused when the result kind, the molecule (without masses,
    which Born-Oppenheimer results do not depend on), the level of theory and
    the engine identity all match. Each record also holds the inputs it was
    computed from, and a record whose inputs do not match is recomputed and
    overwritten rather than trusted. Results from the wrapped engine are
    checked before they are stored; a bad one raises ``ValueError``.

    With :class:`pyvtst.Reaction`, pass ``Reaction(store=...)`` instead; it
    wraps the engine itself. ``CachedEngine`` is for scripts that call engines
    or optimizers directly.
    """

    def __init__(
        self, engine: Engine, store: Store, *, engine_id: str | None = None
    ) -> None:
        self._engine = engine
        self._store = store
        cls = type(engine)
        self.engine_id = engine_id or f"{cls.__module__}.{cls.__qualname__}"

    @property
    def level(self) -> LevelOfTheory:
        return self._engine.level

    def energy(self, mol: Molecule) -> float:
        return self._cached("energy", mol, self._engine.energy, _energy)

    def gradient(self, mol: Molecule) -> FloatArray:
        return self._cached("gradient", mol, self._engine.gradient, _gradient)

    def hessian(self, mol: Molecule) -> FloatArray:
        return self._cached("hessian", mol, self._engine.hessian, _hessian)

    def _cached(
        self,
        kind: str,
        mol: Molecule,
        compute: Callable[[Molecule], object],
        check: Callable[[object, Molecule], T],
    ) -> T:
        level = self._engine.level
        key = cache_key(kind, mol, level, include_masses=False, engine=self.engine_id)
        record = self._store.get(key)
        if record is not None:
            hit = self._verified(record, kind, mol, level, check)
            if hit is not None:
                return hit
        result = check(compute(mol), mol)
        self._store.put(
            key,
            {
                kind: result,
                "level": level.key(),
                "engine": self.engine_id,
                "symbols": " ".join(mol.symbols),
                "coords": np.array(mol.coords),
            },
        )
        return result

    def _verified(
        self,
        record: Record,
        kind: str,
        mol: Molecule,
        level: LevelOfTheory,
        check: Callable[[object, Molecule], T],
    ) -> T | None:
        """Return the stored result if the record's inputs match, else None."""
        symbols, coords = record.get("symbols"), record.get("coords")
        if (
            record.get("level") != level.key()
            or record.get("engine") != self.engine_id
            or not isinstance(symbols, str)
            or not isinstance(coords, np.ndarray)
        ):
            return None
        try:
            stored = Molecule(
                symbols.split(" "),
                coords,
                charge=mol.charge,
                mult=mol.mult,
            )
            if molecule_key(stored, include_masses=False) != molecule_key(
                mol, include_masses=False
            ):
                return None
            return check(record.get(kind), mol)
        except (TypeError, ValueError):
            return None


def _energy(value: object, mol: Molecule) -> float:
    if isinstance(value, bool) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"energy: expected a float, got {type(value).__name__}")
    energy = float(value)
    if not np.isfinite(energy):
        raise ValueError(f"energy: not finite ({energy})")
    return energy


def _gradient(value: object, mol: Molecule) -> FloatArray:
    return _array("gradient", value, (mol.n_atoms, 3))


def _hessian(value: object, mol: Molecule) -> FloatArray:
    n = 3 * mol.n_atoms
    return _array("hessian", value, (n, n))


def _array(kind: str, value: object, shape: tuple[int, int]) -> FloatArray:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
        raise ValueError(f"{kind}: expected a real numpy array, got {value!r:.80}")
    if value.shape != shape:
        raise ValueError(f"{kind}: expected shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{kind}: contains non-finite values")
    return np.array(value, dtype=np.float64)
