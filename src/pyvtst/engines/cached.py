"""An engine that serves results from a store and computes only on a miss."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from pyvtst.core.keys import cache_key, molecule_key
from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine, EngineResult
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

    On a miss the wrapped engine's :meth:`~pyvtst.ports.Engine.compute` runs
    once, and every result it returns is stored: a Hessian request also stores
    the energy and gradient, and an energy request also stores the gradient.

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

    def compute(self, mol: Molecule, *, hessian: bool) -> EngineResult:
        energy = self._stored("energy", mol, _energy)
        gradient = self._stored("gradient", mol, _gradient)
        full = self._stored("hessian", mol, _hessian) if hessian else None
        if energy is None or gradient is None or (hessian and full is None):
            return self._computed(mol, hessian=hessian)
        return EngineResult(energy, gradient, full)

    def energy(self, mol: Molecule) -> float:
        energy = self._stored("energy", mol, _energy)
        if energy is None:
            return self._computed(mol, hessian=False).energy
        return energy

    def gradient(self, mol: Molecule) -> FloatArray:
        gradient = self._stored("gradient", mol, _gradient)
        if gradient is None:
            return self._computed(mol, hessian=False).gradient
        return gradient

    def hessian(self, mol: Molecule) -> FloatArray:
        full = self._stored("hessian", mol, _hessian)
        if full is None:
            # _computed with hessian=True always returns a Hessian.
            return cast(FloatArray, self._computed(mol, hessian=True).hessian)
        return full

    def _key(self, kind: str, mol: Molecule) -> str:
        level = self._engine.level
        return cache_key(kind, mol, level, include_masses=False, engine=self.engine_id)

    def _stored(
        self, kind: str, mol: Molecule, check: Callable[[object, Molecule], T]
    ) -> T | None:
        """Return the stored result of this kind, or None if there is none."""
        record = self._store.get(self._key(kind, mol))
        if record is None:
            return None
        return self._verified(record, kind, mol, self._engine.level, check)

    def _computed(self, mol: Molecule, *, hessian: bool) -> EngineResult:
        """Run the wrapped engine once and store every result it returns."""
        raw = self._engine.compute(mol, hessian=hessian)
        if not isinstance(raw, EngineResult):
            raise ValueError(
                f"compute: expected an EngineResult, got {type(raw).__name__}"
            )
        if hessian and raw.hessian is None:
            raise ValueError("hessian: compute(hessian=True) returned no Hessian")
        # Every result is checked before any is stored.
        result = EngineResult(
            _energy(raw.energy, mol),
            _gradient(raw.gradient, mol),
            None if raw.hessian is None else _hessian(raw.hessian, mol),
        )
        level = self._engine.level
        inputs: Record = {
            "level": level.key(),
            "engine": self.engine_id,
            "symbols": " ".join(mol.symbols),
            "coords": np.array(mol.coords),
        }
        values: dict[str, Value | None] = {
            "energy": result.energy,
            "gradient": result.gradient,
            "hessian": result.hessian,
        }
        for kind, value in values.items():
            if value is not None:
                self._store.put(self._key(kind, mol), {kind: value, **inputs})
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
