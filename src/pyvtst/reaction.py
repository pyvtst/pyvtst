"""The Reaction facade: from geometry guesses to rate constants."""

from __future__ import annotations

import os
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import formula
from pyvtst.core.keys import cache_key
from pyvtst.core.molecule import Molecule
from pyvtst.core.partition import temperatures
from pyvtst.core.rates import tst_rate, zpe_energy
from pyvtst.core.stationary import (
    StationaryPoint,
    check_character,
    species_label,
)
from pyvtst.core.tunneling import eckart, wigner
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from pyvtst.engines.cached import CachedEngine
from pyvtst.optimizers import GeometricOptimizer
from pyvtst.ports.engine import Engine
from pyvtst.ports.errors import ConvergenceError
from pyvtst.ports.optimizer import Optimizer
from pyvtst.ports.store import Store
from pyvtst.stores import open_store

FloatArray = npt.NDArray[np.float64]

THEORIES = ("tst",)
TUNNELING = (None, "wigner", "eckart")
_LATER = "is not implemented yet; it is planned for a later pyvtst release"


@dataclass(frozen=True)
class StationaryPoints:
    """The characterized stationary points of a reaction.

    Attributes:
        reactants: One or two reactant minima.
        ts: The transition state, a first-order saddle point.
        products: Product minima, or ``None`` when no products were given.
    """

    reactants: tuple[StationaryPoint, ...]
    ts: StationaryPoint
    products: tuple[StationaryPoint, ...] | None


@dataclass(frozen=True, eq=False)
class RateResult:
    """Rate constants of a reaction over a set of temperatures.

    Attributes:
        T: Temperatures in kelvin.
        k: Rate constants including tunneling, ``k_tst * kappa``.
        k_tst: Rate constants without tunneling.
        kappa: Tunneling transmission coefficients (all 1 without tunneling).
        theory: The rate theory, e.g. ``"tst"``.
        tunneling: The tunneling correction, or ``None``.
        units: Unit of ``k`` and ``k_tst``: ``"s^-1"`` or
            ``"cm^3 molecule^-1 s^-1"``.
        barrier_kcal: The ZPE-corrected barrier ΔE₀‡ in kcal/mol.

    All arrays have one value per temperature and are read-only.
    """

    T: FloatArray
    k: FloatArray
    k_tst: FloatArray
    kappa: FloatArray
    theory: str
    tunneling: str | None
    units: str
    barrier_kcal: float

    def table(self) -> str:
        """Return a plain-text table: T, k_TST, kappa and k, one row per T."""
        k_tst = f"k_TST ({self.units})"
        k = f"k ({self.units})"
        width = max(len(k_tst), 12)
        lines = [f"{'T (K)':>9}  {k_tst:>{width}}  {'kappa':>10}  {k:>{width}}"]
        for row in zip(self.T, self.k_tst, self.kappa, self.k, strict=True):
            t, a, b, c = (float(v) for v in row)
            lines.append(f"{t:>9.2f}  {a:>{width}.4e}  {b:>10.4f}  {c:>{width}.4e}")
        return "\n".join(lines)


class Reaction:
    """A gas-phase reaction: optimizes stationary points and computes rates.

    Args:
        reactants: One or two reactant molecules (a single molecule is
            accepted too).
        ts: A guess for the transition state.
        engine: Computes energies, gradients and Hessians.
        products: Product molecules, optional. With products, Eckart
            tunneling uses an asymmetric barrier; without, a symmetric one.
        optimizer: Locates minima and saddle points. Defaults to
            :class:`~pyvtst.optimizers.GeometricOptimizer`, built on first use.
        store: Where results are cached. ``None`` keeps them in memory for the
            life of this object; a path keeps them in an HDF5 file (created if
            missing, needs h5py), so later sessions reuse them; a store object
            is used as given. The engine is wrapped in
            :class:`~pyvtst.engines.CachedEngine` over this store, unless it
            already is a ``CachedEngine``, which then keeps its own store.

    Construction checks that reactants, transition state and products have
    the same elements and total charge, and makes no engine calls. A total
    spin that cannot be conserved gives a warning.

    Reaction-path degeneracy enters only through the molecules'
    ``symmetry_number``. A chiral transition state needs an extra factor of 2
    (two mirror-image paths) that cannot be expressed yet; its rate comes out
    a factor of 2 too low.

    Example::

        rxn = Reaction([h2, h], h3_guess, engine, store="h3.h5")
        result = rxn.rates([300, 500, 1000], tunneling="eckart")
        print(result.table())
    """

    def __init__(
        self,
        reactants: Molecule | Sequence[Molecule],
        ts: Molecule,
        engine: Engine,
        products: Sequence[Molecule] | None = None,
        optimizer: Optimizer | None = None,
        store: Store | str | os.PathLike[str] | None = None,
    ) -> None:
        reacts = (reactants,) if isinstance(reactants, Molecule) else tuple(reactants)
        if not 1 <= len(reacts) <= 2:
            raise ValueError(
                f"reactants: expected one or two molecules, got {len(reacts)}"
            )
        prods = None if products is None else tuple(products)
        if prods is not None and not prods:
            raise ValueError("products: expected at least one molecule, or None")
        _check_balance(reacts, ts, "transition state")
        if prods is not None:
            _check_balance(reacts, prods, "products")
        _check_spin(reacts, ts, prods)

        self.reactants = reacts
        self.ts = ts
        self.products = prods
        self.store = open_store(store)
        self.engine: Engine = (
            engine
            if isinstance(engine, CachedEngine)
            else CachedEngine(engine, self.store)
        )
        self._optimizer = optimizer
        self._points: StationaryPoints | None = None

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer; the default one is built on first access."""
        if self._optimizer is None:
            self._optimizer = GeometricOptimizer()
        return self._optimizer

    def optimize(self) -> StationaryPoints:
        """Optimize and characterize every species; cached after the first call.

        Multi-atom reactants and products are optimized to minima and the
        transition-state guess to a first-order saddle point, then Hessians
        are computed at the optimized geometries. Single atoms get only an
        energy. Optimized geometries are kept in the store under a key of the
        input geometry, the level of theory and the optimizer settings.

        Raises:
            StationaryPointError: A reactant or product has an imaginary
                frequency, or the transition state has not exactly one.
            ConvergenceError: An optimization did not converge; the message
                starts with the species.
        """
        if self._points is None:
            reactants = tuple(
                self._characterize(mol, "minimum", f"reactant {i}")
                for i, mol in enumerate(self.reactants, start=1)
            )
            ts = self._characterize(self.ts, "saddle", "transition state")
            products = None
            if self.products is not None:
                products = tuple(
                    self._characterize(mol, "minimum", f"product {i}")
                    for i, mol in enumerate(self.products, start=1)
                )
            self._points = StationaryPoints(reactants, ts, products)
        return self._points

    def rates(
        self,
        T: npt.ArrayLike,
        theory: str = "tst",
        tunneling: Literal["wigner", "eckart"] | str | None = None,
    ) -> RateResult:
        """Return rate constants at temperatures ``T`` (K), optimizing first.

        Args:
            T: A temperature or array of temperatures in kelvin, all > 0.
            theory: ``"tst"`` (conventional transition state theory).
                ``"cvt"`` is reserved and raises ``NotImplementedError``.
            tunneling: ``None``, ``"wigner"`` or ``"eckart"``. ``"zct"`` is
                reserved and raises ``NotImplementedError``.

        Raises:
            ValueError: Unknown ``theory`` or ``tunneling``, invalid ``T``, or
                an Eckart barrier that is not positive in both directions.
        """
        if theory == "cvt":
            raise NotImplementedError(f"theory 'cvt' {_LATER}")
        if theory not in THEORIES:
            raise ValueError(f"theory: expected 'tst', got {theory!r}")
        if tunneling == "zct":
            raise NotImplementedError(f"tunneling 'zct' {_LATER}")
        if tunneling not in TUNNELING:
            raise ValueError(
                f"tunneling: expected None, 'wigner' or 'eckart', got {tunneling!r}"
            )
        t = temperatures(T)

        points = self.optimize()
        tst = tst_rate(points.reactants, points.ts, t)
        # optimize() checked that the transition state is a first-order saddle.
        nu = cast(float, points.ts.imaginary_frequency)
        if tunneling == "wigner":
            kappa = wigner(nu, t)
        elif tunneling == "eckart":
            v1 = tst.barrier_zpe
            v2 = v1
            if points.products is not None:
                v2 = zpe_energy(points.ts) - sum(zpe_energy(p) for p in points.products)
            kappa = eckart(v1, v2, nu, t)
        else:
            kappa = np.ones_like(t)
        k_tst = np.array(tst.k)
        k = k_tst * kappa
        for a in (t, k, k_tst, kappa):
            a.flags.writeable = False
        return RateResult(
            T=t,
            k=k,
            k_tst=k_tst,
            kappa=kappa,
            theory=theory,
            tunneling=tunneling,
            units=tst.units,
            barrier_kcal=tst.barrier_zpe * HARTREE_TO_KCAL_PER_MOL,
        )

    def _characterize(
        self, guess: Molecule, kind: Literal["minimum", "saddle"], role: str
    ) -> StationaryPoint:
        label = species_label(guess, role)
        if guess.n_atoms == 1:
            return StationaryPoint.from_frequencies(
                guess, self.engine.energy(guess), []
            )
        mol, energy = self._optimized(guess, kind, label)
        sp = StationaryPoint.from_hessian(mol, energy, self.engine.hessian(mol))
        check_character(sp, kind, label)
        return sp

    def _optimized(
        self, guess: Molecule, kind: Literal["minimum", "saddle"], label: str
    ) -> tuple[Molecule, float]:
        """Return the optimized molecule and energy, from the store if there."""
        optimizer = self.optimizer
        key = cache_key(
            "opt/min" if kind == "minimum" else "opt/saddle",
            guess,
            self.engine.level,
            optimizer=optimizer.key(),
        )
        record = self.store.get(key)
        if record is not None:
            hit = _from_record(guess, record)
            if hit is not None:
                return hit
        try:
            if kind == "minimum":
                result = optimizer.minimize(guess, self.engine)
            else:
                result = optimizer.saddle(guess, self.engine)
        except ConvergenceError as exc:
            raise ConvergenceError(f"{label}: {exc}", exc.last_molecule) from exc
        mol = guess.replace(coords=result.molecule.coords)
        energy = float(result.energy)
        self.store.put(
            key,
            {
                "coords": np.array(mol.coords),
                "energy": energy,
                "n_iterations": int(result.n_iterations),
            },
        )
        return mol, energy


def _from_record(
    guess: Molecule, record: Mapping[str, object]
) -> tuple[Molecule, float] | None:
    coords, energy = record.get("coords"), record.get("energy")
    if not isinstance(coords, np.ndarray) or not isinstance(energy, float):
        return None
    try:
        return guess.replace(coords=coords), energy
    except ValueError:
        return None


def _check_balance(
    reactants: tuple[Molecule, ...],
    other: Molecule | tuple[Molecule, ...],
    name: str,
) -> None:
    others = (other,) if isinstance(other, Molecule) else other
    left = [s for m in reactants for s in m.symbols]
    right = [s for m in others for s in m.symbols]
    if Counter(left) != Counter(right):
        raise ValueError(
            f"element composition differs: reactants {formula(left)}, "
            f"{name} {formula(right)}"
        )
    q_left = sum(m.charge for m in reactants)
    q_right = sum(m.charge for m in others)
    if q_left != q_right:
        raise ValueError(f"total charge differs: reactants {q_left}, {name} {q_right}")


def _check_spin(
    reactants: tuple[Molecule, ...],
    ts: Molecule,
    products: tuple[Molecule, ...] | None,
) -> None:
    """Warn when the TS multiplicity cannot couple from reactants or products."""
    for name, group in (("reactants", reactants), ("products", products)):
        if group is None:
            continue
        allowed = _coupled_multiplicities([m.mult for m in group])
        if ts.mult not in allowed:
            listed = ", ".join(str(m) for m in sorted(allowed))
            warnings.warn(
                f"total spin is not conserved: the {name} can couple to "
                f"multiplicities {listed}, but the transition state has {ts.mult}",
                stacklevel=3,
            )


def _coupled_multiplicities(mults: list[int]) -> set[int]:
    """Multiplicities 2S + 1 reachable by coupling the spins of ``mults``."""
    allowed = {1}
    for m in mults:
        allowed = {c for a in allowed for c in range(abs(a - m) + 1, a + m, 2)}
    return allowed
