"""The Reaction facade: from geometry guesses to rate constants."""

from __future__ import annotations

import math
import numbers
import os
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import formula
from pyvtst.core.internals import (
    Internal,
    auto_internals,
    check_span,
    validate_internals,
)
from pyvtst.core.keys import cache_key
from pyvtst.core.mep import (
    COORDINATES,
    METHODS,
    Coordinates,
    MEPPoint,
    Method,
    ReactionPath,
    StopReason,
    from_mass_scaled,
    generalized_zpe,
    mass_scale,
    orient,
    to_mass_scaled,
    trace_branch,
    transition_vector,
)
from pyvtst.core.molecule import Molecule
from pyvtst.core.partition import temperatures
from pyvtst.core.rates import CVTResult, cvt_rate, tst_rate, zpe_energy
from pyvtst.core.stationary import (
    StationaryPoint,
    check_character,
    check_freq_scale,
    species_label,
)
from pyvtst.core.tunneling import eckart, wigner
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from pyvtst.core.vib import (
    curvilinear_generalized_modes,
    generalized_normal_modes,
    internal_basis,
)
from pyvtst.engines.cached import CachedEngine
from pyvtst.optimizers import GeometricOptimizer
from pyvtst.ports.engine import Engine
from pyvtst.ports.errors import ConvergenceError
from pyvtst.ports.optimizer import Optimizer
from pyvtst.ports.store import Store
from pyvtst.stores import open_store

FloatArray = npt.NDArray[np.float64]

THEORIES = ("tst", "cvt")
TUNNELING = (None, "wigner", "eckart")
_LATER = "is not implemented yet; it is planned for a later pyvtst release"

DEFAULT_DS: dict[Method, float] = {"lqa": 0.01, "euler": 0.001}  # bohr
DEFAULT_HESSIAN_EVERY = 10
MAX_DS = 0.1  # bohr
_STOP_MEANING = {
    "minimum": "the gradient vanished (a minimum was reached)",
    "energy_rise": "the next step would raise the energy; try a smaller ds",
}


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
        k: Rate constants of ``theory`` including tunneling: ``k_tst * kappa``
            for ``"tst"``, ``cvt.k * kappa`` for ``"cvt"``.
        k_tst: Conventional TST rate constants without tunneling, for either
            theory, so ``cvt.k / k_tst`` is the variational effect.
        kappa: Tunneling transmission coefficients (all 1 without tunneling).
        theory: The rate theory, ``"tst"`` or ``"cvt"``.
        tunneling: The tunneling correction, or ``None``.
        units: Unit of ``k`` and ``k_tst``: ``"s^-1"`` or
            ``"cm^3 molecule^-1 s^-1"``.
        barrier_kcal: The ZPE-corrected barrier ΔE₀‡ in kcal/mol.
        cvt: For ``"cvt"``, the variational details: rate constants without
            tunneling, s*(T), the path points used, k^GT(T, s) and
            ΔG^GT,°(T, s). ``None`` for ``"tst"``.

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
    cvt: CVTResult | None = None

    def table(self) -> str:
        """Return a plain-text table, one row per T.

        Columns T, k_TST, kappa and k; for CVT, T, s*, k_TST, k_CVT, kappa
        and k.
        """
        k_tst = f"k_TST ({self.units})"
        k_cvt = f"k_CVT ({self.units})"
        k = f"k ({self.units})"
        width = max(len(k_tst), 12)
        cvt = self.cvt
        header = f"{'T (K)':>9}  "
        if cvt is not None:
            header += f"{'s* (bohr)':>9}  "
        header += f"{k_tst:>{width}}  "
        if cvt is not None:
            header += f"{k_cvt:>{width}}  "
        lines = [header + f"{'kappa':>10}  {k:>{width}}"]
        for i, t in enumerate(self.T):
            line = f"{float(t):>9.2f}  "
            if cvt is not None:
                line += f"{float(cvt.s_star[i]):>9.4f}  "
            line += f"{float(self.k_tst[i]):>{width}.4e}  "
            if cvt is not None:
                line += f"{float(cvt.k[i]):>{width}.4e}  "
            line += f"{float(self.kappa[i]):>10.4f}  {float(self.k[i]):>{width}.4e}"
            lines.append(line)
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
        freq_scale: Frequency scale factor, finite and > 0. Every harmonic
            frequency of every species and every generalized frequency along
            the path, imaginary ones included, is multiplied by it, so ZPEs,
            partition functions, rate constants and tunneling all use scaled
            frequencies. It is not part of any cache key: a reaction with
            another factor on the same store reuses every engine result.

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
        freq_scale: float = 1.0,
    ) -> None:
        scale = check_freq_scale(freq_scale)
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
        cached = (
            engine
            if isinstance(engine, CachedEngine)
            else CachedEngine(engine, self.store)
        )
        self.engine: Engine = cached
        # Optimizations and path points hold engine results, so like the
        # engine's own records they are keyed by its identity.
        self._engine_id = cached.engine_id
        self._optimizer = optimizer
        self._freq_scale = scale
        self._points: StationaryPoints | None = None

    @property
    def freq_scale(self) -> float:
        """The frequency scale factor, fixed at construction."""
        return self._freq_scale

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
        input geometry, the level of theory, the engine and the optimizer
        settings.

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
        theory: Literal["tst", "cvt"] | str = "tst",
        tunneling: Literal["wigner", "eckart"] | str | None = None,
        path: ReactionPath | None = None,
    ) -> RateResult:
        """Return rate constants at temperatures ``T`` (K), optimizing first.

        Args:
            T: A temperature or array of temperatures in kelvin, all > 0.
            theory: ``"tst"`` (conventional transition state theory) or
                ``"cvt"`` (canonical variational TST along the minimum-energy
                path).
            tunneling: ``None``, ``"wigner"`` or ``"eckart"``. ``"zct"`` is
                reserved and raises ``NotImplementedError``. Wigner and Eckart
                are computed from the saddle point for both theories, so with
                CVT they do not follow the variational transition state.
            path: For ``"cvt"``, the path to use, from :meth:`compute_mep`.
                When ``None``, :meth:`compute_mep` runs with its defaults
                (s from -0.5 to 0.5 bohr), which is free once stored. Pass a
                path to use a wider s range, another step size or other
                coordinates.

        For CVT, a variational transition state at either end of the path
        gives a warning that the path's s range should be widened.

        Raises:
            ValueError: Unknown ``theory`` or ``tunneling``, invalid ``T``,
                ``path`` given without ``theory="cvt"`` or from another
                reaction, or an Eckart barrier that is not positive in both
                directions.
        """
        if theory not in THEORIES:
            raise ValueError(f"theory: expected 'tst' or 'cvt', got {theory!r}")
        if tunneling == "zct":
            raise NotImplementedError(f"tunneling 'zct' {_LATER}")
        if tunneling not in TUNNELING:
            raise ValueError(
                f"tunneling: expected None, 'wigner' or 'eckart', got {tunneling!r}"
            )
        if path is not None:
            if theory != "cvt":
                raise ValueError('path: is used only with theory="cvt"')
            if not isinstance(path, ReactionPath):
                raise TypeError(
                    f"path: expected a ReactionPath, got {type(path).__name__}"
                )
        t = temperatures(T)

        points = self.optimize()
        tst = tst_rate(points.reactants, points.ts, t)
        cvt = None
        if theory == "cvt":
            if path is None:
                path = self.compute_mep()
            elif path.molecules[path.ts_index] != points.ts.molecule:
                raise ValueError(
                    "path: its transition state is not this reaction's optimized "
                    "transition state; the path does not belong to this reaction"
                )
            elif path.freq_scale != self._freq_scale:
                raise ValueError(
                    f"path: its frequencies are scaled by {path.freq_scale}, "
                    f"this reaction's by {self._freq_scale}; recompute it with "
                    "this reaction's compute_mep()"
                )
            cvt = cvt_rate(points.reactants, path, t, points.ts.imag_threshold)
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
        k = (k_tst if cvt is None else cvt.k) * kappa
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
            cvt=cvt,
        )

    def compute_mep(
        self,
        s_min: float = -0.5,
        s_max: float = 0.5,
        ds: float | None = None,
        method: Literal["lqa", "euler"] | str = "lqa",
        hessian_every: int | None = None,
        coordinates: Literal["redundant", "cartesian"] | str = "redundant",
        internals: Sequence[Sequence[object]] | None = None,
    ) -> ReactionPath:
        """Trace the minimum-energy path through the transition state.

        The path is traced in mass-scaled coordinates (μ = 1 amu), so s is in
        bohr, with s = 0 at the transition state and s < 0 on the reactant
        side. Runs :meth:`optimize` first if needed. Every point is stored as
        soon as it is computed: a repeated call makes no engine calls, an
        interrupted call resumes from the last stored point, and a call with
        wider limits computes only the points beyond the stored ones.

        Args:
            s_min: Reactant-side limit in bohr, < 0.
            s_max: Product-side limit in bohr, > 0.
            ds: Step size in bohr, in (0, 0.1]. Defaults to 0.01 for
                ``"lqa"`` and 0.001 for ``"euler"``.
            method: ``"lqa"`` (Page-McIver local quadratic approximation, a
                Hessian at every point) or ``"euler"`` (steepest descent,
                gradient only, a Hessian every ``hessian_every`` points).
            hessian_every: For ``"euler"``, compute a Hessian every this many
                points, counting from the transition state. Defaults to 10.
            coordinates: Coordinates of the generalized frequencies:
                ``"redundant"`` (curvilinear, in redundant internal
                coordinates) or ``"cartesian"``. Cartesian frequencies carry
                the gradient times the curvature of straight-line motion, which
                on curved paths turns real bends imaginary.
            internals: For ``"redundant"``, the internal coordinates as
                ``("bond", i, j)``, ``("angle", i, j, k)``,
                ``("linear_bend", i, j, k)`` and ``("torsion", i, j, k, l)``
                tuples, with atom indices in the transition state's order.
                Generated from the optimized transition state when ``None``;
                see :func:`pyvtst.core.internals.auto_internals`. The same set
                is used at every point.

        Orientation needs the transition state's atoms in the same order, with
        the same elements and masses, as the reactants' atoms concatenated
        (and the products', when given). Otherwise the path is still computed
        but a warning says the sign of s is arbitrary. A side that stops
        before its limit (at a minimum, or because a step would go uphill)
        gives a warning.

        The path itself does not depend on ``coordinates`` or ``internals``:
        changing them recomputes the frequencies from stored Hessians, with no
        engine calls.

        Raises:
            ValueError: An argument is invalid, and nothing has been computed;
                or the internal coordinates do not span every internal motion
                at the transition state (before any path point is computed)
                or at a later point (naming its s).
        """
        lo, hi, step, every, meth = _mep_parameters(
            s_min, s_max, ds, method, hessian_every
        )
        coords, given = _coordinate_parameters(coordinates, internals, self.ts.n_atoms)
        points = self.optimize()
        ts = points.ts
        mol0 = ts.molecule
        used: tuple[Internal, ...] = ()
        if coords == "redundant":
            used = auto_internals(mol0) if given is None else given
            check_span(mol0, used, " at the transition state")
        tv = transition_vector(ts)
        sign = orient(
            mol0,
            tv,
            [r.molecule for r in points.reactants],
            None if points.products is None else [p.molecule for p in points.products],
        )
        if sign is None:
            warnings.warn(
                "the transition state's atoms do not match the reactants' (or "
                "products') atoms in element, mass and order; the path is not "
                "oriented and the sign of s is arbitrary",
                stacklevel=2,
            )
        product_branch = "minus" if sign == -1 else "plus"
        level = self.engine.level
        if meth == "euler":
            prefix = cache_key(
                "mep",
                mol0,
                level,
                method=meth,
                ds=step,
                hessian_every=every,
                engine=self._engine_id,
            )
        else:
            prefix = cache_key(
                "mep", mol0, level, method=meth, ds=step, engine=self._engine_id
            )
        scale = mass_scale(mol0)
        x0 = to_mass_scaled(mol0, mol0.coords)
        start = MEPPoint(0, x0, ts.energy, np.zeros_like(x0))

        def compute(index: int, x: FloatArray) -> MEPPoint:
            mol = mol0.replace(coords=from_mass_scaled(mol0, x))
            energy = self.engine.energy(mol)
            gradient = self.engine.gradient(mol)
            hessian = self.engine.hessian(mol) if index % every == 0 else None
            raw[index] = _PointRecord(index, mol, energy, gradient)
            return MEPPoint(
                index,
                to_mass_scaled(mol0, mol.coords),
                energy,
                gradient.reshape(-1) / scale,
                None if hessian is None else hessian / np.outer(scale, scale),
            )

        def basis(x: FloatArray) -> FloatArray:
            return internal_basis(mol0.replace(coords=from_mass_scaled(mol0, x)))

        raw: dict[int, _PointRecord] = {}
        branches: dict[str, tuple[list[_PointRecord], StopReason]] = {}
        for branch, direction in (("minus", -1.0), ("plus", 1.0)):
            limit = hi if branch == product_branch else -lo
            branches[branch] = self._trace_stored(
                f"{prefix}/{branch}",
                mol0,
                start,
                x0 + direction * step * tv,
                compute,
                raw,
                ds=step,
                n_max=round(limit / step),
                method=meth,
                basis=basis if meth == "lqa" else None,
            )

        reactant_branch = "plus" if product_branch == "minus" else "minus"
        reactant, stop_reactant = branches[reactant_branch]
        product, stop_product = branches[product_branch]
        for side, records, reason, limit in (
            ("reactant", reactant, stop_reactant, lo),
            ("product", product, stop_product, hi),
        ):
            if reason != "s_limit":
                reached = math.copysign(len(records) * step, limit)
                warnings.warn(
                    f"MEP: the {side} side stopped at s = {reached:+.4f} bohr, "
                    f"before the requested {limit:+.4f}: {reason}, "
                    f"{_STOP_MEANING[reason]}",
                    stacklevel=2,
                )

        ts_record = _PointRecord(0, mol0, ts.energy, self.engine.gradient(mol0))
        ordered = [*reversed(reactant), ts_record, *product]
        s = np.array(
            [-r.index * step for r in reversed(reactant)]
            + [0.0]
            + [r.index * step for r in product]
        )
        frequencies: list[FloatArray | None] = []
        zpe = np.full(len(ordered), np.nan)
        imaginary: list[str] = []
        for i, record in enumerate(ordered):
            if record.index % every:
                frequencies.append(None)
                continue
            hessian = self.engine.hessian(record.molecule)
            gradient = None if record.index == 0 else record.gradient
            ts_mode = ts.modes[0] if ts.modes is not None else None
            if coords == "redundant":
                freqs, _ = curvilinear_generalized_modes(
                    record.molecule,
                    hessian,
                    used,
                    gradient,
                    ts_mode,
                    reference=mol0,
                    where=f" at s = {s[i]:+.4f} bohr",
                )
            else:
                freqs, _ = generalized_normal_modes(
                    record.molecule, hessian, gradient, ts_mode
                )
            freqs = freqs * self._freq_scale
            frequencies.append(freqs)
            zpe[i] = generalized_zpe(freqs, ts.imag_threshold)
            large = freqs[freqs < -ts.imag_threshold]
            if record.index and large.size:
                listed = ", ".join(f"{f:.1f}" for f in large)
                imaginary.append(f"s = {s[i]:+.4f} ([{listed}] cm^-1)")
        if imaginary:
            hint = ""
            if coords == "cartesian":
                hint = (
                    "; Cartesian generalized frequencies are often spuriously "
                    'imaginary on curved paths, use coordinates="redundant"'
                )
            warnings.warn(
                "MEP: imaginary generalized frequencies above the "
                f"{ts.imag_threshold:g} cm^-1 threshold, excluded from the "
                f"zero-point energy, at {'; '.join(imaginary)}{hint}",
                stacklevel=2,
            )
        return ReactionPath(
            s=s,
            molecules=tuple(r.molecule for r in ordered),
            energies=np.array([r.energy for r in ordered]),
            gradients=np.array([r.gradient for r in ordered]),
            frequencies=tuple(frequencies),
            zpe=zpe,
            reference_energy=sum(r.energy for r in points.reactants),
            reference_zpe_energy=sum(zpe_energy(r) for r in points.reactants),
            ts_index=len(reactant),
            method=meth,
            ds=step,
            stop_reactant=stop_reactant,
            stop_product=stop_product,
            oriented=sign is not None,
            coordinates=coords,
            internals=used,
            freq_scale=self._freq_scale,
        )

    def _trace_stored(
        self,
        branch_key: str,
        mol0: Molecule,
        start: MEPPoint,
        first_x: FloatArray,
        compute: Callable[[int, FloatArray], MEPPoint],
        raw: dict[int, _PointRecord],
        *,
        ds: float,
        n_max: int,
        method: Method,
        basis: Callable[[FloatArray], FloatArray] | None,
    ) -> tuple[list[_PointRecord], StopReason]:
        """Trace one branch from its stored points, storing each new point.

        ``compute`` puts the Cartesian record of every point it computes in
        ``raw``, so exactly what the engine returned is stored.
        """
        stored = self._stored_points(branch_key, mol0, ds)
        new: list[_PointRecord] = []

        def on_point(p: MEPPoint) -> None:
            record = raw[p.index]
            self.store.put(f"{branch_key}/{p.index:06d}", record.as_record(ds))
            new.append(record)

        traced, reason = trace_branch(
            start,
            first_x,
            compute,
            ds=ds,
            n_max=n_max,
            method=method,
            existing=[r.point(mol0) for r in stored],
            on_point=on_point,
            basis=basis,
        )
        return (stored + new)[: len(traced)], reason

    def _stored_points(
        self, branch_key: str, mol0: Molecule, ds: float
    ) -> list[_PointRecord]:
        """Return the valid stored points of a branch, up to the first bad one."""
        points: list[_PointRecord] = []
        for key in self.store.keys(f"{branch_key}/"):
            index = len(points) + 1
            if key != f"{branch_key}/{index:06d}":
                break
            record = self.store.get(key)
            if record is None:
                break
            point = _PointRecord.read(record, index, mol0, ds)
            if point is None:
                break
            points.append(point)
        return points

    def _characterize(
        self, guess: Molecule, kind: Literal["minimum", "saddle"], role: str
    ) -> StationaryPoint:
        label = species_label(guess, role)
        if guess.n_atoms == 1:
            return StationaryPoint.from_frequencies(
                guess, self.engine.energy(guess), [], freq_scale=self._freq_scale
            )
        mol, energy = self._optimized(guess, kind, label)
        sp = StationaryPoint.from_hessian(
            mol, energy, self.engine.hessian(mol), freq_scale=self._freq_scale
        )
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
            engine=self._engine_id,
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


@dataclass(frozen=True, eq=False)
class _PointRecord:
    """A stored MEP point: Cartesian geometry, energy and gradient."""

    index: int
    molecule: Molecule
    energy: float
    gradient: FloatArray

    def point(self, mol0: Molecule) -> MEPPoint:
        scale = mass_scale(mol0)
        return MEPPoint(
            self.index,
            to_mass_scaled(mol0, self.molecule.coords),
            self.energy,
            self.gradient.reshape(-1) / scale,
        )

    def as_record(self, ds: float) -> dict[str, np.ndarray | float]:
        return {
            "coords": np.array(self.molecule.coords),
            "energy": self.energy,
            "gradient": np.array(self.gradient),
            "s": self.index * ds,
        }

    @classmethod
    def read(
        cls, record: Mapping[str, object], index: int, mol0: Molecule, ds: float
    ) -> _PointRecord | None:
        coords, energy = record.get("coords"), record.get("energy")
        gradient, s = record.get("gradient"), record.get("s")
        shape = (mol0.n_atoms, 3)
        if not (
            isinstance(coords, np.ndarray)
            and isinstance(gradient, np.ndarray)
            and isinstance(energy, float)
            and isinstance(s, float)
            and coords.shape == shape
            and gradient.shape == shape
            and np.all(np.isfinite(gradient))
            and math.isfinite(energy)
            and abs(s - index * ds) <= 1e-12
        ):
            return None
        try:
            mol = mol0.replace(coords=coords)
        except ValueError:
            return None
        return cls(index, mol, energy, np.array(gradient, dtype=np.float64))


def _mep_parameters(
    s_min: object,
    s_max: object,
    ds: object,
    method: object,
    hessian_every: object,
) -> tuple[float, float, float, int, Method]:
    """Validate compute_mep arguments; return them with defaults filled in."""
    if method not in METHODS:
        raise ValueError(f"method: expected 'lqa' or 'euler', got {method!r}")
    meth: Method = method
    lo = _real(s_min, "s_min")
    if not lo < 0:
        raise ValueError(f"s_min: must be negative, got {lo}")
    hi = _real(s_max, "s_max")
    if not hi > 0:
        raise ValueError(f"s_max: must be positive, got {hi}")
    step = DEFAULT_DS[meth] if ds is None else _real(ds, "ds")
    if not 0 < step <= MAX_DS:
        raise ValueError(f"ds: must be in (0, {MAX_DS}] bohr, got {step}")
    if hessian_every is None:
        every = DEFAULT_HESSIAN_EVERY if meth == "euler" else 1
    elif (
        isinstance(hessian_every, bool)
        or not isinstance(hessian_every, numbers.Integral)
        or hessian_every < 1
    ):
        raise ValueError(
            f"hessian_every: must be a positive integer, got {hessian_every!r}"
        )
    else:
        every = int(hessian_every)
    if meth == "lqa" and every != 1:
        raise ValueError(
            "hessian_every: LQA needs a Hessian at every point; use "
            "method='euler' for sparse Hessians"
        )
    return lo, hi, step, every, meth


def _coordinate_parameters(
    coordinates: object, internals: Sequence[Sequence[object]] | None, n_atoms: int
) -> tuple[Coordinates, tuple[Internal, ...] | None]:
    """Validate compute_mep's coordinate arguments."""
    if coordinates not in COORDINATES:
        raise ValueError(
            f"coordinates: expected 'redundant' or 'cartesian', got {coordinates!r}"
        )
    coords: Coordinates = coordinates
    if internals is None:
        return coords, None
    if coords == "cartesian":
        raise ValueError(
            "internals: only used with coordinates='redundant'; drop internals= "
            "for Cartesian frequencies"
        )
    return coords, validate_internals(internals, n_atoms)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: must be finite, got {number}")
    return number


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
