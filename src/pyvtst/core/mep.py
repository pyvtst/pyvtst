"""Minimum-energy paths (MEP) through a saddle point in mass-scaled coordinates.

Mass-scaled Cartesian coordinates use the scaling mass μ = 1 amu,
``x = sqrt(m / amu) · R``, so the reaction coordinate s is in bohr. Gradients
transform as ``g / sqrt(m)`` and Hessians as ``H / sqrt(m_i m_j)``. Directions
in these coordinates are the same as in the electron-mass weighting used by
:mod:`pyvtst.core.vib`, which differs by a constant factor.

Path physics here takes callables instead of an engine, so it works on any
analytic surface; :class:`pyvtst.Reaction` connects it to engines and stores.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.linalg
import scipy.optimize

from pyvtst.core.internals import Internal
from pyvtst.core.molecule import Molecule
from pyvtst.core.stationary import DEFAULT_IMAG_THRESHOLD, StationaryPoint
from pyvtst.core.units import CM1_TO_HARTREE, HARTREE_TO_KCAL_PER_MOL

FloatArray = npt.NDArray[np.float64]
Method = Literal["lqa", "euler"]
StopReason = Literal["s_limit", "minimum", "energy_rise"]
Coordinates = Literal["redundant", "cartesian"]

METHODS: tuple[Method, ...] = ("lqa", "euler")
COORDINATES: tuple[Coordinates, ...] = ("redundant", "cartesian")

# A mass-scaled gradient norm below this (hartree/bohr) means a minimum.
GRADIENT_TOLERANCE = 1e-5
# Mass-scaled displacement (bohr) at which the two sides are compared.
ORIENT_STEP = 0.01
# Score difference below which the two sides are indistinguishable.
_ORIENT_TIE = 1e-12
# Masses (amu) closer than this count as the same isotope.
_MASS_MATCH = 1e-6
_MAX_DOUBLINGS = 100


# Mass-scaled coordinates


def mass_scale(mol: Molecule) -> FloatArray:
    """Return ``sqrt(m / amu)`` for each Cartesian component, shape (3N,)."""
    return np.repeat(np.sqrt(mol.masses), 3)


def to_mass_scaled(mol: Molecule, coords: npt.ArrayLike) -> FloatArray:
    """Map Cartesian coordinates in bohr, shape (N, 3), to mass-scaled ones, (3N,)."""
    return np.asarray(coords, dtype=np.float64).reshape(-1) * mass_scale(mol)


def from_mass_scaled(mol: Molecule, x: npt.ArrayLike) -> FloatArray:
    """Map mass-scaled coordinates, shape (3N,), to Cartesian bohr, (N, 3)."""
    return (np.asarray(x, dtype=np.float64) / mass_scale(mol)).reshape(-1, 3)


def transition_vector(ts: StationaryPoint) -> FloatArray:
    """Return the unit transition vector of a first-order saddle point, (3N,).

    It is the imaginary normal mode, in mass-scaled coordinates, with its sign
    fixed so that its largest-magnitude component (the first, on ties) is
    positive. The same saddle point and Hessian always give the same vector.

    Raises:
        ValueError: ``ts`` has no normal modes or is not a first-order saddle.
    """
    if ts.modes is None:
        raise ValueError(
            "ts: normal modes are needed; build it with StationaryPoint.from_hessian"
        )
    if ts.kind != "saddle":
        raise ValueError(f"ts: expected a first-order saddle point, got {ts.kind}")
    v = np.array(ts.modes[0], dtype=np.float64)
    v /= np.linalg.norm(v)
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return v


# Steps


def lqa_step(
    gradient: npt.ArrayLike,
    hessian: npt.ArrayLike,
    ds: float,
    basis: npt.ArrayLike | None = None,
) -> tuple[FloatArray, bool]:
    """Take a Page-McIver local quadratic approximation (LQA) step.

    The steepest-descent path of the quadratic model at the current point is
    followed for an arc length of ``ds``. On a quadratic surface the new point
    lies exactly on the steepest-descent path.

    Args:
        gradient: Gradient at the current point, shape (n,).
        hessian: Hessian at the current point, shape (n, n).
        ds: Arc length of the step, > 0.
        basis: Orthonormal columns, shape (n, k), spanning the space to step
            in (for a molecule, the space without overall translations and
            rotations). Defaults to the whole space.

    Returns:
        ``(displacement, at_minimum)``. When the model's minimum is closer
        than ``ds`` along its path, ``at_minimum`` is true and the
        displacement leads to that minimum.

    Raises:
        ValueError: The gradient vanishes or ``ds`` is not positive.

    References:
        M. Page and J. W. McIver, J. Chem. Phys. 88, 922 (1988).
    """
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    b = np.eye(g.size) if basis is None else np.asarray(basis, dtype=np.float64)
    if not ds > 0:
        raise ValueError(f"ds: must be positive, got {ds}")
    lam, u = scipy.linalg.eigh(b.T @ np.asarray(hessian, dtype=np.float64) @ b)
    gi = u.T @ (b.T @ g)
    norm = float(np.linalg.norm(gi))
    if norm == 0.0:
        raise ValueError("gradient: vanishes; an LQA step needs a direction")
    # Components this small cannot move the point; dropping them keeps the
    # arc length of exactly flat directions from growing without bound.
    keep = np.abs(gi) > 1e-14 * norm
    lam, u, gi = lam[keep], u[:, keep], gi[keep]

    def displacement(t: float) -> FloatArray:
        return np.asarray(b @ (u @ (-gi * _decay(lam, t))), dtype=np.float64)

    def speed(t: float) -> float:
        return float(np.sqrt(np.sum(gi * gi * np.exp(-2.0 * lam * t))))

    def arc(t: float) -> float:
        value, _ = scipy.integrate.quad(
            speed, 0.0, t, epsabs=1e-14, epsrel=1e-12, limit=200
        )
        return float(value)

    if np.all(lam > 0):
        total, _ = scipy.integrate.quad(
            speed, 0.0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=200
        )
        if total <= ds:
            return np.asarray(b @ (u @ (-gi / lam)), dtype=np.float64), True
    t_hi = ds / norm
    for _ in range(_MAX_DOUBLINGS):
        if arc(t_hi) >= ds:
            break
        t_hi *= 2.0
    else:
        raise ValueError("lqa_step: no step of arc length ds along the model path")
    t = scipy.optimize.brentq(
        lambda t: arc(t) - ds, 0.0, t_hi, xtol=1e-13 * t_hi, rtol=1e-14
    )
    return displacement(float(t)), False


def _decay(lam: FloatArray, t: float) -> FloatArray:
    """Return (1 - exp(-λt))/λ, with the small-λt series near λ = 0."""
    lt = lam * t
    small = np.abs(lt) < 1e-8
    safe = np.where(small, 1.0, lam)
    exact = -np.expm1(-np.where(small, 0.0, lt)) / safe
    return np.asarray(np.where(small, t * (1.0 - 0.5 * lt), exact), dtype=np.float64)


def euler_step(gradient: npt.ArrayLike, ds: float) -> FloatArray:
    """Return the steepest-descent displacement ``-ds · g / |g|``.

    Raises:
        ValueError: The gradient vanishes.
    """
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(g))
    if norm == 0.0:
        raise ValueError("gradient: vanishes; an Euler step needs a direction")
    return -ds * g / norm


# Branches


@dataclass(frozen=True, eq=False)
class MEPPoint:
    """One point of a path branch, in mass-scaled coordinates.

    Attributes:
        index: Step count from the transition state (index 0).
        x: Mass-scaled coordinates, shape (n,).
        energy: Energy in hartree.
        gradient: Mass-scaled gradient, shape (n,).
        hessian: Mass-scaled Hessian, shape (n, n), or ``None``.
    """

    index: int
    x: FloatArray
    energy: float
    gradient: FloatArray
    hessian: FloatArray | None = None


def trace_branch(
    start: MEPPoint,
    first_x: FloatArray,
    compute: Callable[[int, FloatArray], MEPPoint],
    *,
    ds: float,
    n_max: int,
    method: Method = "lqa",
    existing: Sequence[MEPPoint] = (),
    on_point: Callable[[MEPPoint], None] | None = None,
    basis: Callable[[FloatArray], FloatArray] | None = None,
) -> tuple[list[MEPPoint], StopReason]:
    """Trace one branch of a path from the transition state.

    Args:
        start: The transition state, index 0.
        first_x: The first point, ``ds`` from the transition state along the
            transition vector.
        compute: ``compute(index, x)`` returns the point at ``x``. It may
            return its coordinates in a canonical form (for example after a
            round trip through Cartesian coordinates); tracing continues from
            the returned coordinates, so a resumed trace matches an
            uninterrupted one bit for bit. With ``"lqa"`` the point must have
            a Hessian.
        ds: Step size in bohr.
        n_max: Index of the last point allowed (the s limit over ``ds``).
        method: ``"lqa"`` or ``"euler"``.
        existing: Points of this branch already computed, indices 1, 2, …;
            tracing continues from the last one. Points beyond ``n_max`` are
            ignored.
        on_point: Called with every new point before the next step.
        basis: ``basis(x)`` returns orthonormal columns spanning the space an
            LQA step may move in. Defaults to the whole space.

    Returns:
        ``(points, reason)``: the points with index 1 and up, and why tracing
        stopped. ``"s_limit"``: index ``n_max`` was reached. ``"minimum"``:
        the gradient norm fell below :data:`GRADIENT_TOLERANCE`, or the LQA
        model's minimum is closer than ``ds``. ``"energy_rise"``: the next
        step would raise the energy; that point is discarded.
    """
    points = list(existing)[: max(n_max, 0)]
    for i, p in enumerate(points, start=1):
        if p.index != i:
            raise ValueError(f"existing: expected index {i}, got {p.index}")
    current = points[-1] if points else start
    while True:
        if current.index >= n_max:
            return points, "s_limit"
        if current.index > 0 and np.linalg.norm(current.gradient) < GRADIENT_TOLERANCE:
            return points, "minimum"
        if current.index == 0:
            x_new = np.asarray(first_x, dtype=np.float64)
        elif method == "lqa":
            hessian = current.hessian
            if hessian is None:
                hessian = compute(current.index, current.x).hessian
            if hessian is None:
                raise ValueError("compute: LQA needs a Hessian at every point")
            sub = None if basis is None else basis(current.x)
            dx, at_minimum = lqa_step(current.gradient, hessian, ds, sub)
            if at_minimum:
                return points, "minimum"
            x_new = current.x + dx
        else:
            x_new = current.x + euler_step(current.gradient, ds)
        new = compute(current.index + 1, x_new)
        if not new.energy < current.energy:
            return points, "energy_rise"
        if on_point is not None:
            on_point(new)
        points.append(new)
        current = new


# Orientation


def orient(
    ts: Molecule,
    direction: npt.ArrayLike,
    reactants: Sequence[Molecule],
    products: Sequence[Molecule] | None = None,
) -> Literal[1, -1] | None:
    """Decide which side of the transition state leads to the products.

    A side is the reactant side when a small step along it brings the
    reactants' intramolecular atom distances closer to their values in the
    reactants (and, with products, the products' distances further from
    theirs). Atoms are matched by position: the transition state's atoms must
    have the same elements and masses, in the same order, as the reactants'
    atoms concatenated, and the products' likewise.

    Args:
        ts: The transition state geometry.
        direction: The transition vector in mass-scaled coordinates, (3N,).
        reactants: The optimized reactants.
        products: The optimized products, or ``None``.

    Returns:
        ``1`` when the step along ``+direction`` leads to the products,
        ``-1`` when the step along ``-direction`` does, and ``None`` when the
        atoms do not match. When both sides score the same (a symmetric
        reaction), ``1``.
    """
    groups = [tuple(reactants)]
    if products is not None:
        groups.append(tuple(products))
    if not all(_same_atoms(ts, group) for group in groups):
        return None
    step = ORIENT_STEP * from_mass_scaled(ts, direction)

    def score(coords: FloatArray) -> float:
        value = _distortion(coords, reactants)
        if products is not None:
            value -= _distortion(coords, products)
        return value

    plus, minus = score(ts.coords + step), score(ts.coords - step)
    if abs(plus - minus) <= _ORIENT_TIE:
        return 1
    return -1 if plus < minus else 1


def _same_atoms(ts: Molecule, group: tuple[Molecule, ...]) -> bool:
    symbols = tuple(s for m in group for s in m.symbols)
    if symbols != ts.symbols:
        return False
    masses = np.concatenate([m.masses for m in group])
    return bool(np.allclose(masses, ts.masses, rtol=0.0, atol=_MASS_MATCH))


def _distortion(coords: FloatArray, molecules: Sequence[Molecule]) -> float:
    """Sum of squared deviations of intramolecular distances from each molecule's."""
    total = 0.0
    offset = 0
    for mol in molecules:
        for i, j in combinations(range(mol.n_atoms), 2):
            d = np.linalg.norm(coords[offset + i] - coords[offset + j])
            ref = np.linalg.norm(mol.coords[i] - mol.coords[j])
            total += float(d - ref) ** 2
        offset += mol.n_atoms
    return total


# Results


def generalized_zpe(
    frequencies: npt.ArrayLike, imag_threshold: float = DEFAULT_IMAG_THRESHOLD
) -> float:
    """Zero-point energy ½ Σ h c nu over the real generalized frequencies, hartree.

    Imaginary frequencies (negative values) with a magnitude at or below
    ``imag_threshold`` (cm⁻¹) count with their magnitude; larger ones are
    excluded, as for stationary points.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    real = np.abs(f[f >= -imag_threshold])
    return float(0.5 * np.sum(real) * CM1_TO_HARTREE)


@dataclass(frozen=True, eq=False)
class ReactionPath:
    """A minimum-energy path through a transition state, in ascending s.

    Attributes:
        s: Signed arc length in mass-scaled coordinates (μ = 1 amu), bohr.
            s = 0 at the transition state, s < 0 on the reactant side.
        molecules: The geometry at each point.
        energies: Absolute energies in hartree.
        gradients: Cartesian gradients in hartree/bohr, shape (n, N, 3).
        frequencies: Generalized frequencies in cm⁻¹ at each point, or
            ``None`` for points without a Hessian.
        zpe: Generalized zero-point energy in hartree, NaN for points without
            a Hessian.
        reference_energy: Sum of the reactants' energies, hartree.
        reference_zpe_energy: Sum of the reactants' energies plus zero-point
            energies, hartree.
        ts_index: Index of the transition state.
        method: The stepping method, ``"lqa"`` or ``"euler"``.
        ds: The step size in bohr.
        stop_reactant: Why the reactant side (s < 0) stopped.
        stop_product: Why the product side (s > 0) stopped.
        oriented: Whether s < 0 is known to be the reactant side. When false,
            the sign of s is arbitrary.
        coordinates: The coordinates of the generalized frequencies,
            ``"redundant"`` (curvilinear, in redundant internal coordinates)
            or ``"cartesian"``.
        internals: The internal coordinates used, as ``(kind, *atom
            indices)`` tuples; empty for ``"cartesian"``.

    All arrays are read-only. Instances compare by value.
    """

    s: FloatArray
    molecules: tuple[Molecule, ...]
    energies: FloatArray
    gradients: FloatArray
    frequencies: tuple[FloatArray | None, ...]
    zpe: FloatArray
    reference_energy: float
    reference_zpe_energy: float
    ts_index: int
    method: Method
    ds: float
    stop_reactant: StopReason
    stop_product: StopReason
    oriented: bool
    coordinates: Coordinates = "cartesian"
    internals: tuple[Internal, ...] = ()

    def __post_init__(self) -> None:
        if self.coordinates not in COORDINATES:
            raise ValueError(
                "coordinates: expected 'redundant' or 'cartesian', got "
                f"{self.coordinates!r}"
            )
        n = len(self.molecules)
        s = _frozen(self.s)
        energies = _frozen(self.energies)
        zpe = _frozen(self.zpe)
        gradients = _frozen(self.gradients)
        freqs = tuple(None if f is None else _frozen(f) for f in self.frequencies)
        for name, size in (
            ("s", s.size),
            ("energies", energies.size),
            ("zpe", zpe.size),
            ("gradients", gradients.shape[0] if gradients.ndim else 0),
            ("frequencies", len(freqs)),
        ):
            if size != n:
                raise ValueError(f"{name}: expected {n} entries, got {size}")
        if not 0 <= self.ts_index < n:
            raise ValueError(f"ts_index: {self.ts_index} is out of range")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "molecules", tuple(self.molecules))
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "gradients", gradients)
        object.__setattr__(self, "frequencies", freqs)
        object.__setattr__(self, "zpe", zpe)
        object.__setattr__(self, "internals", tuple(map(tuple, self.internals)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReactionPath):
            return NotImplemented
        return (
            np.array_equal(self.s, other.s)
            and self.molecules == other.molecules
            and np.array_equal(self.energies, other.energies)
            and np.array_equal(self.gradients, other.gradients)
            and len(self.frequencies) == len(other.frequencies)
            and all(
                a is b if a is None or b is None else np.array_equal(a, b)
                for a, b in zip(self.frequencies, other.frequencies, strict=True)
            )
            and np.array_equal(self.zpe, other.zpe, equal_nan=True)
            and self.reference_energy == other.reference_energy
            and self.reference_zpe_energy == other.reference_zpe_energy
            and self.ts_index == other.ts_index
            and self.method == other.method
            and self.ds == other.ds
            and self.stop_reactant == other.stop_reactant
            and self.stop_product == other.stop_product
            and self.oriented == other.oriented
            and self.coordinates == other.coordinates
            and self.internals == other.internals
        )

    __hash__ = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.molecules)

    @property
    def v_mep(self) -> FloatArray:
        """V_MEP(s): energy relative to the reactants, hartree."""
        return _frozen(self.energies - self.reference_energy)

    @property
    def v_ag(self) -> FloatArray:
        """V_a^G(s): V_MEP plus generalized ZPE, relative to the reactants' ZPE level.

        In hartree; NaN for points without a Hessian.
        """
        return _frozen(self.energies + self.zpe - self.reference_zpe_energy)

    @property
    def has_frequencies(self) -> npt.NDArray[np.bool_]:
        """Whether each point has generalized frequencies."""
        mask = np.array([f is not None for f in self.frequencies], dtype=bool)
        mask.flags.writeable = False
        return mask

    def table(self) -> str:
        """Return a plain-text table: s, V_MEP, V_a^G and ZPE, one row per point."""
        k = HARTREE_TO_KCAL_PER_MOL
        lines = [
            f"{'s (bohr)':>9}  {'V_MEP (kcal/mol)':>16}  "
            f"{'V_a^G (kcal/mol)':>16}  {'ZPE (kcal/mol)':>14}"
        ]
        columns = (self.s, self.v_mep * k, self.v_ag * k, self.zpe * k)
        for row in zip(*columns, strict=True):
            s, v, va, z = (float(x) for x in row)
            lines.append(f"{s:>9.4f}  {v:>16.4f}  {va:>16.4f}  {z:>14.4f}")
        return "\n".join(lines)


def _frozen(values: npt.ArrayLike) -> FloatArray:
    a = np.array(values, dtype=np.float64)
    a.flags.writeable = False
    return a
