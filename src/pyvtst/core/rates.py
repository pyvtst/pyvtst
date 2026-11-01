"""Conventional and canonical variational transition state theory rate constants.

Canonical variational TST (CVT) places the dividing surface, at each
temperature, where the generalized TST rate along the minimum-energy path is
lowest: D. G. Truhlar and B. C. Garrett, Acc. Chem. Res. 13, 440 (1980).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import constants as sc

from pyvtst.core.mep import ReactionPath
from pyvtst.core.partition import (
    STANDARD_PRESSURE,
    ln_q_elec,
    ln_q_rot,
    ln_q_total,
    ln_q_trans_per_volume,
    ln_q_vib,
    temperatures,
)
from pyvtst.core.stationary import (
    DEFAULT_IMAG_THRESHOLD,
    StationaryPoint,
    check_character,
    species_label,
)
from pyvtst.core.units import BOLTZMANN_AU, CM1_TO_HARTREE, HARTREE_TO_KCAL_PER_MOL

FloatArray = npt.NDArray[np.float64]

UNIMOLECULAR_UNITS = "s^-1"
BIMOLECULAR_UNITS = "cm^3 molecule^-1 s^-1"

_LN_M3_TO_CM3 = float(np.log(1e6))


@dataclass(frozen=True, eq=False)
class TSTResult:
    """Conventional TST rate constants.

    Attributes:
        T: Temperatures in kelvin.
        k: Rate constants in ``units`` (tunneling included if ``kappa`` was
            given).
        ln_k: Natural log of ``k`` in ``units``.
        units: ``"s^-1"`` or ``"cm^3 molecule^-1 s^-1"``.
        barrier_zpe: ZPE-corrected barrier ΔE₀‡ in hartree.
    """

    T: FloatArray
    k: FloatArray
    ln_k: FloatArray
    units: str
    barrier_zpe: float


@dataclass(frozen=True, eq=False)
class CVTResult:
    """Canonical variational TST rate constants.

    Attributes:
        T: Temperatures in kelvin.
        k: CVT rate constants in ``units``, without tunneling.
        ln_k: Natural log of ``k`` in ``units``.
        units: ``"s^-1"`` or ``"cm^3 molecule^-1 s^-1"``.
        s_star: Location s*(T) of the variational transition state, bohr.
        s: s values of the path points used, ascending, bohr.
        ln_k_gt: ln of the generalized TST rate constants k^GT(T, s) in
            ``units``, shape (len(T), len(s)).
        delta_g: Standard-state generalized free energy of activation
            ΔG^GT,°(T, s) relative to the reactants at p° = 1 bar, hartree,
            shape (len(T), len(s)).
        delta_g_star: ΔG^GT,° at s*(T), hartree.

    All arrays are read-only.
    """

    T: FloatArray
    k: FloatArray
    ln_k: FloatArray
    units: str
    s_star: FloatArray
    s: FloatArray
    ln_k_gt: FloatArray
    delta_g: FloatArray
    delta_g_star: FloatArray

    @property
    def k_gt(self) -> FloatArray:
        """Generalized TST rate constants k^GT(T, s) in ``units``."""
        k = np.exp(self.ln_k_gt)
        k.flags.writeable = False
        return k


def zpe_energy(sp: StationaryPoint) -> float:
    """Electronic energy plus zero-point energy, in hartree."""
    return sp.energy + sp.zpe


def tst_rate(
    reactants: Sequence[StationaryPoint],
    ts: StationaryPoint,
    T: npt.ArrayLike,
    kappa: npt.ArrayLike | None = None,
) -> TSTResult:
    """Conventional TST rate constant k(T) = κ (k_B T/h) Q‡/ΠQ_R exp(-ΔE₀‡/k_B T).

    Partition functions are per unit volume, each referenced to its own
    zero-point level, with the transition state's imaginary mode excluded.
    Reaction-path degeneracy enters only through the symmetry numbers.

    Args:
        reactants: One or two reactant minima.
        ts: The transition state, a first-order saddle point.
        T: Temperatures in kelvin.
        kappa: Tunneling transmission coefficients, one per temperature
            (default 1).

    Returns:
        Rate constants in s⁻¹ for one reactant and cm³ molecule⁻¹ s⁻¹ for two.

    Raises:
        ValueError: Not one or two reactants, invalid temperatures or kappa.
        StationaryPointError: A reactant is not a minimum or the transition
            state is not a first-order saddle point.

    A warning is emitted when ΔE₀‡ ≤ 0; the rate constant is still returned.
    """
    reactants = tuple(reactants)
    if not 1 <= len(reactants) <= 2:
        raise ValueError(
            "conventional TST needs one or two reactants; at most two reactants "
            f"are supported, got {len(reactants)}"
        )
    t = temperatures(T)
    for i, sp in enumerate(reactants, start=1):
        check_character(sp, "minimum", species_label(sp.molecule, f"reactant {i}"))
    check_character(ts, "saddle", species_label(ts.molecule, "transition state"))

    ln_kappa = np.zeros_like(t)
    if kappa is not None:
        kap = np.array(kappa, dtype=np.float64, ndmin=1)
        if kap.shape != t.shape or not np.all(np.isfinite(kap) & (kap > 0)):
            raise ValueError(
                f"kappa: expected {t.size} finite positive values, got {kappa!r}"
            )
        ln_kappa = np.log(kap)

    barrier = zpe_energy(ts) - sum(zpe_energy(sp) for sp in reactants)
    if barrier <= 0:
        warnings.warn(
            f"the ZPE-corrected barrier is {barrier * HARTREE_TO_KCAL_PER_MOL:.2f} "
            "kcal/mol (<= 0); conventional TST is unreliable for this reaction",
            stacklevel=2,
        )

    ln_k = ln_kappa + _ln_rate_si(ln_q_total(ts, t), barrier, reactants, t)
    ln_k, units = _to_reported_units(ln_k, len(reactants))
    k = np.exp(ln_k)
    for a in (t, k, ln_k):
        a.flags.writeable = False
    return TSTResult(T=t, k=k, ln_k=ln_k, units=units, barrier_zpe=float(barrier))


def cvt_rate(
    reactants: Sequence[StationaryPoint],
    path: ReactionPath,
    T: npt.ArrayLike,
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
) -> CVTResult:
    """Canonical variational TST rate constant k^CVT(T) = min_s k^GT(T, s).

    k^GT(T, s) = (k_B T/h) Q^GT(T, s)/ΠQ_R exp(-V_a^G(s)/k_B T), with Q^GT the
    partition function per unit volume of the geometry at s over its
    generalized frequencies, referenced to its zero-point level, and V_a^G
    measured from the zero-point level of ``reactants`` (not the path's own
    reference), so the same path serves the forward and reverse reaction.
    The symmetry number is the transition state's at every s.

    A point is usable when it has generalized frequencies and none is below
    ``-imag_threshold`` cm⁻¹; smaller imaginary values count with their
    magnitude. Points with larger imaginary values are left out, with a
    warning.

    The minimum is taken over the usable points. When it lies at a point
    with a usable neighbour on each side, it is refined by the parabola in s through
    ln k^GT at those three points, if that parabola has its minimum between
    the neighbours. A minimum at the first or last usable point gives a
    warning: the variational transition state may lie beyond the path.

    Args:
        reactants: One or two reactant minima.
        path: The minimum-energy path.
        T: Temperatures in kelvin.
        imag_threshold: Imaginary-frequency threshold in cm⁻¹, normally the
            transition state's.

    Raises:
        ValueError: Not one or two reactants, invalid temperatures, or no
            usable path point.
        StationaryPointError: A reactant is not a minimum.
    """
    reactants, t = _check_reactants(reactants, T)
    n = len(reactants)
    s, ln_si = _generalized_si(reactants, path, t, imag_threshold)
    s_star = np.empty_like(t)
    ln_star = np.empty_like(t)
    at_end: dict[int, list[float]] = {0: [], s.size - 1: []}
    for j, row in enumerate(ln_si):
        i = int(np.argmin(row))
        s_star[j], ln_star[j] = s[i], row[i]
        if 0 < i < s.size - 1:
            vertex = _parabola_minimum(s[i - 1 : i + 2], row[i - 1 : i + 2])
            if vertex is not None:
                s_star[j], ln_star[j] = vertex
        for end, temps in at_end.items():
            if i == end:
                temps.append(float(t[j]))
    for end, temps in at_end.items():
        if temps:
            _warn_at_end(path, s, end, temps)

    delta_g = _delta_g(ln_si, t[:, None], n)
    delta_g_star = _delta_g(ln_star, t, n)
    ln_gt, units = _to_reported_units(ln_si, n)
    ln_k, _ = _to_reported_units(ln_star, n)
    k = np.exp(ln_k)
    for a in (t, k, ln_k, s_star, s, ln_gt, delta_g, delta_g_star):
        a.flags.writeable = False
    return CVTResult(
        T=t,
        k=k,
        ln_k=ln_k,
        units=units,
        s_star=s_star,
        s=s,
        ln_k_gt=ln_gt,
        delta_g=delta_g,
        delta_g_star=delta_g_star,
    )


def _check_reactants(
    reactants: Sequence[StationaryPoint], T: npt.ArrayLike
) -> tuple[tuple[StationaryPoint, ...], FloatArray]:
    reactants = tuple(reactants)
    if not 1 <= len(reactants) <= 2:
        raise ValueError(
            "variational TST needs one or two reactants; at most two reactants "
            f"are supported, got {len(reactants)}"
        )
    t = temperatures(T)
    for i, sp in enumerate(reactants, start=1):
        check_character(sp, "minimum", species_label(sp.molecule, f"reactant {i}"))
    return reactants, t


def _generalized_si(
    reactants: tuple[StationaryPoint, ...],
    path: ReactionPath,
    t: FloatArray,
    imag_threshold: float,
) -> tuple[FloatArray, FloatArray]:
    """s and ln k^GT in molecule-based SI units at the usable points."""
    reference = sum(zpe_energy(sp) for sp in reactants)
    s: list[float] = []
    columns: list[FloatArray] = []
    excluded: list[str] = []
    for i, freqs in enumerate(path.frequencies):
        if freqs is None:
            continue
        large = freqs[freqs < -imag_threshold]
        if large.size:
            listed = ", ".join(f"{f:.1f}" for f in large)
            excluded.append(f"s = {path.s[i]:+.4f} ([{listed}] cm^-1)")
            continue
        mol = path.molecules[i]
        nu = np.sort(np.abs(freqs))
        # Summed as in StationaryPoint.zpe and ln_q_total, so the transition
        # state gives exactly the conventional TST rate.
        zpe = float(0.5 * np.sum(nu) * CM1_TO_HARTREE)
        ln_q = (
            ln_q_trans_per_volume(mol, t)
            + ln_q_rot(mol, t)
            + ln_q_vib(nu, t)
            + ln_q_elec(mol, t)
        )
        barrier = (float(path.energies[i]) + zpe) - reference
        s.append(float(path.s[i]))
        columns.append(_ln_rate_si(ln_q, barrier, reactants, t))
    if excluded:
        warnings.warn(
            "CVT: path points with imaginary generalized frequencies above the "
            f"{imag_threshold:g} cm^-1 threshold are left out of the variational "
            f"search: {'; '.join(excluded)}",
            stacklevel=3,
        )
    if not s:
        raise ValueError(
            "path: no point has usable generalized frequencies; CVT needs at least one"
        )
    return np.array(s), np.stack(columns, axis=1)


def _parabola_minimum(x: FloatArray, y: FloatArray) -> tuple[float, float] | None:
    """Vertex of the parabola through three points, if it is a minimum in [x0, x2]."""
    f01 = (y[1] - y[0]) / (x[1] - x[0])
    f12 = (y[2] - y[1]) / (x[2] - x[1])
    a = (f12 - f01) / (x[2] - x[0])
    if not a > 0:
        return None
    xv = 0.5 * (x[0] + x[1]) - f01 / (2.0 * a)
    if not x[0] <= xv <= x[2]:
        return None
    yv = y[0] + f01 * (xv - x[0]) + a * (xv - x[0]) * (xv - x[1])
    return float(xv), float(yv)


def _warn_at_end(
    path: ReactionPath, s: FloatArray, end: int, temps: list[float]
) -> None:
    first = end == 0
    if path.oriented:
        side = "reactant side" if first else "product side"
    else:
        side = "s < 0 end" if first else "s > 0 end"
    listed = ", ".join(f"{v:g}" for v in temps)
    warnings.warn(
        f"CVT: at T = {listed} K the generalized rate is lowest at the "
        f"{'first' if first else 'last'} usable path point, on the {side} "
        f"(s = {s[end]:+.4f} bohr); the variational transition state may lie "
        "beyond the path, widen its s range",
        stacklevel=3,
    )


def _delta_g(ln_k_si: FloatArray, t: FloatArray, n_reactants: int) -> FloatArray:
    """ΔG^GT,° in hartree from ln k^GT in molecule-based SI units, p° = 1 bar."""
    ln_standard = (n_reactants - 1) * np.log(sc.k * t / STANDARD_PRESSURE)
    return np.asarray(
        -BOLTZMANN_AU * t * (ln_k_si - np.log(sc.k * t / sc.h) - ln_standard)
    )


def _ln_rate_si(
    ln_q: FloatArray,
    barrier: float,
    reactants: Sequence[StationaryPoint],
    t: FloatArray,
) -> FloatArray:
    """ln[(k_B T/h) Q/ΠQ_R exp(-barrier/k_B T)] in molecule-based SI units.

    ``ln_q`` is the dividing surface's total partition function per unit
    volume, referenced to its zero-point level, and ``barrier`` its
    ZPE-corrected energy above the reactants' zero-point level, hartree.
    """
    return np.asarray(
        np.log(sc.k * t / sc.h)
        + ln_q
        - sum(ln_q_total(sp, t) for sp in reactants)
        - barrier / (BOLTZMANN_AU * t)
    )


def _to_reported_units(ln_k_si: FloatArray, n_reactants: int) -> tuple[FloatArray, str]:
    """Convert ln k from SI to s⁻¹ or cm³ molecule⁻¹ s⁻¹; return it and the unit."""
    if n_reactants == 2:
        return ln_k_si + _LN_M3_TO_CM3, BIMOLECULAR_UNITS
    return ln_k_si, UNIMOLECULAR_UNITS
