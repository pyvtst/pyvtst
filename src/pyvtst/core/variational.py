"""Improved canonical (ICVT) and microcanonical (µVT) variational TST.

Both theories leave out the generalized-transition-state states below the
threshold energy E_th, the maximum V^AG of the vibrationally adiabatic
ground-state curve (or a higher asymptote). ICVT keeps the canonical
variational search; µVT minimizes the sum of states at each energy and then
averages over a Boltzmann distribution:
B. C. Garrett and D. G. Truhlar, J. Phys. Chem. 83, 1079 (1979);
B. C. Garrett, D. G. Truhlar, R. S. Grev and A. W. Magnuson, J. Phys. Chem.
84, 1730 (1980).

Each path point's sum of states is counted on its own energy grid and
normalized by its own generalized TST rate, so that k^IGT ≤ k^GT and
k^µVT ≤ k^IGT hold exactly at every usable point.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.integrate

from pyvtst.core.mep import ReactionPath
from pyvtst.core.rates import (
    _check_reactants,
    _generalized_si,
    _to_reported_units,
    _variational_minimum,
    path_end_name,
    point_zpe,
    usable_points,
    zpe_energy,
)
from pyvtst.core.states import DEFAULT_GRAIN, sum_of_states
from pyvtst.core.stationary import DEFAULT_IMAG_THRESHOLD, StationaryPoint
from pyvtst.core.units import BOLTZMANN_AU, HARTREE_TO_KCAL_PER_MOL
from pyvtst.core.vib import is_linear

FloatArray = npt.NDArray[np.float64]

# Each grid reaches this many k_B T_max beyond the number of degrees of
# freedom (in units of k_B T) above the threshold.
_SPAN_KT = 40.0
# Extra grains on each point's grid, so the shifted read-out stays inside it.
_MARGIN = 2
# Energies per block in the µVT minimum.
_CHUNK = 8192
# Share of k^µVT from energies whose bottleneck is at a path end that warns.
_END_SHARE = 0.01


@dataclass(frozen=True, eq=False)
class ICVTResult:
    """Improved canonical variational TST rate constants.

    Attributes:
        T: Temperatures in kelvin.
        k: ICVT rate constants in ``units``, without tunneling.
        ln_k: Natural log of ``k`` in ``units``.
        units: ``"s^-1"`` or ``"cm^3 molecule^-1 s^-1"``.
        s_star: Location s*(T) of the variational transition state, bohr.
        s: s values of the path points used, ascending, bohr.
        ln_k_igt: ln k^IGT(T, s) in ``units``, shape (len(T), len(s)).
        f: Fraction F(T, s) of each point's states at or above ``e_th``,
            shape (len(T), len(s)).
        e_th: Threshold energy from the reactants' zero-point level, hartree.
        v_max: Maximum V^AG of V_a^G over the path points used, hartree.
        s_max: s of that maximum, bohr.

    All arrays are read-only.
    """

    T: FloatArray
    k: FloatArray
    ln_k: FloatArray
    units: str
    s_star: FloatArray
    s: FloatArray
    ln_k_igt: FloatArray
    f: FloatArray
    e_th: float
    v_max: float
    s_max: float

    @property
    def k_igt(self) -> FloatArray:
        """Improved generalized TST rate constants k^IGT(T, s) in ``units``."""
        k = np.exp(self.ln_k_igt)
        k.flags.writeable = False
        return k


@dataclass(frozen=True, eq=False)
class MuVTResult:
    """Microcanonical variational TST rate constants.

    Attributes:
        T: Temperatures in kelvin.
        k: µVT rate constants in ``units``, without tunneling.
        ln_k: Natural log of ``k`` in ``units``.
        units: ``"s^-1"`` or ``"cm^3 molecule^-1 s^-1"``.
        energies: Total energies E of the grid, from ``e_th`` upwards,
            measured from the reactants' zero-point level, hartree.
        s_star: s*(E), where the sum of states N^GT(E - V_a^G(s), s) is
            lowest, bohr.
        n_min: That lowest sum of states at each energy.
        e_th: Threshold energy from the reactants' zero-point level, hartree.
        v_max: Maximum V^AG of V_a^G over the path points used, hartree.
        s_max: s of that maximum, bohr.

    All arrays are read-only.
    """

    T: FloatArray
    k: FloatArray
    ln_k: FloatArray
    units: str
    energies: FloatArray
    s_star: FloatArray
    n_min: FloatArray
    e_th: float
    v_max: float
    s_max: float


def threshold_energy(
    reactants: Sequence[StationaryPoint],
    path: ReactionPath,
    products: Sequence[StationaryPoint] | None = None,
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
) -> tuple[float, float, float]:
    """Threshold energy E_th, V^AG and the s of V^AG.

    E_th is the highest of V^AG (the maximum of V_a^G over the usable path
    points), the reactants' zero-point level (0) and, when given, the
    products' zero-point level. Energies are in hartree from the reactants'
    zero-point level. A maximum at the first or last usable point gives a
    warning that the barrier top is not inside the path.

    Raises:
        ValueError: No usable path point.
    """
    usable = usable_points(path, imag_threshold, warn=("VTST", "the threshold"))
    if not usable:
        raise ValueError("path: no point has usable generalized frequencies")
    s = np.array(path.s[usable])
    v = _v_ag(tuple(reactants), path, usable)
    return _threshold(tuple(reactants), products, s, v, path, "VTST")


def icvt_rate(
    reactants: Sequence[StationaryPoint],
    path: ReactionPath,
    T: npt.ArrayLike,
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
    *,
    products: Sequence[StationaryPoint] | None = None,
) -> ICVTResult:
    """ICVT rate constant k^ICVT(T) = min_s k^GT(T, s) · F(T, s).

    F(T, s) is the Boltzmann-weighted fraction of the path point's
    rovibrational states, counted from its own zero-point level V_a^G(s), with
    total energy at or above the threshold E_th (see :func:`threshold_energy`).
    Numerator and normalization come from the same discrete sum of states, so
    0 ≤ F ≤ 1 and F = 1 where V_a^G(s) = E_th. The usable points, the parabola
    refinement of the minimum and the end-of-path warning follow
    :func:`~pyvtst.core.rates.cvt_rate`.

    Args:
        reactants: One or two reactant minima.
        path: The minimum-energy path.
        T: Temperatures in kelvin.
        imag_threshold: Imaginary-frequency threshold in cm⁻¹, normally the
            transition state's.
        products: The product minima, or ``None``; their zero-point level
            bounds E_th from below.

    Raises:
        ValueError: Not one or two reactants, invalid temperatures, or no
            usable path point.
        StationaryPointError: A reactant is not a minimum.
    """
    reactants, t = _check_reactants(reactants, T)
    n = len(reactants)
    s, ln_gt, usable = _generalized_si(reactants, path, t, imag_threshold, "ICVT")
    v = _v_ag(reactants, path, usable)
    e_th, v_max, s_max = _threshold(reactants, products, s, v, path, "ICVT")
    beta = 1.0 / (BOLTZMANN_AU * t)
    span = _span(path, usable, t)

    ln_f = np.empty_like(ln_gt)
    for j, i in enumerate(usable):
        delta = max(0.0, e_th - float(v[j]))
        states = _point_states(path, i, delta + span)
        ln_f[:, j] = _ln_tails(states, beta, delta) - _ln_tails(states, beta, 0.0)
    ln_f = np.minimum(ln_f, 0.0)
    ln_igt_si = ln_gt + ln_f
    s_star, ln_star = _variational_minimum(s, ln_igt_si, t, path, "ICVT")

    ln_igt, units = _to_reported_units(ln_igt_si, n)
    ln_k, _ = _to_reported_units(ln_star, n)
    k = np.exp(ln_k)
    f = np.exp(ln_f)
    for a in (t, k, ln_k, s_star, s, ln_igt, f):
        a.flags.writeable = False
    return ICVTResult(
        T=t,
        k=k,
        ln_k=ln_k,
        units=units,
        s_star=s_star,
        s=s,
        ln_k_igt=ln_igt,
        f=f,
        e_th=e_th,
        v_max=v_max,
        s_max=s_max,
    )


def muvt_rate(
    reactants: Sequence[StationaryPoint],
    path: ReactionPath,
    T: npt.ArrayLike,
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
    *,
    products: Sequence[StationaryPoint] | None = None,
) -> MuVTResult:
    """µVT rate constant k^µVT(T) = ∫_{E_th} min_s [w N^GT(E - V_a^G(s), s)] e^{-βE} dE.

    w(T, s) = k^GT(T, s) / ∫_{V_a^G(s)} N^GT(E - V_a^G(s), s) e^{-βE} dE
    normalizes each point's discrete sum of states to its generalized TST
    rate; for an exact count it is the same at every s. At each energy the
    minimum over the usable points is refined by the parabola in s through
    the lowest point and its neighbours, as in CVT. When more than 1 % of
    k^µVT at some temperature comes from energies whose bottleneck is the
    first or last usable point, a warning says the path is too short.

    Args and raises as :func:`icvt_rate`.
    """
    reactants, t = _check_reactants(reactants, T)
    n = len(reactants)
    s, ln_gt, usable = _generalized_si(reactants, path, t, imag_threshold, "muVT")
    v = _v_ag(reactants, path, usable)
    e_th, v_max, s_max = _threshold(reactants, products, s, v, path, "muVT")
    beta = 1.0 / (BOLTZMANN_AU * t)
    span = _span(path, usable, t)
    grain = DEFAULT_GRAIN
    n_e = math.ceil(span / grain - 1e-9) + 1
    energies = e_th + grain * np.arange(n_e, dtype=np.float64)

    shifted: list[tuple[FloatArray, int, float]] = []
    ln_w = np.empty_like(ln_gt)
    for j, i in enumerate(usable):
        delta = max(0.0, e_th - float(v[j]))
        states = _point_states(path, i, delta + span + _MARGIN * grain)
        start = math.floor(delta / grain)
        shifted.append((states, start, delta / grain - start))
        # ∫_{V} N(E - V) e^{-βE} dE = e^{-βV} · tail(0).
        ln_w[:, j] = ln_gt[:, j] - (_ln_tails(states, beta, 0.0) - beta * v[j])

    ns = s.size
    b = np.empty((t.size, n_e))  # ln of the integrand + β E_th
    at_end = np.zeros((t.size, n_e), dtype=np.int8)  # 1 first, 2 last point
    s_star = np.empty(n_e)
    ln_n_min = np.empty(n_e)
    for lo in range(0, n_e, _CHUNK):
        hi = min(n_e, lo + _CHUNK)
        ln_n = np.empty((ns, hi - lo))
        with np.errstate(divide="ignore"):
            for j, (states, start, frac) in enumerate(shifted):
                a = states[start + lo : start + hi]
                c = states[start + lo + 1 : start + hi + 1]
                ln_n[j] = np.log(a + frac * (c - a))
        s_star[lo:hi], ln_n_min[lo:hi], _ = _refined_minimum(s, ln_n)
        e_rel = energies[lo:hi] - e_th
        for k_t in range(t.size):
            _, low, idx = _refined_minimum(s, ln_w[k_t, :, None] + ln_n)
            b[k_t, lo:hi] = low - beta[k_t] * e_rel
            at_end[k_t, lo:hi] = np.where(idx == 0, 1, np.where(idx == ns - 1, 2, 0))

    weights = np.full(n_e, grain)
    weights[[0, -1]] *= 0.5
    finite = np.where(np.isfinite(b), b, -np.inf)
    top = np.max(finite, axis=1, keepdims=True)
    terms = weights * np.exp(finite - top)
    total = terms.sum(axis=1)
    ln_si = np.log(total) + top[:, 0] - beta * e_th
    if ns > 1:
        _warn_muvt_ends(path, s, t, energies - e_th, terms, total, at_end)

    ln_k, units = _to_reported_units(ln_si, n)
    k = np.exp(ln_k)
    n_min = np.exp(ln_n_min)
    for arr in (t, k, ln_k, energies, s_star, n_min):
        arr.flags.writeable = False
    return MuVTResult(
        T=t,
        k=k,
        ln_k=ln_k,
        units=units,
        energies=energies,
        s_star=s_star,
        n_min=n_min,
        e_th=e_th,
        v_max=v_max,
        s_max=s_max,
    )


def _v_ag(
    reactants: tuple[StationaryPoint, ...], path: ReactionPath, usable: list[int]
) -> FloatArray:
    """V_a^G at the usable points, from the reactants' zero-point level."""
    reference = sum(zpe_energy(sp) for sp in reactants)
    return np.array(
        [
            float(path.energies[i])
            + point_zpe(cast(FloatArray, path.frequencies[i]))
            - reference
            for i in usable
        ]
    )


def _threshold(
    reactants: tuple[StationaryPoint, ...],
    products: Sequence[StationaryPoint] | None,
    s: FloatArray,
    v: FloatArray,
    path: ReactionPath,
    label: str,
) -> tuple[float, float, float]:
    top = int(np.argmax(v))
    v_max = float(v[top])
    e_th = max(v_max, 0.0)
    if products is not None:
        reference = sum(zpe_energy(sp) for sp in reactants)
        e_th = max(e_th, sum(zpe_energy(sp) for sp in products) - reference)
    if s.size > 1 and top in (0, s.size - 1):
        warnings.warn(
            f"{label}: the maximum of V_a^G lies at the "
            f"{path_end_name(path, top == 0)} of the path (s = {s[top]:+.4f} "
            "bohr); the barrier top is not inside the path, so the threshold "
            "energy may be too low, widen its s range",
            stacklevel=3,
        )
    return e_th, v_max, float(s[top])


def _span(path: ReactionPath, usable: list[int], t: FloatArray) -> float:
    """Energy range above the threshold that the grids cover, hartree."""
    dof = 0.0
    for i in usable:
        freqs = cast(FloatArray, path.frequencies[i])
        half = 1.0 if is_linear(path.molecules[i]) else 1.5
        dof = max(dof, half + freqs.size)
    return (dof + _SPAN_KT) * BOLTZMANN_AU * float(np.max(t))


def _point_states(path: ReactionPath, i: int, e_max: float) -> FloatArray:
    freqs = cast(FloatArray, path.frequencies[i])
    return sum_of_states(path.molecules[i], freqs, e_max, DEFAULT_GRAIN)


def _ln_tails(states: FloatArray, beta: FloatArray, x: float) -> FloatArray:
    """ln ∫_x^{ε_max} N(ε) e^{-βε} dε for each β, N on the default grid.

    N(ε) e^{-βε} is taken linear between grid points (the trapezoid rule), so
    x = 0 gives exactly the trapezoid integral of the whole grid and a larger
    x never gives more.
    """
    grain = DEFAULT_GRAIN
    k = min(int(x // grain), states.size - 2)
    eps = grain * np.arange(k, states.size, dtype=np.float64)
    g = states[k:] * np.exp(-beta[:, None] * (eps - x))
    frac = (x - eps[0]) / grain
    gx = g[:, 0] + frac * (g[:, 1] - g[:, 0])
    head = 0.5 * (gx + g[:, 1]) * (eps[1] - x)
    body = scipy.integrate.trapezoid(g[:, 1:], dx=grain, axis=1)
    return np.asarray(np.log(head + body) - beta * x)


def _refined_minimum(
    s: FloatArray, values: FloatArray
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.intp]]:
    """Minimum over s (axis 0) of each column, refined as in CVT.

    Returns s* and the minimum per column, and the index of the lowest point.
    A column whose lowest point has a neighbour on each side is refined by the
    parabola through the three values when all are finite and it has a
    minimum between the neighbours.
    """
    idx = np.argmin(values, axis=0)
    cols = np.arange(values.shape[1])
    s_star = s[idx].astype(np.float64)
    low = values[idx, cols].astype(np.float64)
    inner = (idx > 0) & (idx < s.size - 1)
    if not np.any(inner):
        return s_star, low, idx
    i, c = idx[inner], cols[inner]
    x0, x1, x2 = s[i - 1], s[i], s[i + 1]
    y0, y1, y2 = values[i - 1, c], values[i, c], values[i + 1, c]
    with np.errstate(invalid="ignore", divide="ignore"):
        f01 = (y1 - y0) / (x1 - x0)
        f12 = (y2 - y1) / (x2 - x1)
        a = (f12 - f01) / (x2 - x0)
        xv = 0.5 * (x0 + x1) - f01 / (2.0 * a)
        yv = y0 + f01 * (xv - x0) + a * (xv - x0) * (xv - x1)
    ok = (
        np.isfinite(y0)
        & np.isfinite(y1)
        & np.isfinite(y2)
        & (a > 0)
        & (x0 <= xv)
        & (xv <= x2)
    )
    target = np.flatnonzero(inner)[ok]
    s_star[target] = xv[ok]
    low[target] = yv[ok]
    return s_star, low, idx


def _warn_muvt_ends(
    path: ReactionPath,
    s: FloatArray,
    t: FloatArray,
    above: FloatArray,
    terms: FloatArray,
    total: FloatArray,
    at_end: npt.NDArray[np.int8],
) -> None:
    """Warn per path end about temperatures with > 1 % of k^µVT from that end."""
    for first in (True, False):
        end = 0 if first else s.size - 1
        temps: list[float] = []
        lo, hi = math.inf, -math.inf
        for k_t in range(t.size):
            mask = at_end[k_t] == (1 if first else 2)
            share = terms[k_t, mask].sum() / total[k_t]
            if share > _END_SHARE:
                temps.append(float(t[k_t]))
                relevant = mask & (terms[k_t] > 1e-6 * terms[k_t].max())
                lo = min(lo, float(above[relevant].min()))
                hi = max(hi, float(above[relevant].max()))
        if temps:
            listed = ", ".join(f"{v:g}" for v in temps)
            warnings.warn(
                f"muVT: at T = {listed} K more than 1% of the rate comes from "
                "energies whose bottleneck is the "
                f"{'first' if first else 'last'} usable path point, on the "
                f"{path_end_name(path, first)} (s = {s[end]:+.4f} bohr), from "
                f"{lo * HARTREE_TO_KCAL_PER_MOL:.2f} to "
                f"{hi * HARTREE_TO_KCAL_PER_MOL:.2f} kcal/mol above the "
                "threshold; the variational transition state may lie beyond "
                "the path, widen its s range",
                stacklevel=4,
            )
