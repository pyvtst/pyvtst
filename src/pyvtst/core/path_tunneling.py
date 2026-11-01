"""Multidimensional tunneling along a minimum-energy path: ZCT and SCT.

The ground-state transmission probability through the vibrationally adiabatic
curve V_a^G(s) is the uniform semiclassical P(E) = 1/(1 + exp 2θ(E)), with the
imaginary action θ(E) = (1/ħ) ∫ √(2 μ_eff(s) (V_a^G(s) - E)) ds over the
region where V_a^G(s) > E. Zero-curvature tunneling (ZCT) uses the scaling mass
μ = 1 amu of the path's coordinates; small-curvature tunneling (SCT) uses the
centrifugal-dominant small-curvature effective mass μ_eff(s) ≤ μ, which lets
the tunneling path cut the corner of a curved reaction path.

References:
    A. Fernández-Ramos, B. A. Ellingson, B. C. Garrett and D. G. Truhlar,
    Rev. Comput. Chem. 23, 125 (2007).
    Y.-P. Liu, G. C. Lynch, T. N. Truong, D.-h. Lu, D. G. Truhlar and
    B. C. Garrett, J. Am. Chem. Soc. 115, 2408 (1993).
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.interpolate

from pyvtst.core.mep import ReactionPath
from pyvtst.core.partition import temperatures
from pyvtst.core.rates import path_end_name, point_zpe, usable_points, zpe_energy
from pyvtst.core.stationary import DEFAULT_IMAG_THRESHOLD, StationaryPoint
from pyvtst.core.units import AMU_TO_ME, BOLTZMANN_AU, CM1_TO_HARTREE
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL as _KCAL

FloatArray = npt.NDArray[np.float64]
Method = Literal["zct", "sct"]

METHODS: tuple[Method, ...] = ("zct", "sct")

# The scaling mass of the path's coordinates, in electron masses (ħ = 1).
_MU = AMU_TO_ME
# θ(E) is computed at this many Chebyshev points between E₀ and V^AG.
_NODES = 64
# The Boltzmann average is integrated up to this many kT above the barrier
# region; the remaining tail, with P = 1, is added analytically.
_GRID_POINTS = 2001
_EPSREL = 1e-10


@dataclass(frozen=True, eq=False)
class PathTunnelingResult:
    """ZCT or SCT transmission coefficients along a path.

    Energies are in hartree, measured from the reactants' zero-point level.

    Attributes:
        method: ``"zct"`` or ``"sct"``.
        T: Temperatures in kelvin.
        kappa: Transmission coefficients κ(T), including ``kappa_cag``.
        kappa_cag: Classical adiabatic ground-state factor
            exp(-[V^AG - V_a^G(s_ref)]/k_B T), which refers κ to the dividing
            surface of the rate theory it multiplies.
        e0: Lowest tunneling energy E₀.
        v_max: Maximum V^AG of the adiabatic curve.
        s_max: s of that maximum, bohr.
        s: s of the usable path points, bohr.
        v_ag: V_a^G at those points.
        mu_eff: μ_eff/μ at those points (all 1 for ZCT).
        energies: Energies at which P(E) was evaluated, ascending.
        probability: P(E) at ``energies``.

    All arrays are read-only.
    """

    method: str
    T: FloatArray
    kappa: FloatArray
    kappa_cag: FloatArray
    e0: float
    v_max: float
    s_max: float
    s: FloatArray
    v_ag: FloatArray
    mu_eff: FloatArray
    energies: FloatArray
    probability: FloatArray


def path_tunneling(
    reactants: Sequence[StationaryPoint],
    path: ReactionPath,
    T: npt.ArrayLike,
    method: Method | str = "zct",
    *,
    products: Sequence[StationaryPoint] | None = None,
    s_ref: npt.ArrayLike = 0.0,
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
    warn_excluded: bool = True,
) -> PathTunnelingResult:
    """ZCT or SCT transmission coefficient κ(T) along a minimum-energy path.

    V_a^G(s) = E(s) + ZPE^GT(s) - Σ(E_R + ZPE_R) at the path's usable points
    (as for CVT), interpolated by a not-a-knot cubic spline. With V^AG its
    maximum and E₀ the lowest tunneling energy, P(E) = 1/(1 + exp 2θ(E)) for
    E₀ ≤ E ≤ V^AG, 1 - P(2V^AG - E) up to 2V^AG - E₀ and 1 above, and
    κ(T) = κ^CAG(T) · β exp(β V^AG) ∫ P(E) exp(-βE) dE from E₀.

    E₀ is the higher of the reactants' and the products' zero-point levels.
    When V_a^G at an end of the path lies above it, the path is too short:
    E₀ is raised to the higher end, with a warning, and κ is underestimated at
    low temperature.

    Args:
        reactants: One or two reactant minima; energies are measured from
            their zero-point level.
        path: The minimum-energy path. SCT needs its curvature, from
            :meth:`pyvtst.Reaction.compute_mep`.
        T: Temperatures in kelvin.
        method: ``"zct"`` or ``"sct"``.
        products: The product minima, or ``None``.
        s_ref: s of the dividing surface κ refers to, bohr: 0 for
            conventional TST, s*(T) for CVT (one value per temperature).
        imag_threshold: Imaginary-frequency threshold in cm⁻¹, normally the
            transition state's.
        warn_excluded: Warn about points left out for an imaginary
            frequency. :meth:`pyvtst.Reaction.rates` turns it off when CVT has
            already warned.

    Swapping ``reactants`` and ``products`` gives the same κ, so forward and
    reverse rate constants keep detailed balance.

    Raises:
        ValueError: Unknown ``method``, invalid temperatures, fewer than three
            usable points, ``s_ref`` outside the usable points, V^AG not above
            E₀, or SCT on a path without curvature.
    """
    if method not in METHODS:
        raise ValueError(f"method: expected 'zct' or 'sct', got {method!r}")
    t = temperatures(T)
    ref = np.broadcast_to(np.asarray(s_ref, dtype=np.float64), t.shape)
    reference = sum(zpe_energy(sp) for sp in reactants)
    usable = usable_points(
        path,
        imag_threshold,
        warn=("tunneling", "the adiabatic profile") if warn_excluded else None,
        stacklevel=3,
    )
    if len(usable) < 3:
        raise ValueError(
            "path: tunneling needs at least 3 points with usable generalized "
            f"frequencies, got {len(usable)}"
        )
    s = np.array(path.s[usable])
    v = np.array(
        [
            float(path.energies[i])
            + point_zpe(cast(FloatArray, path.frequencies[i]))
            - reference
            for i in usable
        ]
    )
    if np.any(ref < s[0]) or np.any(ref > s[-1]):
        raise ValueError(
            f"s_ref: must lie within the usable points, [{s[0]:+.4f}, "
            f"{s[-1]:+.4f}] bohr"
        )
    mass = _ZeroCurvature() if method == "zct" else _SCTMass.from_path(path, usable)
    profile = _Profile(s, v)

    e_asym = 0.0
    if products is not None:
        e_asym = max(e_asym, sum(zpe_energy(sp) for sp in products) - reference)
    e0 = max(e_asym, float(v[0]), float(v[-1]))
    if e0 > e_asym:
        first = v[0] >= v[-1]
        warnings.warn(
            f"tunneling: the path ends at V_a^G = {e0 * _KCAL:.2f} kcal/mol on "
            f"the {path_end_name(path, first)} (s = {s[0 if first else -1]:+.4f} "
            f"bohr), {(e0 - e_asym) * _KCAL:.2f} kcal/mol above the lowest "
            "zero-point level; E0 is raised to it, so kappa is underestimated "
            "at low temperature. Pass a path with a wider s range via path=",
            stacklevel=3,
        )
    if profile.top_index in (0, s.size - 1):
        warnings.warn(
            "tunneling: the maximum of V_a^G lies at the "
            f"{path_end_name(path, profile.top_index == 0)} of the path "
            f"(s = {profile.s_max:+.4f} bohr); the barrier top is not inside the "
            "path, widen its s range",
            stacklevel=3,
        )
    if not profile.v_max > e0:
        raise ValueError(
            f"tunneling: the adiabatic barrier V^AG = {profile.v_max * _KCAL:.3f} "
            f"kcal/mol is not above the lowest tunneling energy E0 = "
            f"{e0 * _KCAL:.3f} kcal/mol"
        )

    theta = _ThetaTable(profile, mass, e0)
    kappa_raw = np.array([theta.boltzmann_average(BOLTZMANN_AU * ti) for ti in t])
    kappa_cag = np.exp(-(profile.v_max - profile.at(ref)) / (BOLTZMANN_AU * t))
    energies, probability = theta.probabilities()
    mu_eff = np.asarray(mass(s), dtype=np.float64)
    arrays = (t, kappa_raw * kappa_cag, kappa_cag, s, v, mu_eff, energies, probability)
    for a in arrays:
        a.flags.writeable = False
    return PathTunnelingResult(
        method=method,
        T=t,
        kappa=arrays[1],
        kappa_cag=kappa_cag,
        e0=e0,
        v_max=profile.v_max,
        s_max=profile.s_max,
        s=s,
        v_ag=v,
        mu_eff=mu_eff,
        energies=energies,
        probability=probability,
    )


def sct_effective_mass(
    s: npt.ArrayLike,
    frequencies: Sequence[npt.ArrayLike],
    curvature: Sequence[npt.ArrayLike],
    at: npt.ArrayLike,
) -> FloatArray:
    """Centrifugal-dominant small-curvature effective mass μ_eff/μ.

    μ_eff/μ = min{exp[-2ā - ā² + (dt̄/ds)²], 1}, with ā = |κ t̄|, κ the total
    curvature, t̄ = [Σ (κ_m/κ)² t_m⁻⁴]^(-1/4) and t_m = (ħ/μ ω_m)^(1/2) the
    ground-state turning point of mode m (frequency magnitudes are used).
    ā and t̄ are not-a-knot cubic splines through the given points, so the
    result is defined between and beyond them.

    Args:
        s: s of the points with curvature, ascending, bohr.
        frequencies: Generalized frequencies in cm⁻¹ at each point.
        curvature: Curvature components κ_m in bohr⁻¹ at each point, aligned
            with ``frequencies``.
        at: Where to evaluate μ_eff/μ, bohr.

    Raises:
        ValueError: Fewer than two points, or mismatched lengths.
    """
    mass = _SCTMass(
        np.asarray(s, dtype=np.float64),
        [np.asarray(f, dtype=np.float64) for f in frequencies],
        [np.asarray(c, dtype=np.float64) for c in curvature],
    )
    return mass(np.asarray(at, dtype=np.float64))


class _ZeroCurvature:
    def __call__(self, s: FloatArray) -> FloatArray:
        return np.ones_like(np.asarray(s, dtype=np.float64))


class _SCTMass:
    """μ_eff/μ from splines of ā(s) and t̄(s)."""

    def __init__(
        self, s: FloatArray, frequencies: list[FloatArray], curvature: list[FloatArray]
    ) -> None:
        if not len(frequencies) == len(curvature) == s.size:
            raise ValueError(
                "s, frequencies and curvature: expected one entry per point, got "
                f"{s.size}, {len(frequencies)} and {len(curvature)}"
            )
        if s.size < 2:
            raise ValueError(
                "SCT needs the path curvature at two or more usable points, got "
                f"{s.size}"
            )
        a_bar = np.zeros(s.size)
        t_s: list[float] = []
        t_bar: list[float] = []
        for i, (freqs, kappa_m) in enumerate(zip(frequencies, curvature, strict=True)):
            if freqs.shape != kappa_m.shape:
                raise ValueError(
                    f"curvature: point {i} has {kappa_m.size} components for "
                    f"{freqs.size} frequencies"
                )
            kappa = float(np.linalg.norm(kappa_m))
            # t_m⁻⁴ = (μ ω_m)² with ħ = 1.
            inv_t4 = (
                float(
                    np.sum(
                        (kappa_m / kappa) ** 2
                        * (_MU * np.abs(freqs) * CM1_TO_HARTREE) ** 2
                    )
                )
                if kappa > 0
                else 0.0
            )
            if inv_t4 > 0:
                t = inv_t4**-0.25
                a_bar[i] = kappa * t
                t_s.append(float(s[i]))
                t_bar.append(t)
        self._a = scipy.interpolate.CubicSpline(s, a_bar)
        self._dt: scipy.interpolate.PPoly | None = None
        if len(t_s) >= 2:
            self._dt = scipy.interpolate.CubicSpline(t_s, t_bar).derivative()

    @classmethod
    def from_path(cls, path: ReactionPath, usable: list[int]) -> _SCTMass:
        if path.curvature is None:
            raise ValueError(
                "path: SCT needs the path's curvature, which comes from "
                "Reaction.compute_mep(); this path has none"
            )
        points = [i for i in usable if path.curvature[i] is not None]
        return cls(
            np.array(path.s[points]),
            [np.asarray(path.frequencies[i]) for i in points],
            [np.asarray(path.curvature[i]) for i in points],
        )

    def __call__(self, s: FloatArray) -> FloatArray:
        a = np.maximum(self._a(s), 0.0)
        dt = 0.0 if self._dt is None else self._dt(s)
        return np.asarray(np.minimum(np.exp(-2.0 * a - a * a + dt * dt), 1.0))


class _Profile:
    """V_a^G(s): a not-a-knot cubic spline through the usable points."""

    def __init__(self, s: FloatArray, v: FloatArray) -> None:
        self.s = s
        self.v = v
        self.spline = scipy.interpolate.CubicSpline(s, v)
        i = int(np.argmax(v))
        self.top_index = i
        best_s, best_v = float(s[i]), float(v[i])
        slope = self.spline.derivative()
        for lo, hi in ((max(i - 1, 0), i), (i, min(i + 1, s.size - 1))):
            if lo == hi:
                continue
            for root in slope.solve(0.0, extrapolate=False):
                if s[lo] < root < s[hi] and self.spline(root) > best_v:
                    best_s, best_v = float(root), float(self.spline(root))
        self.s_max = best_s
        self.v_max = best_v

    def at(self, x: FloatArray) -> FloatArray:
        """V_a^G at ``x``, exactly the point values at the points."""
        x = np.asarray(x, dtype=np.float64)
        out = np.asarray(self.spline(x), dtype=np.float64)
        idx = np.searchsorted(self.s, x)
        idx = np.clip(idx, 0, self.s.size - 1)
        hit = self.s[idx] == x
        out[hit] = self.v[idx[hit]]
        return out

    def above(self, e: float) -> list[tuple[float, float]]:
        """The s intervals where V_a^G > ``e``."""
        roots = [float(r) for r in self.spline.solve(e, extrapolate=False)]
        edges = sorted({float(self.s[0]), *roots, float(self.s[-1])})
        return [
            (a, b)
            for a, b in itertools.pairwise(edges)
            if b > a and self.spline(0.5 * (a + b)) > e
        ]


class _ThetaTable:
    """θ(E) on Chebyshev points between E₀ and V^AG, and the Boltzmann average."""

    def __init__(self, profile: _Profile, mass: _ZeroCurvature | _SCTMass, e0: float):
        self.profile = profile
        self.mass = mass
        self.e0 = e0
        self.v_max = profile.v_max
        k = np.arange(_NODES)
        x = np.cos(np.pi * k / (_NODES - 1))  # 1 … -1
        self.nodes = e0 + 0.5 * (self.v_max - e0) * (1.0 - x)  # E₀ … V^AG
        theta = np.array([self._theta(e) for e in self.nodes[:-1]] + [0.0])
        self.theta_nodes = theta
        self._interp = scipy.interpolate.BarycentricInterpolator(self.nodes, theta)

    def _theta(self, e: float) -> float:
        spline, mass = self.profile.spline, self.mass

        def integrand(x: float) -> float:
            dv = max(float(spline(x)) - e, 0.0)
            return float(np.sqrt(2.0 * _MU * float(mass(np.asarray(x))) * dv))

        def piece(lo: float, hi: float, left: bool, right: bool) -> float:
            """∫ from lo to hi; at a turning point x = end ± u² removes the √."""
            if left and right:
                mid = 0.5 * (lo + hi)
                return piece(lo, mid, True, False) + piece(mid, hi, False, True)
            if not (left or right):
                value, _ = scipy.integrate.quad(
                    integrand, lo, hi, epsabs=1e-12, epsrel=_EPSREL, limit=200
                )
                return float(value)
            end, sign = (lo, 1.0) if left else (hi, -1.0)

            def substituted(u: float) -> float:
                return 2.0 * u * integrand(end + sign * u * u)

            value, _ = scipy.integrate.quad(
                substituted,
                0.0,
                float(np.sqrt(hi - lo)),
                epsabs=1e-12,
                epsrel=_EPSREL,
                limit=200,
            )
            return float(value)

        # The profile's third derivative jumps at the path points, so each
        # stretch between them is integrated on its own.
        knots = self.profile.s
        total = 0.0
        for a, b in self.profile.above(e):
            edges = [a, *knots[(knots > a) & (knots < b)], b]
            turning_a = float(spline(a)) - e <= 1e-14
            turning_b = float(spline(b)) - e <= 1e-14
            n = len(edges) - 1
            for j, (lo, hi) in enumerate(itertools.pairwise(edges)):
                total += piece(lo, hi, j == 0 and turning_a, j == n - 1 and turning_b)
        return total

    def theta(self, e: FloatArray) -> FloatArray:
        e = np.clip(e, self.e0, self.v_max)
        return np.maximum(np.asarray(self._interp(e), dtype=np.float64), 0.0)

    def ln_p(self, e: FloatArray) -> FloatArray:
        """ln P(E) for E₀ ≤ E ≤ 2V^AG - E₀ (the uniform semiclassical formula)."""
        e = np.asarray(e, dtype=np.float64)
        below = e <= self.v_max
        mirrored = np.where(below, e, 2.0 * self.v_max - e)
        two_theta = 2.0 * self.theta(mirrored)
        return np.where(
            below, -np.logaddexp(0.0, two_theta), -np.logaddexp(0.0, -two_theta)
        )

    def probabilities(self) -> tuple[FloatArray, FloatArray]:
        above = 2.0 * self.v_max - self.nodes[-2::-1]
        energies = np.concatenate([self.nodes, above])
        return energies, np.exp(self.ln_p(energies))

    def boltzmann_average(self, kt: float) -> float:
        """β exp(β V^AG) ∫ P(E) exp(-βE) dE from E₀, in u = (E - V^AG)/kT."""
        u0 = (self.e0 - self.v_max) / kt  # < 0

        def f(u: FloatArray | float) -> FloatArray:
            return np.asarray(self.ln_p(self.v_max + np.asarray(u) * kt) - u)

        grid = np.linspace(u0, -u0, _GRID_POINTS)
        values = f(grid)
        peak = int(np.argmax(values))
        c = float(values[peak])
        u_peak = float(grid[peak])

        def integrand(u: float) -> float:
            return float(np.exp(f(u) - c))

        breaks = [u_peak] if u0 < u_peak < -u0 and u_peak != 0.0 else []
        value, _ = scipy.integrate.quad(
            integrand,
            u0,
            -u0,
            epsabs=0.0,
            epsrel=_EPSREL,
            limit=200,
            points=[0.0, *breaks],
        )
        # P = 1 beyond 2V^AG - E₀: ∫ exp(-u) du from -u0 to ∞.
        return float(np.exp(c) * value + np.exp(u0))
