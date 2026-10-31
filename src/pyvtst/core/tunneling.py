"""One-dimensional tunneling transmission coefficients κ(T).

Wigner: E. Wigner, Z. Phys. Chem. B 19, 203 (1932).
Eckart: C. Eckart, Phys. Rev. 35, 1303 (1930), with the barrier fitted to the
forward and reverse barrier heights and the imaginary frequency as in
H. S. Johnston and J. Heicklen, J. Phys. Chem. 66, 532 (1962).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate
from scipy import constants as sc

from pyvtst.core.partition import temperatures
from pyvtst.core.units import BOLTZMANN_AU, CM1_TO_HARTREE, HARTREE_TO_KCAL_PER_MOL

FloatArray = npt.NDArray[np.float64]

# hc/k in cm·K: h c nu/kT with nu in cm⁻¹.
_HC_OVER_K_CM = sc.h * sc.c * 100.0 / sc.k

# The Boltzmann average is integrated up to this many kT above the barrier
# top; the remaining tail, with P = 1, contributes exp(-50) relative.
_UPPER_KT = 50.0
_GRID_POINTS = 2001
_LN2 = float(np.log(2.0))


def wigner(nu_imag: float, T: npt.ArrayLike) -> FloatArray:
    """Wigner transmission coefficient 1 + (1/24)(hc|nu‡|/k_B T)².

    Args:
        nu_imag: Imaginary frequency of the transition state in cm⁻¹ (its
            magnitude is used).
        T: Temperatures in kelvin.
    """
    t = temperatures(T)
    u = _HC_OVER_K_CM * abs(float(nu_imag)) / t
    return np.asarray(1.0 + u * u / 24.0)


def eckart_probability(
    E: npt.ArrayLike, V1: float, V2: float, nu_imag: float
) -> FloatArray:
    """Transmission probability P(E) through an Eckart barrier.

    Args:
        E: Energies in hartree, measured from the reactant zero-point level.
        V1: Forward barrier height in hartree (> 0).
        V2: Reverse barrier height in hartree (> 0); the products lie at
            V1 - V2.
        nu_imag: Imaginary frequency of the barrier top in cm⁻¹ (magnitude).

    Returns:
        P(E) in [0, 1], zero below the higher of the two asymptotes.
    """
    hw = _check_barrier(V1, V2, nu_imag)
    eps = np.asarray(E, dtype=np.float64) - V1
    return np.asarray(np.exp(_ln_probability(eps / hw, V1 / hw, V2 / hw)))


def eckart(V1: float, V2: float, nu_imag: float, T: npt.ArrayLike) -> FloatArray:
    """Eckart transmission coefficient κ(T).

    κ(T) = (exp(V₁/k_B T)/k_B T) ∫ P(E) exp(-E/k_B T) dE from
    E_min = max(0, V₁ - V₂) to ∞, with E from the reactant zero-point level.

    Args:
        V1: ZPE-corrected forward barrier in hartree (> 0).
        V2: ZPE-corrected reverse barrier in hartree (> 0). Pass ``V1`` for a
            symmetric barrier.
        nu_imag: Imaginary frequency of the transition state in cm⁻¹
            (magnitude).
        T: Temperatures in kelvin.

    The result is the same for (V1, V2) and (V2, V1), so forward and reverse
    rate constants with Eckart tunneling satisfy detailed balance.

    Raises:
        ValueError: A barrier is not positive, or ``nu_imag`` is zero.
    """
    t = temperatures(T)
    hw = _check_barrier(V1, V2, nu_imag)
    v1, v2 = V1 / hw, V2 / hw
    return np.array([_kappa(v1, v2, BOLTZMANN_AU * ti / hw) for ti in t])


def _check_barrier(V1: float, V2: float, nu_imag: float) -> float:
    """Validate the barrier and return ħω‡ in hartree."""
    if not (np.isfinite(V1) and np.isfinite(V2) and V1 > 0 and V2 > 0):
        raise ValueError(
            "an Eckart barrier needs positive ZPE-corrected barriers in both "
            f"directions, got V1 = {V1 * HARTREE_TO_KCAL_PER_MOL:.3f} kcal/mol and "
            f"V2 = {V2 * HARTREE_TO_KCAL_PER_MOL:.3f} kcal/mol"
        )
    nu = abs(float(nu_imag))
    if not (np.isfinite(nu) and nu > 0):
        raise ValueError(f"nu_imag: must be finite and non-zero, got {nu_imag}")
    return nu * CM1_TO_HARTREE


def _kappa(v1: float, v2: float, kt: float) -> float:
    """κ at one temperature; energies in units of ħω‡, ε from the barrier top.

    κ = ∫ P(ε) exp(-ε/kT) dε / kT from -min(V1, V2) to ∞. In u = ε/kT the
    integrand is exp(f(u)) with f = ln P - u; it is integrated as
    exp(c) ∫ exp(f - c) du with c the maximum of f, so nothing overflows.
    """
    u_lo = -min(v1, v2) / kt

    def f(u: FloatArray | float) -> FloatArray:
        return np.asarray(_ln_probability(np.asarray(u) * kt, v1, v2) - u)

    grid = np.linspace(u_lo, 0.0, _GRID_POINTS)
    values = f(grid)
    peak = int(np.argmax(values))
    c = float(max(values[peak], f(0.0)))
    u_peak = float(grid[peak])

    def integrand(u: float) -> float:
        return float(np.exp(f(u) - c))

    def integral(a: float, b: float, points: list[float] | None = None) -> float:
        value, _ = scipy.integrate.quad(
            integrand, a, b, epsabs=0.0, epsrel=1e-11, limit=200, points=points
        )
        return value

    below = integral(u_lo, 0.0, [u_peak] if u_lo < u_peak < 0.0 else None)
    above = integral(0.0, _UPPER_KT)
    tail = np.exp(-_UPPER_KT - c)
    return float(np.exp(c) * (below + above + tail))


def _ln_probability(eps: FloatArray | float, v1: float, v2: float) -> FloatArray:
    """ln P for energies ``eps`` above the barrier top, all in units of ħω‡.

    P = 2 sinh(a) sinh(b) / (cosh(a + b) + cosh(d)), with
    a = 2π g √(ε + V1), b = 2π g √(ε + V2), g = 2/(V1^(-1/2) + V2^(-1/2)),
    d = 2π √(4 V1 V2 - 1/4) (cosh d becomes cos |d| when the root is
    imaginary). Everything is symmetric in V1 and V2. -inf below threshold.
    """
    e = np.asarray(eps, dtype=np.float64)
    g = 2.0 / (1.0 / np.sqrt(v1) + 1.0 / np.sqrt(v2))
    k1 = np.maximum(e + v1, 0.0)
    k2 = np.maximum(e + v2, 0.0)
    a = 2.0 * np.pi * g * np.sqrt(k1)
    b = 2.0 * np.pi * g * np.sqrt(k2)
    d2 = 4.0 * (v1 * v2) - 0.25
    ln_cosh_sum = _ln_cosh(a + b)
    if d2 >= 0:
        ln_den = np.logaddexp(ln_cosh_sum, _ln_cosh(2.0 * np.pi * np.sqrt(d2)))
    else:
        cos_d = np.cos(2.0 * np.pi * np.sqrt(-d2))
        ln_den = ln_cosh_sum + np.log1p(cos_d * np.exp(-ln_cosh_sum))
    with np.errstate(divide="ignore"):
        ln_p = _LN2 + (_ln_sinh(a) + _ln_sinh(b)) - ln_den
    return np.minimum(ln_p, 0.0)


def _ln_cosh(x: FloatArray) -> FloatArray:
    ax = np.abs(x)
    return np.asarray(ax + np.log1p(np.exp(-2.0 * ax)) - _LN2)


def _ln_sinh(x: FloatArray) -> FloatArray:
    """ln sinh x for x ≥ 0 (-inf at 0)."""
    with np.errstate(divide="ignore"):
        return np.asarray(x + np.log(-np.expm1(-2.0 * x)) - _LN2)
