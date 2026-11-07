"""Arrhenius fits of rate constants k(T).

Two forms are fitted, both by least squares in ln k so that every temperature
carries the same relative weight:

- ``"arrhenius"``: k = A exp(-Ea / RT)
- ``"modified"``: k = A (T / T_ref)ⁿ exp(-Ea / RT), with T_ref = 300 K

Both are linear in (ln A, n, Ea), so each fit is an exact linear least-squares
solve with no starting guess.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import linalg

from pyvtst.core.units import GAS_CONSTANT_KCAL

FloatArray = npt.NDArray[np.float64]

MODELS: Final = ("arrhenius", "modified")
N_PARAMETERS: Final = {"arrhenius": 2, "modified": 3}
T_REF: Final = 300.0
POOR_FIT: Final = 0.05

_FORMULAS: Final = {
    "arrhenius": "k = A exp(-Ea/RT)",
    "modified": "k = A (T/T_ref)^n exp(-Ea/RT)",
}
_HINTS: Final = {
    "arrhenius": "try model='modified'",
    "modified": "fit a narrower temperature range",
}


@dataclass(frozen=True, eq=False)
class ArrheniusFit:
    """A fitted Arrhenius expression for k(T).

    Attributes:
        model: ``"arrhenius"`` or ``"modified"``.
        A: Pre-exponential factor, in the unit of k.
        n: Temperature exponent (0 for ``"arrhenius"``).
        Ea_kcal: Activation energy Ea in kcal/mol.
        T_ref: Reference temperature in K.
        units: Unit of k and A, or ``None`` when not known.
        T_min: Lowest fitted temperature in K.
        T_max: Highest fitted temperature in K.
        max_rel_dev: Largest ``abs(k_fit / k - 1)`` over the fitted temperatures.
        rss: Sum of squared ln k residuals.
    """

    model: str
    A: float
    n: float
    Ea_kcal: float
    T_ref: float
    units: str | None
    T_min: float
    T_max: float
    max_rel_dev: float
    rss: float

    def __call__(self, T: npt.ArrayLike) -> FloatArray:
        """Return the fitted k at temperatures ``T`` (K)."""
        t = np.asarray(T, dtype=np.float64)
        return np.asarray(
            self.A
            * (t / self.T_ref) ** self.n
            * np.exp(-self.Ea_kcal / (GAS_CONSTANT_KCAL * t))
        )

    def activation_energy(self, T: npt.ArrayLike) -> FloatArray:
        """Return the activation energy at temperatures ``T`` (K), in kcal/mol.

        This is the local slope of the fitted Arrhenius plot,
        Ea(T) = -R d ln k / d(1/T) = Ea + nRT. It equals ``Ea_kcal`` at every
        T for ``"arrhenius"``; for ``"modified"`` it rises by nRT, so
        ``Ea_kcal`` is its value extrapolated to T = 0.
        """
        t = np.asarray(T, dtype=np.float64)
        return np.asarray(self.Ea_kcal + self.n * GAS_CONSTANT_KCAL * t)

    def __str__(self) -> str:
        unit = self.units if self.units is not None else "unit of k"
        lines = [
            f"Arrhenius fit ({self.model}): {_FORMULAS[self.model]}",
            f"  A     = {self.A:.6e} {unit}",
        ]
        if self.model == "modified":
            lines.append(f"  n     = {self.n:.6f}")
        lines.append(f"  Ea    = {self.Ea_kcal:.6f} kcal/mol")
        if self.model == "modified":
            lines.append(f"  T_ref = {self.T_ref:g} K")
        lines.append(
            f"  fitted over {self.T_min:g}-{self.T_max:g} K, "
            f"max deviation {100 * self.max_rel_dev:.2f}%"
        )
        return "\n".join(lines)


def fit_arrhenius(
    T: npt.ArrayLike,
    k: npt.ArrayLike,
    model: str = "modified",
    *,
    units: str | None = None,
) -> ArrheniusFit:
    """Fit k(T) to an Arrhenius form by least squares in ln k.

    Args:
        T: Temperatures in K, 1-D.
        k: Rate constants, one per temperature, all > 0.
        model: ``"arrhenius"`` or ``"modified"`` (default).
        units: Unit of k, carried into the result for printing.

    Returns:
        The fit. A ``UserWarning`` is issued when it deviates from k by more
        than 5 % at some temperature.

    Raises:
        ValueError: Unknown model, ``T`` and ``k`` not 1-D or of different
            lengths, a value not finite, T ≤ 0, k ≤ 0, or fewer distinct
            temperatures than the model has parameters.
    """
    return _fit(T, k, model, units, stacklevel=3)


def _fit(
    T: npt.ArrayLike,
    k: npt.ArrayLike,
    model: str,
    units: str | None,
    stacklevel: int,
) -> ArrheniusFit:
    """Do the work of :func:`fit_arrhenius`.

    Kept separate so that every public entry point (:func:`fit_arrhenius`,
    ``RateResult.fit_arrhenius``) can make the poor-fit warning point at the
    user's line: ``stacklevel`` counts frames up from the ``warnings.warn``
    call here, so an entry point that calls this directly passes 3.
    """
    t, ln_k = _checked(T, k, model)
    # Solve in reduced variables, all of order 1: tau = T / T_ref and
    # e = Ea / (R T_ref), so ln k = ln A + n ln tau - e / tau.
    tau = t / T_REF
    columns = [np.ones_like(tau), -1.0 / tau]
    if model == "modified":
        columns.append(np.log(tau))
    coef = linalg.lstsq(np.column_stack(columns), ln_k)[0]
    ln_a, e = float(coef[0]), float(coef[1])
    n = float(coef[2]) if model == "modified" else 0.0
    resid = ln_a + n * np.log(tau) - e / tau - ln_k
    result = ArrheniusFit(
        model=model,
        A=float(np.exp(ln_a)),
        n=n,
        Ea_kcal=e * GAS_CONSTANT_KCAL * T_REF,
        T_ref=T_REF,
        units=units,
        T_min=float(np.min(t)),
        T_max=float(np.max(t)),
        max_rel_dev=float(np.max(np.abs(np.expm1(resid)))),
        rss=float(resid @ resid),
    )
    if result.max_rel_dev > POOR_FIT:
        warnings.warn(
            f"Arrhenius fit ({model!r}) deviates from k by up to "
            f"{100 * result.max_rel_dev:.1f}% at some temperature; "
            f"{_HINTS[model]}",
            UserWarning,
            stacklevel=stacklevel,
        )
    return result


def check_model(model: str) -> None:
    """Raise ``ValueError`` unless ``model`` is a known fit model."""
    if model not in MODELS:
        names = ", ".join(repr(m) for m in MODELS)
        raise ValueError(f"Arrhenius model: expected one of {names}, got {model!r}")


def check_count(model: str, n_distinct: int) -> None:
    """Raise ``ValueError`` if ``n_distinct`` temperatures cannot fit ``model``."""
    need = N_PARAMETERS[model]
    if n_distinct < need:
        raise ValueError(
            f"Arrhenius model {model!r} needs at least {need} distinct "
            f"temperatures, got {n_distinct}"
        )


def _checked(
    T: npt.ArrayLike, k: npt.ArrayLike, model: str
) -> tuple[FloatArray, FloatArray]:
    check_model(model)
    t = np.asarray(T, dtype=np.float64)
    rates = np.asarray(k, dtype=np.float64)
    if t.ndim != 1 or rates.ndim != 1:
        raise ValueError(f"T and k must be 1-D, got {t.ndim}-D T and {rates.ndim}-D k")
    if t.size != rates.size:
        raise ValueError(f"T and k differ in length: {t.size} T and {rates.size} k")
    for name, values, unit in (("T", t, " K"), ("k", rates, "")):
        bad = values[~(np.isfinite(values) & (values > 0))]
        if bad.size:
            listed = ", ".join(f"{v:g}" for v in bad)
            raise ValueError(
                f"{name}: values must be finite and > 0{unit}, got {listed}"
            )
    check_count(model, int(np.unique(t).size))
    return t, np.log(rates)
