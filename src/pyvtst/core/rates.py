"""Conventional transition state theory rate constants."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import constants as sc

from pyvtst.core.partition import ln_q_total, temperatures
from pyvtst.core.stationary import StationaryPoint, check_character, species_label
from pyvtst.core.units import BOLTZMANN_AU, HARTREE_TO_KCAL_PER_MOL

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

    ln_k = (
        ln_kappa
        + np.log(sc.k * t / sc.h)
        + ln_q_total(ts, t)
        - sum(ln_q_total(sp, t) for sp in reactants)
        - barrier / (BOLTZMANN_AU * t)
    )
    units = UNIMOLECULAR_UNITS
    if len(reactants) == 2:
        ln_k = ln_k + _LN_M3_TO_CM3
        units = BIMOLECULAR_UNITS
    k = np.exp(ln_k)
    for a in (t, k, ln_k):
        a.flags.writeable = False
    return TSTResult(T=t, k=k, ln_k=ln_k, units=units, barrier_zpe=float(barrier))
