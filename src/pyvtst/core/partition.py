"""Rigid-rotor / harmonic-oscillator partition functions and thermochemistry.

Partition functions are evaluated in SI units with constants from
``scipy.constants``: the translational partition function is per unit volume
(m⁻³), the others are dimensionless. Vibrational partition functions are
referenced to the zero-point level. Every function takes temperatures in
kelvin as a scalar or 1-D array and returns a 1-D array of the same length;
the ``ln_*`` forms are what rate expressions should combine, since products of
many partition functions can overflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import constants as sc

from pyvtst.core.molecule import Molecule
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import AMU_TO_KG, BOHR_TO_M, BOLTZMANN_AU
from pyvtst.core.vib import is_linear, principal_axes

FloatArray = npt.NDArray[np.float64]

STANDARD_PRESSURE = 1e5  # Pa (1 bar)

# hc/k in m·K: x = h c nu/kT with nu in m⁻¹.
_HC_OVER_K = sc.h * sc.c / sc.k


def temperatures(T: npt.ArrayLike) -> FloatArray:
    """Return ``T`` as a new 1-D float array, checking every value is finite and > 0.

    Raises:
        ValueError: ``T`` is not a scalar or 1-D array, or a value is not a
            finite positive temperature in kelvin.
    """
    t = np.array(T, dtype=np.float64, ndmin=1)
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"T: expected a scalar or a non-empty 1-D array, got {T!r}")
    bad = t[~(np.isfinite(t) & (t > 0))]
    if bad.size:
        listed = ", ".join(f"{v:g}" for v in bad)
        raise ValueError(f"T: temperatures must be finite and > 0 K, got {listed}")
    return t


def ln_q_trans_per_volume(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """ln of (2π m k_B T / h²)^{3/2}, the translational partition function in m⁻³."""
    t = temperatures(T)
    m = mol.total_mass * AMU_TO_KG
    return np.asarray(1.5 * np.log(2.0 * np.pi * m * sc.k * t / sc.h**2))


def q_trans_per_volume(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """Translational partition function per unit volume, in m⁻³."""
    return np.exp(ln_q_trans_per_volume(mol, T))


def ln_q_rot(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """ln of the classical rigid-rotor partition function.

    T/(sigma Θ) for linear molecules, (√π/sigma)(T³/(Θ_A Θ_B Θ_C))^{1/2} for
    non-linear ones and 1 for a single atom, with Θ = h²/(8π² I k_B) and sigma the
    molecule's symmetry number.
    """
    t = temperatures(T)
    if mol.n_atoms == 1:
        return np.zeros_like(t)
    moments, _ = principal_axes(mol)
    inertia = moments * AMU_TO_KG * BOHR_TO_M**2  # kg m²
    ln_sigma = np.log(mol.symmetry_number)
    if is_linear(mol):
        theta = sc.h**2 / (8.0 * np.pi**2 * inertia[2] * sc.k)
        return np.asarray(np.log(t / theta) - ln_sigma)
    theta = sc.h**2 / (8.0 * np.pi**2 * inertia * sc.k)
    return np.asarray(
        0.5 * np.log(np.pi) - ln_sigma + 0.5 * (3.0 * np.log(t) - np.sum(np.log(theta)))
    )


def q_rot(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """Classical rigid-rotor partition function (dimensionless)."""
    return np.exp(ln_q_rot(mol, T))


def ln_q_vib(frequencies: npt.ArrayLike, T: npt.ArrayLike) -> FloatArray:
    """ln of Π 1/(1 - exp(-h c nu/k_B T)), referenced to the zero-point level.

    Args:
        frequencies: Real harmonic frequencies in cm⁻¹, all > 0.
        T: Temperatures in kelvin.
    """
    t = temperatures(T)
    nu = _frequencies(frequencies)
    x = _HC_OVER_K * 100.0 * nu[None, :] / t[:, None]
    return np.asarray(-np.sum(np.log1p(-np.exp(-x)), axis=1))


def q_vib(frequencies: npt.ArrayLike, T: npt.ArrayLike) -> FloatArray:
    """Harmonic vibrational partition function from the zero-point level."""
    return np.exp(ln_q_vib(frequencies, T))


def ln_q_elec(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """ln of the electronic partition function: the spin multiplicity."""
    return np.full_like(temperatures(T), np.log(mol.mult))


def q_elec(mol: Molecule, T: npt.ArrayLike) -> FloatArray:
    """Electronic partition function: the spin multiplicity (ground state only)."""
    return np.exp(ln_q_elec(mol, T))


def ln_q_total(sp: StationaryPoint, T: npt.ArrayLike) -> FloatArray:
    """ln of the total partition function per unit volume (m⁻³) of ``sp``.

    Uses :attr:`StationaryPoint.real_frequencies`, so a saddle point's
    imaginary mode is excluded.
    """
    mol = sp.molecule
    return (
        ln_q_trans_per_volume(mol, T)
        + ln_q_rot(mol, T)
        + ln_q_vib(sp.real_frequencies, T)
        + ln_q_elec(mol, T)
    )


@dataclass(frozen=True, eq=False)
class Thermo:
    """Ideal-gas thermochemical corrections at the standard state p° = 1 bar.

    Attributes:
        T: Temperatures in kelvin.
        zpe: Zero-point energy in hartree.
        h_corr: Enthalpy relative to the electronic energy (ZPE included),
            hartree.
        s: Entropy in hartree/K.
        g_corr: Gibbs energy relative to the electronic energy, h_corr - T·s,
            hartree.
    """

    T: FloatArray
    zpe: float
    h_corr: FloatArray
    s: FloatArray
    g_corr: FloatArray


def thermo(sp: StationaryPoint, T: npt.ArrayLike) -> Thermo:
    """Rigid-rotor / harmonic-oscillator thermochemistry of ``sp``.

    The translational entropy uses the ideal-gas standard state p° = 1 bar
    (1e5 Pa), i.e. the volume V = k_B T / p°. Values are per molecule, in
    hartree and hartree/K. A saddle point's imaginary mode is excluded.
    """
    t = temperatures(T)
    mol = sp.molecule
    k = BOLTZMANN_AU

    ln_qv_trans = ln_q_trans_per_volume(mol, t) + np.log(sc.k * t / STANDARD_PRESSURE)
    e_trans = 1.5 * k * t
    s_trans = k * (ln_qv_trans + 2.5)

    if mol.n_atoms == 1:
        e_rot = np.zeros_like(t)
        s_rot = np.zeros_like(t)
    else:
        dof = 1.0 if is_linear(mol) else 1.5
        e_rot = dof * k * t
        s_rot = k * (ln_q_rot(mol, t) + dof)

    nu = sp.real_frequencies
    if nu.size:
        x = _HC_OVER_K * 100.0 * _frequencies(nu)[None, :] / t[:, None]
        boltz = np.exp(-x)
        x_occ = x * boltz / -np.expm1(-x)  # x/(eˣ - 1), without overflow
        e_vib = k * t * np.sum(x_occ, axis=1)
        s_vib = k * np.sum(x_occ - np.log1p(-boltz), axis=1)
    else:
        e_vib = np.zeros_like(t)
        s_vib = np.zeros_like(t)

    s_elec = k * np.log(mol.mult)
    h_corr = sp.zpe + e_trans + e_rot + e_vib + k * t
    s = s_trans + s_rot + s_vib + s_elec
    for a in (t, h_corr, s):
        a.flags.writeable = False
    g_corr = h_corr - t * s
    g_corr.flags.writeable = False
    return Thermo(T=t, zpe=sp.zpe, h_corr=h_corr, s=s, g_corr=g_corr)


def _frequencies(frequencies: npt.ArrayLike) -> FloatArray:
    nu = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if nu.ndim != 1:
        raise ValueError("frequencies: expected a 1-D array")
    if not np.all(np.isfinite(nu) & (nu > 0)):
        raise ValueError(
            f"frequencies: all must be finite and > 0 cm^-1, got {nu.tolist()}"
        )
    return nu
