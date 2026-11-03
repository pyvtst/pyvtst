"""Rovibrational sums of states on an energy grid.

The harmonic vibrational levels are counted exactly on a uniform grid with
the Beyer-Swinehart recursion, starting from the classical rigid-rotor sum of
states, which convolves the two (the Stein-Rabinovitch extension):
T. Beyer and D. F. Swinehart, Commun. ACM 16, 379 (1973);
S. E. Stein and B. S. Rabinovitch, J. Chem. Phys. 58, 2438 (1973).
The rotor is the one whose partition function :func:`ln_q_rot` gives, so that
the Laplace transform of the sum of states reproduces Q_rot · Q_vib.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pyvtst.core.molecule import Molecule
from pyvtst.core.partition import ln_q_rot
from pyvtst.core.units import BOLTZMANN_AU, CM1_TO_HARTREE
from pyvtst.core.vib import is_linear

FloatArray = npt.NDArray[np.float64]

#: Default and largest grid spacing, 1 cm⁻¹ in hartree.
DEFAULT_GRAIN = CM1_TO_HARTREE
# Allows a grain computed from 1 cm⁻¹ by another route to pass the limit.
_GRAIN_SLACK = 1e-12


def sum_of_states(
    mol: Molecule,
    frequencies: npt.ArrayLike,
    e_max: float,
    grain: float = DEFAULT_GRAIN,
) -> FloatArray:
    """Rovibrational sum of states N(ε) on the grid ε_i = i · ``grain``.

    N(ε) is the number of states with internal energy ≤ ε above the zero-point
    level: harmonic vibrations over ``frequencies`` (their magnitudes, each
    placed at the nearest whole number of grains and at least one), convolved
    with the classical rigid rotor of ``mol``, with its symmetry number. The
    spin degeneracy is not included.

    Args:
        mol: The geometry; not a single atom.
        frequencies: Harmonic frequencies in cm⁻¹, non-zero and finite;
            imaginary values (negative) count with their magnitude.
        e_max: Highest energy of the grid in hartree, > 0.
        grain: Grid spacing in hartree, > 0 and at most 1 cm⁻¹.

    Returns:
        N at i = 0 … ⌈e_max / grain⌉, with N[0] = 0.

    Raises:
        ValueError: Invalid ``grain``, ``e_max`` or frequency, or a single atom.
    """
    if not (math.isfinite(grain) and 0.0 < grain <= DEFAULT_GRAIN * (1 + _GRAIN_SLACK)):
        raise ValueError(
            f"grain: expected a finite spacing in (0, 1 cm^-1], got {grain!r} hartree"
        )
    if not (math.isfinite(e_max) and e_max > 0.0):
        raise ValueError(f"e_max: expected a finite energy > 0, got {e_max!r}")
    nu = np.array(frequencies, dtype=np.float64, ndmin=1)
    if nu.ndim != 1:
        raise ValueError(f"frequencies: expected a 1-D array, got {frequencies!r}")
    bad = nu[~np.isfinite(nu) | (nu == 0.0)]
    if bad.size:
        listed = ", ".join(f"{v:g}" for v in bad)
        raise ValueError(
            f"frequencies: must be finite and non-zero, got {listed} cm^-1"
        )
    if mol.n_atoms == 1:
        raise ValueError(
            "mol: a single atom has no rotor; the sum of states needs a molecule"
        )

    n = math.ceil(e_max / grain - 1e-9)
    eps = grain * np.arange(n + 1, dtype=np.float64)
    states = rotor_sum_of_states(mol, eps)
    ladder = np.maximum(1, np.rint(np.abs(nu) * CM1_TO_HARTREE / grain))
    for steps in ladder.astype(int):
        states = _add_oscillator(states, int(steps))
    return states


def rotor_sum_of_states(mol: Molecule, energies: npt.ArrayLike) -> FloatArray:
    """Classical rigid-rotor sum of states at ``energies`` (hartree, ≥ 0).

    With Q_rot = C (k_B T)^(r/2), r = 2 for a linear and 3 for a non-linear
    geometry, N_rot(ε) = C ε^(r/2) / Γ(r/2 + 1). C is taken from
    :func:`ln_q_rot`, so the symmetry number, the moments of inertia and the
    linearity test are the partition function's own.
    """
    e = np.asarray(energies, dtype=np.float64)
    half = 1.0 if is_linear(mol) else 1.5
    t_ref = 1000.0
    ln_c = float(ln_q_rot(mol, t_ref)[0]) - half * math.log(BOLTZMANN_AU * t_ref)
    return np.asarray(np.exp(ln_c - math.lgamma(half + 1.0)) * e**half)


def _add_oscillator(states: FloatArray, steps: int) -> FloatArray:
    """Convolve with one harmonic ladder: N'[i] = Σ_{k ≥ 0} N[i - k·steps].

    The Beyer-Swinehart recursion N[i] += N[i - steps] is a cumulative sum
    within each residue class modulo ``steps``.
    """
    size = states.size
    pad = (-size) % steps
    grid = np.concatenate([states, np.zeros(pad)]).reshape(-1, steps)
    return np.cumsum(grid, axis=0).ravel()[:size]
