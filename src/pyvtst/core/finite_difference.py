"""Finite-difference derivatives built from other derivatives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def fd_hessian(
    gradient_fn: Callable[[FloatArray], npt.ArrayLike],
    coords: npt.ArrayLike,
    step: float = 5e-3,
) -> FloatArray:
    """Return the Cartesian Hessian of a gradient function by central differences.

    Args:
        gradient_fn: Maps (N, 3) coordinates in bohr to an (N, 3) gradient in
            hartree/bohr.
        coords: Coordinates in bohr, shape (N, 3).
        step: Displacement in bohr.

    Returns:
        The (3N, 3N) Hessian in hartree/bohr², symmetrized as (H + Hᵀ)/2, with
        row and column index ``3*i + k`` for atom ``i`` and component ``k``.
        The gradient function is called 6N times.
    """
    if not step > 0:
        raise ValueError(f"step: must be positive, got {step}")
    x0 = np.array(coords, dtype=np.float64)
    n = x0.size
    hessian = np.empty((n, n))
    for j in range(n):
        x = x0.ravel().copy()
        x[j] += step
        g_plus = np.asarray(gradient_fn(x.reshape(x0.shape)), dtype=np.float64)
        x[j] -= 2.0 * step
        g_minus = np.asarray(gradient_fn(x.reshape(x0.shape)), dtype=np.float64)
        hessian[:, j] = (g_plus - g_minus).ravel() / (2.0 * step)
    return 0.5 * (hessian + hessian.T)
