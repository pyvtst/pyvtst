"""Harmonic vibrational analysis of Cartesian Hessians.

Hessians are mass-weighted with masses in electron masses, so the mass-weighted
Hessian is in atomic units and its eigenvalues are squared angular frequencies
in atomic units (hartree/ħ)². Overall translations and rotations about the
center of mass are removed exactly, by diagonalizing the Hessian in an
orthonormal basis of the internal (vibrational) space, so the number of modes
never depends on a numerical threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import AMU_TO_ME, HARTREE_TO_CM1

if TYPE_CHECKING:
    from pyvtst.core.internals import Internal

FloatArray = npt.NDArray[np.float64]

# Atoms within this distance (bohr) of one line make a linear molecule.
LINEAR_TOLERANCE = 1e-3


def is_linear(mol: Molecule) -> bool:
    """Whether all atoms lie on one line to within 1e-3 bohr.

    Diatomics are always linear; a single atom is not.
    """
    if mol.n_atoms == 1:
        return False
    if mol.n_atoms == 2:
        return True
    centered = mol.coords - mol.coords.mean(axis=0)
    singular = scipy.linalg.svdvals(centered)
    return bool(singular[1] <= LINEAR_TOLERANCE)


def center_of_mass(mol: Molecule) -> FloatArray:
    """Center of mass in bohr, shape (3,)."""
    return np.asarray(mol.masses @ mol.coords / mol.total_mass, dtype=np.float64)


def principal_axes(mol: Molecule) -> tuple[FloatArray, FloatArray]:
    """Return the principal moments of inertia and axes about the center of mass.

    Returns:
        ``(moments, axes)``: moments in amu·bohr², ascending, shape (3,), and
        the corresponding unit axes as columns, shape (3, 3).
    """
    m = mol.masses
    r = mol.coords - center_of_mass(mol)
    inertia = np.sum(m * np.sum(r * r, axis=1)) * np.eye(3) - (m[:, None] * r).T @ r
    moments, axes = scipy.linalg.eigh(inertia)
    return np.clip(moments, 0.0, None), axes


def n_vibrations(mol: Molecule) -> int:
    """Number of vibrational modes: 3N-6, 3N-5 if linear, 0 for one atom."""
    if mol.n_atoms == 1:
        return 0
    return 3 * mol.n_atoms - (5 if is_linear(mol) else 6)


def mass_weighted_hessian(mol: Molecule, hessian: npt.ArrayLike) -> FloatArray:
    """Return M^(-1/2) H M^(-1/2) in atomic units (masses in electron masses).

    Args:
        mol: The molecule the Hessian belongs to.
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).

    Raises:
        ValueError: The Hessian has the wrong shape or is not finite.
    """
    n = 3 * mol.n_atoms
    h = np.array(hessian, dtype=np.float64)
    if h.shape != (n, n):
        raise ValueError(f"hessian: expected shape ({n}, {n}), got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("hessian: contains non-finite values")
    h = 0.5 * (h + h.T)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    return inv_sqrt_m[:, None] * h * inv_sqrt_m[None, :]


def external_projector(mol: Molecule, extra: npt.ArrayLike | None = None) -> FloatArray:
    """Return the projector P = I - V Vᵀ onto the internal space.

    V is an orthonormal basis, in mass-weighted Cartesian coordinates, of the
    3 overall translations, the 3 rotations about the center of mass (2 for a
    linear molecule) and any ``extra`` vectors.

    Args:
        mol: The molecule.
        extra: Additional directions to project out, in mass-weighted
            Cartesian coordinates, shape (3N,) or (k, 3N).
    """
    basis = _external_basis(mol, extra)
    return np.eye(3 * mol.n_atoms) - basis @ basis.T


def normal_modes(
    mol: Molecule, hessian: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return harmonic frequencies and mass-weighted normal modes.

    Args:
        mol: The molecule the Hessian belongs to.
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).

    Returns:
        ``(frequencies, modes)``: frequencies in cm⁻¹, ascending, with
        imaginary modes as negative values, shape (n,), and the orthonormal
        mass-weighted normal modes as rows, shape (n, 3N), with n = 3N-6, 3N-5
        for a linear molecule or 0 for a single atom.
    """
    f = mass_weighted_hessian(mol, hessian)
    n = 3 * mol.n_atoms
    if mol.n_atoms == 1:
        return np.empty(0), np.empty((0, n))
    return _modes_in(internal_basis(mol), f)


def internal_basis(mol: Molecule, extra: npt.ArrayLike | None = None) -> FloatArray:
    """Return an orthonormal basis of the internal space, as columns.

    The internal space is the complement, in mass-weighted Cartesian
    coordinates, of the overall translations, the rotations about the center
    of mass and any ``extra`` directions. It has exactly 3N-6 dimensions (3N-5
    for a linear molecule) minus the number of independent ``extra`` vectors.
    """
    external = _external_basis(mol, extra)
    # The trailing columns of a full QR are an orthonormal basis of the
    # complement of the external space: exactly the internal space.
    q, _ = scipy.linalg.qr(external, mode="full")
    return np.asarray(q[:, external.shape[1] :], dtype=np.float64)


def generalized_normal_modes(
    mol: Molecule,
    hessian: npt.ArrayLike,
    gradient: npt.ArrayLike | None = None,
    direction: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return generalized frequencies and modes at a point on a reaction path.

    Overall translations, rotations and the reaction-path direction are
    projected out of the mass-weighted Hessian. The path direction is the
    mass-weighted gradient, or ``direction`` when no gradient is given or its
    norm is below 1e-8 hartree/bohr (at the transition state, where it is the
    transition vector).

    Args:
        mol: The geometry.
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).
        gradient: Cartesian gradient in hartree/bohr, shape (N, 3) or (3N,).
        direction: Path direction in mass-weighted Cartesian coordinates,
            shape (3N,), used when the gradient is absent or vanishes.

    Returns:
        ``(frequencies, modes)`` as in :func:`normal_modes`, with one mode
        fewer: 3N-7, or 3N-6 for a linear geometry.

    Raises:
        ValueError: Neither a non-vanishing gradient nor a direction is given,
            or an array has the wrong shape.
    """
    f = mass_weighted_hessian(mol, hessian)
    d = _path_direction(mol, gradient, direction)
    return _modes_in(internal_basis(mol, extra=d), f)


def curvilinear_generalized_modes(
    mol: Molecule,
    hessian: npt.ArrayLike,
    internals: Sequence[Internal],
    gradient: npt.ArrayLike | None = None,
    direction: npt.ArrayLike | None = None,
    *,
    reference: Molecule | None = None,
    where: str = "",
) -> tuple[FloatArray, FloatArray]:
    """Return generalized frequencies and modes in redundant internal coordinates.

    Away from a stationary point the Cartesian Hessian holds the gradient
    times the curvature of the internal coordinates, Σᵢ (∂V/∂qᵢ) ∂²qᵢ/∂x²,
    which describes straight-line motion rather than vibration and turns bends
    on curved paths imaginary. Here that term is removed, the coordinates are
    reduced to a non-redundant set, the path direction is projected out in the
    metric of the Wilson G matrix and the projected G·F problem is solved
    (C. F. Jackels, Z. Gu and D. G. Truhlar, J. Chem. Phys. 102, 3188 (1995)).
    At a stationary point the result equals :func:`generalized_normal_modes`
    for any set that spans the internal space. Away from one it depends on the
    set, redundant coordinates included, since the set decides which motions
    count as curvilinear.

    Args:
        mol: The geometry.
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).
        internals: Internal coordinates, see :mod:`pyvtst.core.internals`.
        gradient: Cartesian gradient in hartree/bohr, shape (N, 3) or (3N,).
        direction: Path direction in mass-weighted Cartesian coordinates,
            shape (3N,), used when the gradient is absent or its norm is below
            1e-8 hartree/bohr.
        reference: Geometry that fixes the linear-bend axes; ``mol`` if
            ``None``.
        where: Appended to the coverage error, e.g. ``" at s = 0.1000 bohr"``.

    Returns:
        ``(frequencies, modes)`` as in :func:`generalized_normal_modes`:
        3N-7 frequencies, or 3N-6 for a linear geometry, whatever the number
        of redundant coordinates, and the modes in mass-weighted Cartesian
        coordinates.

    Raises:
        ValueError: The coordinates do not span the internal space, neither a
            non-vanishing gradient nor a direction is given, or an array has
            the wrong shape.
    """
    # internals builds on this module's rigid-body bases, so it is imported
    # here rather than at the top.
    from pyvtst.core.internals import b_derivatives, nonredundant  # noqa: PLC0415

    f = mass_weighted_hessian(mol, hessian)
    t, w = nonredundant(mol, internals, where, reference)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    g = _gradient(mol, gradient)
    if g is not None and np.linalg.norm(g) >= 1e-8:
        g_y = t @ (inv_sqrt_m * g)
        curvature = np.tensordot(w @ g_y, b_derivatives(mol, internals, reference), 1)
        f = f - inv_sqrt_m[:, None] * curvature * inv_sqrt_m[None, :]
    d = t @ _path_direction(mol, gradient, direction)
    # The trailing columns of a full QR are an orthonormal basis of the
    # complement of d.
    q, _ = scipy.linalg.qr(d[:, None], mode="full")
    return _modes_in(t.T @ q[:, 1:], f)


def _modes_in(internal: FloatArray, f: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Diagonalize the mass-weighted Hessian ``f`` within the ``internal`` columns."""
    eigvals, eigvecs = scipy.linalg.eigh(internal.T @ f @ internal)
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * HARTREE_TO_CM1
    return freqs, (internal @ eigvecs).T


def _gradient(mol: Molecule, gradient: npt.ArrayLike | None) -> FloatArray | None:
    if gradient is None:
        return None
    n = 3 * mol.n_atoms
    g = np.array(gradient, dtype=np.float64).reshape(-1)
    if g.shape != (n,):
        raise ValueError(f"gradient: expected {n} components, got {g.size}")
    return g


def _path_direction(
    mol: Molecule, gradient: npt.ArrayLike | None, direction: npt.ArrayLike | None
) -> FloatArray:
    n = 3 * mol.n_atoms
    g = _gradient(mol, gradient)
    if g is not None and np.linalg.norm(g) >= 1e-8:
        g_mw = g / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
        return np.asarray(g_mw / np.linalg.norm(g_mw), dtype=np.float64)
    if direction is None:
        raise ValueError("a path direction is needed: the gradient vanishes")
    d = np.array(direction, dtype=np.float64).reshape(-1)
    if d.shape != (n,):
        raise ValueError(f"direction: expected {n} components, got {d.size}")
    return np.asarray(d / np.linalg.norm(d), dtype=np.float64)


def _external_basis(mol: Molecule, extra: npt.ArrayLike | None = None) -> FloatArray:
    """Orthonormal columns spanning translations, rotations and ``extra``."""
    n = 3 * mol.n_atoms
    sqrt_m = np.sqrt(mol.masses * AMU_TO_ME)
    vectors = [np.ravel(sqrt_m[:, None] * np.eye(3)[k]) for k in range(3)]
    if mol.n_atoms > 1:
        r = mol.coords - center_of_mass(mol)
        _, axes = principal_axes(mol)
        # For a linear molecule the axis with the smallest moment is the
        # molecular axis; rotation about it moves no atom.
        n_rot = 2 if is_linear(mol) else 3
        for axis in axes.T[3 - n_rot :]:
            vectors.append(np.ravel(sqrt_m[:, None] * np.cross(axis, r)))
    if extra is not None:
        e = np.array(extra, dtype=np.float64)
        e = e.reshape(-1, n) if e.size else np.empty((0, n))
        vectors.extend(e)
    basis, _ = scipy.linalg.qr(np.column_stack(vectors), mode="economic")
    return np.asarray(basis, dtype=np.float64)
