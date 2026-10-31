"""Harmonic vibrational analysis of Cartesian Hessians.

Hessians are mass-weighted with masses in electron masses, so the mass-weighted
Hessian is in atomic units and its eigenvalues are squared angular frequencies
in atomic units (hartree/ħ)². Overall translations and rotations about the
center of mass are removed exactly, by diagonalizing the Hessian in an
orthonormal basis of the internal (vibrational) space, so the number of modes
never depends on a numerical threshold.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pyvtst.core.molecule import Molecule
from pyvtst.core.units import AMU_TO_ME, HARTREE_TO_CM1

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
    external = _external_basis(mol)
    # The trailing columns of a full QR are an orthonormal basis of the
    # complement of the external space: exactly the internal space.
    q, _ = scipy.linalg.qr(external, mode="full")
    internal = q[:, external.shape[1] :]
    eigvals, eigvecs = scipy.linalg.eigh(internal.T @ f @ internal)
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * HARTREE_TO_CM1
    modes = (internal @ eigvecs).T
    return freqs, modes


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
