"""Redundant internal coordinates: bonds, angles, linear bends and torsions.

A coordinate is a plain tuple ``(kind, *atom indices)`` with zero-based atom
indices: ``("bond", i, j)``, ``("angle", i, j, k)`` with apex j,
``("linear_bend", i, j, k)`` with middle atom j, and ``("torsion", i, j, k, l)``
about the j-k bond. Bonds are in bohr, angles in radians in [0, π] and torsions
in radians in (-π, π].

A linear bend has two components, ``e_aᵀ (u_ji + u_jk)`` for a = 1, 2, where
u_ji and u_jk are the unit vectors from j to i and from j to k, and e_1, e_2
are perpendicular unit vectors normal to the i-k axis of a reference geometry.
Both components vanish when the three atoms are collinear and, unlike an
angle, are smooth there. When the reference is bent, e_1 points along its bend,
so the components turn with the molecule; when it is collinear, e_1 is normal
to the axis and to the Cartesian unit vector least parallel to it. Either way
the axes are fixed by the reference, so every function takes the reference
geometry as ``reference``, defaulting to the geometry itself.

The Wilson B matrix has one row per component, so two rows per linear bend,
and holds the first derivatives with respect to the Cartesian coordinates in
bohr. Second derivatives are central differences of the analytic B rows.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse.csgraph

from pyvtst.core.elements import element
from pyvtst.core.molecule import Molecule
from pyvtst.core.units import AMU_TO_ME, ANGSTROM_TO_BOHR
from pyvtst.core.vib import external_projector, internal_basis, n_vibrations

FloatArray = npt.NDArray[np.float64]
Internal = tuple[str, *tuple[int, ...]]

KINDS = ("bond", "angle", "linear_bend", "torsion")
_N_INDICES = {"bond": 2, "angle": 3, "linear_bend": 3, "torsion": 4}

# Atoms closer than this times the sum of their covalent radii are bonded.
BOND_FACTOR = 1.3
# Bond angles at or above this (radians) give a linear bend instead of an angle.
LINEAR_ANGLE = np.deg2rad(170.0)
# Eigenvalues of G below this fraction of the largest are redundancies.
RANK_TOLERANCE = 1e-10
# Cartesian step (bohr) for the second derivatives.
FD_STEP = 1e-5
# A reference bend vector shorter than this counts as collinear.
_COLLINEAR = 1e-8


def validate_internals(
    internals: Iterable[object], n_atoms: int
) -> tuple[Internal, ...]:
    """Check a set of internal coordinates and return it as tuples.

    Args:
        internals: Coordinates as ``(kind, *atom indices)`` sequences.
        n_atoms: Number of atoms of the molecule they describe.

    Raises:
        ValueError: An entry has an unknown kind, the wrong number of atom
            indices, an index that is not an atom or the same atom twice, or
            describes the same coordinate as an earlier entry (reversed atom
            order counts as the same). The message names the entry.
    """
    if isinstance(internals, str) or not isinstance(internals, Iterable):
        raise ValueError(
            f"internals: expected a sequence of (kind, *atom indices) tuples, "
            f"got {internals!r}"
        )
    seen: dict[tuple[str, tuple[int, ...]], Internal] = {}
    for entry in internals:
        coord = _validated(entry, n_atoms)
        atoms = _indices(coord)
        key = (coord[0], min(atoms, atoms[::-1]))
        if key in seen:
            raise ValueError(
                f"internals: {coord!r} describes the same coordinate as {seen[key]!r}"
            )
        seen[key] = coord
    return tuple(seen.values())


def values(
    mol: Molecule, internals: Sequence[Internal], reference: Molecule | None = None
) -> FloatArray:
    """Values of the coordinates, one per B row, shape (m,)."""
    coords = validate_internals(internals, mol.n_atoms)
    ref = _reference(mol, reference)
    return np.concatenate(
        [_coordinate(mol.coords, ref, c)[0] for c in coords] or [np.empty(0)]
    )


def wilson_b(
    mol: Molecule, internals: Sequence[Internal], reference: Molecule | None = None
) -> FloatArray:
    """Wilson B matrix, the first derivatives in 1/bohr or rad/bohr, shape (m, 3N)."""
    coords = validate_internals(internals, mol.n_atoms)
    return _b_matrix(mol.coords, _reference(mol, reference), coords)


def b_derivatives(
    mol: Molecule, internals: Sequence[Internal], reference: Molecule | None = None
) -> FloatArray:
    """Second derivatives of each B row, shape (m, 3N, 3N), symmetric in the last two.

    Central differences of the analytic B rows with a step of 1e-5 bohr.
    """
    coords = validate_internals(internals, mol.n_atoms)
    ref = _reference(mol, reference)
    n = 3 * mol.n_atoms
    c = np.empty((_n_rows(coords), n, n))
    for p in range(n):
        step = np.zeros(n)
        step[p] = FD_STEP
        shift = step.reshape(-1, 3)
        plus = _b_matrix(mol.coords + shift, ref, coords)
        minus = _b_matrix(mol.coords - shift, ref, coords)
        c[:, p, :] = (plus - minus) / (2.0 * FD_STEP)
    return np.asarray(0.5 * (c + c.transpose(0, 2, 1)), dtype=np.float64)


def auto_internals(mol: Molecule) -> tuple[Internal, ...]:
    """Generate redundant internal coordinates from a geometry.

    Atoms closer than 1.3 times the sum of their covalent radii are bonded.
    While the bonds leave more than one fragment, the closest pair of atoms in
    different fragments is bonded, ties going to the lowest (i, j). Each pair
    of bonds sharing an atom gives an angle, or a linear bend when the angle
    is 170° or more. Each chain of three bonds whose two angles are below 170°
    gives a torsion.

    Returns:
        Bonds, angles, linear bends and torsions, in that order, each sorted
        by atom indices.
    """
    x = mol.coords
    n = mol.n_atoms
    radii = np.array([element(s).covalent_radius for s in mol.symbols])
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    bonded = dist < BOND_FACTOR * ANGSTROM_TO_BOHR * (radii[:, None] + radii[None, :])
    np.fill_diagonal(bonded, False)
    while True:
        n_fragments, labels = scipy.sparse.csgraph.connected_components(bonded)
        if n_fragments <= 1:
            break
        _, i, j = min(
            (dist[i, j], i, j)
            for i, j in combinations(range(n), 2)
            if labels[i] != labels[j]
        )
        bonded[i, j] = bonded[j, i] = True

    neighbours = [np.flatnonzero(bonded[j]).tolist() for j in range(n)]
    bonds = [("bond", i, j) for i, j in combinations(range(n), 2) if bonded[i, j]]
    angles: list[Internal] = []
    linear: list[Internal] = []
    for j in range(n):
        for i, k in combinations(neighbours[j], 2):
            if _angle(x, i, j, k) >= LINEAR_ANGLE:
                linear.append(("linear_bend", i, j, k))
            else:
                angles.append(("angle", i, j, k))
    torsions: list[Internal] = []
    for _, j, k in bonds:
        for i in neighbours[j]:
            for m in neighbours[k]:
                if (
                    len({i, j, k, m}) == 4
                    and _angle(x, i, j, k) < LINEAR_ANGLE
                    and _angle(x, j, k, m) < LINEAR_ANGLE
                ):
                    torsions.append(("torsion", i, j, k, m))
    return (
        *bonds,
        *sorted(angles, key=_indices),
        *sorted(linear, key=_indices),
        *sorted(torsions, key=_indices),
    )


def check_span(
    mol: Molecule,
    internals: Sequence[Internal],
    where: str = "",
    reference: Molecule | None = None,
) -> None:
    """Check that the coordinates span every internal motion of ``mol``.

    The rank of the B matrix, with overall rotations removed, must be 3N-6
    (3N-5 for a linear geometry).

    Args:
        mol: The geometry.
        internals: The coordinates.
        where: Appended to the error message, e.g. ``" at s = -0.1000 bohr"``.
        reference: Geometry that fixes the linear-bend axes; ``mol`` if
            ``None``.

    Raises:
        ValueError: The coordinates miss internal directions. The message says
            how many of how many are spanned and names the atoms that move
            most in the first missing direction.
    """
    nonredundant(mol, internals, where, reference)


def nonredundant(
    mol: Molecule,
    internals: Sequence[Internal],
    where: str = "",
    reference: Molecule | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Reduce redundant coordinates to an orthonormal basis of the internal space.

    With inverse masses u (electron masses), B_mw = B u^½ restricted to the
    internal space and G = B_mw B_mwᵀ = V Λ Vᵀ, the eigenvectors with
    eigenvalues above 1e-10 of the largest give the non-redundant coordinates
    Q = V_rᵀ q.

    Returns:
        ``(t, w)``: ``t = Λ_r^(-½) V_rᵀ B_mw``, shape (r, 3N), whose
        orthonormal rows are the mass-weighted Cartesian directions of the
        unit-metric coordinates y = Λ_r^(-½) Q, and ``w = V_r Λ_r^(-½)``,
        shape (m, r), which turns a gradient with respect to y into one with
        respect to each B row.

    Raises:
        ValueError: The rank r is below 3N-6 (3N-5 if linear); see
            :func:`check_span`.
    """
    coords = validate_internals(internals, mol.n_atoms)
    b = _b_matrix(mol.coords, _reference(mol, reference), coords)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    # Linear-bend axes are fixed in space, so away from collinearity their
    # rows pick up overall rotation; the projection removes it and leaves the
    # rows of every other kind unchanged.
    b_mw = (b * inv_sqrt_m) @ external_projector(mol)
    lam, vecs = scipy.linalg.eigh(b_mw @ b_mw.T)
    keep = lam > RANK_TOLERANCE * np.max(lam, initial=0.0)
    lam, vecs = lam[keep], vecs[:, keep]
    expected = n_vibrations(mol)
    if lam.size < expected:
        raise ValueError(_span_message(mol, b_mw, lam.size, expected, where))
    t = (vecs / np.sqrt(lam)).T @ b_mw
    return np.asarray(t, dtype=np.float64), np.asarray(vecs / np.sqrt(lam))


def _span_message(
    mol: Molecule, b_mw: FloatArray, rank: int, expected: int, where: str
) -> str:
    basis = internal_basis(mol)
    # The missing directions are the internal motions no B row sees; the
    # trailing right singular vectors span them.
    _, _, vt = scipy.linalg.svd(b_mw @ basis)
    missing = basis @ vt[-1]
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    moves = np.linalg.norm((inv_sqrt_m * missing).reshape(-1, 3), axis=1)
    atoms = np.argsort(-moves, kind="stable")[: min(3, mol.n_atoms)]
    names = ", ".join(f"{a} ({mol.symbols[a]})" for a in atoms)
    count = expected - rank
    return (
        f"internal coordinates span {rank} of {expected} internal directions"
        f"{where}; {count} direction{'s are' if count > 1 else ' is'} missing, "
        f"moving mostly atoms {names}; add coordinates that involve them with "
        "internals="
    )


def _validated(entry: object, n_atoms: int) -> Internal:
    if isinstance(entry, str) or not isinstance(entry, Sequence) or not entry:
        raise ValueError(
            f"internals: expected a (kind, *atom indices) tuple, got {entry!r}"
        )
    kind, *indices = entry
    shown = tuple(entry)
    if not isinstance(kind, str) or kind not in _N_INDICES:
        allowed = ", ".join(repr(k) for k in KINDS)
        raise ValueError(
            f"internals: unknown kind {kind!r} in {shown!r}; expected one of {allowed}"
        )
    if len(indices) != _N_INDICES[kind]:
        raise ValueError(
            f"internals: {shown!r}: a {kind} takes {_N_INDICES[kind]} atom "
            f"indices, got {len(indices)}"
        )
    atoms: list[int] = []
    for i in indices:
        if (
            isinstance(i, bool)
            or not isinstance(i, numbers.Integral)
            or not 0 <= int(i) < n_atoms
        ):
            raise ValueError(
                f"internals: {shown!r}: {i!r} is not an atom index of a "
                f"{n_atoms}-atom molecule"
            )
        atoms.append(int(i))
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"internals: {shown!r} uses an atom more than once")
    return (kind, *atoms)


def _indices(coord: Internal) -> tuple[int, ...]:
    return coord[1:]


def _n_rows(coords: Sequence[Internal]) -> int:
    return sum(2 if c[0] == "linear_bend" else 1 for c in coords)


def _reference(mol: Molecule, reference: Molecule | None) -> FloatArray:
    if reference is None:
        return mol.coords
    if reference.n_atoms != mol.n_atoms:
        raise ValueError(
            f"reference: expected {mol.n_atoms} atoms, got {reference.n_atoms}"
        )
    return reference.coords


def _b_matrix(x: FloatArray, ref: FloatArray, coords: Sequence[Internal]) -> FloatArray:
    rows = [_coordinate(x, ref, c)[1] for c in coords]
    if not rows:
        return np.empty((0, x.size))
    return np.concatenate(rows).reshape(-1, x.size)


def _angle(x: FloatArray, i: int, j: int, k: int) -> float:
    u = x[i] - x[j]
    v = x[k] - x[j]
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def _coordinate(
    x: FloatArray, ref: FloatArray, coord: Internal
) -> tuple[FloatArray, FloatArray]:
    """Values, shape (k,), and B rows, shape (k, N, 3), of one coordinate."""
    kind, *atoms = coord
    b = np.zeros((2 if kind == "linear_bend" else 1, *x.shape))
    if kind == "bond":
        i, j = atoms
        d = x[i] - x[j]
        r = np.linalg.norm(d)
        b[0, i] = d / r
        b[0, j] = -d / r
        return np.array([r]), b
    if kind == "torsion":
        return _torsion(x, b, *atoms)
    i, j, k = atoms
    u = x[i] - x[j]
    v = x[k] - x[j]
    lu = np.linalg.norm(u)
    lv = np.linalg.norm(v)
    eu = u / lu
    ev = v / lv
    if kind == "angle":
        cos = eu @ ev
        sin = np.linalg.norm(np.cross(eu, ev))
        b[0, i] = (cos * eu - ev) / (lu * sin)
        b[0, k] = (cos * ev - eu) / (lv * sin)
        b[0, j] = -b[0, i] - b[0, k]
        return np.array([np.arctan2(sin, cos)]), b
    axes = _bend_axes(ref, i, j, k)
    b[:, i] = (axes - np.outer(axes @ eu, eu)) / lu
    b[:, k] = (axes - np.outer(axes @ ev, ev)) / lv
    b[:, j] = -b[:, i] - b[:, k]
    return axes @ (eu + ev), b


def _torsion(
    x: FloatArray, b: FloatArray, i: int, j: int, k: int, m: int
) -> tuple[FloatArray, FloatArray]:
    # Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996).
    f = x[i] - x[j]
    g = x[j] - x[k]
    h = x[m] - x[k]
    a = np.cross(f, g)
    c = np.cross(h, g)
    lg = np.linalg.norm(g)
    aa = a @ a
    cc = c @ c
    phi = np.arctan2(np.cross(c, a) @ g / lg, a @ c)
    b[0, i] = -lg / aa * a
    b[0, m] = lg / cc * c
    b[0, j] = (lg / aa + (f @ g) / (aa * lg)) * a - (h @ g) / (cc * lg) * c
    b[0, k] = ((h @ g) / (cc * lg) - lg / cc) * c - (f @ g) / (aa * lg) * a
    return np.array([np.pi if phi <= -np.pi else phi]), b


def _bend_axes(ref: FloatArray, i: int, j: int, k: int) -> FloatArray:
    """The two linear-bend axes of i-j-k in the reference geometry, as rows."""
    axis = ref[k] - ref[i]
    axis /= np.linalg.norm(axis)
    bend = sum((ref[a] - ref[j]) / np.linalg.norm(ref[a] - ref[j]) for a in (i, k))
    e1 = bend - (bend @ axis) * axis
    if np.linalg.norm(e1) < _COLLINEAR:
        e1 = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(axis, e1)])
