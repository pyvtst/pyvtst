"""Synthetic Cartesian Hessians with known physics, for vibrational analysis tests.

Each Hessian is Bᵀ K B for internal coordinates with Wilson B rows that are
invariant under overall translation and rotation, so the external modes are
exact zero modes, as at a true stationary point.
"""

import numpy as np

from pyvtst import Molecule
from pyvtst.core.units import ANGSTROM_TO_BOHR

WATER_COORDS = np.array(
    [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]
)


def stretch_row(coords, i, j):
    """Wilson B row of the distance between atoms i and j."""
    u = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
    row = np.zeros(coords.size)
    row[3 * i : 3 * i + 3] = u
    row[3 * j : 3 * j + 3] = -u
    return row


def spring_hessian(coords, pairs, force_constants):
    """Hessian of harmonic springs at their rest lengths, hartree/bohr²."""
    b = np.array([stretch_row(coords, i, j) for i, j in pairs])
    return b.T @ np.diag(force_constants) @ b


def water():
    return Molecule(["O", "H", "H"], WATER_COORDS, name="water")


def water_hessian():
    return spring_hessian(WATER_COORDS, [(0, 1), (0, 2), (1, 2)], [0.5, 0.5, 0.05])


def linear_triatomic(r1=1.75, r2=1.75, symbols=("H", "H", "H"), mult=2):
    coords = np.array([[-r1, 0.0, 0.0], [0.0, 0.0, 0.0], [r2, 0.0, 0.0]])
    return Molecule(symbols, coords, mult=mult, name="H3")


def linear_saddle_hessian(mol, k_stretch=0.1, k_coupling=0.2, k_bend=0.02):
    """Collinear A-B-C saddle along x: one unbound stretch, two bound bends.

    The stretch block is [[k, c], [c, k]] with |c| > k, so the antisymmetric
    stretch has a negative force constant.
    """
    x = mol.coords
    s1 = stretch_row(x, 0, 1)
    s2 = stretch_row(x, 2, 1)
    r1 = np.linalg.norm(x[0] - x[1])
    r2 = np.linalg.norm(x[2] - x[1])
    bends = []
    for k in (1, 2):  # y and z
        row = np.zeros(9)
        row[k] = 1.0 / r1
        row[3 + k] = -1.0 / r1 - 1.0 / r2
        row[6 + k] = 1.0 / r2
        bends.append(row)
    b = np.array([s1, s2, *bends])
    kmat = np.zeros((4, 4))
    kmat[:2, :2] = [[k_stretch, k_coupling], [k_coupling, k_stretch]]
    kmat[2, 2] = kmat[3, 3] = k_bend
    return b.T @ kmat @ b


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


H2_BOND = 0.7414 * ANGSTROM_TO_BOHR
