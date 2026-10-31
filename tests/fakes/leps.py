"""Analytic LEPS potential for H + H2, a fast stand-in for a quantum engine.

London-Eyring-Polanyi-Sato surface with Sato parameter k. Each H-H pair has a
singlet Morse curve E1 and an anti-Morse triplet curve E3:

    E1 = D (exp(-2b(r - r0)) - 2 exp(-b(r - r0)))
    E3 = (D/2) (exp(-2b(r - r0)) + 2 exp(-b(r - r0)))
    Q = (E1 (1 + k) + E3 (1 - k)) / 2,   J = (E1 (1 + k) - E3 (1 - k)) / 2
    E = (Q_AB + Q_BC + Q_AC - sqrt(((J_AB - J_BC)^2 + (J_BC - J_AC)^2
         + (J_AC - J_AB)^2) / 2)) / (1 + k)

Parameters: D = 4.7466 eV, b = 1.9420 1/Å, r0 = 0.7416 Å are a Morse fit to
H2 (they agree with De, re and ωe = 4401 cm⁻¹ of ground-state H2 to 0.1%),
and k = 0.18 is the usual Sato parameter for H3 (S. Sato, J. Chem. Phys. 23,
592 (1955)). The surface has a collinear, symmetric saddle at r = 1.73 bohr,
5.3 kcal/mol above H + H2. It is a model for tests, not an accurate H3
surface.
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt

from pyvtst import Molecule
from pyvtst.core.finite_difference import fd_hessian
from pyvtst.core.units import ANGSTROM_TO_BOHR, EV_TO_HARTREE
from pyvtst.ports import LevelOfTheory

FloatArray = npt.NDArray[np.float64]

D = 4.7466 * EV_TO_HARTREE  # hartree
BETA = 1.9420 / ANGSTROM_TO_BOHR  # 1/bohr
R0 = 0.7416 * ANGSTROM_TO_BOHR  # bohr
K = 0.18


def _pair(r: float) -> tuple[float, float, float, float]:
    """Return Q, J and their derivatives dQ/dr, dJ/dr for one pair."""
    a = np.exp(-BETA * (r - R0))
    e1 = D * (a * a - 2.0 * a)
    e3 = 0.5 * D * (a * a + 2.0 * a)
    de1 = -2.0 * BETA * D * a * (a - 1.0)
    de3 = -BETA * D * a * (a + 1.0)
    q = 0.5 * (e1 * (1 + K) + e3 * (1 - K))
    j = 0.5 * (e1 * (1 + K) - e3 * (1 - K))
    dq = 0.5 * (de1 * (1 + K) + de3 * (1 - K))
    dj = 0.5 * (de1 * (1 + K) - de3 * (1 - K))
    return q, j, dq, dj


class LEPSEngine:
    """LEPS H + H2 surface for 1, 2 or 3 hydrogen atoms in any 3D geometry.

    One atom has energy 0 and two atoms follow the Morse curve. Charge and
    multiplicity are ignored. The gradient is analytic; the Hessian is the
    finite difference of the gradient with a 1e-4 bohr step.
    """

    def __init__(self) -> None:
        self._level = LevelOfTheory("leps", "none", (("k", K),))

    @property
    def level(self) -> LevelOfTheory:
        return self._level

    def energy(self, mol: Molecule) -> float:
        return self._energy_gradient(mol)[0]

    def gradient(self, mol: Molecule) -> FloatArray:
        return self._energy_gradient(mol)[1]

    def hessian(self, mol: Molecule) -> FloatArray:
        return fd_hessian(
            lambda x: self.gradient(mol.replace(coords=x)), mol.coords, 1e-4
        )

    def _energy_gradient(self, mol: Molecule) -> tuple[float, FloatArray]:
        if any(s != "H" for s in mol.symbols):
            raise ValueError(f"LEPSEngine: only H atoms, got {mol.symbols}")
        if mol.n_atoms > 3:
            raise ValueError(f"LEPSEngine: at most 3 atoms, got {mol.n_atoms}")
        x = mol.coords
        grad = np.zeros_like(x)
        if mol.n_atoms == 1:
            return 0.0, grad

        pairs = list(itertools.combinations(range(mol.n_atoms), 2))
        units = []
        terms = []
        for p1, p2 in pairs:
            d = x[p1] - x[p2]
            r = float(np.linalg.norm(d))
            units.append(d / r)  # dr/dx_i; dr/dx_j is its negative
            terms.append(_pair(r))

        if mol.n_atoms == 2:
            # Q + J = E1 (1 + k), the singlet Morse curve.
            q0, j0, dq0, dj0 = terms[0]
            grad[0] = (dq0 + dj0) / (1 + K) * units[0]
            grad[1] = -grad[0]
            return (q0 + j0) / (1 + K), grad

        q = np.array([t[0] for t in terms])
        j = np.array([t[1] for t in terms])
        dq = np.array([t[2] for t in terms])
        dj = np.array([t[3] for t in terms])
        w = 0.5 * ((j[0] - j[1]) ** 2 + (j[1] - j[2]) ** 2 + (j[2] - j[0]) ** 2)
        root = np.sqrt(w)
        energy = (q.sum() - root) / (1 + K)
        # dW/dJ_p = 2 J_p - (sum of the other two J). When all J are equal
        # (equilateral, or all atoms far apart) the root has a cusp; take the
        # symmetric choice of a zero contribution there instead of 0/0.
        dw_dj = 3.0 * j - j.sum()
        droot = 0.5 * dw_dj * dj / root if root > 0 else np.zeros(3)
        de_dr = (dq - droot) / (1 + K)
        for (p1, p2), g, u in zip(pairs, de_dr, units, strict=True):
            grad[p1] += g * u
            grad[p2] -= g * u
        return float(energy), grad
