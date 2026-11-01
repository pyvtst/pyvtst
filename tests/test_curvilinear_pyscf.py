"""OH + H2 -> H2O + H generalized frequencies at UHF/6-31G(d) with PySCF (slow)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("geometric")

from pyvtst import Molecule, Reaction
from pyvtst.core.units import ANGSTROM_TO_BOHR
from pyvtst.engines import PySCFEngine

pytestmark = pytest.mark.slow


def planar(points_angstrom):
    return np.array([[x, y, 0.0] for x, y in points_angstrom]) * ANGSTROM_TO_BOHR


@pytest.fixture(scope="module")
def rxn():
    oh = Molecule(["O", "H"], planar([(0.0, 0.0), (0.97, 0.0)]), mult=2)
    h2 = Molecule(["H", "H"], planar([(0.0, 0.0), (0.74, 0.0)]), symmetry_number=2)
    # O, H_a, then the abstracted H_b and the leaving H_c: O-H_b-H_c is bent.
    guess = Molecule(
        ["O", "H", "H", "H"],
        planar([(0.0, 0.0), (-0.12, 0.96), (1.32, 0.0), (2.13, -0.18)]),
        mult=2,
    )
    return Reaction([oh, h2], guess, PySCFEngine("HF", "6-31G(d)"))


def test_spurious_bends_removed_on_a_curved_path_curvilinear(rxn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartesian = rxn.compute_mep(coordinates="cartesian")
        redundant = rxn.compute_mep()

    assert cartesian.oriented
    far = np.abs(cartesian.s) > 0.3
    lowest_far = min(np.min(f) for f in np.array(cartesian.frequencies)[far])
    assert lowest_far < -100.0

    away = np.flatnonzero(redundant.s != 0.0)
    lowest_away = min(np.min(redundant.frequencies[i]) for i in away)
    assert lowest_away > -50.0
