import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pyvtst import Molecule
from tests.fakes.leps import BETA, R0, D


def linear_h3(r1, r2):
    return Molecule(
        ["H"] * 3, [[-r1, 0.0, 0.0], [0.0, 0.0, 0.0], [r2, 0.0, 0.0]], mult=2
    )


@pytest.mark.parametrize("r", [1.0, 1.4, 2.0, 4.0])
def test_two_atoms_follow_the_morse_curve(leps_engine, r):
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, r, 0.0]])
    x = np.exp(-BETA * (r - R0))
    assert leps_engine.energy(mol) == pytest.approx(D * (x * x - 2 * x), abs=1e-14)


def test_morse_minimum_is_minus_d(leps_engine):
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [R0, 0.0, 0.0]])
    assert leps_engine.energy(mol) == pytest.approx(-D, abs=1e-14)
    np.testing.assert_allclose(leps_engine.gradient(mol), 0.0, atol=1e-14)


def test_one_atom_has_zero_energy(leps_engine, h_atom):
    assert leps_engine.energy(h_atom) == 0.0
    np.testing.assert_array_equal(leps_engine.gradient(h_atom), np.zeros((1, 3)))


def test_far_atom_reduces_to_h2(leps_engine):
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    assert leps_engine.energy(linear_h3(1.5, 40.0)) == pytest.approx(
        leps_engine.energy(h2), abs=1e-12
    )


def test_two_atom_gradient_matches_morse_derivative(leps_engine):
    r = 1.4
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    x = np.exp(-BETA * (r - R0))
    de_dr = -2.0 * BETA * D * x * (x - 1.0)
    np.testing.assert_allclose(
        leps_engine.gradient(mol), [[-de_dr, 0, 0], [de_dr, 0, 0]], atol=1e-14
    )


@pytest.mark.parametrize("side", [1.6, 40.0])
def test_equal_distances_give_finite_gradient(leps_engine, side):
    # Equilateral H3: all J are equal and the LEPS root has a cusp.
    h = side * np.sqrt(3.0) / 2.0
    mol = Molecule(
        ["H"] * 3, [[0.0, 0.0, 0.0], [side, 0.0, 0.0], [side / 2, h, 0.0]], mult=2
    )
    grad = leps_engine.gradient(mol)
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-14)


def test_non_hydrogen_rejected(leps_engine):
    with pytest.raises(ValueError, match="only H"):
        leps_engine.energy(Molecule(["He"], [[0.0, 0.0, 0.0]]))


def test_barrier_maximum_at_equal_distances(leps_engine):
    # Collinear barrier profile: for each r1, relax r2. The profile peaks
    # where the two distances are equal (the symmetric saddle near 1.73 bohr).
    def relaxed(r1):
        res = minimize_scalar(
            lambda r2: leps_engine.energy(linear_h3(r1, r2)),
            bounds=(1.0, 4.0),
            method="bounded",
            options={"xatol": 1e-8},
        )
        return res.fun, res.x

    r1s = np.linspace(1.5, 2.0, 51)
    profile = [relaxed(r1) for r1 in r1s]
    top = int(np.argmax([e for e, _ in profile]))
    assert 0 < top < len(r1s) - 1
    assert profile[top][1] == pytest.approx(r1s[top], abs=0.02)


def test_level(leps_engine):
    assert leps_engine.level.key() == "leps/none?k=f:0.18"
