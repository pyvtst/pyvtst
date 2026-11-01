import numpy as np
import pytest

from pyvtst.core.mep import generalized_zpe, transition_vector
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import AMU_TO_ME, CM1_TO_HARTREE
from pyvtst.core.vib import generalized_normal_modes, normal_modes
from tests.core.hessians import random_rotation, water, water_hessian
from tests.fakes.leps import LEPSEngine, collinear_h3, saddle_distance


@pytest.fixture(scope="module")
def leps_ts():
    engine = LEPSEngine()
    r = saddle_distance()
    mol = collinear_h3(r, r)
    return StationaryPoint.from_hessian(mol, engine.energy(mol), engine.hessian(mol))


def test_saddle_point_limit(leps_ts):
    freqs, modes = generalized_normal_modes(
        leps_ts.molecule,
        LEPSEngine().hessian(leps_ts.molecule),
        direction=leps_ts.modes[0],
    )
    np.testing.assert_allclose(freqs, leps_ts.frequencies[1:], rtol=0, atol=1e-3)
    assert modes.shape == (3, 9)


def test_vanishing_gradient_falls_back_to_direction(leps_ts):
    mol = leps_ts.molecule
    hessian = LEPSEngine().hessian(mol)
    with_tiny = generalized_normal_modes(
        mol, hessian, gradient=np.full(9, 1e-10), direction=leps_ts.modes[0]
    )
    without = generalized_normal_modes(mol, hessian, direction=leps_ts.modes[0])
    np.testing.assert_array_equal(with_tiny[0], without[0])


def test_collinear_point_has_three_modes():
    engine = LEPSEngine()
    mol = collinear_h3(1.5, 2.1)
    freqs, _ = generalized_normal_modes(
        mol, engine.hessian(mol), gradient=engine.gradient(mol)
    )
    assert freqs.shape == (3,)
    # The two bends are degenerate.
    assert freqs[0] == pytest.approx(freqs[1], abs=1e-3)


def test_known_projected_frequencies():
    # On a quadratic surface the Hessian is the same everywhere; with the
    # gradient along one normal mode, that mode drops out.
    mol = water()
    hessian = water_hessian()
    freqs, modes = normal_modes(mol, hessian)
    sqrt_m = np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    for k in range(3):
        gradient = 0.01 * sqrt_m * modes[k]
        projected, _ = generalized_normal_modes(mol, hessian, gradient=gradient)
        np.testing.assert_allclose(projected, np.delete(freqs, k), rtol=0, atol=1e-6)


def test_rigid_body_invariance():
    mol = water()
    hessian = water_hessian()
    gradient = np.random.default_rng(3).normal(size=9) * 1e-2
    freqs, _ = generalized_normal_modes(mol, hessian, gradient=gradient)

    rot = random_rotation(11)
    big = np.kron(np.eye(3), rot)
    moved = mol.replace(coords=mol.coords @ rot.T + [0.7, -1.2, 2.3])
    freqs_moved, _ = generalized_normal_modes(
        moved, big @ hessian @ big.T, gradient=big @ gradient
    )
    np.testing.assert_allclose(freqs_moved, freqs, rtol=0, atol=1e-6)


def test_bent_geometry_has_3n_minus_7_modes():
    gradient = np.random.default_rng(5).normal(size=9) * 1e-2
    freqs, modes = generalized_normal_modes(water(), water_hessian(), gradient=gradient)
    assert freqs.shape == (2,)
    assert modes.shape == (2, 9)


def test_needs_a_direction_when_the_gradient_vanishes():
    with pytest.raises(ValueError, match="direction"):
        generalized_normal_modes(water(), water_hessian(), gradient=np.zeros(9))
    with pytest.raises(ValueError, match="direction"):
        generalized_normal_modes(water(), water_hessian())


def test_rejects_wrong_gradient_shape():
    with pytest.raises(ValueError, match="gradient"):
        generalized_normal_modes(water(), water_hessian(), gradient=np.ones(6))


def test_transition_vector_matches_the_imaginary_mode(leps_ts):
    v = transition_vector(leps_ts)
    assert abs(v @ leps_ts.modes[0]) == pytest.approx(1.0, abs=1e-12)


# Generalized zero-point energy


def test_large_imaginary_frequency_is_excluded():
    assert generalized_zpe([-200.0, 1000.0, 2000.0]) == pytest.approx(
        0.5 * 3000.0 * CM1_TO_HARTREE, rel=1e-14
    )


def test_small_imaginary_frequency_counts_with_its_magnitude():
    assert generalized_zpe([-20.0, 1000.0]) == pytest.approx(
        0.5 * 1020.0 * CM1_TO_HARTREE, rel=1e-14
    )


def test_threshold_is_configurable():
    assert generalized_zpe([-100.0, 1000.0], imag_threshold=150.0) == pytest.approx(
        0.5 * 1100.0 * CM1_TO_HARTREE, rel=1e-14
    )
