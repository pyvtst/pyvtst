import numpy as np
import pytest
import scipy.linalg

from pyvtst import Molecule
from pyvtst.core.internals import auto_internals, values, wilson_b
from pyvtst.core.mep import generalized_zpe, transition_vector
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import AMU_TO_ME, CM1_TO_HARTREE, HARTREE_TO_CM1
from pyvtst.core.vib import (
    curvilinear_generalized_modes,
    generalized_normal_modes,
    normal_modes,
    path_curvature,
)
from tests.core.hessians import (
    TRIATOMIC_INTERNALS,
    quadratic_triatomic,
    random_rotation,
    water,
    water_hessian,
)
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


# Curvilinear generalized modes


def test_curvilinear_agrees_at_the_saddle_point(leps_ts):
    mol = leps_ts.molecule
    hessian = LEPSEngine().hessian(mol)
    cartesian, _ = generalized_normal_modes(mol, hessian, direction=leps_ts.modes[0])
    curvilinear, modes = curvilinear_generalized_modes(
        mol, hessian, auto_internals(mol), direction=leps_ts.modes[0]
    )
    np.testing.assert_allclose(curvilinear, cartesian, rtol=0, atol=1e-3)
    assert modes.shape == (3, 9)
    np.testing.assert_allclose(modes @ modes.T, np.eye(3), atol=1e-12)


def test_curvilinear_linear_geometry_has_3n_minus_6_modes(leps_ts):
    engine = LEPSEngine()
    mol = collinear_h3(1.5, 2.1)
    freqs, _ = curvilinear_generalized_modes(
        mol,
        engine.hessian(mol),
        auto_internals(leps_ts.molecule),
        gradient=engine.gradient(mol),
        reference=leps_ts.molecule,
    )
    assert freqs.shape == (3,)
    assert freqs[0] == pytest.approx(freqs[1], abs=1e-3)


def test_curvilinear_bent_geometry_has_3n_minus_7_modes():
    gradient, hessian = quadratic_triatomic_point()
    freqs, modes = curvilinear_generalized_modes(
        water(), hessian, TRIATOMIC_INTERNALS, gradient=gradient
    )
    assert freqs.shape == (2,)
    assert modes.shape == (2, 9)


def test_linear_bend_follows_a_bent_geometry(leps_ts):
    # Bonds plus both components of a linear bend from the collinear saddle
    # over-span a bent geometry, which still has 3N-7 modes.
    engine = LEPSEngine()
    ts = leps_ts.molecule
    shift = np.zeros((3, 3))
    shift[0] = [0.1, 0.15, 0.0]
    bent = ts.replace(coords=ts.coords + shift)
    freqs, _ = curvilinear_generalized_modes(
        bent,
        engine.hessian(bent),
        auto_internals(ts),
        gradient=engine.gradient(bent),
        reference=ts,
    )
    assert freqs.shape == (2,)


def test_curvilinear_coverage_error():
    gradient, hessian = quadratic_triatomic_point()
    with pytest.raises(ValueError, match=r"2 of 3 internal directions at s = 0\.1"):
        curvilinear_generalized_modes(
            water(),
            hessian,
            TRIATOMIC_INTERNALS[:2],
            gradient=gradient,
            where=" at s = 0.1",
        )


def test_curvilinear_vanishing_gradient_uses_the_direction(leps_ts):
    mol = leps_ts.molecule
    hessian = LEPSEngine().hessian(mol)
    internals = auto_internals(mol)
    with_tiny = curvilinear_generalized_modes(
        mol, hessian, internals, gradient=np.full(9, 1e-10), direction=leps_ts.modes[0]
    )
    without = curvilinear_generalized_modes(
        mol, hessian, internals, direction=leps_ts.modes[0]
    )
    np.testing.assert_array_equal(with_tiny[0], without[0])
    with pytest.raises(ValueError, match="direction"):
        curvilinear_generalized_modes(mol, hessian, internals)


QUADRATIC_K = np.diag([0.5, 0.5, 0.16])


def quadratic_rest():
    # The first bond is 0.1 bohr longer than at rest: the gradient is along it.
    return values(water(), TRIATOMIC_INTERNALS) - [0.1, 0.0, 0.0]


def quadratic_triatomic_point():
    return quadratic_triatomic(water(), QUADRATIC_K, quadratic_rest())


def test_potential_quadratic_in_internal_coordinates():
    mol = water()
    gradient, hessian = quadratic_triatomic_point()
    curvilinear, _ = curvilinear_generalized_modes(
        mol, hessian, TRIATOMIC_INTERNALS, gradient=gradient
    )
    cartesian, _ = generalized_normal_modes(mol, hessian, gradient=gradient)

    # Wilson GF in unit-metric coordinates y = G^(-1/2) q, with the gradient
    # direction projected out.
    b = wilson_b(mol, TRIATOMIC_INTERNALS)
    inv_m = 1.0 / np.repeat(mol.masses * AMU_TO_ME, 3)
    g_half = np.real(scipy.linalg.sqrtm(b @ (inv_m[:, None] * b.T)))
    d = g_half @ QUADRATIC_K @ (values(mol, TRIATOMIC_INTERNALS) - quadratic_rest())
    d /= np.linalg.norm(d)
    projector = np.eye(3) - np.outer(d, d)
    eigvals = np.linalg.eigvalsh(projector @ g_half @ QUADRATIC_K @ g_half @ projector)
    expected = np.sqrt(np.delete(eigvals, np.argmin(np.abs(eigvals)))) * HARTREE_TO_CM1

    np.testing.assert_allclose(curvilinear, expected, rtol=0, atol=1e-6)
    assert np.max(np.abs(cartesian - expected)) > 1.0


def test_redundancy_does_not_change_the_result_at_a_stationary_point():
    mol = water()
    hessian = water_hessian()
    _, modes = normal_modes(mol, hessian)
    with_hh = (*TRIATOMIC_INTERNALS, ("bond", 1, 2))
    non_redundant, _ = curvilinear_generalized_modes(
        mol, hessian, TRIATOMIC_INTERNALS, direction=modes[2]
    )
    redundant, _ = curvilinear_generalized_modes(
        mol, hessian, with_hh, direction=modes[2]
    )
    np.testing.assert_allclose(redundant, non_redundant, rtol=0, atol=1e-6)

    # Away from it the gradient term depends on the set, but not the count.
    gradient, hessian = quadratic_triatomic_point()
    freqs, _ = curvilinear_generalized_modes(mol, hessian, with_hh, gradient=gradient)
    assert freqs.shape == (2,)


def test_rigid_body_invariance_of_curvilinear_frequencies():
    mol = water()
    gradient, hessian = quadratic_triatomic_point()
    freqs, _ = curvilinear_generalized_modes(
        mol, hessian, TRIATOMIC_INTERNALS, gradient=gradient
    )

    rot = random_rotation(11)
    big = np.kron(np.eye(3), rot)
    moved = mol.replace(coords=mol.coords @ rot.T + [0.7, -1.2, 2.3])
    freqs_moved, _ = curvilinear_generalized_modes(
        moved, big @ hessian @ big.T, TRIATOMIC_INTERNALS, gradient=big @ gradient
    )
    np.testing.assert_allclose(freqs_moved, freqs, rtol=0, atol=1e-6)


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


# Path curvature


def _unit_direction(engine, mol):
    g = engine.gradient(mol).reshape(-1) / np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    return g / np.linalg.norm(g)


def test_path_curvature_matches_finite_difference_of_the_direction():
    engine = LEPSEngine()
    mol = Molecule(
        ["H", "H", "H"], [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.3, 0.0]], mult=2
    )
    gradient = engine.gradient(mol)
    hessian = engine.hessian(mol)
    _, modes = generalized_normal_modes(mol, hessian, gradient)
    kappa = path_curvature(mol, hessian, gradient, modes)
    assert kappa.shape == (modes.shape[0],)

    # Step along the steepest-descent direction in electron-mass-weighted
    # coordinates; dv/ds is the central difference of the unit direction.
    sqrt_m = np.sqrt(np.repeat(mol.masses * AMU_TO_ME, 3))
    v = _unit_direction(engine, mol)
    eps = 1e-3

    def moved(step):
        return mol.replace(coords=mol.coords + (step * v / sqrt_m).reshape(-1, 3))

    dv = _unit_direction(engine, moved(-eps)) - _unit_direction(engine, moved(eps))
    dv /= 2.0 * eps
    expected = modes @ dv * np.sqrt(AMU_TO_ME)
    assert np.linalg.norm(expected) > 0.1
    np.testing.assert_allclose(
        kappa, expected, rtol=0, atol=1e-4 * np.abs(expected).max()
    )


def test_path_curvature_needs_a_gradient(leps_ts):
    mol = leps_ts.molecule
    with pytest.raises(ValueError, match="gradient"):
        path_curvature(mol, LEPSEngine().hessian(mol), np.zeros(9), leps_ts.modes[1:])


def test_path_curvature_checks_mode_shape():
    engine = LEPSEngine()
    mol = collinear_h3(1.5, 2.1)
    with pytest.raises(ValueError, match="modes"):
        path_curvature(mol, engine.hessian(mol), engine.gradient(mol), np.ones((2, 6)))
