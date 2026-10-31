import numpy as np
import pytest
from scipy import constants as sc

from pyvtst import Molecule
from pyvtst.core.units import AMU_TO_KG, BOHR_TO_M, HARTREE_TO_J
from pyvtst.core.vib import (
    external_projector,
    is_linear,
    n_vibrations,
    normal_modes,
    principal_axes,
)
from tests.core.hessians import (
    H2_BOND,
    linear_saddle_hessian,
    linear_triatomic,
    random_rotation,
    spring_hessian,
    water,
    water_hessian,
)

# Harmonic frequencies


def test_diatomic_matches_analytic_frequency():
    k = 0.37  # hartree/bohr²
    mol = Molecule(["H", "F"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.73]])
    freqs, modes = normal_modes(mol, spring_hessian(mol.coords, [(0, 1)], [k]))

    m1, m2 = mol.masses * AMU_TO_KG
    mu = m1 * m2 / (m1 + m2)
    k_si = k * HARTREE_TO_J / BOHR_TO_M**2
    expected = np.sqrt(k_si / mu) / (2 * np.pi * sc.c * 100.0)  # cm⁻¹
    assert freqs.shape == (1,)
    assert freqs[0] == pytest.approx(expected, rel=1e-6)
    assert modes.shape == (1, 6)


def test_rotation_and_translation_invariance():
    mol = water()
    freqs, _ = normal_modes(mol, water_hessian())

    rot = random_rotation(7)
    big = np.kron(np.eye(3), rot)
    moved = mol.replace(coords=mol.coords @ rot.T + [1.3, -2.1, 0.4])
    freqs_moved, _ = normal_modes(moved, big @ water_hessian() @ big.T)

    np.testing.assert_allclose(freqs_moved, freqs, rtol=0, atol=1e-6)


def test_water_has_three_ascending_frequencies():
    freqs, modes = normal_modes(water(), water_hessian())
    assert freqs.shape == (3,)
    assert np.all(np.diff(freqs) >= 0)
    assert np.all(freqs > 0)
    np.testing.assert_allclose(modes @ modes.T, np.eye(3), atol=1e-12)


def test_modes_are_orthogonal_to_external_motion():
    mol = water()
    _, modes = normal_modes(mol, water_hessian())
    p = external_projector(mol)
    np.testing.assert_allclose(modes @ p, modes, atol=1e-12)


def test_linear_transition_state():
    mol = linear_triatomic()
    freqs, modes = normal_modes(mol, linear_saddle_hessian(mol))
    assert freqs.shape == (4,)
    assert modes.shape == (4, 9)
    assert np.sum(freqs < 0) == 1
    assert freqs[0] < 0


def test_single_atom_has_no_modes():
    atom = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    freqs, modes = normal_modes(atom, np.zeros((3, 3)))
    assert freqs.shape == (0,)
    assert modes.shape == (0, 3)


def test_rejects_wrong_hessian_shape():
    with pytest.raises(ValueError, match=r"expected shape \(9, 9\)"):
        normal_modes(water(), np.zeros((6, 6)))


def test_rejects_non_finite_hessian():
    h = water_hessian()
    h[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        normal_modes(water(), h)


# Linearity


def test_collinear_triatomic_within_tolerance_is_linear():
    mol = Molecule(
        ["H", "H", "H"], [[-1.7, 0.0, 0.0], [0.0, 1e-4, 0.0], [1.7, 0.0, 0.0]], mult=2
    )
    assert is_linear(mol)
    assert n_vibrations(mol) == 4


def test_bent_triatomic_is_not_linear():
    theta = np.deg2rad(104.5)
    r = 1.81
    coords = [
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [r * np.cos(theta), r * np.sin(theta), 0.0],
    ]
    mol = Molecule(["O", "H", "H"], coords)
    assert not is_linear(mol)
    assert n_vibrations(mol) == 3


def test_diatomic_is_linear_and_atom_is_not():
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND]])
    atom = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    assert is_linear(h2)
    assert not is_linear(atom)
    assert n_vibrations(atom) == 0


# Projector


def test_projector_is_symmetric_idempotent_with_internal_rank():
    mol = water()
    p = external_projector(mol)
    np.testing.assert_allclose(p, p.T, atol=1e-14)
    np.testing.assert_allclose(p @ p, p, atol=1e-12)
    assert np.trace(p) == pytest.approx(3.0)


def test_projector_removes_extra_direction():
    mol = water()
    extra = np.zeros(9)
    extra[2] = 1.0
    p = external_projector(mol, extra=extra)
    assert np.trace(p) == pytest.approx(2.0)
    np.testing.assert_allclose(p @ (p @ extra), 0.0, atol=1e-12)


def test_principal_moments_of_water_are_ascending_and_positive():
    moments, axes = principal_axes(water())
    assert np.all(np.diff(moments) >= 0)
    assert np.all(moments > 0)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
