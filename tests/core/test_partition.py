import numpy as np
import pytest
from scipy import constants as sc

from pyvtst import Molecule
from pyvtst.core.partition import (
    ln_q_total,
    q_elec,
    q_rot,
    q_trans_per_volume,
    q_vib,
    temperatures,
    thermo,
)
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import AMU_TO_KG, BOHR_TO_M, HARTREE_TO_J
from pyvtst.core.vib import principal_axes
from tests.core.hessians import H2_BOND, water

T_GRID = np.array([200.0, 298.15, 1000.0, 2000.0])


def h2(sigma=2):
    return Molecule(
        ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND]], symmetry_number=sigma
    )


def argon():
    return Molecule(["Ar"], [[0.0, 0.0, 0.0]])


# Translational


def test_translational_formula():
    m = argon().total_mass * sc.atomic_mass
    expected = (2 * np.pi * m * sc.k * 300.0 / sc.h**2) ** 1.5
    q = q_trans_per_volume(argon(), 300.0)
    assert q.shape == (1,)
    assert q[0] == pytest.approx(expected, rel=1e-12)


# Rotational


def test_symmetry_number_divides_rotational_partition_function():
    np.testing.assert_allclose(
        q_rot(h2(sigma=2), T_GRID), 0.5 * q_rot(h2(sigma=1), T_GRID), rtol=1e-15
    )


def test_linear_rotor_formula():
    mol = h2(sigma=1)
    mu = 0.5 * mol.masses[0] * AMU_TO_KG
    inertia = mu * (H2_BOND * BOHR_TO_M) ** 2
    theta = sc.h**2 / (8 * np.pi**2 * inertia * sc.k)
    np.testing.assert_allclose(q_rot(mol, T_GRID), T_GRID / theta, rtol=1e-12)


def test_nonlinear_rotor_formula():
    mol = water().replace(symmetry_number=2)
    moments, _ = principal_axes(mol)
    theta = sc.h**2 / (8 * np.pi**2 * moments * AMU_TO_KG * BOHR_TO_M**2 * sc.k)
    expected = np.sqrt(np.pi) / 2 * np.sqrt(T_GRID**3 / np.prod(theta))
    np.testing.assert_allclose(q_rot(mol, T_GRID), expected, rtol=1e-12)


def test_single_atom_rotational_is_one():
    np.testing.assert_array_equal(q_rot(argon(), T_GRID), 1.0)


# Vibrational


def test_vibrational_limits():
    assert q_vib([1000.0], 1.0)[0] == pytest.approx(1.0, abs=1e-12)
    classical = sc.k * 1e6 / (sc.h * sc.c * 1000.0 * 100.0)
    assert q_vib([1000.0], 1e6)[0] == pytest.approx(classical, rel=1e-3)


def test_vibrational_is_product_over_modes():
    np.testing.assert_allclose(
        q_vib([500.0, 1500.0], T_GRID),
        q_vib([500.0], T_GRID) * q_vib([1500.0], T_GRID),
        rtol=1e-14,
    )


def test_vibrational_with_no_modes_is_one():
    np.testing.assert_array_equal(q_vib([], T_GRID), 1.0)


def test_vibrational_rejects_non_positive_frequency():
    with pytest.raises(ValueError, match="> 0"):
        q_vib([0.0, 1000.0], 300.0)


# Electronic


def test_electronic_partition_function():
    doublet = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    np.testing.assert_array_equal(q_elec(doublet, T_GRID), 2.0)


# Temperatures


@pytest.mark.parametrize("bad", [0.0, -5.0, np.nan, np.inf])
def test_non_positive_temperature_is_rejected(bad):
    with pytest.raises(ValueError, match="> 0 K"):
        q_vib([1000.0], [300.0, bad])


def test_invalid_temperature_is_named():
    with pytest.raises(ValueError, match="-5"):
        temperatures([300.0, -5.0])


def test_two_dimensional_temperature_is_rejected():
    with pytest.raises(ValueError, match="1-D"):
        temperatures(np.ones((2, 2)))


# Total


def test_total_is_sum_of_logs():
    sp = StationaryPoint.from_frequencies(
        water().replace(symmetry_number=2, mult=1), -76.0, [1600.0, 3700.0, 3800.0]
    )
    mol = sp.molecule
    expected = (
        q_trans_per_volume(mol, T_GRID)
        * q_rot(mol, T_GRID)
        * q_vib(sp.real_frequencies, T_GRID)
        * q_elec(mol, T_GRID)
    )
    np.testing.assert_allclose(np.exp(ln_q_total(sp, T_GRID)), expected, rtol=1e-12)


# Standard-state thermochemistry


def test_gibbs_energy_uses_one_bar():
    freqs = np.array([1600.0, 3700.0, 3800.0])
    mol = water().replace(symmetry_number=2)
    sp = StationaryPoint.from_frequencies(mol, -76.0, freqs)
    t = 298.15
    kt = sc.k * t  # J

    m = mol.total_mass * sc.atomic_mass
    volume = kt / 1e5
    q_t = (2 * np.pi * m * kt / sc.h**2) ** 1.5 * volume
    s_t = sc.k * (np.log(q_t) + 2.5)
    s_r = sc.k * (np.log(q_rot(mol, t)[0]) + 1.5)
    x = sc.h * sc.c * freqs * 100.0 / kt
    s_v = sc.k * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x)))
    e_v = kt * np.sum(x / np.expm1(x))
    zpe = 0.5 * sc.h * sc.c * np.sum(freqs) * 100.0
    h = zpe + 1.5 * kt + 1.5 * kt + e_v + kt
    g = (h - t * (s_t + s_r + s_v)) / HARTREE_TO_J

    result = thermo(sp, t)
    assert result.g_corr[0] == pytest.approx(g, rel=1e-10)
    assert result.h_corr[0] == pytest.approx(h / HARTREE_TO_J, rel=1e-10)
    assert result.zpe == pytest.approx(zpe / HARTREE_TO_J, rel=1e-10)


def test_thermo_of_an_atom_has_only_translation_and_spin():
    atom = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    sp = StationaryPoint.from_frequencies(atom, -0.5, [])
    result = thermo(sp, T_GRID)
    k_au = sc.k / HARTREE_TO_J
    np.testing.assert_allclose(result.h_corr, 2.5 * k_au * T_GRID, rtol=1e-12)
    assert result.zpe == 0.0


def test_thermo_documents_standard_state():
    assert "p° = 1 bar" in (thermo.__doc__ or "")
