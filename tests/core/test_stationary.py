import warnings

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.stationary import (
    StationaryPoint,
    StationaryPointError,
    check_character,
)
from pyvtst.core.units import CM1_TO_HARTREE
from tests.core.hessians import (
    H2_BOND,
    linear_saddle_hessian,
    linear_triatomic,
    water,
    water_hessian,
)


def h2():
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND]], name="H2")


def bent():
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]]
    )


# Construction


def test_build_from_hessian():
    mol = water()
    sp = StationaryPoint.from_hessian(mol, -76.0, water_hessian())
    assert sp.molecule is mol
    assert sp.energy == -76.0
    assert sp.frequencies.shape == (3,)
    assert sp.modes is not None
    assert sp.modes.shape == (3, 9)


def test_build_from_frequencies_sorts_ascending():
    sp = StationaryPoint.from_frequencies(bent(), -1.0, [3800.0, 1600.0, 3700.0])
    np.testing.assert_array_equal(sp.frequencies, [1600.0, 3700.0, 3800.0])
    assert sp.modes is None


def test_wrong_frequency_count_is_rejected():
    with pytest.raises(ValueError, match="expected 3"):
        StationaryPoint.from_frequencies(bent(), -1.0, [1.0, 2.0, 3.0, 4.0])


def test_frequencies_are_read_only():
    sp = StationaryPoint.from_frequencies(h2(), -1.0, [4400.0])
    with pytest.raises(ValueError, match="read-only"):
        sp.frequencies[0] = 1.0


def test_rejects_non_finite_energy():
    with pytest.raises(ValueError, match="energy"):
        StationaryPoint.from_frequencies(h2(), np.inf, [4400.0])


def test_equal_by_value():
    a = StationaryPoint.from_frequencies(h2(), -1.0, [4400.0])
    b = StationaryPoint.from_frequencies(h2(), -1.0, [4400.0])
    c = StationaryPoint.from_frequencies(h2(), -1.0, [4401.0])
    assert a == b
    assert a != c


# Classification


def test_first_order_saddle_point():
    mol = linear_triatomic()
    sp = StationaryPoint.from_frequencies(mol, -1.6, [-1500.0, 400.0, 400.0, 2000.0])
    assert sp.n_imaginary == 1
    assert sp.kind == "saddle"
    assert sp.imaginary_frequency == 1500.0
    np.testing.assert_array_equal(sp.real_frequencies, [400.0, 400.0, 2000.0])


def test_small_imaginary_frequency_is_noise():
    with pytest.warns(UserWarning, match=r"'water'.*-20\.0"):
        sp = StationaryPoint.from_frequencies(water(), -76.0, [-20.0, 300.0, 1600.0])
    assert sp.n_imaginary == 0
    assert sp.kind == "minimum"
    assert sp.imaginary_frequency is None
    np.testing.assert_array_equal(sp.real_frequencies, [20.0, 300.0, 1600.0])


def test_threshold_is_configurable():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp = StationaryPoint.from_frequencies(
            water(), -76.0, [-20.0, 300.0, 1600.0], imag_threshold=10.0
        )
    assert sp.kind == "saddle"


def test_higher_order_saddle():
    sp = StationaryPoint.from_frequencies(water(), -76.0, [-900.0, -300.0, 1600.0])
    assert sp.kind == "higher-order"
    assert sp.imaginary_frequency is None


def test_linear_saddle_from_hessian():
    mol = linear_triatomic()
    sp = StationaryPoint.from_hessian(mol, -1.6, linear_saddle_hessian(mol))
    assert sp.is_linear
    assert sp.kind == "saddle"
    assert sp.real_frequencies.shape == (3,)


def test_check_character_names_species_and_counts():
    sp = StationaryPoint.from_frequencies(water(), -76.0, [300.0, 1600.0, 3700.0])
    check_character(sp, "minimum", "reactant 1")
    with pytest.raises(
        StationaryPointError,
        match="transition state has 0 imaginary frequencies, expected 1",
    ):
        check_character(sp, "saddle", "transition state")


def test_check_character_lists_imaginary_frequencies():
    sp = StationaryPoint.from_frequencies(water(), -76.0, [-700.0, 1600.0, 3700.0])
    with pytest.raises(StationaryPointError, match=r"1 imaginary.*-700\.0"):
        check_character(sp, "minimum", "reactant 1")


# Zero-point energy


def test_zpe_of_a_diatomic():
    sp = StationaryPoint.from_frequencies(h2(), -1.17, [4400.0])
    assert sp.zpe == pytest.approx(2200.0 * CM1_TO_HARTREE, rel=1e-12)


def test_zpe_excludes_the_imaginary_mode():
    mol = linear_triatomic()
    sp = StationaryPoint.from_frequencies(mol, -1.6, [-1500.0, 400.0, 400.0, 2000.0])
    assert sp.zpe == pytest.approx(1400.0 * CM1_TO_HARTREE, rel=1e-12)


def test_single_atom_has_zero_zpe():
    atom = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    sp = StationaryPoint.from_hessian(atom, -0.5, np.zeros((3, 3)))
    assert sp.frequencies.shape == (0,)
    assert sp.zpe == 0.0
    assert sp.kind == "minimum"


# Frequency scale factor


def test_default_scale_changes_nothing():
    mol = linear_triatomic()
    hessian = linear_saddle_hessian(mol)
    plain = StationaryPoint.from_hessian(mol, -1.6, hessian)
    unit = StationaryPoint.from_hessian(mol, -1.6, hessian, freq_scale=1.0)
    assert plain.freq_scale == unit.freq_scale == 1.0
    np.testing.assert_array_equal(plain.frequencies, unit.frequencies)
    assert plain.modes is not None
    assert unit.modes is not None
    np.testing.assert_array_equal(plain.modes, unit.modes)
    assert plain.zpe == unit.zpe


def test_every_frequency_is_scaled():
    mol = linear_triatomic()
    hessian = linear_saddle_hessian(mol)
    plain = StationaryPoint.from_hessian(mol, -1.6, hessian)
    scaled = StationaryPoint.from_hessian(mol, -1.6, hessian, freq_scale=0.95)
    assert scaled.freq_scale == 0.95
    assert scaled.kind == "saddle"
    np.testing.assert_allclose(scaled.frequencies, 0.95 * plain.frequencies, rtol=1e-10)
    assert plain.imaginary_frequency is not None
    assert scaled.imaginary_frequency == pytest.approx(
        0.95 * plain.imaginary_frequency, rel=1e-10
    )
    assert scaled.modes is not None
    assert plain.modes is not None
    np.testing.assert_array_equal(scaled.modes, plain.modes)
    assert scaled.zpe == pytest.approx(0.95 * plain.zpe, rel=1e-10)


def test_frequencies_given_directly_are_scaled():
    sp = StationaryPoint.from_frequencies(
        bent(), -1.0, [1000.0, 2000.0, 3000.0], freq_scale=0.9
    )
    np.testing.assert_allclose(sp.frequencies, [900.0, 1800.0, 2700.0], rtol=1e-12)
    assert sp.freq_scale == 0.9


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "x"])
def test_invalid_scale_is_rejected(bad):
    with pytest.raises(ValueError, match="freq_scale"):
        StationaryPoint.from_frequencies(bent(), -1.0, [1.0, 2.0, 3.0], freq_scale=bad)
    with pytest.raises(ValueError, match="freq_scale"):
        StationaryPoint(bent(), -1.0, np.array([1.0, 2.0, 3.0]), freq_scale=bad)


def test_invalid_scale_message_gives_the_value():
    with pytest.raises(ValueError, match=r"-1\.0"):
        StationaryPoint.from_frequencies(bent(), -1.0, [1.0, 2.0, 3.0], freq_scale=-1)


def test_scale_is_part_of_equality():
    a = StationaryPoint(bent(), -1.0, np.array([900.0, 1800.0, 2700.0]))
    b = StationaryPoint(bent(), -1.0, np.array([900.0, 1800.0, 2700.0]), freq_scale=0.9)
    assert a != b


def test_threshold_applies_to_scaled_frequencies():
    freqs = [-48.0, 1600.0, 3700.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = StationaryPoint.from_frequencies(bent(), -1.0, freqs)
    scaled = StationaryPoint.from_frequencies(bent(), -1.0, freqs, freq_scale=1.078)
    assert plain.kind == "minimum"
    assert scaled.kind == "saddle"
    assert scaled.imaginary_frequency == pytest.approx(48.0 * 1.078)
