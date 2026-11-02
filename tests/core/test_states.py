import numpy as np
import pytest
import scipy.integrate

from pyvtst import Molecule
from pyvtst.core.partition import ln_q_rot, ln_q_vib
from pyvtst.core.states import DEFAULT_GRAIN, sum_of_states
from pyvtst.core.units import AMU_TO_ME, BOLTZMANN_AU, CM1_TO_HARTREE
from pyvtst.core.vib import principal_axes
from tests.core.test_rates import bent, h3_ts

CM1 = CM1_TO_HARTREE


def rotational_constants(mol):
    """Rotational constants ħ²/2I in hartree, ascending moments."""
    moments, _ = principal_axes(mol)
    return 1.0 / (2.0 * moments * AMU_TO_ME)


def test_rotor_alone():
    mol = bent(sigma=2)
    n = sum_of_states(mol, [], 3000 * CM1)
    eps = DEFAULT_GRAIN * np.arange(n.size)
    abc = np.prod(rotational_constants(mol))
    expected = (4.0 / 3.0) * eps**1.5 / (2 * np.sqrt(abc))
    np.testing.assert_allclose(n, expected, rtol=1e-12, atol=0.0)


def test_one_oscillator_and_a_linear_rotor():
    mol = h3_ts().molecule
    n = sum_of_states(mol, [1000.0], 5000 * CM1)
    eps = DEFAULT_GRAIN * np.arange(n.size)
    b = rotational_constants(mol)[2]
    expected = np.zeros_like(eps)
    for quanta in range(6):
        e_v = quanta * 1000 * CM1
        expected += np.where(eps >= e_v, (eps - e_v) / (2 * b), 0.0)
    assert n.size == 5001
    assert n[0] == 0.0
    np.testing.assert_allclose(n, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("t", [200.0, 1000.0, 5000.0])
def test_consistent_with_the_partition_function(t):
    ts = h3_ts()
    kt = BOLTZMANN_AU * t
    n = sum_of_states(ts.molecule, ts.real_frequencies, 60 * kt)
    eps = DEFAULT_GRAIN * np.arange(n.size)
    laplace = scipy.integrate.trapezoid(n * np.exp(-eps / kt), eps) / kt
    expected = np.exp(ln_q_rot(ts.molecule, t) + ln_q_vib(ts.real_frequencies, t))[0]
    assert laplace == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("freqs", [[900.0, 900.0, 2000.0], [3.0, 900.0, -40.0]])
def test_monotonic_count(freqs):
    n = sum_of_states(h3_ts().molecule, freqs, 8000 * CM1)
    assert np.all(np.isfinite(n))
    assert np.all(n >= 0.0)
    assert np.all(np.diff(n) >= 0.0)


def test_imaginary_frequency_counts_with_its_magnitude():
    mol = h3_ts().molecule
    a = sum_of_states(mol, [-300.0, 900.0], 4000 * CM1)
    b = sum_of_states(mol, [300.0, 900.0], 4000 * CM1)
    np.testing.assert_array_equal(a, b)


def test_coarser_grain_is_accepted_up_to_one_wavenumber():
    mol = h3_ts().molecule
    n = sum_of_states(mol, [1000.0], 2000 * CM1, grain=0.5 * CM1)
    assert n.size == 4001


# Input validation


def test_zero_frequency():
    with pytest.raises(ValueError, match=r"0 cm\^-1"):
        sum_of_states(h3_ts().molecule, [900.0, 0.0], 1000 * CM1)


def test_non_finite_frequency():
    with pytest.raises(ValueError, match="nan"):
        sum_of_states(h3_ts().molecule, [np.nan], 1000 * CM1)


@pytest.mark.parametrize("grain", [0.0, -CM1, np.inf, 2 * CM1])
def test_invalid_grain(grain):
    with pytest.raises(ValueError, match="grain"):
        sum_of_states(h3_ts().molecule, [900.0], 1000 * CM1, grain=grain)


@pytest.mark.parametrize("e_max", [0.0, -1.0, np.nan])
def test_invalid_e_max(e_max):
    with pytest.raises(ValueError, match="e_max"):
        sum_of_states(h3_ts().molecule, [900.0], e_max)


def test_single_atom():
    with pytest.raises(ValueError, match="single atom"):
        sum_of_states(Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2), [], 1000 * CM1)
