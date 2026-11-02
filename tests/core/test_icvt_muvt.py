import time
import tracemalloc
import warnings

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.partition import ln_q_total
from pyvtst.core.rates import cvt_rate, point_zpe, zpe_energy
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import BOLTZMANN_AU, KCAL_PER_MOL_TO_HARTREE
from pyvtst.core.variational import icvt_rate, muvt_rate, threshold_energy
from tests.core.paths import make_path, reactants

KCAL = KCAL_PER_MOL_TO_HARTREE
REF = sum(zpe_energy(r) for r in reactants())
S_GRID = np.linspace(-0.5, 0.5, 21)
T_GRID = np.array([200.0, 300.0, 600.0, 1000.0, 2000.0])
RIGID = [900.0, 900.0, 2000.0]


def vag_path(s, v_kcal, freqs, **kwargs):
    """A hand-made path with V_a^G(s) = ``v_kcal`` above the reactants."""
    energies = [
        REF + v * KCAL - point_zpe(np.asarray(f, dtype=np.float64))
        for v, f in zip(v_kcal, freqs, strict=True)
    ]
    return make_path(s, energies, freqs, **kwargs)


def rigid_path(s0=0.0, s=S_GRID):
    """Same geometry and frequencies everywhere; V_a^G = 10 - 4 (s - s0)² kcal/mol."""
    return vag_path(s, 10.0 - 4.0 * (s - s0) ** 2, [RIGID] * s.size)


def moving_path():
    """A soft mode stiffens along s while V_a^G falls: the bottleneck moves."""
    soft = 300.0 + 400.0 * S_GRID
    return vag_path(
        S_GRID, 10.0 - 4.0 * S_GRID**2, [[nu, 900.0, 2000.0] for nu in soft]
    )


def products():
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.45]], symmetry_number=2)
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    return [
        StationaryPoint.from_frequencies(h2, -1.18, [4300.0]),
        StationaryPoint.from_frequencies(h, -0.49, []),
    ]


def index_of(s, value):
    return int(np.argmin(np.abs(s - value)))


# Threshold energy


def test_barrier_above_both_asymptotes():
    path = vag_path(S_GRID, 9.8 - 4.0 * S_GRID**2, [RIGID] * S_GRID.size)
    e_th, v_max, s_max = threshold_energy(reactants(), path, products())
    assert e_th == v_max
    assert e_th == pytest.approx(9.8 * KCAL, rel=1e-12)
    assert s_max == 0.0


def test_submerged_barrier():
    v_max = threshold_energy(reactants(), rigid_path())[1]
    h2, h = products()
    # Raise the products' zero-point level to 2 kcal/mol above V^AG.
    shift = v_max + 2.0 * KCAL - (zpe_energy(h2) + zpe_energy(h) - REF)
    lifted = StationaryPoint.from_frequencies(
        h2.molecule, h2.energy + shift, h2.frequencies
    )
    e_th = threshold_energy(reactants(), rigid_path(), [lifted, h])[0]
    assert e_th == pytest.approx(v_max + 2.0 * KCAL, rel=1e-12)


def test_maximum_at_the_path_end():
    with pytest.warns(UserWarning, match=r"product side of the path \(s = \+0\.5000"):
        e_th, _, s_max = threshold_energy(reactants(), rigid_path(s0=0.8))
    assert s_max == 0.5
    assert e_th > 0.0


def test_threshold_is_the_same_in_absolute_energy_both_ways():
    path = moving_path()
    forward = threshold_energy(reactants(), path, products())[0]
    reverse = threshold_energy(products(), path, reactants())[0]
    shift = sum(zpe_energy(p) for p in products()) - REF
    assert forward == pytest.approx(reverse + shift, rel=0.0, abs=1e-14)


# ICVT


def test_not_above_the_generalized_rate():
    path = moving_path()
    icvt = icvt_rate(reactants(), path, T_GRID)
    cvt = cvt_rate(reactants(), path, T_GRID)
    np.testing.assert_array_equal(icvt.s, cvt.s)
    assert np.all(icvt.ln_k_igt <= cvt.ln_k_gt + 1e-12)
    assert np.all((icvt.f >= 0.0) & (icvt.f <= 1.0))


def test_full_fraction_at_the_threshold_point():
    result = icvt_rate(reactants(), moving_path(), T_GRID, products=products())
    top = index_of(result.s, result.s_max)
    assert result.s_max == 0.0
    np.testing.assert_array_equal(result.f[:, top], 1.0)


def test_correction_vanishes_at_high_temperature():
    t = np.array([300.0, 1000.0, 3000.0, 10000.0])
    result = icvt_rate(reactants(), rigid_path(), t)
    point = index_of(result.s, 0.5)  # V^AG - V_a^G = 1 kcal/mol
    missing = 1.0 - result.f[:, point]
    assert np.all(np.diff(missing) < 0.0)
    assert missing[-1] < 1e-2
    assert missing[0] > 0.3


def test_low_temperature_correction():
    # V_a^G falls by 1 kcal/mol within 0.1 bohr while two soft modes stiffen,
    # so the CVT bottleneck sits below the threshold.
    v = 10.0 - np.minimum(100.0 * S_GRID**2, 1.0)
    soft = 30.0 + 570.0 * np.clip(S_GRID / 0.1, 0.0, 1.0)
    path = vag_path(S_GRID, v, [[nu, nu, 2000.0] for nu in soft])
    t = np.array([200.0, 1000.0])
    cvt = cvt_rate(reactants(), path, t)
    icvt = icvt_rate(reactants(), path, t)
    drop = 10.0 - np.interp(cvt.s_star, S_GRID, v)
    assert drop[0] >= 0.3
    assert drop[1] >= 1.0 - 1e-12
    assert icvt.k[0] < 0.9 * cvt.k[0]
    assert icvt.k[1] < cvt.k[1]


def test_icvt_minimum_at_the_end():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = icvt_rate(reactants(), rigid_path(s0=-0.8), 300.0)
    assert any("barrier top is not inside the path" in str(w.message) for w in record)
    messages = [str(w.message) for w in record]
    ends = [m for m in messages if m.startswith("ICVT: at T")]
    assert len(ends) == 1
    assert "300 K" in ends[0]
    assert "reactant side" in ends[0]
    assert "s = -0.5000" in ends[0]
    assert result.s_star[0] == -0.5


def test_icvt_interior_minimum_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        icvt_rate(reactants(), moving_path(), T_GRID)


def test_icvt_result_arrays_are_read_only():
    result = icvt_rate(reactants(), moving_path(), T_GRID)
    for a in (result.T, result.k, result.ln_k, result.s_star, result.s):
        assert not a.flags.writeable
    for a in (result.ln_k_igt, result.f, result.k_igt):
        assert not a.flags.writeable
    assert result.units == "cm^3 molecule^-1 s^-1"


# µVT


@pytest.mark.filterwarnings("ignore:muVT. at T")
def test_not_above_any_icvt_generalized_rate():
    path = moving_path()
    muvt = muvt_rate(reactants(), path, T_GRID)
    icvt = icvt_rate(reactants(), path, T_GRID)
    assert np.all(muvt.ln_k[:, None] <= icvt.ln_k_igt + 1e-12)


def test_rigid_profile_collapses_to_one_point():
    path = rigid_path()
    cvt = cvt_rate(reactants(), path, T_GRID)
    icvt = icvt_rate(reactants(), path, T_GRID)
    muvt = muvt_rate(reactants(), path, T_GRID)
    zero = index_of(cvt.s, 0.0)
    expected = cvt.ln_k_gt[:, zero]
    for ln_k in (cvt.ln_k, icvt.ln_k, muvt.ln_k):
        np.testing.assert_allclose(ln_k, expected, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(muvt.s_star, 0.0, rtol=0.0, atol=1e-10)


def test_bottleneck_moves_with_energy():
    path = moving_path()
    muvt = muvt_rate(reactants(), path, 300.0)
    icvt = icvt_rate(reactants(), path, 300.0)
    high = index_of(muvt.energies, muvt.e_th + 20.0 * KCAL)
    assert muvt.energies[high] - muvt.e_th == pytest.approx(20.0 * KCAL, abs=1e-5)
    assert muvt.s_star[high] - muvt.s_star[0] > 0.05
    assert muvt.k[0] < icvt.k[0]


def test_muvt_bottleneck_beyond_the_path():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = muvt_rate(reactants(), rigid_path(s0=0.8), 1000.0)
    ends = [str(w.message) for w in record if str(w.message).startswith("muVT: at")]
    assert len(ends) == 1
    assert "1000 K" in ends[0]
    assert "product side" in ends[0]
    assert "s = +0.5000" in ends[0]
    assert "kcal/mol above the threshold" in ends[0]
    assert np.isfinite(result.k[0])


@pytest.mark.filterwarnings("ignore:muVT. at T")
def test_muvt_result():
    result = muvt_rate(reactants(), moving_path(), T_GRID)
    assert result.energies[0] == result.e_th
    assert np.all(np.diff(result.energies) > 0.0)
    assert result.s_star.shape == result.n_min.shape == result.energies.shape
    assert np.all(np.diff(result.n_min) >= -1e-9 * result.n_min[1:])
    for a in (result.T, result.k, result.ln_k, result.energies, result.s_star):
        assert not a.flags.writeable
    assert not result.n_min.flags.writeable


# Detailed balance


@pytest.mark.filterwarnings("ignore:muVT. at T")
def test_forward_reverse_ratio():
    path = moving_path()
    reacts, prods = reactants(), products()
    d_e0 = sum(zpe_energy(p) for p in prods) - REF
    ln_k_eq = (
        sum(ln_q_total(p, T_GRID) for p in prods)
        - sum(ln_q_total(r, T_GRID) for r in reacts)
        - d_e0 / (BOLTZMANN_AU * T_GRID)
    )
    for rate in (icvt_rate, muvt_rate):
        forward = rate(reacts, path, T_GRID, products=prods)
        reverse = rate(prods, path, T_GRID, products=reacts)
        np.testing.assert_allclose(
            forward.ln_k - reverse.ln_k, ln_k_eq, rtol=0.0, atol=1e-8
        )
        if rate is icvt_rate:
            np.testing.assert_allclose(
                forward.s_star, reverse.s_star, rtol=0.0, atol=1e-12
            )


# Cost


@pytest.mark.slow
def test_muvt_time_and_memory():
    s = np.linspace(-1.0, 1.0, 100)
    path = vag_path(
        s, 10.0 - 4.0 * s**2, [[300.0 + 400.0 * x, 900.0, 2000.0] for x in s]
    )
    t = np.linspace(200.0, 2000.0, 20)
    tracemalloc.start()
    start = time.perf_counter()
    muvt_rate(reactants(), path, t)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 10.0
    assert peak < 150e6
