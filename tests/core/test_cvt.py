import warnings

import numpy as np
import pytest
from scipy import constants as sc

from pyvtst import Molecule
from pyvtst.core.mep import ReactionPath, generalized_zpe
from pyvtst.core.partition import (
    ln_q_elec,
    ln_q_rot,
    ln_q_total,
    ln_q_trans_per_volume,
    ln_q_vib,
    thermo,
)
from pyvtst.core.rates import (
    _parabola_minimum,
    cvt_rate,
    tst_rate,
    zpe_energy,
)
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import BOLTZMANN_AU, CM1_TO_HARTREE
from pyvtst.core.vib import is_linear
from tests.core.test_rates import h2, h3_ts, h_atom

T_GRID = np.array([200.0, 300.0, 600.0, 1000.0, 2000.0])
S_GRID = np.linspace(-0.5, 0.5, 21)
TS_FREQS = [-1500.0, 900.0, 900.0, 2000.0]
GENERALIZED = [900.0, 900.0, 2000.0]
E_TS = -1.16


def reactants():
    return [h2(), h_atom()]


def make_path(s, energies, frequencies, molecules=None, oriented=True, freq_scale=1.0):
    """A hand-made path; ``frequencies`` holds one array (or None) per point."""
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    mols = (h3_ts().molecule,) * n if molecules is None else tuple(molecules)
    freqs = tuple(None if f is None else np.asarray(f, float) for f in frequencies)
    zpe = [np.nan if f is None else generalized_zpe(f) for f in freqs]
    reacts = reactants()
    return ReactionPath(
        s=s,
        molecules=mols,
        energies=np.asarray(energies, dtype=np.float64),
        gradients=np.zeros((n, mols[0].n_atoms, 3)),
        frequencies=freqs,
        zpe=np.array(zpe),
        reference_energy=sum(r.energy for r in reacts),
        reference_zpe_energy=sum(zpe_energy(r) for r in reacts),
        ts_index=int(np.argmin(np.abs(s))),
        method="lqa",
        ds=0.05,
        stop_reactant="s_limit",
        stop_product="s_limit",
        oriented=oriented,
        coordinates="redundant",
        freq_scale=freq_scale,
    )


def quadratic_path(s0=0.0, a=0.02):
    """Fixed geometry and frequencies; V_MEP = E_TS - a (s - s0)^2."""
    energies = E_TS - a * (S_GRID - s0) ** 2
    return make_path(S_GRID, energies, [GENERALIZED] * S_GRID.size)


def ts_point(energy=E_TS):
    return StationaryPoint.from_frequencies(h3_ts().molecule, energy, TS_FREQS)


# Path builder


def test_builder_v_ag_is_energy_plus_zpe():
    path = quadratic_path()
    reference = sum(zpe_energy(r) for r in reactants())
    zpe = 0.5 * sum(GENERALIZED) * CM1_TO_HARTREE
    np.testing.assert_allclose(path.v_ag, path.energies + zpe - reference, rtol=1e-14)


# Generalized TST


def test_saddle_point_reproduces_conventional_tst():
    path = quadratic_path()
    result = cvt_rate(reactants(), path, T_GRID)
    s, ln_k = result.s, result.ln_k_gt
    tst = tst_rate(reactants(), ts_point(), T_GRID)
    zero = int(np.argmin(np.abs(s)))
    assert s[zero] == 0.0
    np.testing.assert_allclose(np.exp(ln_k[:, zero]), tst.k, rtol=1e-10)


def test_points_without_frequencies_are_skipped():
    freqs = [GENERALIZED if i % 10 == 0 else None for i in range(S_GRID.size)]
    path = make_path(S_GRID, E_TS - 0.02 * S_GRID**2, freqs)
    result = cvt_rate(reactants(), path, T_GRID)
    np.testing.assert_array_equal(result.s, [-0.5, 0.0, 0.5])
    assert result.ln_k_gt.shape == (T_GRID.size, 3)


@pytest.mark.filterwarnings("ignore:CVT. at T")
def test_linearity_follows_the_geometry():
    linear = h3_ts().molecule
    bent = linear.replace(coords=[[-1.75, 0.3, 0.0], [0.0, 0.0, 0.0], [1.75, 0.3, 0.0]])
    assert is_linear(linear)
    assert not is_linear(bent)
    bent_freqs = [500.0, 900.0, 2000.0]
    path = make_path(
        [-0.05, 0.0, 0.05],
        [E_TS - 0.001, E_TS, E_TS - 0.001],
        [bent_freqs, GENERALIZED, GENERALIZED],
        molecules=[bent, linear, linear],
    )
    ln_k = cvt_rate(reactants(), path, T_GRID).ln_k_gt

    reacts = reactants()
    barrier = E_TS - 0.001 + 0.5 * sum(bent_freqs) * CM1_TO_HARTREE
    barrier -= sum(zpe_energy(r) for r in reacts)
    expected = (
        np.log(sc.k * T_GRID / sc.h)
        + ln_q_trans_per_volume(bent, T_GRID)
        + ln_q_rot(bent, T_GRID)
        + ln_q_vib(bent_freqs, T_GRID)
        + ln_q_elec(bent, T_GRID)
        - sum(ln_q_total(r, T_GRID) for r in reacts)
        - barrier / (BOLTZMANN_AU * T_GRID)
        + np.log(1e6)
    )
    np.testing.assert_allclose(ln_k[:, 0], expected, rtol=1e-12)


def test_imaginary_point_is_excluded():
    freqs = [list(GENERALIZED) for _ in S_GRID]
    bad = int(np.argmin(np.abs(S_GRID + 0.3)))
    freqs[bad] = [-200.0, 900.0, 2000.0]
    # Without the exclusion, the spike at s = -0.3 would be the bottleneck.
    energies = E_TS - 0.02 * (S_GRID - 0.137) ** 2
    energies[bad] = E_TS + 0.05
    path = make_path(S_GRID, energies, freqs)
    with pytest.warns(UserWarning, match=r"s = -0\.3000 \(\[-200\.0\] cm\^-1\)"):
        result = cvt_rate(reactants(), path, T_GRID)
    assert not np.any(np.isclose(result.s, -0.3))
    assert not np.any(np.isclose(result.s_star, -0.3, atol=1e-12))
    np.testing.assert_allclose(result.s_star, 0.137, rtol=0.0, atol=1e-10)


@pytest.mark.filterwarnings("ignore:CVT. at T")
def test_small_imaginary_frequency_counts_with_its_magnitude():
    freqs = [[-30.0, 900.0, 2000.0]] * 3
    path = make_path([-0.05, 0.0, 0.05], [E_TS] * 3, freqs)
    ln_k = cvt_rate(reactants(), path, T_GRID).ln_k_gt
    ref = make_path([-0.05, 0.0, 0.05], [E_TS] * 3, [[30.0, 900.0, 2000.0]] * 3)
    ln_ref = cvt_rate(reactants(), ref, T_GRID).ln_k_gt
    np.testing.assert_allclose(ln_k, ln_ref, rtol=1e-14)


@pytest.mark.filterwarnings("ignore:CVT. path points with imaginary")
def test_no_usable_point():
    path = make_path([0.0], [E_TS], [[-500.0, 900.0, 2000.0]])
    with pytest.raises(ValueError, match="no point has usable"):
        cvt_rate(reactants(), path, 300.0)


def test_three_reactants_are_rejected():
    with pytest.raises(ValueError, match="at most two reactants"):
        cvt_rate([h2(), h_atom(), h_atom()], quadratic_path(), 300.0)


# CVT


def test_variational_bound():
    path = quadratic_path(s0=0.137)
    result = cvt_rate(reactants(), path, T_GRID)
    assert np.all(result.k[:, None] <= result.k_gt * (1 + 1e-14))
    zero = int(np.argmin(np.abs(result.s)))
    assert result.s[zero] == 0.0
    assert np.all(result.k <= result.k_gt[:, zero])


def test_exact_optimum_of_a_quadratic_profile():
    result = cvt_rate(reactants(), quadratic_path(s0=0.137), T_GRID)
    np.testing.assert_allclose(result.s_star, 0.137, rtol=0.0, atol=1e-10)
    # The vertex is the conventional TST rate of a saddle with energy E_TS.
    expected = tst_rate(reactants(), ts_point(), T_GRID).k
    np.testing.assert_allclose(result.k, expected, rtol=1e-10)
    assert result.units == "cm^3 molecule^-1 s^-1"


def test_symmetric_profile():
    result = cvt_rate(reactants(), quadratic_path(), T_GRID)
    np.testing.assert_allclose(result.s_star, 0.0, rtol=0.0, atol=1e-10)
    zero = int(np.argmin(np.abs(result.s)))
    np.testing.assert_allclose(result.k, result.k_gt[:, zero], rtol=1e-12)


def test_bottleneck_moves_with_temperature():
    # A soft mode stiffens along s; V_MEP falls so that V_a^G is symmetric
    # about s = 0. Its entropy pushes the bottleneck to s > 0 as T rises.
    soft = 300.0 + 400.0 * S_GRID
    energies = E_TS - 0.02 * S_GRID**2 - 0.5 * 400.0 * S_GRID * CM1_TO_HARTREE
    freqs = [[nu, 900.0, 2000.0] for nu in soft]
    path = make_path(S_GRID, energies, freqs)
    t = np.array([200.0, 2000.0])
    result = cvt_rate(reactants(), path, t)
    assert result.s_star[1] - result.s_star[0] > 0.05
    zero = int(np.argmin(np.abs(result.s)))
    assert result.k[1] / result.k_gt[1, zero] < 1.0


def test_minimum_at_the_end_of_the_path():
    path = quadratic_path(s0=0.8)
    with pytest.warns(UserWarning, match="widen its s range") as record:
        result = cvt_rate(reactants(), path, [300.0, 1000.0])
    messages = [str(w.message) for w in record]
    assert len(messages) == 1
    assert "300, 1000 K" in messages[0]
    assert "product side" in messages[0]
    assert "s = +0.5000" in messages[0]
    np.testing.assert_array_equal(result.s_star, [0.5, 0.5])


def test_end_of_unoriented_path():
    path = make_path(
        S_GRID,
        E_TS - 0.02 * (S_GRID + 0.8) ** 2,
        [GENERALIZED] * S_GRID.size,
        oriented=False,
    )
    with pytest.warns(UserWarning, match=r"s < 0 end \(s = -0\.5000"):
        cvt_rate(reactants(), path, 300.0)


def test_interior_minimum_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cvt_rate(reactants(), quadratic_path(s0=0.137), T_GRID)


def test_result_arrays_are_read_only():
    result = cvt_rate(reactants(), quadratic_path(), T_GRID)
    for a in (result.T, result.k, result.ln_k, result.s_star, result.s):
        assert not a.flags.writeable
    for a in (result.ln_k_gt, result.delta_g, result.delta_g_star):
        assert not a.flags.writeable


# Free energy of activation


def test_delta_g_matches_thermochemistry_at_the_saddle_point():
    result = cvt_rate(reactants(), quadratic_path(), T_GRID)
    zero = int(np.argmin(np.abs(result.s)))
    ts = ts_point()
    g_ts = ts.energy + thermo(ts, T_GRID).g_corr
    g_r = sum(r.energy + thermo(r, T_GRID).g_corr for r in reactants())
    np.testing.assert_allclose(result.delta_g[:, zero], g_ts - g_r, rtol=0, atol=1e-10)


def test_delta_g_is_largest_where_the_rate_is_lowest():
    soft = 300.0 + 400.0 * S_GRID
    path = make_path(
        S_GRID, E_TS - 0.02 * S_GRID**2, [[nu, 900.0, 2000.0] for nu in soft]
    )
    result = cvt_rate(reactants(), path, T_GRID)
    np.testing.assert_array_equal(
        np.argmax(result.delta_g, axis=1), np.argmin(result.ln_k_gt, axis=1)
    )
    assert np.all(result.delta_g_star >= result.delta_g.max(axis=1) - 1e-15)


def test_unimolecular_delta_g_has_no_standard_state_term():
    reactant = StationaryPoint.from_frequencies(
        h3_ts().molecule.replace(name="R"), E_TS - 0.02, [*GENERALIZED, 1e5]
    )
    result = cvt_rate([reactant], quadratic_path(), T_GRID)
    zero = int(np.argmin(np.abs(result.s)))
    kt = BOLTZMANN_AU * T_GRID
    expected = -kt * np.log(result.k_gt[:, zero] * sc.h / (sc.k * T_GRID))
    np.testing.assert_allclose(result.delta_g[:, zero], expected, rtol=1e-12)
    assert result.units == "s^-1"


# Detailed balance


def test_forward_reverse_cvt_ratio():
    soft = 300.0 + 400.0 * S_GRID
    path = make_path(
        S_GRID, E_TS - 0.02 * S_GRID**2, [[nu, 900.0, 2000.0] for nu in soft]
    )
    products = [
        StationaryPoint.from_frequencies(
            Molecule(
                ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.45]], symmetry_number=2
            ),
            -1.18,
            [4300.0],
        ),
        StationaryPoint.from_frequencies(
            Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2), -0.49, []
        ),
    ]
    forward = cvt_rate(reactants(), path, T_GRID)
    reverse = cvt_rate(products, path, T_GRID)
    np.testing.assert_allclose(forward.s_star, reverse.s_star, rtol=0.0, atol=1e-12)

    d_e0 = sum(zpe_energy(p) for p in products) - sum(
        zpe_energy(r) for r in reactants()
    )
    ln_k_eq = (
        sum(ln_q_total(p, T_GRID) for p in products)
        - sum(ln_q_total(r, T_GRID) for r in reactants())
        - d_e0 / (BOLTZMANN_AU * T_GRID)
    )
    # The toy barrier underflows k at 200 K; the ratio is compared as logs.
    np.testing.assert_allclose(
        forward.ln_k - reverse.ln_k, ln_k_eq, rtol=0.0, atol=1e-10
    )


def test_parabola_without_an_interior_minimum_is_rejected():
    x = np.array([0.0, 0.1, 0.2])
    assert _parabola_minimum(x, np.array([1.0, 1.0, 1.0])) is None
    assert _parabola_minimum(x, np.array([1.0, 2.0, 1.0])) is None
    assert _parabola_minimum(x, np.array([3.0, 2.0, 1.5])) is None
    vertex = _parabola_minimum(x, np.array([2.0, 1.0, 2.0]))
    assert vertex == pytest.approx((0.1, 1.0))


# Frequency scale


def test_cvt_reads_scaled_frequencies_as_stored():
    energies = E_TS - 0.02 * S_GRID**2
    scaled = [0.95 * np.asarray(GENERALIZED)] * S_GRID.size
    recorded = make_path(S_GRID, energies, scaled, freq_scale=0.95)
    by_hand = make_path(S_GRID, energies, scaled)
    a = cvt_rate(reactants(), recorded, T_GRID)
    b = cvt_rate(reactants(), by_hand, T_GRID)
    np.testing.assert_array_equal(a.k, b.k)
    np.testing.assert_array_equal(a.s_star, b.s_star)
