import warnings

import numpy as np
import pytest
from scipy import constants as sc

from pyvtst import Molecule
from pyvtst.core.partition import ln_q_total
from pyvtst.core.rates import tst_rate, zpe_energy
from pyvtst.core.stationary import StationaryPoint, StationaryPointError
from pyvtst.core.tunneling import eckart
from pyvtst.core.units import BOLTZMANN_AU, KCAL_PER_MOL_TO_HARTREE
from pyvtst.core.vib import normal_modes
from tests.core.hessians import (
    H2_BOND,
    linear_saddle_hessian,
    linear_triatomic,
    spring_hessian,
    water,
)

T_GRID = np.linspace(200.0, 2000.0, 10)


def bent(sigma=1, name=""):
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]],
        symmetry_number=sigma,
        name=name,
    )


def unimolecular_pair(ts_sigma=1, ts_shift=0.0):
    """Reactant and TS whose partition functions differ only by the lost mode.

    The reactant's extra mode is so stiff that it contributes exactly 1 to
    Q_vib; the TS energy cancels the ZPE difference so ΔE₀‡ = ts_shift.
    """
    reactant = StationaryPoint.from_frequencies(
        bent(name="R"), 0.0, [1600.0, 3700.0, 1e5]
    )
    ts_mol = bent(sigma=ts_sigma, name="TS")
    zpe_ts = StationaryPoint.from_frequencies(
        ts_mol, 0.0, [-1200.0, 1600.0, 3700.0]
    ).zpe
    ts = StationaryPoint.from_frequencies(
        ts_mol, reactant.zpe - zpe_ts + ts_shift, [-1200.0, 1600.0, 3700.0]
    )
    return reactant, ts


def h_atom(name="H"):
    return StationaryPoint.from_frequencies(
        Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, name=name), -0.5, []
    )


def h2(name="H2"):
    mol = Molecule(
        ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND]], symmetry_number=2, name=name
    )
    return StationaryPoint.from_frequencies(mol, -1.17, [4400.0])


def h3_ts(energy=-1.16):
    mol = Molecule(
        ["H", "H", "H"],
        [[-1.75, 0.0, 0.0], [0.0, 0.0, 0.0], [1.75, 0.0, 0.0]],
        mult=2,
        symmetry_number=2,
        name="H3",
    )
    return StationaryPoint.from_frequencies(
        mol, energy, [-1500.0, 900.0, 900.0, 2000.0]
    )


# Rate expression


@pytest.mark.filterwarnings("ignore:the ZPE-corrected barrier")
def test_unimolecular_eyring_limit():
    reactant, ts = unimolecular_pair()
    result = tst_rate([reactant], ts, T_GRID)
    np.testing.assert_allclose(result.k, sc.k * T_GRID / sc.h, rtol=1e-12)
    assert result.units == "s^-1"


@pytest.mark.filterwarnings("ignore:the ZPE-corrected barrier")
def test_symmetry_number_scales_the_rate():
    reactant, ts1 = unimolecular_pair(ts_sigma=1)
    _, ts2 = unimolecular_pair(ts_sigma=2)
    k1 = tst_rate([reactant], ts1, T_GRID).k
    k2 = tst_rate([reactant], ts2, T_GRID).k
    np.testing.assert_allclose(k2, 0.5 * k1, rtol=1e-12)


def test_barrier_dependence():
    delta = 3.0 * KCAL_PER_MOL_TO_HARTREE
    reactant, ts0 = unimolecular_pair(ts_shift=0.01)
    _, ts1 = unimolecular_pair(ts_shift=0.01 + delta)
    k0 = tst_rate([reactant], ts0, T_GRID).k
    k1 = tst_rate([reactant], ts1, T_GRID).k
    np.testing.assert_allclose(
        k1, k0 * np.exp(-delta / (BOLTZMANN_AU * T_GRID)), rtol=1e-12
    )


def test_kappa_multiplies_the_rate():
    reactant, ts = unimolecular_pair(ts_shift=0.01)
    kappa = np.linspace(1.0, 3.0, T_GRID.size)
    np.testing.assert_allclose(
        tst_rate([reactant], ts, T_GRID, kappa=kappa).k,
        kappa * tst_rate([reactant], ts, T_GRID).k,
        rtol=1e-12,
    )


@pytest.mark.parametrize("kappa", [[1.0, 2.0], np.zeros(10), np.full(10, np.nan)])
def test_rejects_bad_kappa(kappa):
    reactant, ts = unimolecular_pair(ts_shift=0.01)
    with pytest.raises(ValueError, match="kappa"):
        tst_rate([reactant], ts, T_GRID, kappa=kappa)


def test_result_fields():
    result = tst_rate([h2(), h_atom()], h3_ts(), [300.0, 600.0])
    np.testing.assert_array_equal(result.T, [300.0, 600.0])
    np.testing.assert_allclose(np.log(result.k), result.ln_k, rtol=1e-14)
    expected_barrier = zpe_energy(h3_ts()) - zpe_energy(h2()) - zpe_energy(h_atom())
    assert result.barrier_zpe == pytest.approx(expected_barrier, rel=1e-14)
    with pytest.raises(ValueError, match="read-only"):
        result.k[0] = 0.0


# Molecularity and units


def test_bimolecular_units():
    assert tst_rate([h2(), h_atom()], h3_ts(), 300.0).units == "cm^3 molecule^-1 s^-1"


def test_bimolecular_conversion_to_cm3():
    t = np.array([300.0, 900.0])
    reactants, ts = [h2(), h_atom()], h3_ts()
    barrier = zpe_energy(ts) - zpe_energy(reactants[0]) - zpe_energy(reactants[1])
    ln_k_si = (
        np.log(sc.k * t / sc.h)
        + ln_q_total(ts, t)
        - ln_q_total(reactants[0], t)
        - ln_q_total(reactants[1], t)
        - barrier / (BOLTZMANN_AU * t)
    )
    np.testing.assert_allclose(
        tst_rate(reactants, ts, t).k, np.exp(ln_k_si) * 1e6, rtol=1e-12
    )


def test_unimolecular_units():
    reactant, ts = unimolecular_pair(ts_shift=0.01)
    assert tst_rate([reactant], ts, 300.0).units == "s^-1"


def test_termolecular_is_rejected():
    with pytest.raises(ValueError, match="at most two reactants"):
        tst_rate([h2(), h_atom(), h_atom()], h3_ts(), 300.0)


def test_no_reactants_is_rejected():
    with pytest.raises(ValueError, match="one or two reactants"):
        tst_rate([], h3_ts(), 300.0)


# Stationary-point character


def test_minimum_passed_as_transition_state():
    reactant, _ = unimolecular_pair()
    minimum = StationaryPoint.from_frequencies(water(), -76.0, [1600.0, 3700.0, 3800.0])
    with pytest.raises(
        StationaryPointError,
        match="transition state 'water' has 0 imaginary frequencies, expected 1",
    ):
        tst_rate([reactant], minimum, 300.0)


def test_saddle_passed_as_reactant():
    _, ts = unimolecular_pair()
    with pytest.raises(StationaryPointError, match=r"reactant 1 'TS' has 1 .*-1200\.0"):
        tst_rate([ts], ts, 300.0)


# Negative barrier


def test_barrierless_after_zpe_correction():
    reactant, ts = unimolecular_pair(ts_shift=-0.2 * KCAL_PER_MOL_TO_HARTREE)
    with pytest.warns(UserWarning, match=r"-0\.20 kcal/mol"):
        result = tst_rate([reactant], ts, T_GRID)
    assert np.all(np.isfinite(result.k) & (result.k > 0))


def test_no_warning_for_a_positive_barrier():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tst_rate([h2(), h_atom()], h3_ts(), 300.0)


# Detailed balance


def _species():
    """Synthetic A + B ⇌ C + D with distinct masses, shapes and spins."""
    a = StationaryPoint.from_frequencies(
        Molecule(["O"], [[0.0, 0.0, 0.0]], mult=3, name="A"), -75.0, []
    )
    b = StationaryPoint.from_frequencies(
        bent(sigma=2, name="B"), -76.3, [1600.0, 3700.0, 3800.0]
    )
    c = StationaryPoint.from_frequencies(
        Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], mult=2, name="C"),
        -75.7,
        [3700.0],
    )
    d = StationaryPoint.from_frequencies(
        Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.85]], mult=2, name="D"),
        -75.6,
        [3600.0],
    )
    ts = StationaryPoint.from_frequencies(
        Molecule(
            ["O", "H", "O", "H"],
            [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [4.2, 0.3, 0.0], [5.0, 1.9, 0.5]],
            mult=3,
            name="TS",
        ),
        -151.27,
        [-1700.0, 300.0, 600.0, 900.0, 3500.0, 3650.0],
    )
    return a, b, c, d, ts


def _equilibrium_ratio(reactants, products, t):
    delta = sum(zpe_energy(p) for p in products) - sum(zpe_energy(r) for r in reactants)
    ln_ratio = (
        sum(ln_q_total(p, t) for p in products)
        - sum(ln_q_total(r, t) for r in reactants)
        - delta / (BOLTZMANN_AU * t)
    )
    return np.exp(ln_ratio)


def test_forward_reverse_ratio_without_tunneling():
    a, b, c, d, ts = _species()
    t = np.linspace(200.0, 2000.0, 19)
    k_f = tst_rate([a, b], ts, t).k
    k_r = tst_rate([c, d], ts, t).k
    np.testing.assert_allclose(
        k_f / k_r, _equilibrium_ratio([a, b], [c, d], t), rtol=1e-10
    )


def test_forward_reverse_ratio_with_eckart():
    a, b, c, d, ts = _species()
    t = np.linspace(200.0, 2000.0, 19)
    v1 = zpe_energy(ts) - zpe_energy(a) - zpe_energy(b)
    v2 = zpe_energy(ts) - zpe_energy(c) - zpe_energy(d)
    assert v1 > 0
    assert v2 > 0
    nu = ts.imaginary_frequency
    assert nu is not None
    k_f = tst_rate([a, b], ts, t, kappa=eckart(v1, v2, nu, t)).k
    k_r = tst_rate([c, d], ts, t, kappa=eckart(v2, v1, nu, t)).k
    np.testing.assert_allclose(
        k_f / k_r, _equilibrium_ratio([a, b], [c, d], t), rtol=1e-10
    )


# Frequency scale factor


def test_scaling_matches_hand_scaled_frequencies():
    f = 1.078
    h2_mol = Molecule(
        ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND]], symmetry_number=2
    )
    h2_hessian = spring_hessian(h2_mol.coords, [(0, 1)], [0.37])
    ts_mol = linear_triatomic().replace(symmetry_number=2)
    ts_hessian = linear_saddle_hessian(ts_mol)
    atom = h_atom()

    scaled = [
        StationaryPoint.from_hessian(h2_mol, -1.17, h2_hessian, freq_scale=f),
        StationaryPoint.from_hessian(ts_mol, -1.16, ts_hessian, freq_scale=f),
    ]
    # The hand-scaled side multiplies the unscaled analysis output itself.
    by_hand = [
        StationaryPoint.from_frequencies(mol, e, f * normal_modes(mol, h)[0])
        for mol, e, h in ((h2_mol, -1.17, h2_hessian), (ts_mol, -1.16, ts_hessian))
    ]
    t = [50.0, 300.0, 1000.0]
    a = tst_rate([scaled[0], atom], scaled[1], t)
    b = tst_rate([by_hand[0], atom], by_hand[1], t)
    np.testing.assert_allclose(a.k, b.k, rtol=1e-12)
    unscaled_ts = StationaryPoint.from_hessian(ts_mol, -1.16, ts_hessian)
    assert unscaled_ts.imaginary_frequency is not None
    assert scaled[1].imaginary_frequency == pytest.approx(
        f * unscaled_ts.imaginary_frequency, rel=1e-12
    )
