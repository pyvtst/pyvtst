import warnings

import numpy as np
import pytest
import scipy.integrate

from pyvtst.core.path_tunneling import (
    _Profile,
    _ThetaTable,
    _ZeroCurvature,
    path_tunneling,
    sct_effective_mass,
)
from pyvtst.core.rates import cvt_rate, point_zpe, zpe_energy
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import (
    AMU_TO_ME,
    BOLTZMANN_AU,
    CM1_TO_HARTREE,
    KCAL_PER_MOL_TO_HARTREE,
)
from tests.core.paths import make_path, reactants

KCAL = KCAL_PER_MOL_TO_HARTREE
MU = AMU_TO_ME
GENERALIZED = [900.0, 900.0, 2000.0]
T_GRID = np.array([200.0, 300.0, 1000.0])
REFERENCE = sum(zpe_energy(r) for r in reactants())


def profile_path(s, v_ag, frequencies=None, curvature=None, oriented=True):
    """A hand-made path whose V_a^G, from the test reactants, is ``v_ag``."""
    s = np.asarray(s, dtype=np.float64)
    freqs = [GENERALIZED] * s.size if frequencies is None else frequencies
    zpe = np.array([point_zpe(np.asarray(f, float)) for f in freqs])
    energies = REFERENCE + np.asarray(v_ag) - zpe
    return make_path(s, energies, freqs, curvature=curvature, oriented=oriented)


def parabola(s, v_top=10.0 * KCAL, nu=1500.0):
    omega = nu * CM1_TO_HARTREE
    return v_top - 0.5 * MU * omega**2 * np.asarray(s) ** 2


def products_at(kcal):
    """H2 + H products whose zero-point level lies ``kcal`` from the reactants'."""
    h2, h = reactants()
    shifted = StationaryPoint.from_frequencies(
        h2.molecule, h2.energy + kcal * KCAL, h2.frequencies
    )
    return [shifted, h]


S_LONG = np.linspace(-1.0, 1.0, 41)


def exact_parabolic_kappa(v_top, omega, kt, e0=0.0):
    """β exp(βV‡) ∫ P(E) exp(-βE) dE from E0 with the parabolic θ(E)."""

    def ln_p(e):
        below = e <= v_top
        theta = np.pi * np.abs(v_top - e) / omega
        return np.where(
            below, -np.logaddexp(0, 2 * theta), -np.logaddexp(0, -2 * theta)
        )

    def f(e):
        return ln_p(e) - (e - v_top) / kt

    hi = 2 * v_top - e0
    grid = np.linspace(e0, hi, 20001)
    c = float(np.max(f(grid)))
    value, _ = scipy.integrate.quad(
        lambda e: float(np.exp(f(e) - c)), e0, hi, epsabs=0, epsrel=1e-12, limit=500
    )
    return float(np.exp(c) * value / kt + np.exp((e0 - v_top) / kt))


# Action integral


def test_parabolic_theta():
    v = parabola(S_LONG)
    omega = 1500.0 * CM1_TO_HARTREE
    table = _ThetaTable(_Profile(S_LONG, v), _ZeroCurvature(), 0.0)
    v_top = 10.0 * KCAL
    for e in np.linspace(0.0, v_top, 13)[:-1]:
        expected = np.pi * (v_top - e) / omega
        assert table._theta(float(e)) == pytest.approx(expected, rel=1e-6)
        assert float(table.theta(np.array(e))) == pytest.approx(expected, rel=1e-6)


def test_wells_contribute_nothing():
    # A barrier with a well on the product side, deeper than E.
    s = np.linspace(-1.0, 1.5, 51)
    v = parabola(s) + np.where(s > 0.9, 8.0 * KCAL * np.sin(np.pi * (s - 0.9)), 0.0)
    profile = _Profile(s, v)
    e = 0.5 * KCAL
    intervals = profile.above(e)
    assert len(intervals) >= 1
    for a, b in intervals:
        assert profile.spline(0.5 * (a + b)) > e


# ZCT


def test_parabolic_barrier_kappa():
    path = profile_path(S_LONG, parabola(S_LONG))
    result = path_tunneling(reactants(), path, T_GRID)
    omega = 1500.0 * CM1_TO_HARTREE
    for t, kappa in zip(T_GRID, result.kappa, strict=True):
        expected = exact_parabolic_kappa(10.0 * KCAL, omega, BOLTZMANN_AU * t)
        assert kappa == pytest.approx(expected, rel=1e-6)
    assert result.method == "zct"
    assert result.e0 == 0.0
    assert result.v_max == pytest.approx(10.0 * KCAL, rel=1e-12)


def test_reference_at_the_saddle_point():
    path = profile_path(S_LONG, parabola(S_LONG))
    result = path_tunneling(reactants(), path, T_GRID)
    np.testing.assert_allclose(result.kappa_cag, 1.0, rtol=0, atol=1e-12)
    assert result.s_max == pytest.approx(0.0, abs=1e-10)


def test_reference_at_the_variational_transition_state():
    path = profile_path(S_LONG, parabola(S_LONG))
    s_star = np.array([0.0, 0.1, -0.25])
    result = path_tunneling(reactants(), path, T_GRID, s_ref=s_star)
    v = parabola(s_star)
    expected = np.exp(-(10.0 * KCAL - v) / (BOLTZMANN_AU * T_GRID))
    np.testing.assert_allclose(result.kappa_cag, expected, rtol=1e-10)
    assert np.all(result.kappa_cag[1:] < 1.0)
    base = path_tunneling(reactants(), path, T_GRID)
    np.testing.assert_allclose(result.kappa, base.kappa * expected, rtol=1e-10)


def test_high_temperature_limit():
    path = profile_path(S_LONG, parabola(S_LONG))
    result = path_tunneling(reactants(), path, 1e5)
    assert result.kappa[0] == pytest.approx(1.0, abs=1e-2)


def test_reported_profile():
    curvature = [[0.0, 0.0, 0.5]] * S_LONG.size
    path = profile_path(S_LONG, parabola(S_LONG), curvature=curvature)
    result = path_tunneling(reactants(), path, T_GRID, "sct")
    np.testing.assert_array_equal(result.s, path.s)
    np.testing.assert_allclose(result.v_ag, path.v_ag, rtol=0, atol=1e-14)
    assert np.all(result.mu_eff > 0)
    assert np.all(result.mu_eff <= 1)
    assert np.all(np.diff(result.energies) > 0)
    assert np.all((result.probability >= 0) & (result.probability <= 1))
    for name in ("T", "kappa", "kappa_cag", "s", "v_ag", "mu_eff", "probability"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(result, name)[0] = 0.0


def test_low_temperature_high_barrier():
    v_top = 50.0 * KCAL
    s = np.linspace(-2.0, 2.0, 81)
    path = profile_path(s, parabola(s, v_top=v_top, nu=3000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = path_tunneling(reactants(), path, 100.0)
    assert np.isfinite(result.kappa[0])
    assert result.kappa[0] > 0


def test_unknown_method():
    path = profile_path(S_LONG, parabola(S_LONG))
    with pytest.raises(ValueError, match="'zct' or 'sct'"):
        path_tunneling(reactants(), path, 300.0, "lct")


def test_reference_outside_the_path():
    path = profile_path(S_LONG, parabola(S_LONG))
    with pytest.raises(ValueError, match="s_ref"):
        path_tunneling(reactants(), path, 300.0, s_ref=1.5)


# E0 and the path ends


def short_path():
    s = np.linspace(-1.0, 0.5, 31)
    v = 10.0 * KCAL - 28.0 * KCAL * s**2  # 3 kcal/mol at s = +0.5
    return profile_path(s, v)


def test_path_too_short():
    with pytest.warns(UserWarning, match="tunneling: the path ends") as record:
        result = path_tunneling(
            reactants(), short_path(), 300.0, products=products_at(-5.0)
        )
    assert result.e0 == pytest.approx(3.0 * KCAL, rel=1e-12)
    messages = [str(w.message) for w in record if "path ends" in str(w.message)]
    assert len(messages) == 1
    assert "product side" in messages[0]
    assert "s = +0.5000" in messages[0]
    assert "3.00 kcal/mol above" in messages[0]


def test_unoriented_end_is_named_by_sign():
    s = np.linspace(-1.0, 0.5, 31)
    path = profile_path(s, 10.0 * KCAL - 28.0 * KCAL * s**2, oriented=False)
    with pytest.warns(UserWarning, match="s > 0 end"):
        path_tunneling(reactants(), path, 300.0)


def test_long_enough_path():
    path = profile_path(S_LONG, parabola(S_LONG))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = path_tunneling(reactants(), path, 300.0, products=products_at(-2.0))
    assert result.e0 == 0.0


def test_products_above_reactants_set_e0():
    path = profile_path(S_LONG, parabola(S_LONG))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = path_tunneling(reactants(), path, 300.0, products=products_at(2.0))
    assert result.e0 == pytest.approx(2.0 * KCAL, rel=1e-12)


@pytest.mark.filterwarnings("ignore:tunneling. the path ends")
def test_barrier_top_outside_the_path():
    s = np.linspace(-1.0, 0.0, 21)
    path = profile_path(s, 10.0 * KCAL + 5.0 * KCAL * s)
    # E0 is raised to the path end, which is the maximum itself, so the
    # warning comes with the error.
    with (
        pytest.warns(UserWarning, match="barrier top is not inside the path"),
        pytest.raises(ValueError, match="not above"),
    ):
        path_tunneling(reactants(), path, 300.0)


def test_no_barrier_above_e0():
    path = profile_path(S_LONG, parabola(S_LONG))
    with pytest.raises(ValueError, match=r"V\^AG = 10\.000 kcal/mol.*E0 = 12\.000"):
        path_tunneling(reactants(), path, 300.0, products=products_at(12.0))


def test_too_few_usable_points():
    path = profile_path([-0.1, 0.0], parabola(np.array([-0.1, 0.0])))
    with pytest.raises(ValueError, match="at least 3"):
        path_tunneling(reactants(), path, 300.0)


# Detailed balance


def soft_path(curvature=None):
    freqs = [[500.0 + 200.0 * x, 900.0, 2000.0] for x in S_LONG]
    return profile_path(S_LONG, parabola(S_LONG), freqs, curvature=curvature)


@pytest.mark.parametrize("method", ["zct", "sct"])
@pytest.mark.parametrize("theory", ["tst", "cvt"])
def test_forward_and_reverse(method, theory):
    curvature = [[0.1 * x, 0.2, 0.4 + 0.3 * x] for x in S_LONG]
    path = soft_path(curvature)
    products = products_at(-3.0)
    s_fwd = s_rev = np.zeros_like(T_GRID)
    if theory == "cvt":
        s_fwd = cvt_rate(reactants(), path, T_GRID).s_star
        s_rev = cvt_rate(products, path, T_GRID).s_star
        assert np.any(s_fwd != 0.0)
    forward = path_tunneling(
        reactants(), path, T_GRID, method, products=products, s_ref=s_fwd
    )
    reverse = path_tunneling(
        products, path, T_GRID, method, products=reactants(), s_ref=s_rev
    )
    np.testing.assert_allclose(forward.kappa, reverse.kappa, rtol=1e-10)


# SCT


def test_zero_curvature():
    zeros = [[0.0, 0.0, 0.0]] * S_LONG.size
    path = soft_path(zeros)
    zct = path_tunneling(reactants(), path, T_GRID, "zct")
    sct = path_tunneling(reactants(), path, T_GRID, "sct")
    np.testing.assert_array_equal(sct.mu_eff, 1.0)
    np.testing.assert_allclose(sct.kappa, zct.kappa, rtol=1e-12)


def test_constant_curvature():
    freqs = np.array(GENERALIZED)
    kappa_m = np.array([0.1, -0.2, 0.6])
    s = np.linspace(-0.5, 0.5, 11)
    mu_eff = sct_effective_mass(s, [freqs] * s.size, [kappa_m] * s.size, s)
    kappa = np.linalg.norm(kappa_m)
    inv_t4 = np.sum((kappa_m / kappa) ** 2 * (MU * freqs * CM1_TO_HARTREE) ** 2)
    a = kappa * inv_t4**-0.25
    np.testing.assert_allclose(mu_eff, np.exp(-2 * a - a * a), rtol=1e-10)
    assert np.all(mu_eff < 1)


def test_corner_cutting_raises_tunneling():
    curvature = [[0.3 * np.cos(x), 0.5, 1.5 * np.exp(-4.0 * x * x)] for x in S_LONG]
    path = soft_path(curvature)
    zct = path_tunneling(reactants(), path, T_GRID, "zct")
    sct = path_tunneling(reactants(), path, T_GRID, "sct")
    assert np.all(sct.mu_eff <= 1)
    assert np.any(sct.mu_eff < 0.9)
    assert np.all(sct.kappa >= zct.kappa)
    assert np.all(sct.kappa > 1.01 * zct.kappa)


def test_sct_bridges_points_without_curvature():
    curvature = [None if abs(x) < 1e-12 else [0.0, 0.2, 0.5] for x in S_LONG]
    path = soft_path(curvature)
    result = path_tunneling(reactants(), path, T_GRID, "sct")
    assert np.all(np.isfinite(result.mu_eff))
    i = int(np.argmin(np.abs(result.s)))
    assert result.mu_eff[i] == pytest.approx(result.mu_eff[i + 1], rel=0.05)


def test_path_without_curvature():
    path = soft_path()
    with pytest.raises(ValueError, match=r"curvature.*compute_mep"):
        path_tunneling(reactants(), path, 300.0, "sct")


def test_effective_mass_needs_two_points():
    with pytest.raises(ValueError, match="two or more"):
        sct_effective_mass([0.0], [GENERALIZED], [[0.1, 0.2, 0.3]], 0.0)


def test_effective_mass_checks_its_inputs():
    with pytest.raises(ValueError, match="one entry per point"):
        sct_effective_mass([0.0, 0.1], [GENERALIZED], [[0.1, 0.2, 0.3]] * 2, 0.0)
    with pytest.raises(ValueError, match="point 1 has 2 components for 3"):
        sct_effective_mass(
            [0.0, 0.1], [GENERALIZED] * 2, [[0.1, 0.2, 0.3], [0.1, 0.2]], 0.0
        )


# A barrier top between path points


S_OFF = 0.013  # bohr, between the points at 0 and 0.05


def test_barrier_top_between_path_points():
    v = parabola(S_LONG - S_OFF)
    profile = _Profile(S_LONG, v)
    assert profile.s_max == pytest.approx(S_OFF, abs=1e-10)
    assert profile.v_max == pytest.approx(10.0 * KCAL, rel=1e-12)
    assert profile.v_max > v.max()


def test_turning_points_between_the_same_path_points():
    # Just below the top both turning points lie between s = 0 and 0.05.
    v_top = 10.0 * KCAL
    omega = 1500.0 * CM1_TO_HARTREE
    table = _ThetaTable(
        _Profile(S_LONG, parabola(S_LONG - S_OFF)), _ZeroCurvature(), 0.0
    )
    e = v_top - 1e-6
    roots = table.profile.spline.solve(e, extrapolate=False)
    assert np.all((roots > 0.0) & (roots < 0.05))
    assert table._theta(e) == pytest.approx(np.pi * (v_top - e) / omega, rel=1e-6)
