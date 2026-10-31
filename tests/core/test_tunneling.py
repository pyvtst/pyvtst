import warnings

import numpy as np
import pytest
from scipy import constants as sc

from pyvtst.core.tunneling import eckart, eckart_probability, wigner
from pyvtst.core.units import KCAL_PER_MOL_TO_HARTREE as KCAL

# Wigner


def test_wigner_formula():
    u = sc.h * sc.c * 1500.0 * 100.0 / (sc.k * 300.0)
    assert wigner(1500.0, 300.0)[0] == pytest.approx(1 + u**2 / 24, rel=1e-12)


def test_wigner_high_temperature_limit():
    assert wigner(1500.0, 1e6)[0] == pytest.approx(1.0, abs=1e-6)


def test_wigner_uses_the_magnitude():
    np.testing.assert_array_equal(wigner(-1500.0, [300.0]), wigner(1500.0, [300.0]))


# Eckart transmission probability


def test_transmission_probability_limits():
    v1, v2 = 10 * KCAL, 4 * KCAL
    e = np.linspace(6 * KCAL, 60 * KCAL, 2000)  # products lie at 6 kcal/mol
    p = eckart_probability(e, v1, v2, 1500.0)
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] < 1e-3
    assert eckart_probability(60 * KCAL, v1, v2, 1500.0)[()] == pytest.approx(1.0)
    assert np.all(np.diff(p) >= -1e-13)


def test_transmission_is_zero_below_the_higher_asymptote():
    p = eckart_probability([0.0, 3 * KCAL], 10 * KCAL, 4 * KCAL, 1500.0)
    np.testing.assert_array_equal(p, 0.0)


def test_symmetric_transmission_is_one_half_at_the_top_for_a_wide_barrier():
    # A parabolic-like top: P(V1) → 1/2 as the barrier becomes wide.
    p = eckart_probability(40 * KCAL, 40 * KCAL, 40 * KCAL, 300.0)[()]
    assert p == pytest.approx(0.5, abs=5e-3)


# Eckart transmission coefficient


def test_symmetric_barrier_without_products():
    t = [300.0, 600.0]
    np.testing.assert_array_equal(
        eckart(10 * KCAL, 10 * KCAL, 1500.0, t),
        eckart(10 * KCAL, 10 * KCAL - 0.0, 1500.0, t),
    )


def test_eckart_high_temperature_limit():
    assert eckart(10 * KCAL, 10 * KCAL, 1500.0, 1e5)[0] == pytest.approx(1.0, abs=1e-2)


def test_eckart_agrees_with_wigner_for_weak_tunneling():
    nu = 500.0
    t = sc.h * sc.c * nu * 100.0 / (sc.k * 0.5)
    kappa_e = eckart(20 * KCAL, 20 * KCAL, nu, t)[0]
    kappa_w = wigner(nu, t)[0]
    assert kappa_e == pytest.approx(kappa_w, rel=1e-2)


def test_eckart_is_monotonic_in_temperature():
    t = np.linspace(200.0, 2000.0, 37)
    kappa = eckart(10 * KCAL, 6 * KCAL, 1500.0, t)
    assert np.all(np.diff(kappa) <= 0)
    assert np.all(kappa >= 1.0)


def test_eckart_is_the_same_in_both_directions():
    t = np.linspace(200.0, 2000.0, 10)
    np.testing.assert_array_equal(
        eckart(12 * KCAL, 3 * KCAL, 1800.0, t), eckart(3 * KCAL, 12 * KCAL, 1800.0, t)
    )


def test_eckart_matches_a_direct_quadrature():
    v1, v2, nu, t = 8 * KCAL, 5 * KCAL, 1200.0, 400.0
    kt = sc.k * t / (sc.physical_constants["Hartree energy"][0])
    e = np.linspace(v1 - v2, v1 + 60 * kt, 400_001)
    integrand = eckart_probability(e, v1, v2, nu) * np.exp((v1 - e) / kt) / kt
    direct = np.trapezoid(integrand, e) + np.exp(-60.0)
    assert eckart(v1, v2, nu, t)[0] == pytest.approx(direct, rel=1e-6)


@pytest.mark.parametrize(("v1", "v2"), [(-1.0, 5.0), (5.0, -0.5), (5.0, 0.0)])
def test_eckart_rejects_non_positive_barriers(v1, v2):
    with pytest.raises(ValueError, match="kcal/mol") as info:
        eckart(v1 * KCAL, v2 * KCAL, 1500.0, 300.0)
    message = str(info.value)
    assert f"{v1:.3f}" in message
    assert f"{v2:.3f}" in message
    assert "positive ZPE-corrected barriers in both directions" in message


def test_eckart_rejects_zero_frequency():
    with pytest.raises(ValueError, match="nu_imag"):
        eckart(5 * KCAL, 5 * KCAL, 0.0, 300.0)


@pytest.mark.parametrize(
    ("v1", "v2", "nu", "t"),
    [
        (50.0, 50.0, 3000.0, 100.0),
        (50.0, 0.5, 3000.0, 100.0),
        (0.1, 0.1, 3000.0, 100.0),
        (50.0, 50.0, 50.0, 5000.0),
        (1.0, 50.0, 3000.0, 5000.0),
    ],
)
def test_low_temperature_high_barrier_is_finite_without_warnings(v1, v2, nu, t):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kappa = eckart(v1 * KCAL, v2 * KCAL, nu, t)
    assert np.all(np.isfinite(kappa))
    assert np.all(kappa > 0)
