import warnings

import numpy as np
import pytest
from scipy import constants as sc

from pyvtst.core.arrhenius import ArrheniusFit, fit_arrhenius
from pyvtst.core.tunneling import eckart
from pyvtst.core.units import GAS_CONSTANT_KCAL, KCAL_PER_MOL_TO_HARTREE

R = GAS_CONSTANT_KCAL
T_GRID = np.linspace(200.0, 2000.0, 10)


def modified_k(T, A=2e13, n=1.7, Ea=20.0):
    return A * (T / 300.0) ** n * np.exp(-Ea / (R * T))


def tunneling_k(T):
    """TST times an Eckart κ: strongly curved at low T, like an SCT k(T)."""
    barrier = 10.0
    kappa = eckart(
        barrier * KCAL_PER_MOL_TO_HARTREE, barrier * KCAL_PER_MOL_TO_HARTREE, 2000.0, T
    )
    return sc.k * T / sc.h * np.exp(-barrier / (R * T)) * kappa


T_CURVED = np.linspace(150.0, 1500.0, 13)


def quiet_fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_arrhenius(*args, **kwargs)


# --- models and recovery -----------------------------------------------------


def test_default_model_is_modified():
    assert fit_arrhenius(T_GRID, modified_k(T_GRID)).model == "modified"


def test_unknown_model():
    with pytest.raises(ValueError, match=r"'arrhenius', 'modified', got 'truhlar'"):
        fit_arrhenius(T_GRID, np.ones(10), "truhlar")


def test_recover_arrhenius():
    k = 1e-11 * np.exp(-5.0 / (R * T_GRID))
    fit = fit_arrhenius(T_GRID, k, "arrhenius", units="cm^3 molecule^-1 s^-1")
    assert (fit.A, fit.Ea_kcal) == pytest.approx((1e-11, 5.0), rel=1e-8)
    assert fit.n == 0.0


def test_recover_modified():
    fit = fit_arrhenius(T_GRID, modified_k(T_GRID), "modified")
    assert (fit.A, fit.n, fit.Ea_kcal) == pytest.approx((2e13, 1.7, 20.0), rel=1e-8)
    assert fit.T_ref == 300.0


def test_modified_never_worse_than_arrhenius():
    k = tunneling_k(T_CURVED)
    arrhenius = quiet_fit(T_CURVED, k, "arrhenius")
    modified = quiet_fit(T_CURVED, k, "modified")
    assert modified.rss <= arrhenius.rss


# --- validation --------------------------------------------------------------


def test_zero_rate_constant():
    k = np.ones(10)
    k[3] = 0.0
    with pytest.raises(ValueError, match=r"k: values must be finite and > 0, got 0"):
        fit_arrhenius(T_GRID, k)


def test_non_positive_temperature():
    T = T_GRID.copy()
    T[0] = -5.0
    with pytest.raises(ValueError, match=r"T: values must be finite and > 0 K, got -5"):
        fit_arrhenius(T, np.ones(10))


def test_non_finite_value():
    k = np.ones(10)
    k[1] = np.nan
    with pytest.raises(ValueError, match="got nan"):
        fit_arrhenius(T_GRID, k)


def test_too_few_temperatures():
    with pytest.raises(
        ValueError, match="'modified' needs at least 3 distinct temperatures, got 2"
    ):
        fit_arrhenius([300.0, 500.0], [1.0, 3.0], "modified")


def test_repeated_temperatures_do_not_count():
    with pytest.raises(ValueError, match="at least 3 distinct temperatures, got 2"):
        fit_arrhenius([300.0, 300.0, 500.0], [1.0, 1.0, 3.0])


def test_two_temperatures_fit_arrhenius_exactly():
    fit = fit_arrhenius([300.0, 500.0], [1.0, 3.0], "arrhenius")
    assert fit([300.0, 500.0]) == pytest.approx([1.0, 3.0], rel=1e-12)


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="5 T and 4 k"):
        fit_arrhenius(T_GRID[:5], np.ones(4))


def test_not_one_dimensional():
    with pytest.raises(ValueError, match="2-D T"):
        fit_arrhenius(np.ones((2, 5)), np.ones(10))


# --- result ------------------------------------------------------------------


def test_evaluate_modified():
    fit = fit_arrhenius(T_GRID, modified_k(T_GRID))
    expected = fit.A * (T_GRID / 300.0) ** fit.n * np.exp(-fit.Ea_kcal / (R * T_GRID))
    assert fit(T_GRID) == pytest.approx(expected, rel=1e-12)
    assert (fit.T_min, fit.T_max) == (200.0, 2000.0)
    assert fit.max_rel_dev < 1e-9


def test_activation_energy_arrhenius_is_constant():
    fit = fit_arrhenius(T_GRID, 1e-11 * np.exp(-5.0 / (R * T_GRID)), "arrhenius")
    assert fit.activation_energy(T_GRID) == pytest.approx(
        np.full(10, fit.Ea_kcal), rel=1e-12
    )


def test_activation_energy_modified():
    fit = fit_arrhenius(T_GRID, modified_k(T_GRID))
    expected = fit.Ea_kcal + fit.n * R * T_GRID
    assert fit.activation_energy(T_GRID) == pytest.approx(expected, rel=1e-10)


def test_activation_energy_matches_finite_difference():
    fit = fit_arrhenius(T_GRID, modified_k(T_GRID))
    t = np.array([250.0, 500.0, 1500.0])
    h = 1e-4 / t  # step in 1/T
    up, down = fit(1.0 / (1.0 / t + h)), fit(1.0 / (1.0 / t - h))
    numeric = -R * (np.log(up) - np.log(down)) / (2 * h)
    assert fit.activation_energy(t) == pytest.approx(numeric, rel=1e-6)


def test_result_is_immutable():
    fit = fit_arrhenius(T_GRID, 1e-11 * np.exp(-5.0 / (R * T_GRID)), "arrhenius")
    with pytest.raises(AttributeError):
        fit.A = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model", "present", "absent"),
    [
        ("arrhenius", ["Ea    ="], ["n     =", "T_ref"]),
        ("modified", ["n     =", "Ea    =", "T_ref = 300 K"], []),
    ],
)
def test_printed_parameters(model, present, absent):
    fit = quiet_fit(T_GRID, modified_k(T_GRID), model, units="s^-1")
    text = str(fit)
    assert f"({model})" in text
    assert "s^-1" in text
    assert "kcal/mol" in text
    assert "200-2000 K" in text
    assert "max deviation" in text
    for item in present:
        assert item in text
    for item in absent:
        assert item not in text


def test_printed_without_units():
    fit = fit_arrhenius(T_GRID, 1e-11 * np.exp(-5.0 / (R * T_GRID)), "arrhenius")
    assert "unit of k" in str(fit)


# --- poor-fit warning --------------------------------------------------------


def test_poor_arrhenius_fit_suggests_modified():
    with pytest.warns(UserWarning, match=r"'arrhenius'.*%.*model='modified'"):
        fit_arrhenius(T_GRID, modified_k(T_GRID, n=3.0), "arrhenius")


def test_poor_modified_fit_suggests_a_narrower_range():
    k = tunneling_k(T_CURVED)
    with pytest.warns(UserWarning, match=r"'modified'.*narrower") as record:
        fit = fit_arrhenius(T_CURVED, k, "modified")
    assert fit.max_rel_dev > 0.05
    assert f"{100 * fit.max_rel_dev:.1f}%" in str(record[0].message)
    assert record[0].filename == __file__


def test_good_fit_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_arrhenius(T_GRID, modified_k(T_GRID))


def test_fit_type():
    assert isinstance(
        fit_arrhenius(T_GRID, np.exp(-1000.0 / T_GRID), "arrhenius"), ArrheniusFit
    )
