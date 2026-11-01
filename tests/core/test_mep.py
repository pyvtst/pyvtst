from typing import Any

import numpy as np
import pytest
import scipy.integrate
import scipy.linalg
import scipy.optimize

from pyvtst import Molecule
from pyvtst.core.mep import (
    MEPPoint,
    ReactionPath,
    euler_step,
    from_mass_scaled,
    lqa_step,
    orient,
    to_mass_scaled,
    trace_branch,
    transition_vector,
)
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from tests.core.hessians import random_rotation
from tests.fakes.leps import R0, LEPSEngine, collinear_h3, saddle_distance

D_MASS = 2.014101778  # amu
H_MASS = 1.00782503223  # amu


def anisotropic_matrix():
    rot = random_rotation(4)
    return rot @ np.diag([0.5, 2.0, 5.0]) @ rot.T


def analytic_arc(a, x0, t):
    """Arc length of x(t) = expm(-A t) x0, the steepest-descent path of ½xᵀAx."""
    lam, u = scipy.linalg.eigh(a)
    c = u.T @ x0

    def speed(tau):
        return np.linalg.norm(lam * c * np.exp(-lam * tau))

    return scipy.integrate.quad(speed, 0.0, t, epsabs=1e-15, epsrel=1e-13)[0]


# Mass-scaled coordinates and the transition vector


def test_mass_scaled_round_trip():
    mol = collinear_h3(1.6, 1.9, masses=[D_MASS, H_MASS, 15.9949])
    coords = np.random.default_rng(1).normal(size=(3, 3))
    back = from_mass_scaled(mol, to_mass_scaled(mol, coords))
    np.testing.assert_allclose(back, coords, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        to_mass_scaled(mol, coords)[:3], np.sqrt(D_MASS) * coords[0], rtol=1e-15
    )


@pytest.fixture(scope="module")
def leps_ts():
    engine = LEPSEngine()
    r = saddle_distance()
    mol = collinear_h3(r, r)
    return StationaryPoint.from_hessian(mol, engine.energy(mol), engine.hessian(mol))


def test_transition_vector_sign_rule(leps_ts):
    v = transition_vector(leps_ts)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-14)
    assert v[int(np.argmax(np.abs(v)))] > 0


def test_transition_vector_ignores_the_sign_of_the_mode(leps_ts):
    modes = np.array(leps_ts.modes)
    modes[0] = -modes[0]
    flipped = StationaryPoint(
        leps_ts.molecule, leps_ts.energy, leps_ts.frequencies, modes
    )
    np.testing.assert_array_equal(
        transition_vector(flipped), transition_vector(leps_ts)
    )


def test_transition_vector_needs_modes_and_a_saddle(leps_ts):
    no_modes = StationaryPoint.from_frequencies(
        leps_ts.molecule, leps_ts.energy, leps_ts.frequencies
    )
    with pytest.raises(ValueError, match="normal modes"):
        transition_vector(no_modes)
    minimum = StationaryPoint.from_frequencies(
        leps_ts.molecule, 0.0, [500.0, 500.0, 1000.0, 2000.0]
    )
    minimum = StationaryPoint(minimum.molecule, 0.0, minimum.frequencies, leps_ts.modes)
    with pytest.raises(ValueError, match="saddle"):
        transition_vector(minimum)


# LQA step


def test_lqa_is_exact_on_a_quadratic_surface():
    a = anisotropic_matrix()
    x = np.array([0.3, -0.2, 0.1])
    dx, at_minimum = lqa_step(a @ x, a, 0.05)
    new = x + dx
    assert not at_minimum

    def miss(t):
        return np.linalg.norm(scipy.linalg.expm(-a * t) @ x - new)

    t = scipy.optimize.minimize_scalar(
        miss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-14}
    ).x
    assert miss(t) < 1e-8
    assert analytic_arc(a, x, t) == pytest.approx(0.05, abs=1e-8)


def test_lqa_step_has_arc_length_ds():
    a = anisotropic_matrix()
    x = np.array([0.3, -0.2, 0.1])
    dx, _ = lqa_step(a @ x, a, 0.05)
    lam, u = scipy.linalg.eigh(a)
    c = u.T @ x
    # Recover t from the slowest-decaying component, then integrate the path.
    t = -np.log((u.T @ (x + dx))[0] / c[0]) / lam[0]
    assert analytic_arc(a, x, t) == pytest.approx(0.05, abs=1e-10)
    assert np.linalg.norm(dx) <= 0.05


def test_lqa_reports_a_model_minimum_closer_than_ds():
    a = anisotropic_matrix()
    x = np.array([1e-3, -2e-3, 5e-4])
    dx, at_minimum = lqa_step(a @ x, a, 0.05)
    assert at_minimum
    np.testing.assert_allclose(x + dx, 0.0, atol=1e-15)


def test_lqa_handles_a_zero_eigenvalue():
    h = np.diag([0.0, 1.0])
    g = np.array([0.1, 0.1])
    dx, at_minimum = lqa_step(g, h, 0.05)
    assert not at_minimum
    # x(t) = (-0.1 t, -0.1 (1 - exp(-t))); recover t from the flat direction.
    t = -dx[0] / 0.1
    assert dx[1] == pytest.approx(-0.1 * (1.0 - np.exp(-t)), rel=1e-12)
    arc = scipy.integrate.quad(
        lambda tau: 0.1 * np.hypot(1.0, np.exp(-tau)), 0.0, t, epsabs=1e-15
    )[0]
    assert arc == pytest.approx(0.05, abs=1e-10)


def test_lqa_step_in_a_subspace():
    a = np.diag([1.0, 3.0, 0.0])
    basis = np.eye(3)[:, :2]
    g = np.array([0.1, 0.2, 0.7])
    dx, _ = lqa_step(g, a, 0.01, basis)
    assert dx[2] == 0.0


def test_lqa_rejects_a_vanishing_gradient():
    with pytest.raises(ValueError, match="vanishes"):
        lqa_step(np.zeros(2), np.eye(2), 0.01)


# Euler step


def test_euler_step():
    g = np.array([0.3, -0.4, 1.2])
    dx = euler_step(g, 0.02)
    assert np.linalg.norm(dx) == pytest.approx(0.02, abs=1e-14)
    np.testing.assert_allclose(dx / np.linalg.norm(dx), -g / np.linalg.norm(g))
    with pytest.raises(ValueError, match="vanishes"):
        euler_step(np.zeros(3), 0.02)


# Branch tracing on analytic surfaces


def double_well(x):
    """V = -x²/2 + x⁴/4 + 2y² + 0.3x²y: saddle at 0, minima near (±1, ∓0.07)."""
    a, b = x
    energy = -0.5 * a**2 + 0.25 * a**4 + 2.0 * b**2 + 0.3 * a**2 * b
    gradient = np.array([-a + a**3 + 0.6 * a * b, 4.0 * b + 0.3 * a**2])
    hessian = np.array([[-1.0 + 3.0 * a**2 + 0.6 * b, 0.6 * a], [0.6 * a, 4.0]])
    return energy, gradient, hessian


class Surface:
    def __init__(self, fn, hessians=True):
        self.fn = fn
        self.hessians = hessians
        self.calls = []

    def __call__(self, index, x):
        self.calls.append(index)
        e, g, h = self.fn(x)
        return MEPPoint(index, np.array(x), e, g, h if self.hessians else None)


def double_well_start():
    e, g, h = double_well(np.zeros(2))
    return MEPPoint(0, np.zeros(2), e, g, h)


@pytest.mark.parametrize("method", ["lqa", "euler"])
def test_points_are_ds_apart_and_downhill(method):
    ds = 0.02
    points, reason = trace_branch(
        double_well_start(),
        np.array([ds, 0.0]),
        Surface(double_well),
        ds=ds,
        n_max=30,
        method=method,
    )
    assert reason == "s_limit"
    assert [p.index for p in points] == list(range(1, 31))
    xs = np.array([np.zeros(2)] + [p.x for p in points])
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(xs, axis=0), axis=1), ds, rtol=0, atol=1e-4
    )
    energies = [double_well_start().energy] + [p.energy for p in points]
    assert np.all(np.diff(energies) < 0)


def test_branch_into_a_shallow_well_stops_at_the_minimum():
    points, reason = trace_branch(
        double_well_start(),
        np.array([0.05, 0.0]),
        Surface(double_well),
        ds=0.05,
        n_max=1000,
    )
    assert reason == "minimum"
    _, g, _ = double_well(points[-1].x)
    assert np.linalg.norm(points[-1].x - [1.0, 0.0]) < 0.15
    assert len(points) < 1000
    assert np.linalg.norm(g) < 0.5


def narrow_valley(x):
    """V = 50 y² - 0.01 x: a 0.1 step across the valley overshoots its floor."""
    a, b = x
    return 50.0 * b**2 - 0.01 * a, np.array([-0.01, 100.0 * b]), None


def test_step_too_large_stops_with_an_energy_rise():
    e0, g0, _ = narrow_valley(np.array([0.0, 0.05]))
    start = MEPPoint(0, np.array([0.0, 0.05]), e0, g0)
    surface = Surface(narrow_valley, hessians=False)
    points, reason = trace_branch(
        start, np.array([0.1, 0.02]), surface, ds=0.1, n_max=10, method="euler"
    )
    assert reason == "energy_rise"
    assert len(points) == 1
    assert points[0].energy < e0
    # The uphill point was computed and discarded.
    assert surface.calls == [1, 2]


def test_first_step_uphill_gives_no_points():
    start = MEPPoint(0, np.zeros(2), -1.0, np.zeros(2))
    points, reason = trace_branch(
        start, np.array([0.01, 0.0]), Surface(double_well), ds=0.01, n_max=5
    )
    assert (points, reason) == ([], "energy_rise")


@pytest.mark.parametrize("method", ["lqa", "euler"])
def test_resume_is_bit_identical(method):
    ds = 0.02
    first = np.array([ds, 0.0])
    full, reason = trace_branch(
        double_well_start(), first, Surface(double_well), ds=ds, n_max=20, method=method
    )
    stored = [MEPPoint(p.index, p.x, p.energy, p.gradient) for p in full[:7]]
    surface = Surface(double_well)
    seen: list[MEPPoint] = []
    resumed, reason2 = trace_branch(
        double_well_start(),
        first,
        surface,
        ds=ds,
        n_max=20,
        method=method,
        existing=stored,
        on_point=seen.append,
    )
    assert reason2 == reason
    assert [p.index for p in seen] == list(range(8, 21))
    for a, b in zip(full, resumed, strict=True):
        np.testing.assert_array_equal(a.x, b.x)
        assert a.energy == b.energy
        np.testing.assert_array_equal(a.gradient, b.gradient)
    # LQA asks for the Hessian of the last stored point; Euler does not.
    expected = ([7] if method == "lqa" else []) + list(range(8, 21))
    assert surface.calls == expected


def test_existing_points_beyond_the_limit_are_ignored():
    ds = 0.02
    full, _ = trace_branch(
        double_well_start(), np.array([ds, 0.0]), Surface(double_well), ds=ds, n_max=10
    )
    surface = Surface(double_well)
    points, reason = trace_branch(
        double_well_start(),
        np.array([ds, 0.0]),
        surface,
        ds=ds,
        n_max=4,
        existing=full,
    )
    assert reason == "s_limit"
    assert len(points) == 4
    assert surface.calls == []


def test_existing_points_must_be_contiguous():
    p = MEPPoint(2, np.zeros(2), 0.0, np.zeros(2))
    with pytest.raises(ValueError, match="expected index 1"):
        trace_branch(
            double_well_start(),
            np.zeros(2),
            Surface(double_well),
            ds=0.01,
            n_max=5,
            existing=[p],
        )


def test_canonical_coordinates_from_compute_are_followed():
    def compute(index, x):
        snapped = np.round(x, 6)
        e, g, h = double_well(snapped)
        return MEPPoint(index, snapped, e, g, h)

    points, _ = trace_branch(
        double_well_start(), np.array([0.02, 0.0]), compute, ds=0.02, n_max=5
    )
    for p in points:
        np.testing.assert_array_equal(p.x, np.round(p.x, 6))


# Orientation


def isotopic_ts(masses):
    engine = LEPSEngine()
    r = saddle_distance()
    mol = collinear_h3(r, r, masses=masses)
    sp = StationaryPoint.from_hessian(mol, engine.energy(mol), engine.hessian(mol))
    return mol, transition_vector(sp)


def d2():
    return Molecule(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [R0, 0.0, 0.0]],
        masses=[D_MASS, D_MASS],
        name="D2",
    )


def h_atom():
    return Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)


def test_asymmetric_isotopic_reaction_is_oriented():
    ts, v = isotopic_ts([D_MASS, D_MASS, H_MASS])
    sign = orient(ts, v, [d2(), h_atom()])
    assert sign in (1, -1)
    # The reactant side is the step along -sign·v.
    reactant = ts.coords - 0.05 * sign * from_mass_scaled(ts, v)
    product = ts.coords + 0.05 * sign * from_mass_scaled(ts, v)

    def dd(c):
        return abs(np.linalg.norm(c[0] - c[1]) - R0)

    assert dd(reactant) < dd(product)


def test_orientation_with_products_agrees():
    ts, v = isotopic_ts([D_MASS, D_MASS, H_MASS])
    dh = Molecule(
        ["H", "H"], [[0.0, 0.0, 0.0], [R0, 0.0, 0.0]], masses=[D_MASS, H_MASS]
    )
    d_atom = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, masses=[D_MASS])
    assert orient(ts, v, [d2(), h_atom()], [d_atom, dh]) == orient(
        ts, v, [d2(), h_atom()]
    )


def test_atom_order_mismatch_is_not_oriented():
    ts, v = isotopic_ts([H_MASS, D_MASS, D_MASS])
    assert orient(ts, v, [d2(), h_atom()]) is None


def test_product_mismatch_is_not_oriented():
    ts, v = isotopic_ts([D_MASS, D_MASS, H_MASS])
    assert orient(ts, v, [d2(), h_atom()], [h_atom(), d2()]) is None


def test_symmetric_case_keeps_the_plus_branch_as_products():
    # Atoms 0 and 1 are the two ends of the symmetric saddle, so the reactant
    # H2 distance changes the same way on both sides.
    r = saddle_distance()
    ts = Molecule(
        ["H", "H", "H"], [[-r, 0.0, 0.0], [r, 0.0, 0.0], [0.0, 0.0, 0.0]], mult=2
    )
    engine = LEPSEngine()
    sp = StationaryPoint.from_hessian(ts, engine.energy(ts), engine.hessian(ts))
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [R0, 0.0, 0.0]])
    v = transition_vector(sp)
    assert orient(ts, v, [h2, h_atom()]) == 1
    assert orient(ts, -v, [h2, h_atom()]) == 1


# ReactionPath


def make_path(**changes):
    mols = tuple(collinear_h3(1.7 + 0.01 * i, 1.7 - 0.01 * i) for i in range(-2, 3))
    fields: dict[str, Any] = {
        "s": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "molecules": mols,
        "energies": [-1.60, -1.59, -1.58, -1.59, -1.61],
        "gradients": np.zeros((5, 3, 3)),
        "frequencies": (
            np.array([1.0, 2.0, 3.0]),
            None,
            np.array([1.0, 2.0, 4.0]),
            None,
            np.array([1.0, 2.0, 5.0]),
        ),
        "zpe": [0.010, np.nan, 0.011, np.nan, 0.012],
        "reference_energy": -1.6,
        "reference_zpe_energy": -1.59,
        "ts_index": 2,
        "method": "euler",
        "ds": 0.01,
        "stop_reactant": "s_limit",
        "stop_product": "s_limit",
        "oriented": True,
    }
    fields.update(changes)
    return ReactionPath(**fields)


def test_read_only_arrays():
    path = make_path()
    for name in (
        "s",
        "energies",
        "gradients",
        "zpe",
        "v_mep",
        "v_ag",
        "has_frequencies",
    ):
        with pytest.raises(ValueError, match="read-only"):
            getattr(path, name)[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        path.frequencies[0][0] = 1.0


def test_relative_energies():
    path = make_path()
    np.testing.assert_allclose(path.v_mep, np.array(path.energies) + 1.6)
    np.testing.assert_allclose(
        path.v_ag[[0, 2, 4]],
        np.array([-1.60 + 0.010, -1.58 + 0.011, -1.61 + 0.012]) + 1.59,
    )
    assert np.isnan(path.v_ag[1])
    np.testing.assert_array_equal(
        path.has_frequencies, [True, False, True, False, True]
    )


def test_table_output():
    path = make_path()
    lines = path.table().splitlines()
    assert len(lines) == 1 + len(path)
    assert "kcal/mol" in lines[0]
    assert float(lines[1].split()[1]) == pytest.approx(0.0, abs=1e-4)
    kcal = 0.02 * HARTREE_TO_KCAL_PER_MOL
    assert float(lines[3].split()[1]) == pytest.approx(kcal, abs=1e-4)


def test_value_equality():
    assert make_path() == make_path()
    assert make_path() != make_path(oriented=False)
    assert make_path() != make_path(energies=[-1.60, -1.59, -1.58, -1.59, -1.62])


def test_inputs_are_copied():
    energies = np.array([-1.60, -1.59, -1.58, -1.59, -1.61])
    path = make_path(energies=energies)
    energies[0] = 0.0
    assert path.energies[0] == -1.60


def test_inconsistent_lengths_are_rejected():
    with pytest.raises(ValueError, match="energies"):
        make_path(energies=[0.0, 1.0])
    with pytest.raises(ValueError, match="ts_index"):
        make_path(ts_index=5)


def test_coordinate_defaults():
    path = make_path()
    assert path.coordinates == "cartesian"
    assert path.internals == ()


def test_coordinates_are_part_of_equality():
    internals = (("bond", 0, 1), ("bond", 1, 2), ("linear_bend", 0, 1, 2))
    redundant = make_path(coordinates="redundant", internals=internals)
    assert redundant == make_path(coordinates="redundant", internals=list(internals))
    assert redundant != make_path(coordinates="redundant", internals=internals[:2])
    assert make_path() != make_path(coordinates="redundant")


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError, match="coordinates"):
        make_path(coordinates="polar")


def test_path_frequency_scale_defaults_to_one():
    assert make_path().freq_scale == 1.0


def test_path_frequency_scale_is_part_of_equality():
    assert make_path() == make_path(freq_scale=1.0)
    assert make_path() != make_path(freq_scale=0.9)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan")])
def test_path_rejects_invalid_frequency_scale(bad):
    with pytest.raises(ValueError, match="freq_scale"):
        make_path(freq_scale=bad)


CURVATURE = (np.array([0.0, 0.1, 0.2]), None, None, None, np.array([0.0, 0.3, 0.4]))


def test_curvature_defaults_to_none():
    path = make_path()
    assert path.curvature is None
    np.testing.assert_array_equal(path.has_curvature, [False] * 5)


def test_curvature_is_frozen_and_masked():
    path = make_path(curvature=CURVATURE)
    np.testing.assert_array_equal(path.has_curvature, [True, False, False, False, True])
    with pytest.raises(ValueError, match="read-only"):
        path.curvature[0][0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        path.has_curvature[0] = False


def test_curvature_is_part_of_equality():
    assert make_path(curvature=CURVATURE) == make_path(curvature=list(CURVATURE))
    assert make_path(curvature=CURVATURE) != make_path()
    other = (*CURVATURE[:4], np.array([0.0, 0.3, 0.5]))
    assert make_path(curvature=CURVATURE) != make_path(curvature=other)


def test_mis_sized_curvature_is_rejected():
    with pytest.raises(ValueError, match="curvature: expected 5 entries"):
        make_path(curvature=CURVATURE[:4])
    with pytest.raises(ValueError, match="point 0 has 2 components, expected 3"):
        make_path(curvature=(np.array([0.1, 0.2]), *CURVATURE[1:]))
    with pytest.raises(ValueError, match="point 1 has 3 components, expected none"):
        make_path(curvature=(CURVATURE[0], np.zeros(3), *CURVATURE[2:]))
