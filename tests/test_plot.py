import dataclasses
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pyvtst  # noqa: E402
from pyvtst.plot import UNITS, plot_free_energy, plot_path, plot_paths  # noqa: E402
from tests.fakes.leps import LEPSEngine  # noqa: E402
from tests.test_reaction import mep_reaction  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def lqa():
    return mep_reaction(LEPSEngine()).compute_mep(-0.3, 0.3)


@pytest.fixture(scope="module")
def euler():
    return mep_reaction(LEPSEngine()).compute_mep(
        -0.3, 0.3, method="euler", hessian_every=10
    )


def xdata(line):
    return np.asarray(line.get_xdata(), dtype=float)


def ydata(line):
    return np.asarray(line.get_ydata(), dtype=float)


def offsets(collection):
    return np.asarray(collection.get_offsets(), dtype=float)


def marker_size(collection):
    return float(collection.get_sizes()[0])


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_single_path(lqa):
    ax = plot_path(lqa)
    assert len(ax.lines) == 2
    mep, ag = ax.lines
    k = UNITS["kcal/mol"]
    np.testing.assert_allclose(xdata(mep), lqa.s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ydata(mep), lqa.v_mep * k, rtol=0, atol=1e-12)
    np.testing.assert_allclose(xdata(ag), lqa.s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ydata(ag), lqa.v_ag * k, rtol=0, atol=1e-12)
    assert mep.get_linestyle() == "-"
    assert ag.get_linestyle() == "--"
    assert "kcal/mol" in ax.get_ylabel()
    assert ax.get_xlabel() == "s (bohr)"
    assert ax.get_legend() is None


def test_transition_state_marker(lqa):
    ax = plot_path(lqa)
    (marker,) = ax.collections
    np.testing.assert_allclose(
        offsets(marker),
        [[0.0, lqa.v_mep[lqa.ts_index] * UNITS["kcal/mol"]]],
        rtol=0,
        atol=1e-12,
    )


def test_euler_points_without_a_hessian(euler):
    ax = plot_path(euler)
    ag = ax.lines[1]
    np.testing.assert_array_equal(xdata(ag), euler.s[euler.has_frequencies])
    assert np.all(np.isfinite(ydata(ag)))
    assert len(xdata(ax.lines[0])) == len(euler)


def test_comparing_two_paths(lqa, euler):
    ax = plot_paths([lqa, euler], labels=["LQA", "Euler"])
    assert len(ax.lines) == 4
    colors = [line.get_color() for line in ax.lines]
    assert colors[0] == colors[1]
    assert colors[2] == colors[3]
    assert colors[0] != colors[2]
    np.testing.assert_array_equal(xdata(ax.lines[0]), lqa.s)
    np.testing.assert_array_equal(xdata(ax.lines[2]), euler.s)
    texts = legend_texts(ax)
    assert "LQA V_MEP" in texts
    assert "Euler V_a^G" in texts
    assert len(ax.collections) == 2


def test_overlapping_paths_stay_visible(lqa, euler):
    ax = plot_paths([lqa, euler, lqa])
    widths = [line.get_linewidth() for line in ax.lines]
    base = plt.rcParams["lines.linewidth"]
    assert widths[0] == widths[1] == pytest.approx(2.5 * base)
    assert widths[2] == widths[3] == pytest.approx(1.75 * base)
    assert widths[4] == widths[5] == pytest.approx(base)
    sizes = [marker_size(c) for c in ax.collections]
    assert sizes[0] > sizes[1] > sizes[2]


def test_single_path_uses_the_default_width(lqa):
    ax = plot_path(lqa)
    assert ax.lines[0].get_linewidth() == pytest.approx(plt.rcParams["lines.linewidth"])


def test_adding_to_existing_axes(lqa, euler):
    ax = plot_path(lqa, label="a")
    again = plot_path(euler, ax=ax, label="b")
    assert again is ax
    assert len(ax.lines) == 4
    assert ax.lines[0].get_color() != ax.lines[2].get_color()
    assert {"a V_MEP", "b V_MEP"} <= set(legend_texts(ax))


@pytest.mark.parametrize("units", list(UNITS))
def test_units(lqa, units):
    ax = plot_path(lqa, units=units)
    np.testing.assert_allclose(
        ydata(ax.lines[0]), lqa.v_mep * UNITS[units], rtol=1e-15, atol=0
    )
    assert units in ax.get_ylabel()


def test_single_curve(lqa):
    ax = plot_path(lqa, curves=("v_ag",))
    (line,) = ax.lines
    assert line.get_linestyle() == "--"
    assert offsets(ax.collections[0])[0, 1] == pytest.approx(
        lqa.v_ag[lqa.ts_index] * UNITS["kcal/mol"]
    )
    assert len(plot_path(lqa, curves="v_mep").lines) == 1


def test_path_that_is_not_oriented(lqa, euler):
    unoriented = dataclasses.replace(euler, oriented=False)
    with pytest.warns(UserWarning, match=r"'MP2' is not oriented.*arbitrary"):
        ax = plot_paths([lqa, unoriented], labels=["B3LYP", "MP2"])
    assert len(ax.lines) == 4
    with pytest.warns(UserWarning, match=r"path 1 is not oriented"):
        plot_paths([lqa, unoriented])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"labels": ["only one"]}, "labels"),
        ({"curves": ("v_ag2",)}, "curves"),
        ({"curves": ()}, "curves"),
        ({"units": "K"}, "units"),
    ],
)
def test_invalid_plot_arguments(lqa, euler, kwargs, message):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=message):
        plot_paths([lqa, euler], ax=ax, **kwargs)
    assert len(ax.lines) == 0


def test_empty_paths_are_rejected():
    with pytest.raises(ValueError, match="paths"):
        plot_paths([])


def test_matplotlib_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    path = mep_reaction(LEPSEngine()).compute_mep(-0.02, 0.02)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[plot\]"') as excinfo:
        plot_path(path)
    assert excinfo.value.__cause__ is not None


def test_top_level_names():
    assert pyvtst.plot_path is plot_path
    assert pyvtst.plot_paths is plot_paths
    assert pyvtst.plot_free_energy is plot_free_energy


# Free-energy profiles


@pytest.fixture(scope="module")
def cvt_result():
    return mep_reaction(LEPSEngine()).rates([300.0, 600.0, 1000.0], theory="cvt")


def test_one_curve_per_temperature(cvt_result):
    ax = plot_free_energy(cvt_result)
    lines = ax.get_lines()
    assert [line.get_label() for line in lines] == ["300 K", "600 K", "1000 K"]
    cvt = cvt_result.cvt
    for i, line in enumerate(lines):
        np.testing.assert_array_equal(xdata(line), cvt.s)
        np.testing.assert_allclose(ydata(line), cvt.delta_g[i] * UNITS["kcal/mol"])
    markers = ax.collections
    assert len(markers) == 3
    for i, marker in enumerate(markers):
        x, y = offsets(marker)[0]
        assert x == pytest.approx(cvt.s_star[i])
        assert y == pytest.approx(cvt.delta_g_star[i] * UNITS["kcal/mol"])
        face = np.asarray(marker.get_facecolor())[0, :3]
        assert np.allclose(face, to_rgb(lines[i]))
    assert ax.get_xlabel() == "s (bohr)"
    assert "kcal/mol" in ax.get_ylabel()


def to_rgb(line):
    return matplotlib.colors.to_rgb(line.get_color())


def test_free_energy_from_cvt_details_and_given_axes(cvt_result):
    _, ax = plt.subplots()
    out = plot_free_energy(cvt_result.cvt, ax=ax, units="eV")
    assert out is ax
    np.testing.assert_allclose(
        ydata(ax.get_lines()[0]), cvt_result.cvt.delta_g[0] * UNITS["eV"]
    )


def test_tst_result_is_rejected():
    result = mep_reaction(LEPSEngine()).rates(300.0)
    with pytest.raises(ValueError, match="CVT results are needed"):
        plot_free_energy(result)
    assert not plt.get_fignums()


def test_free_energy_unknown_unit(cvt_result):
    with pytest.raises(ValueError, match="units"):
        plot_free_energy(cvt_result, units="cal")
    assert not plt.get_fignums()


def test_free_energy_without_matplotlib(monkeypatch, cvt_result):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[plot\]"'):
        plot_free_energy(cvt_result)
