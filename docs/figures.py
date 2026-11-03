"""Interactive figures for the introduction to reaction rates.

Run by ``conf.py`` at the start of every build. Each figure is written as an
HTML fragment holding a light and a dark plotly figure; ``_static/figures.js``
draws the one matching the theme and redraws it when the theme changes. All
data are computed here from simple models or from pyvtst itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.integrate
import scipy.linalg
from scipy import constants, special

from pyvtst.core.tunneling import eckart

FloatArray = npt.NDArray[np.float64]

KJ_PER_HARTREE = constants.physical_constants["Hartree energy"][0] * constants.N_A / 1e3
R_KJ = constants.R / 1e3  # kJ/(mol K)
EV_TO_KJ = constants.e * constants.N_A / 1e3
HC_OVER_K = constants.h * constants.c * 100 / constants.k  # cm K

# Categorical slots 1-3, text inks and a sequential blue ramp, each validated
# for its mode's page background.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "text": "#0b0b0b",
        "muted": "#52514e",
        "axis": "#b5b4ae",
        "grid": "#e8e7e3",
        "ramp": ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "axis": "#5c5b57",
        "grid": "#2a2e35",
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _layout(mode: str, height: int = 380, **kwargs: object) -> dict[str, object]:
    """Shared layout; dict-valued ``kwargs`` are merged into the defaults."""
    t = THEMES[mode]
    axis = {
        "color": t["muted"],
        "linecolor": t["axis"],
        "gridcolor": t["grid"],
        "zeroline": False,
        "showline": True,
        "ticks": "outside",
        "tickcolor": t["axis"],
        "title": {"font": {"color": t["text"]}},
    }
    layout: dict[str, object] = {
        "height": height,
        "margin": {"l": 70, "r": 20, "t": 20, "b": 55},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 13, "color": t["text"]},
        "showlegend": False,
        "hovermode": "x unified",
        "hoverlabel": {"font": {"family": FONT}},
        "xaxis": dict(axis),
        "yaxis": dict(axis),
    }
    for key, value in kwargs.items():
        base = layout.get(key)
        if isinstance(value, dict) and isinstance(base, dict):
            merged = {**base, **value}
            if "title" in value and "title" in base:
                merged["title"] = {**base["title"], **value["title"]}
            layout[key] = merged
        else:
            layout[key] = value
    return layout


def _label(
    x: float, y: float, text: str, mode: str, muted: bool = False, **kwargs: object
) -> dict[str, object]:
    """A text annotation without an arrow."""
    t = THEMES[mode]
    return {
        "x": x,
        "y": y,
        "text": text,
        "showarrow": False,
        "font": {"color": t["muted"] if muted else t["text"]},
        **kwargs,
    }


def _rgba(color: str, alpha: float) -> str:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _vline(x: float, color: str) -> dict[str, object]:
    return {
        "type": "line",
        "x0": x,
        "x1": x,
        "y0": 0,
        "y1": 1,
        "yref": "paper",
        "line": {"color": color, "width": 1, "dash": "dash"},
    }


# --- Energy profile ----------------------------------------------------------


def profile(mode: str) -> go.Figure:
    """Energy along the reaction coordinate for an exothermic reaction."""
    t = THEMES[mode]
    s = np.linspace(-4, 4, 400)
    barrier, reaction = 50.0, -30.0  # kJ/mol
    # Asymmetric Eckart shape: V(s) = A y/(1+y) + B y/(1+y)², y = exp(2s).
    b = (2 * barrier - reaction) + 2 * np.sqrt(barrier * (barrier - reaction))
    y = np.exp(2 * s)
    v = reaction * y / (1 + y) + b * y / (1 + y) ** 2
    i = int(np.argmax(v))
    x_b, x_r = float(s[i]) - 1.1, 3.4

    def arrow(x: float, y1: float) -> dict[str, object]:
        return {
            "x": x,
            "y": y1,
            "ax": x,
            "ay": 0,
            "axref": "x",
            "ayref": "y",
            "text": "",
            "showarrow": True,
            "arrowhead": 2,
            "arrowside": "end+start",
            "arrowwidth": 1.2,
            "arrowcolor": t["muted"],
        }

    def level(e: float) -> dict[str, object]:
        return {
            "type": "line",
            "x0": -4,
            "x1": 4.8,
            "y0": e,
            "y1": e,
            "layer": "below",
            "line": {"color": t["axis"], "width": 1, "dash": "dot"},
        }

    curve = go.Scatter(
        x=s,
        y=v,
        mode="lines",
        line={"color": t["series"][0], "width": 2.5},
        hovertemplate="%{y:.1f} kJ/mol<extra></extra>",
    )
    top = go.Scatter(
        x=[s[i]],
        y=[v[i]],
        mode="markers",
        marker={"color": t["series"][0], "size": 10},
        hoverinfo="skip",
    )
    layout = _layout(
        mode,
        xaxis={
            "title": {"text": "Reaction coordinate"},
            "showticklabels": False,
            "ticks": "",
            "showgrid": False,
            "range": [-4, 4.8],
        },
        yaxis={
            "title": {"text": "Energy (kJ/mol)"},
            "showgrid": False,
            "range": [reaction - 10, barrier + 12],
        },
        shapes=[level(0), level(reaction)],
        annotations=[
            _label(-3.4, 4, "Reactants", mode),
            _label(2.6, reaction - 5, "Products", mode),
            _label(s[i] + 0.3, v[i] + 4, "Transition state", mode, xanchor="left"),
            arrow(x_b, float(v[i])),
            _label(x_b - 0.15, v[i] / 2, "Barrier", mode, True, xanchor="right"),
            arrow(x_r, reaction),
            _label(
                x_r + 0.15,
                reaction / 2,
                "Reaction<br>energy",
                mode,
                True,
                xanchor="left",
                align="left",
            ),
        ],
    )
    return go.Figure([curve, top], layout)


# --- Potential energy surface of collinear H + H2 ----------------------------

# LEPS surface with the Sato parameter, for H-H-H on a line (Å, eV).
_D, _BETA, _RE, _K = 4.7466, 1.9420, 0.7419, 0.180


def _qj(r: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = np.exp(-_BETA * (r - _RE))
    c = _D / (4 * (1 + _K))
    q = c * ((3 + _K) * x * x - (2 + 6 * _K) * x)
    j = c * ((1 + 3 * _K) * x * x - (6 + 2 * _K) * x)
    return q, j


def leps(r1: npt.ArrayLike, r2: npt.ArrayLike) -> FloatArray:
    """LEPS energy in eV relative to H + H2 at equilibrium."""
    a, b = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    q1, j1 = _qj(a)
    q2, j2 = _qj(b)
    q3, j3 = _qj(a + b)
    exch = np.sqrt(0.5 * ((j1 - j2) ** 2 + (j2 - j3) ** 2 + (j3 - j1) ** 2))
    return np.asarray(q1 + q2 + q3 - exch + _D)


def _grad(r: FloatArray, h: float = 1e-5) -> FloatArray:
    return np.array(
        [
            (leps(r[0] + h, r[1]) - leps(r[0] - h, r[1])) / (2 * h),
            (leps(r[0], r[1] + h) - leps(r[0], r[1] - h)) / (2 * h),
        ]
    )


def saddle() -> FloatArray:
    """Symmetric saddle point: on the line r1 = r2 it is the energy minimum."""
    r = np.linspace(0.7, 1.3, 60001)
    return np.array([r[np.argmin(leps(r, r))]] * 2)


_M_H = 1.00782503  # amu
# Wilson G matrix of collinear A-B-C in the bond lengths (1/amu). With
# q = G^(-1/2) r the kinetic energy is ½|q̇|², so q are mass-weighted
# coordinates and steepest descent in them is dr ∝ -G ∇V.
_G = np.array([[2.0, -1.0], [-1.0, 2.0]]) / _M_H
_G_HALF = np.real(scipy.linalg.sqrtm(_G))
# eV/(Å² amu) to s⁻², then angular frequency to cm⁻¹.
_EV_PER_A2_AMU = constants.e / (1e-20 * constants.atomic_mass)
_BOHR_PER_A = 1e-10 / constants.physical_constants["Bohr radius"][0]


def _hess(r: FloatArray, h: float = 1e-3) -> FloatArray:
    def v(di: float, dj: float, i: int, j: int) -> float:
        rr = np.array(r, dtype=float)
        rr[i] += di
        rr[j] += dj
        return float(leps(rr[0], rr[1]))

    return np.array(
        [
            [
                (v(h, h, i, j) - v(h, -h, i, j) - v(-h, h, i, j) + v(-h, -h, i, j))
                / (4 * h * h)
                for j in range(2)
            ]
            for i in range(2)
        ]
    )


def _wavenumber(lam: float) -> float:
    """Mass-weighted Hessian eigenvalue (eV/(Å² amu)) to cm⁻¹, imaginary < 0."""
    omega = np.sqrt(abs(lam) * _EV_PER_A2_AMU)
    return float(np.sign(lam) * omega / (2 * np.pi * constants.c * 100))


def mep_leps(r_max: float = 2.5, ds: float = 0.01) -> tuple[FloatArray, FloatArray]:
    """Minimum-energy path through the saddle point.

    The steepest-descent equation is integrated with a stiff solver, so the
    path stays on the valley floor, then resampled every ``ds`` bohr of s.

    Returns:
        ``(s, r)``: s in mass-scaled bohr (μ = 1 amu), 0 at the saddle point
        and < 0 on the reactant side (large r_BC), and the (r_AB, r_BC) rows
        in Å.
    """
    r0 = saddle()
    lam, vec = np.linalg.eigh(_G_HALF @ _hess(r0) @ _G_HALF)
    u = _G_HALF @ vec[:, int(np.argmin(lam))]  # the imaginary mode, in r

    def far(_t: float, r: FloatArray) -> float:
        return float(max(r) - r_max)

    far.terminal = True  # type: ignore[attr-defined]
    branches = []
    for sign in (1.0, -1.0):
        sol = scipy.integrate.solve_ivp(
            lambda _t, r: -_G @ _grad(r),
            (0.0, 1e3),
            r0 + sign * 1e-4 * u / np.linalg.norm(u),
            method="Radau",
            events=far,
            rtol=1e-9,
            atol=1e-12,
        )
        branches.append(sol.y.T)
    # Reactant side (r_BC grows) first.
    if branches[0][-1, 1] > branches[0][-1, 0]:
        branches.reverse()
    path = np.vstack([branches[1][::-1], r0, branches[0]])

    # Arc length in mass-scaled Cartesian coordinates, centre of mass fixed.
    def cartesian(r: FloatArray) -> FloatArray:
        return np.array([-(2 * r[0] + r[1]), r[0] - r[1], r[0] + 2 * r[1]]) / 3

    x = np.array([cartesian(p) for p in path]) * np.sqrt(_M_H) * _BOHR_PER_A
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))])
    s -= s[len(branches[1])]
    grid = np.arange(np.ceil(s[0] / ds), np.floor(s[-1] / ds) + 1) * ds
    r = np.column_stack([np.interp(grid, s, path[:, k]) for k in range(2)])
    return grid, r


def transverse_wavenumber(r: FloatArray) -> float:
    """Generalized frequency (cm⁻¹) of the one vibration across the path."""
    f = _G_HALF @ _hess(r) @ _G_HALF
    g = _G_HALF @ _grad(r)
    if np.linalg.norm(g) < 1e-3:  # at the saddle point: across the imaginary mode
        lam = np.linalg.eigvalsh(f)
        return _wavenumber(float(lam[1]))
    v = g / np.linalg.norm(g)
    across = np.array([-v[1], v[0]])
    return _wavenumber(float(across @ f @ across))


def pes(mode: str) -> go.Figure:
    t = THEMES[mode]
    v_max = 200.0  # kJ/mol; the surface is cut flat above this
    r = np.linspace(0.5, 2.5, 121)
    r1, r2 = np.meshgrid(r, r)
    v = np.minimum(leps(r1, r2) * EV_TO_KJ, v_max)
    _, path = mep_leps()
    v_path = leps(path[:, 0], path[:, 1]) * EV_TO_KJ
    ts = saddle()
    v_ts = float(leps(ts[0], ts[1]) * EV_TO_KJ)
    n = len(t["ramp"]) - 1
    hover = (
        "r<sub>AB</sub> = %{x:.2f} Å<br>r<sub>BC</sub> = %{y:.2f} Å<br>"
        "V = %{z:.0f} kJ/mol<extra></extra>"
    )
    surface = go.Surface(
        x=r,
        y=r,
        z=v,
        colorscale=[[k / n, c] for k, c in enumerate(t["ramp"])],
        cmin=0,
        cmax=v_max,
        lighting={"specular": 0.05, "roughness": 0.9},
        hovertemplate=hover,
        contours={
            "z": {
                "show": True,
                "start": 20,
                "end": 180,
                "size": 20,
                "color": t["axis"],
                "width": 1,
            }
        },
        colorbar={
            "title": {"text": "kJ/mol", "font": {"color": t["text"]}},
            "tickfont": {"color": t["muted"]},
            "len": 0.6,
            "thickness": 14,
            "outlinewidth": 0,
        },
    )
    # Lifted slightly so the surface does not hide them.
    mep = go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=v_path + 3,
        mode="lines",
        line={"color": t["series"][1], "width": 6},
        hovertemplate="Minimum-energy path<br>" + hover,
    )
    top = go.Scatter3d(
        x=[ts[0]],
        y=[ts[1]],
        z=[v_ts + 3],
        mode="markers",
        marker={"color": t["series"][1], "size": 6},
        hovertemplate="Saddle point<br>" + hover,
    )
    axis = {
        "color": t["muted"],
        "gridcolor": t["grid"],
        "showbackground": False,
        "title": {"font": {"color": t["text"]}},
    }
    layout = _layout(
        mode,
        height=520,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hovermode="closest",
        scene={
            "xaxis": {**axis, "title": {**axis["title"], "text": "r<sub>AB</sub> (Å)"}},
            "yaxis": {**axis, "title": {**axis["title"], "text": "r<sub>BC</sub> (Å)"}},
            "zaxis": {
                **axis,
                "title": {**axis["title"], "text": "V (kJ/mol)"},
                "range": [0, v_max * 1.05],
            },
            # Looking up the product valley toward the saddle point.
            "camera": {"eye": {"x": 2.0, "y": -0.6, "z": 1.0}},
            "aspectratio": {"x": 1, "y": 1, "z": 0.7},
        },
    )
    return go.Figure([surface, mep, top], layout)


# --- Energies along the path ---------------------------------------------------


def path_energies(mode: str) -> go.Figure:
    """V_MEP(s) and V_a^G(s) along the collinear H + H2 path."""
    t = THEMES[mode]
    s, path = mep_leps()
    v_mep = leps(path[:, 0], path[:, 1]) * EV_TO_KJ
    nu = np.array([transverse_wavenumber(r) for r in path])
    nu_h2 = transverse_wavenumber(np.array([_RE, 10.0]))
    zpe = 0.5 * constants.h * constants.c * 100 * constants.N_A / 1e3  # kJ/mol per cm⁻¹
    v_ag = v_mep + zpe * (nu - nu_h2)
    i0 = int(np.argmin(abs(s)))
    custom = np.column_stack([nu])
    mep = go.Scatter(
        x=s,
        y=v_mep,
        mode="lines",
        line={"color": t["series"][0], "width": 2.5},
        customdata=custom,
        hovertemplate="V<sub>MEP</sub> = %{y:.1f} kJ/mol<extra></extra>",
    )
    ag = go.Scatter(
        x=s,
        y=v_ag,
        mode="lines",
        line={"color": t["series"][1], "width": 2.5},
        customdata=custom,
        hovertemplate=(
            "V<sub>a</sub><sup>G</sup> = %{y:.1f} kJ/mol<br>"
            "&#957;<sub>⊥</sub> = %{customdata[0]:.0f} cm⁻¹<extra></extra>"
        ),
    )
    layout = _layout(
        mode,
        xaxis={"title": {"text": "Reaction coordinate s (bohr)"}, "showgrid": False},
        yaxis={"title": {"text": "Energy (kJ/mol)"}},
        shapes=[_vline(0, t["muted"])],
        annotations=[
            _label(
                0.08,
                v_mep[i0] + 1,
                "V<sub>MEP</sub>",
                mode,
                xanchor="left",
                yanchor="bottom",
            ),
            _label(
                -0.08,
                v_ag[i0] + 1,
                "V<sub>a</sub><sup>G</sup> = V<sub>MEP</sub> + ZPE",
                mode,
                xanchor="right",
                yanchor="bottom",
            ),
            _label(
                0.03,
                0.02,
                "Saddle point",
                mode,
                True,
                yref="paper",
                xanchor="left",
                yanchor="bottom",
            ),
        ],
    )
    return go.Figure([mep, ag], layout)


# --- Boltzmann distribution ----------------------------------------------------


def boltzmann(mode: str) -> go.Figure:
    t = THEMES[mode]
    ea = 10.0  # kJ/mol
    e = np.linspace(0, 25, 501)
    traces: list[go.Scatter] = []
    notes: list[dict[str, object]] = []
    for i, temp in enumerate((300, 600)):
        kt = R_KJ * temp
        f = 2 * np.sqrt(e / np.pi) * kt**-1.5 * np.exp(-e / kt)
        x = ea / kt
        above = special.erfc(np.sqrt(x)) + 2 * np.sqrt(x / np.pi) * np.exp(-x)
        color = t["series"][i]
        tail = e >= ea
        traces.append(
            go.Scatter(
                x=e[tail],
                y=f[tail],
                mode="lines",
                fill="tozeroy",
                line={"width": 0},
                fillcolor=_rgba(color, 0.3),
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter(
                x=e,
                y=f,
                mode="lines",
                line={"color": color, "width": 2.5},
                hovertemplate=f"{temp} K: %{{y:.3f}}<extra></extra>",
            )
        )
        j = int(np.searchsorted(e, 4.0 if temp == 300 else 8.0))
        notes.append(
            {
                "x": e[j],
                "y": f[j],
                "ax": 50,
                "ay": -45,
                "showarrow": True,
                "arrowhead": 0,
                "arrowcolor": t["text"],
                "xanchor": "left",
                "text": f"{temp} K: {above:.0%} above the barrier",
                "font": {"color": t["text"]},
            }
        )
    barrier = _label(
        ea + 0.3, 1, "Barrier", mode, True, yref="paper", xanchor="left", yanchor="top"
    )
    layout = _layout(
        mode,
        xaxis={
            "title": {"text": "Kinetic energy of a molecule (kJ/mol)"},
            "range": [0, 25],
            "showgrid": False,
        },
        yaxis={
            "title": {"text": "Fraction of molecules per kJ/mol"},
            "rangemode": "tozero",
        },
        shapes=[_vline(ea, t["muted"])],
        annotations=[*notes, barrier],
    )
    return go.Figure(traces, layout)


# --- Generalized free energy and the variational transition state --------------


def free_energy_model(s: FloatArray, temp: float) -> FloatArray:
    """Model ΔG^GT(T, s) in kJ/mol: a symmetric barrier plus one transverse mode.

    The transverse mode is loose (150 cm⁻¹) on the reactant side and stiffens
    to 800 cm⁻¹ past the saddle point. Its free energy rises along s, more so
    at high T where the lost entropy weighs more, so the maximum moves to
    larger s as T rises.
    """
    v = 10.0 / np.cosh(s) ** 2
    nu_r, nu_p = 150.0, 800.0  # cm⁻¹
    nu = nu_r + (nu_p - nu_r) / (1 + np.exp(-s / 0.4))

    def f_vib(n: npt.ArrayLike) -> FloatArray:
        u = HC_OVER_K * np.asarray(n) / temp
        return np.asarray(R_KJ * temp * np.log(2 * np.sinh(u / 2)))

    return np.asarray(v + f_vib(nu) - f_vib(nu_r))


def free_energy(mode: str) -> go.Figure:
    t = THEMES[mode]
    s = np.linspace(-2.0, 2.0, 801)
    traces: list[go.Scatter] = []
    notes: list[dict[str, object]] = []
    for i, temp in enumerate((200, 600, 1500)):
        g = free_energy_model(s, temp)
        j = int(np.argmax(g))
        color = t["series"][i]
        traces.append(
            go.Scatter(
                x=s,
                y=g,
                mode="lines",
                line={"color": color, "width": 2.5},
                hovertemplate=f"{temp} K: %{{y:.1f}} kJ/mol<extra></extra>",
            )
        )
        traces.append(
            go.Scatter(
                x=[s[j]],
                y=[g[j]],
                mode="markers",
                marker={"color": color, "size": 10},
                hoverinfo="skip",
            )
        )
        notes.append(
            _label(
                s[j] + 0.08,
                g[j],
                f"{temp} K, s* = {s[j]:.2f} bohr",
                mode,
                xanchor="left",
                yanchor="bottom",
            )
        )
    saddle_label = _label(
        0.05,
        0.02,
        "Saddle point",
        mode,
        True,
        yref="paper",
        xanchor="left",
        yanchor="bottom",
    )
    layout = _layout(
        mode,
        xaxis={"title": {"text": "Reaction coordinate s (bohr)"}, "showgrid": False},
        yaxis={"title": {"text": "ΔG<sup>GT</sup> (kJ/mol)"}, "rangemode": "tozero"},
        shapes=[_vline(0, t["muted"])],
        annotations=[*notes, saddle_label],
    )
    return go.Figure(traces, layout)


# --- Arrhenius plot with tunneling ---------------------------------------------


def arrhenius(mode: str) -> go.Figure:
    t = THEMES[mode]
    temp = np.linspace(200, 1000, 161)
    barrier = 40.0  # kJ/mol
    nu_imag = 1500.0  # cm⁻¹
    k_tst = constants.k * temp / constants.h * np.exp(-barrier / (R_KJ * temp))
    kappa = eckart(barrier / KJ_PER_HARTREE, barrier / KJ_PER_HARTREE, nu_imag, temp)
    x = 1000 / temp
    custom = np.column_stack([temp, kappa])
    j = int(np.argmin(abs(temp - 280)))
    plain = go.Scatter(
        x=x,
        y=np.log10(k_tst),
        mode="lines",
        line={"color": t["series"][0], "width": 2.5},
        hovertemplate="TST: 10^%{y:.2f} s⁻¹<extra></extra>",
    )
    tunnel = go.Scatter(
        x=x,
        y=np.log10(k_tst * kappa),
        mode="lines",
        line={"color": t["series"][1], "width": 2.5},
        customdata=custom,
        hovertemplate=(
            "With tunneling: 10^%{y:.2f} s⁻¹<br>"
            "T = %{customdata[0]:.0f} K, κ = %{customdata[1]:.1f}<extra></extra>"
        ),
    )
    layout = _layout(
        mode,
        xaxis={"title": {"text": "1000 / T (K⁻¹)"}, "showgrid": False},
        yaxis={"title": {"text": "log<sub>10</sub> k (s⁻¹)"}},
        annotations=[
            _label(
                x[j] + 0.1,
                np.log10(k_tst[j] * kappa[j]) + 0.3,
                f"TST with Eckart tunneling<br>κ = {kappa[0]:.0f} at 200 K",
                mode,
                xanchor="left",
                yanchor="bottom",
                align="left",
            ),
            _label(
                x[j] - 0.1,
                np.log10(k_tst[j]) - 0.3,
                "TST, no tunneling",
                mode,
                xanchor="right",
                yanchor="top",
            ),
        ],
    )
    return go.Figure([plain, tunnel], layout)


FIGURES: dict[str, Callable[[str], go.Figure]] = {
    "profile": profile,
    "path_energies": path_energies,
    "pes": pes,
    "boltzmann": boltzmann,
    "free_energy": free_energy,
    "arrhenius": arrhenius,
}


def fragment(name: str) -> str:
    """HTML for one figure: a placeholder and its light and dark specs."""
    specs = {mode: json.loads(FIGURES[name](mode).to_json()) for mode in THEMES}
    # "</" may not appear inside a <script> element.
    payload = json.dumps(specs, separators=(",", ":")).replace("</", "<\\/")
    return (
        f'<div class="pyvtst-figure" data-figure="{name}"></div>\n'
        f'<script type="application/json" data-figure-spec="{name}">'
        f"{payload}</script>\n"
    )


def build(outdir: Path) -> None:
    """Write every figure's HTML fragment to ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        (outdir / f"{name}.html").write_text(fragment(name), encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).parent / "_generated" / "figures")
