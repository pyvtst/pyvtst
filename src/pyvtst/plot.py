"""Plots of minimum-energy paths: V_MEP(s) and V_a^G(s) with matplotlib.

matplotlib is optional (``pip install "pyvtst[plot]"``) and imported only
when a plot function is called.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pyvtst.core.mep import ReactionPath
from pyvtst.core.units import (
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    HARTREE_TO_KJ_PER_MOL,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CURVES = ("v_mep", "v_ag")
UNITS = {
    "kcal/mol": HARTREE_TO_KCAL_PER_MOL,
    "kJ/mol": HARTREE_TO_KJ_PER_MOL,
    "eV": HARTREE_TO_EV,
    "hartree": 1.0,
}
_NAMES = {"v_mep": "V_MEP", "v_ag": "V_a^G"}
_STYLES = {"v_mep": "-", "v_ag": "--"}


def plot_path(
    path: ReactionPath,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    curves: Sequence[str] = CURVES,
    units: str = "kcal/mol",
) -> Axes:
    """Plot V_MEP(s) and V_a^G(s) of one path; return the matplotlib ``Axes``.

    Args:
        path: The path, from :meth:`pyvtst.Reaction.compute_mep`.
        ax: Axes to draw on; a new figure is created when omitted. What is
            already drawn stays, so repeated calls overlay paths.
        label: Name of the path in the legend, e.g. ``"LQA"``.
        curves: Any of ``"v_mep"`` (solid line) and ``"v_ag"`` (dashed).
        units: Energy unit: ``"kcal/mol"``, ``"kJ/mol"``, ``"eV"`` or
            ``"hartree"``.

    The path is drawn in one color with its transition state marked at
    s = 0. Points without a Hessian are left out of V_a^G. A path that is not
    oriented gives a warning, since its sign of s is arbitrary.

    Raises:
        ValueError: An unknown curve or unit.
        ImportError: matplotlib is not installed.
    """
    labels = None if label is None else [label]
    return _plot([path], labels, ax, curves, units, "plot_path")


def plot_paths(
    paths: Sequence[ReactionPath],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    curves: Sequence[str] = CURVES,
    units: str = "kcal/mol",
) -> Axes:
    """Plot several paths on one ``Axes``, one color per path; return it.

    Use it to compare paths from different methods, step sizes or levels of
    theory. Each path is drawn at its own s values, without interpolation,
    and each later path thinner than the one before, so paths that agree
    stay visible on top of each other.
    Arguments are as in :func:`plot_path`; ``labels`` gives one legend name
    per path.

    Raises:
        ValueError: ``paths`` is empty, ``labels`` has a different length, or
            a curve or unit is unknown.
        ImportError: matplotlib is not installed.
    """
    return _plot(list(paths), labels, ax, curves, units, "plot_paths")


def _plot(
    paths: list[ReactionPath],
    labels: Sequence[str] | None,
    ax: Axes | None,
    curves: Sequence[str],
    units: str,
    caller: str,
) -> Axes:
    chosen = (curves,) if isinstance(curves, str) else tuple(curves)
    if not paths:
        raise ValueError("paths: expected at least one ReactionPath")
    for p in paths:
        if not isinstance(p, ReactionPath):
            raise TypeError(f"paths: expected ReactionPath, got {type(p).__name__}")
    if not chosen or any(c not in CURVES for c in chosen):
        raise ValueError(f"curves: expected 'v_mep' and/or 'v_ag', got {curves!r}")
    if units not in UNITS:
        listed = ", ".join(repr(u) for u in UNITS)
        raise ValueError(f"units: expected one of {listed}, got {units!r}")
    if labels is not None and len(labels) != len(paths):
        raise ValueError(
            f"labels: expected {len(paths)} labels, one per path, got {len(labels)}"
        )
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f'{caller} needs matplotlib: pip install "pyvtst[plot]"'
        ) from exc

    if ax is None:
        _, ax = plt.subplots()
    factor = UNITS[units]
    base_width = float(plt.rcParams["lines.linewidth"])
    base_size = float(plt.rcParams["lines.markersize"]) ** 2
    for i, path in enumerate(paths):
        # Paths that agree would hide each other; each later path is drawn
        # thinner, so the earlier ones stay visible around it.
        scale = _overlay_scale(i, len(paths))
        name = None if labels is None else labels[i]
        if not path.oriented:
            who = repr(name) if name else f"path {i}"
            warnings.warn(
                f"{who} is not oriented: its sign of s is arbitrary", stacklevel=3
            )
        color = None
        for curve in chosen:
            values = getattr(path, curve) * factor
            shown = np.isfinite(values)
            legend = _NAMES[curve] if name is None else f"{name} {_NAMES[curve]}"
            (line,) = ax.plot(
                path.s[shown],
                values[shown],
                linestyle=_STYLES[curve],
                linewidth=base_width * scale,
                color=color,
                label=legend,
            )
            color = line.get_color()
        ts_value = getattr(path, chosen[0])[path.ts_index] * factor
        ax.scatter(
            [0.0],
            [ts_value],
            s=base_size * scale**2,
            color=color,
            zorder=3,
            label="_nolegend_",
        )
    ax.set_xlabel("s (bohr)")
    ax.set_ylabel(f"Energy ({units})")
    if labels is not None:
        ax.legend()
    return ax


def _overlay_scale(i: int, n: int) -> float:
    """Line-width factor of path ``i`` of ``n``: from 2.5 for the first to 1."""
    return 1.0 if n == 1 else 1.0 + 1.5 * (n - 1 - i) / (n - 1)
