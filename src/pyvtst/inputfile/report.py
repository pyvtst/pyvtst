"""The plain-text report of an input-file run."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyvtst.core.mep import ReactionPath
from pyvtst.core.molecule import Molecule
from pyvtst.core.stationary import StationaryPoint
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from pyvtst.inputfile.build import RunLog
from pyvtst.inputfile.registry import DEFAULT_REGISTRY, Registry
from pyvtst.inputfile.schema import AdapterSpec
from pyvtst.reaction import StationaryPoints

WIDTH = 78
RULE = "=" * WIDTH
FREQS_PER_LINE = 6
_STOP_TEXT = {
    "s_limit": "reached the s limit",
    "minimum": "reached a minimum (the gradient vanished)",
    "energy_rise": "stopped: the next step would raise the energy",
}


def write_report(log: RunLog, output: str | os.PathLike[str]) -> Path:
    """Write the report of ``log`` to ``output``, creating its directory.

    When the run computed a reaction path, its geometries are also written
    next to the report, as :func:`mep_xyz_path` names them.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = None
    if log.path is not None:
        trajectory = mep_xyz_path(path)
        trajectory.write_text(mep_xyz(log.path), encoding="utf-8")
    path.write_text(render(log, trajectory), encoding="utf-8")
    return path


def mep_xyz_path(report: Path) -> Path:
    """The path-geometry file of a report: ``h3.out`` gives ``h3_mep.xyz``."""
    return report.with_name(f"{report.stem}_mep.xyz")


def mep_xyz(path: ReactionPath) -> str:
    """Multi-frame XYZ text of a reaction path, one frame per point in s order.

    Coordinates are in ångström. Each comment line holds ``key=value`` pairs
    (``index``, ``s_bohr``, ``energy_hartree``, ``v_mep_kcal_mol``, relative
    to the reactants), which ASE reads into ``atoms.info``.
    """
    lines: list[str] = []
    for i in np.argsort(path.s, kind="stable"):
        mol = path.molecules[i]
        v_mep = (path.energies[i] - path.reference_energy) * HARTREE_TO_KCAL_PER_MOL
        lines.append(str(mol.n_atoms))
        lines.append(
            f"index={int(i)} s_bohr={float(path.s[i]):.6f} "
            f"energy_hartree={float(path.energies[i]):.10f} "
            f"v_mep_kcal_mol={float(v_mep):.6f}"
        )
        for symbol, (x, y, z) in zip(mol.symbols, mol.coords_angstrom, strict=True):
            lines.append(f"{symbol:<3}{x:14.8f}{y:14.8f}{z:14.8f}")
    return "\n".join(lines) + "\n"


def render(log: RunLog, trajectory: Path | None = None) -> str:
    """Return the report text: the sections in order, ending with the outcome.

    ``trajectory`` names the file the path geometries were written to.
    """
    from pyvtst import __version__  # noqa: PLC0415

    spec = log.spec
    out: list[str] = []
    header = [
        f"pyvtst {__version__}",
        f"date:  {log.started.isoformat(timespec='seconds')}",
        f"input: {spec.path}",
    ]
    if spec.title is not None:
        header.append(f"title: {spec.title}")
    _section(out, "PYVTST", header)
    _section(out, "INPUT", spec.text.rstrip("\n").splitlines())
    _section(out, "SETTINGS", _settings(log))
    if log.points is not None:
        _section(out, "STATIONARY POINTS", _stationary_points(log, log.points))
    if log.results:
        barrier = log.results[0].barrier_kcal
        _section(out, "BARRIER", [f"ZPE-corrected barrier dE0: {barrier:.4f} kcal/mol"])
    if log.path is not None:
        _section(out, "REACTION PATH", _path(log.path, trajectory))
    if log.results:
        _section(out, "RATE CONSTANTS", _rates(log))
    _section(out, "WARNINGS", log.warnings or ["none"])
    out.append(RULE)
    exc = log.exception
    if exc is None:
        out.append("pyvtst finished normally")
    elif isinstance(exc, KeyboardInterrupt):
        out.append("pyvtst stopped with an error: interrupted")
    else:
        out.append(f"pyvtst stopped with an error: {exc}")
    return "\n".join(out) + "\n"


def _section(out: list[str], title: str, lines: list[str]) -> None:
    out.extend([RULE, title, RULE, *lines, ""])


def _adapter(label: str, spec: AdapterSpec) -> list[str]:
    lines = [f"{label}: {spec.type}"]
    lines += [f"  {name} = {value!r}" for name, value in spec.options]
    return lines


def _default_optimizer(registry: Registry) -> list[str]:
    # Reaction builds GeometricOptimizer() when no optimizer is given.
    entry = (
        registry.optimizers.get("geometric") or DEFAULT_REGISTRY.optimizers["geometric"]
    )
    lines = ["optimizer: geometric (default)"]
    lines += [f"  {f.name} = {f.default!r}" for f in entry.fields]
    return lines


def _settings(log: RunLog) -> list[str]:
    spec = log.spec
    lines = _adapter("engine", spec.engine)
    if spec.optimizer is not None:
        lines += _adapter("optimizer", spec.optimizer)
    else:
        lines += _default_optimizer(log.registry)
    lines.append(f"freq_scale: {spec.freq_scale:g}")
    lines.append(f"store: {spec.store if spec.store is not None else 'in memory'}")
    lines.append("inline geometries: angstrom unless 'units' says otherwise")
    if spec.needs_path:
        if spec.mep is None:
            lines.append("reaction path: compute_mep() defaults")
        else:
            lines.append("reaction path:")
            for name, value in spec.mep.kwargs.items():
                lines.append(f"  {name} = {value!r}")
    return lines


def _stationary_points(log: RunLog, points: StationaryPoints) -> list[str]:
    spec = log.spec
    lines: list[str] = []
    entries: list[tuple[str, Molecule, StationaryPoint]] = [
        ("reactant", m, p)
        for m, p in zip(spec.reactants, points.reactants, strict=True)
    ]
    entries.append(("transition state", spec.ts, points.ts))
    if spec.products is not None and points.products is not None:
        entries += [
            ("product", m, p)
            for m, p in zip(spec.products, points.products, strict=True)
        ]
    for role, guess, sp in entries:
        mol = sp.molecule
        lines.append(f"{role} {guess.name!r}")
        lines.append(
            f"  charge {mol.charge}  mult {mol.mult}  "
            f"symmetry number {mol.symmetry_number}"
        )
        lines.append(f"  energy: {sp.energy:.10f} hartree")
        lines.append("  geometry (angstrom):")
        for symbol, (x, y, z) in zip(mol.symbols, mol.coords_angstrom, strict=True):
            lines.append(f"    {symbol:<3}{x:14.6f}{y:14.6f}{z:14.6f}")
        lines.append(f"  frequencies (cm^-1, scale {sp.freq_scale:g}):")
        freqs = [_frequency(float(v)) for v in sp.frequencies]
        if not freqs:
            lines.append("    none")
        for i in range(0, len(freqs), FREQS_PER_LINE):
            lines.append("    " + "".join(freqs[i : i + FREQS_PER_LINE]))
        lines.append("")
    return lines[:-1]


def _frequency(value: float) -> str:
    return f"{abs(value):11.2f}" + ("i" if value < 0 else " ")


def _path(path: ReactionPath, trajectory: Path | None) -> list[str]:
    lines = [
        f"s from {float(path.s[0]):.4f} to {float(path.s[-1]):.4f} bohr, "
        f"{len(path.s)} points, method {path.method}, ds {path.ds:g} bohr",
        f"reactant side: {_STOP_TEXT.get(path.stop_reactant, path.stop_reactant)}",
        f"product side:  {_STOP_TEXT.get(path.stop_product, path.stop_product)}",
    ]
    if trajectory is not None:
        lines.append(f"geometries (angstrom): {trajectory.name}")
    return lines


def _rates(log: RunLog) -> list[str]:
    lines: list[str] = []
    for i, (job, result) in enumerate(
        zip(log.spec.rates, log.results, strict=False), start=1
    ):
        tunneling = job.tunneling if job.tunneling is not None else "none"
        lines.append(f"job {i}: theory {job.theory}, tunneling {tunneling}")
        lines.append(f"k in {result.units}")
        lines.extend(result.table().splitlines())
        lines.append("")
    return lines[:-1]
