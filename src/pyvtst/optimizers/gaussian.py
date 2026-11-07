"""Gaussian 16's own optimizer, at the level of theory of a Gaussian engine."""

from __future__ import annotations

import os
import re

from pyvtst.core.elements import formula
from pyvtst.core.level import LevelOfTheory, render_options
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine
from pyvtst.ports.errors import ConvergenceError, EngineError
from pyvtst.ports.optimizer import OptimizationResult
from pyvtst.programs import gaussian

CONVERGENCE = ("default", "tight", "verytight")

_NOT_CONVERGED = ("Number of steps exceeded", "Optimization stopped")


class GaussianOptimizer:
    """Minima and transition states located by Gaussian's ``Opt``.

    Args:
        convergence: ``"default"``, ``"tight"`` or ``"verytight"``,
            case-insensitively: Gaussian's standard, ``Opt=Tight`` or
            ``Opt=VeryTight`` criteria.
        max_cycles: Maximum number of optimization steps (``MaxCycles``).
        nprocs: Processors per job (``%NProcShared``).
        memory: Memory per job (``%Mem``), e.g. ``"4GB"``.
        executable: The Gaussian executable. Default: ``$PYVTST_G16``, else
            ``g16`` on ``PATH``.
        formchk: The ``formchk`` utility. Default: ``$PYVTST_FORMCHK``, else
            ``formchk`` on ``PATH``.
        workdir: Keep each optimization's input, log and formatted checkpoint
            in a new subdirectory here. By default a temporary directory is
            used and removed afterwards.
        timeout: Seconds after which a job is stopped and an
            :class:`~pyvtst.ports.EngineError` raised. ``None`` waits forever.
        verbose: Pass Gaussian's own output through and say where each job
            runs.

    The engine passed to :meth:`minimize` and :meth:`saddle` must be a
    :class:`~pyvtst.engines.GaussianEngine`, or a wrapper such as
    :class:`~pyvtst.engines.CachedEngine` that forwards its ``level``;
    Gaussian runs at that level's method, basis, keywords and reference. The
    engine itself is not called. Saddle searches use
    ``Opt=(TS,CalcFC,NoEigenTest)``.
    """

    def __init__(
        self,
        *,
        convergence: str = "tight",
        max_cycles: int = 100,
        nprocs: int = 1,
        memory: str = "2GB",
        executable: str | os.PathLike[str] | None = None,
        formchk: str | os.PathLike[str] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        name = convergence.lower() if isinstance(convergence, str) else convergence
        if name not in CONVERGENCE:
            raise ValueError(
                f"convergence: unknown convergence {convergence!r}; expected one "
                f"of {', '.join(CONVERGENCE)}"
            )
        if isinstance(max_cycles, bool) or not isinstance(max_cycles, int):
            raise ValueError(f"max_cycles: expected an integer, got {max_cycles!r}")
        if max_cycles < 1:
            raise ValueError(f"max_cycles: must be at least 1, got {max_cycles}")
        gaussian.check_run_settings(nprocs, memory, timeout)
        self.convergence: str = name
        self.max_cycles = max_cycles
        g16, formchk_path = gaussian.find_programs(executable, formchk)
        self._settings = gaussian.RunSettings(
            g16=g16,
            formchk=formchk_path,
            nprocs=int(nprocs),
            memory=memory,
            workdir=workdir,
            timeout=None if timeout is None else float(timeout),
            verbose=verbose,
        )

    def key(self) -> str:
        """Identify the settings that can change a converged result.

        Only ``convergence`` is part of it.
        """
        return "gaussian" + render_options((("convergence", self.convergence),))

    def minimize(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        return self._run(mol, engine, saddle=False)

    def saddle(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        return self._run(mol, engine, saddle=True)

    def _run(
        self, mol: Molecule, engine: Engine, *, saddle: bool
    ) -> OptimizationResult:
        level = engine.level
        reference, keywords = _gaussian_options(level)
        job = gaussian.opt_job(
            saddle=saddle, max_cycles=self.max_cycles, convergence=self.convergence
        )
        title = "ts" if saddle else "opt"
        deck = gaussian.build_deck(
            mol,
            method=level.method,
            basis=level.basis,
            reference=reference,
            keywords=keywords,
            job=job,
            title=title,
            nprocs=self._settings.nprocs,
            memory=self._settings.memory,
        )
        context = f"{level.key()}: Gaussian {job} for {gaussian.label(mol)}"
        result = gaussian.run_job(
            self._settings,
            deck,
            f"{title}-{formula(mol.symbols)}",
            formchk_on_failure=True,
        )
        steps = [int(n) for n in re.findall(r"Step number\s+(\d+)", result.log)]
        if result.failure is not None:
            if any(marker in result.log for marker in _NOT_CONVERGED):
                raise ConvergenceError(
                    result.describe(
                        f"{context}: Gaussian did not converge in "
                        f"{max(steps, default=0)} steps"
                    ),
                    _last_molecule(result.fchk, mol),
                )
            raise EngineError(result.describe(f"{context}: {result.failure}"))
        if "Stationary point found" not in result.log:
            raise EngineError(
                result.describe(f"{context}: the log reports no stationary point")
            )
        if result.fchk is None:
            raise EngineError(result.describe(f"{context}: {result.fchk_failure}"))
        try:
            coords = gaussian.fchk_coordinates(result.fchk, mol)
            energy = result.fchk.scalar("Total Energy")
        except ValueError as exc:
            raise EngineError(f"{context}: {gaussian.FCHK_NAME}: {exc}") from None
        return OptimizationResult(
            molecule=mol.replace(coords=coords),
            energy=energy,
            n_iterations=steps[-1] if steps else 0,
        )


def _gaussian_options(level: LevelOfTheory) -> tuple[str, str]:
    """The reference and keywords of a Gaussian level; ``TypeError`` otherwise."""
    options = dict(level.options)
    reference, keywords = options.get("reference"), options.get("keywords")
    if (
        options.get("program") != "gaussian"
        or reference not in gaussian.REFERENCES
        or not isinstance(keywords, str)
    ):
        raise TypeError(
            "GaussianOptimizer needs the Gaussian engine (pyvtst.engines."
            f"GaussianEngine), got an engine at level {level.key()!r}"
        )
    return str(reference), keywords


def _last_molecule(fchk: gaussian.Fchk | None, mol: Molecule) -> Molecule | None:
    """The checkpoint's current geometry as a molecule like ``mol``, if readable."""
    if fchk is None:
        return None
    try:
        return mol.replace(coords=gaussian.fchk_coordinates(fchk, mol))
    except ValueError:
        return None
