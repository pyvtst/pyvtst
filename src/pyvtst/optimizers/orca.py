"""ORCA 6's own optimizer, at the level of theory of an ORCA engine."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pyvtst.core.level import LevelOfTheory, render_options
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine
from pyvtst.ports.errors import ConvergenceError, EngineError
from pyvtst.ports.optimizer import OptimizationResult
from pyvtst.programs import orca

CONVERGENCE = orca.CONVERGENCE

CONVERGED = "THE OPTIMIZATION HAS CONVERGED"
CYCLE_LIMIT = "The optimization did not converge but reached the maximum number of"


class OrcaOptimizer:
    """Minima and transition states located by ORCA's ``Opt`` and ``OptTS``.

    Args:
        convergence: ``"normal"``, ``"tight"`` or ``"verytight"``,
            case-insensitively: ORCA's ``Opt``, ``TightOpt`` or
            ``VeryTightOpt`` criteria, for minima and saddle points.
        max_cycles: Maximum number of optimization cycles (``%geom MaxIter``).
        nprocs: Processes per job (``%pal nprocs``).
        maxcore: Memory per process in MB (``%maxcore``).
        executable: The ORCA executable. Default: ``$PYVTST_ORCA``, else
            ``orca`` on ``PATH``. Symbolic links are resolved.
        workdir: Keep each optimization's input, log and result files in a
            new subdirectory here. By default a temporary directory is used
            and removed afterwards.
        timeout: Seconds after which a job is stopped and an
            :class:`~pyvtst.ports.EngineError` raised. ``None`` waits forever.
        verbose: Say where each job runs and where its log is.

    The engine passed to :meth:`minimize` and :meth:`saddle` must be an
    :class:`~pyvtst.engines.OrcaEngine`, or a wrapper such as
    :class:`~pyvtst.engines.CachedEngine` that forwards its ``level``; ORCA
    runs at that level's method, basis, keywords, blocks and reference. The
    engine itself is not called. Saddle searches use ``OptTS`` with an initial
    Hessian (``Calc_Hess true``), computed numerically when the engine's
    ``hessian`` is ``"numerical"``.
    """

    def __init__(
        self,
        *,
        convergence: str = "tight",
        max_cycles: int = 100,
        nprocs: int = 1,
        maxcore: int = 2000,
        executable: str | os.PathLike[str] | None = None,
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
        orca.check_run_settings(nprocs, maxcore, timeout)
        self.convergence: str = name
        self.max_cycles = max_cycles
        self._settings = orca.RunSettings(
            orca=orca.find_orca(executable),
            nprocs=int(nprocs),
            maxcore=int(maxcore),
            workdir=workdir,
            timeout=None if timeout is None else float(timeout),
            verbose=verbose,
        )

    def key(self) -> str:
        """Identify the settings that can change a converged result.

        Only ``convergence`` is part of it.
        """
        return "orca" + render_options((("convergence", self.convergence),))

    def minimize(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        return self._run(mol, engine, saddle=False)

    def saddle(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        return self._run(mol, engine, saddle=True)

    def _run(
        self, mol: Molecule, engine: Engine, *, saddle: bool
    ) -> OptimizationResult:
        level = engine.level
        options = _orca_options(level)
        deck = orca.opt_deck(
            mol,
            method=level.method,
            basis=level.basis,
            reference=options["reference"],
            keywords=options["keywords"],
            blocks=options["blocks"],
            saddle=saddle,
            convergence=self.convergence,
            max_cycles=self.max_cycles,
            numerical=options["hessian"] == "numerical",
            nprocs=self._settings.nprocs,
            maxcore=self._settings.maxcore,
        )
        job = "OptTS" if saddle else "Opt"
        title = "ts" if saddle else "opt"
        context = f"{level.key()}: ORCA {job} for {orca.label(mol)}"
        result = orca.run_job(self._settings, deck, orca.job_name(title, mol))
        if result.failure is not None:
            raise EngineError(result.describe(f"{context}: {result.failure}"))
        cycles = orca.cycles(result.log)
        if CYCLE_LIMIT in result.log:
            raise ConvergenceError(
                result.describe(
                    f"{context}: ORCA did not converge in {max(cycles, default=0)} "
                    "cycles"
                ),
                _last_molecule(result.files, mol),
            )
        if CONVERGED not in result.log:
            raise EngineError(
                result.describe(f"{context}: the log reports no converged optimization")
            )
        missing = [
            n for n in (orca.XYZ_NAME, orca.ENGRAD_NAME) if n not in result.files
        ]
        if missing:
            raise EngineError(
                result.describe(f"{context}: ORCA wrote no {', '.join(missing)}")
            )
        try:
            coords = orca.xyz_coordinates(result.files[orca.XYZ_NAME], mol)
            final = mol.replace(coords=coords)
            # After convergence ORCA evaluates the final geometry once more, so
            # job.engrad belongs to the geometry in job.xyz.
            energy, _ = orca.engrad_results(
                orca.parse_engrad(result.files[orca.ENGRAD_NAME]), final
            )
        except ValueError as exc:
            raise EngineError(f"{context}: {exc}") from None
        return OptimizationResult(
            molecule=final,
            energy=energy,
            n_iterations=cycles[-1] if cycles else 0,
        )


def _orca_options(level: LevelOfTheory) -> dict[str, str]:
    """The reference, Hessian mode, keywords and blocks of an ORCA level;
    ``TypeError`` otherwise."""
    options = dict(level.options)
    values = {k: options.get(k) for k in ("reference", "hessian", "keywords", "blocks")}
    if (
        options.get("program") != "orca"
        or values["reference"] not in orca.REFERENCES
        or values["hessian"] not in orca.HESSIANS
        or not isinstance(values["keywords"], str)
        or not isinstance(values["blocks"], str)
    ):
        raise TypeError(
            "OrcaOptimizer needs the ORCA engine (pyvtst.engines.OrcaEngine), "
            f"got an engine at level {level.key()!r}"
        )
    return {k: str(v) for k, v in values.items()}


def _last_molecule(files: Mapping[str, str], mol: Molecule) -> Molecule | None:
    """The last evaluated geometry, from ``job.xyz``, as a molecule like ``mol``.

    ``job.engrad`` is not used: at the cycle limit ORCA leaves the coordinates
    of the next, unevaluated step in it.
    """
    text = files.get(orca.XYZ_NAME)
    if text is None:
        return None
    try:
        return mol.replace(coords=orca.xyz_coordinates(text, mol))
    except ValueError:
        return None
