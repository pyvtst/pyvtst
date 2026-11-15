"""Gaussian 16 engine. Gaussian is run as an external program, one job per
geometry, and results are read from its formatted checkpoint."""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine, EngineResult
from pyvtst.ports.errors import EngineError
from pyvtst.programs import gaussian

FloatArray = npt.NDArray[np.float64]


class GaussianEngine(Engine):
    """Energies, gradients and Hessians from Gaussian 16.

    Args:
        method: Any Gaussian method name, e.g. ``"B3LYP"`` or ``"MP2"``.
        basis: Any Gaussian basis-set name, e.g. ``"def2SVP"``. ``Gen`` and
            ``GenECP`` are not supported.
        keywords: Extra route keywords, e.g. ``"EmpiricalDispersion=GD3BJ
            Int=UltraFine"``. Job types and keywords the engine sets itself
            (:data:`~pyvtst.programs.gaussian.RESERVED_KEYWORDS`) are rejected.
        reference: ``"auto"`` leaves the choice to Gaussian (restricted for
            singlets, unrestricted otherwise); ``"restricted"`` or
            ``"unrestricted"`` force it.
        nprocs: Processors per job (``%NProcShared``).
        memory: Memory per job (``%Mem``), e.g. ``"4GB"``.
        executable: The Gaussian executable. Default: ``$PYVTST_G16``, else
            ``g16`` on ``PATH``.
        formchk: The ``formchk`` utility. Default: ``$PYVTST_FORMCHK``, else
            ``formchk`` on ``PATH``.
        workdir: Keep each job's input, log and formatted checkpoint in a new
            subdirectory here. By default each job runs in a temporary
            directory that is removed afterwards.
        timeout: Seconds after which a job is stopped and an
            :class:`~pyvtst.ports.EngineError` raised. ``None`` waits forever.
        verbose: Pass Gaussian's own output through and say where each job
            runs.

    Only ``method``, ``basis``, ``keywords`` and ``reference`` are part of
    :attr:`level`. Gaussian runs with the caller's environment, so
    ``GAUSS_SCRDIR`` and the rest of a ``g16.profile`` setup apply.

    Each call runs one job: a ``Force`` job for the energy and gradient, and a
    ``Freq`` job when the Hessian is asked for. Wrap the engine in
    :class:`~pyvtst.engines.CachedEngine` (as :class:`pyvtst.Reaction` does)
    so a ``Freq`` job also serves later energy and gradient requests.
    """

    def __init__(
        self,
        method: str,
        basis: str,
        *,
        keywords: str = "",
        reference: str = "auto",
        nprocs: int = 1,
        memory: str = "2GB",
        executable: str | os.PathLike[str] | None = None,
        formchk: str | os.PathLike[str] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        if reference not in gaussian.REFERENCES:
            raise ValueError(f"reference: unknown reference {reference!r}")
        gaussian.check_run_settings(nprocs, memory, timeout)
        self._level = LevelOfTheory(
            method,
            basis,
            (
                ("program", "gaussian"),
                ("reference", reference),
                ("keywords", gaussian.normalize_keywords(keywords)),
            ),
        )
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

    @property
    def level(self) -> LevelOfTheory:
        return self._level

    def compute(self, mol: Molecule, *, hessian: bool) -> EngineResult:
        level = self._level
        options = dict(level.options)
        job = gaussian.FREQ_JOB if hessian else gaussian.FORCE_JOB
        title = "freq" if hessian else "force"
        deck = gaussian.build_deck(
            mol,
            method=level.method,
            basis=level.basis,
            reference=str(options["reference"]),
            keywords=str(options["keywords"]),
            job=job,
            title=title,
            nprocs=self._settings.nprocs,
            memory=self._settings.memory,
        )
        context = f"{level.key()}: Gaussian {job} for {gaussian.label(mol)}"
        result = gaussian.run_job(self._settings, deck, gaussian.job_name(title, mol))
        if result.failure is not None:
            raise EngineError(result.describe(f"{context}: {result.failure}"))
        if result.fchk is None:
            raise EngineError(result.describe(f"{context}: {result.fchk_failure}"))
        try:
            energy, gradient, full = gaussian.fchk_results(
                result.fchk, mol, hessian=hessian
            )
        except ValueError as exc:
            raise EngineError(f"{context}: {gaussian.FCHK_NAME}: {exc}") from None
        return EngineResult(energy, gradient, full)
