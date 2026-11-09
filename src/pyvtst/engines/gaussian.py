"""Gaussian 16 engine. Gaussian is run as an external program, one job per
geometry, and results are read from its formatted checkpoint."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule
from pyvtst.ports.errors import EngineError
from pyvtst.programs import gaussian

FloatArray = npt.NDArray[np.float64]

# Recent runs kept per engine, so energy, gradient and Hessian requests at one
# geometry share a run.
MEMO_SIZE = 8

_MemoKey = tuple[tuple[str, ...], int, int, bytes]


@dataclass(frozen=True)
class _Run:
    energy: float
    gradient: FloatArray
    hessian: FloatArray | None


class GaussianEngine:
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

    A ``Force`` job serves energy and gradient requests, and a ``Freq`` job
    the Hessian together with the energy and gradient. The last
    :data:`MEMO_SIZE` results are kept, so asking for all three at one
    geometry runs Gaussian once when the Hessian is asked for first. The
    engine is not thread-safe.
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
        self._memo: OrderedDict[_MemoKey, _Run] = OrderedDict()

    @property
    def level(self) -> LevelOfTheory:
        return self._level

    def energy(self, mol: Molecule) -> float:
        return self._result(mol, hessian=False).energy

    def gradient(self, mol: Molecule) -> FloatArray:
        return self._result(mol, hessian=False).gradient.copy()

    def hessian(self, mol: Molecule) -> FloatArray:
        # _result with hessian=True always returns a run with a Hessian.
        return cast(FloatArray, self._result(mol, hessian=True).hessian).copy()

    def _result(self, mol: Molecule, *, hessian: bool) -> _Run:
        key = (
            tuple(mol.symbols),
            mol.charge,
            mol.mult,
            np.ascontiguousarray(mol.coords, dtype=np.float64).tobytes(),
        )
        hit = self._memo.get(key)
        if hit is not None and (hit.hessian is not None or not hessian):
            self._memo.move_to_end(key)
            return hit
        run = self._run(mol, hessian=hessian)
        self._memo[key] = run
        self._memo.move_to_end(key)
        while len(self._memo) > MEMO_SIZE:
            self._memo.popitem(last=False)
        return run

    def _run(self, mol: Molecule, *, hessian: bool) -> _Run:
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
        return _Run(energy, gradient, full)
