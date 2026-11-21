"""ORCA 6 engine. ORCA is run as an external program, one job per geometry,
and results are read from its ``.engrad`` and ``.hess`` files."""

from __future__ import annotations

import os

from pyvtst.core.level import LevelOfTheory
from pyvtst.core.molecule import Molecule
from pyvtst.ports.engine import Engine, EngineResult
from pyvtst.ports.errors import EngineError
from pyvtst.programs import orca


class OrcaEngine(Engine):
    """Energies, gradients and Hessians from ORCA 6.

    Args:
        method: Any ORCA method name, e.g. ``"B3LYP"``, ``"MP2"`` or the
            composite ``"r2SCAN-3c"``.
        basis: Any ORCA basis-set name, e.g. ``"def2-SVP"``, or ``None`` for a
            method that sets its own basis (``r2SCAN-3c``, ``B97-3c``,
            ``HF-3c``). The level's basis is then ``"-"``.
        keywords: Extra simple-input keywords for the ``!`` line, e.g.
            ``"D3BJ RIJCOSX def2/J"``. Job types, coordinate units, ``PAL<n>``
            and reference keywords are set by pyvtst and rejected
            (:data:`~pyvtst.programs.orca.RESERVED_KEYWORDS`).
        blocks: Extra ``%`` input blocks, e.g. ``"%scf\\n MaxIter 300\\nend"``.
            ``%pal``, ``%maxcore``, ``%geom``, ``%freq`` and ``%coords`` are set
            by pyvtst and rejected.
        reference: ``"auto"`` leaves the choice to ORCA (restricted for
            singlets, unrestricted otherwise); ``"restricted"`` or
            ``"unrestricted"`` force it (``RHF``/``UHF``, which also select
            RKS/UKS for DFT).
        hessian: ``"analytic"`` (``Freq``) or ``"numerical"`` (``NumFreq``:
            central differences of analytic gradients). ORCA has analytic
            Hessians for SCF methods only; MP2 and double hybrids need
            ``"numerical"``.
        nprocs: Processes per job (``%pal nprocs``). ORCA needs OpenMPI for
            more than 1.
        maxcore: Memory per process in MB (``%maxcore``).
        executable: The ORCA executable. Default: ``$PYVTST_ORCA``, else
            ``orca`` on ``PATH``. Symbolic links are resolved, because ORCA
            finds its helper programs next to the path it is called with.
        workdir: Keep each job's input, log and result files in a new
            subdirectory here. By default each job runs in a temporary
            directory that is removed afterwards.
        timeout: Seconds after which a job is stopped and an
            :class:`~pyvtst.ports.EngineError` raised. ``None`` waits forever.
        verbose: Say where each job runs and where its log is.

    Only ``method``, ``basis``, ``keywords``, ``blocks``, ``reference`` and
    ``hessian`` are part of :attr:`level`.

    Each call runs one job: ``EnGrad`` for the energy and gradient, and
    ``EnGrad Freq`` (or ``EnGrad NumFreq``) when the Hessian is asked for. Every
    job is a gradient job, so the energy and gradient do not depend on whether
    a Hessian was asked for. Wrap the engine in
    :class:`~pyvtst.engines.CachedEngine` (as :class:`pyvtst.Reaction` does) so
    a Hessian job also serves later energy and gradient requests.
    """

    def __init__(
        self,
        method: str,
        basis: str | None = None,
        *,
        keywords: str = "",
        blocks: str = "",
        reference: str = "auto",
        hessian: str = "analytic",
        nprocs: int = 1,
        maxcore: int = 2000,
        executable: str | os.PathLike[str] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        if basis is not None and (not isinstance(basis, str) or not basis.strip()):
            raise ValueError(
                f"basis: expected a non-empty string, or None for a method that "
                f"sets its own basis, got {basis!r}"
            )
        if reference not in orca.REFERENCES:
            raise ValueError(f"reference: unknown reference {reference!r}")
        if hessian not in orca.HESSIANS:
            raise ValueError(
                f"hessian: unknown Hessian mode {hessian!r}; expected one of "
                f"{', '.join(orca.HESSIANS)}"
            )
        orca.check_run_settings(nprocs, maxcore, timeout)
        self._level = LevelOfTheory(
            method,
            orca.NO_BASIS if basis is None else basis,
            (
                ("program", "orca"),
                ("reference", reference),
                ("hessian", hessian),
                ("keywords", orca.normalize_keywords(keywords)),
                ("blocks", orca.normalize_blocks(blocks)),
            ),
        )
        self._settings = orca.RunSettings(
            orca=orca.find_orca(executable),
            nprocs=int(nprocs),
            maxcore=int(maxcore),
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
        numerical = options["hessian"] == "numerical"
        deck = orca.engine_deck(
            mol,
            method=level.method,
            basis=level.basis,
            reference=str(options["reference"]),
            keywords=str(options["keywords"]),
            blocks=str(options["blocks"]),
            hessian=hessian,
            numerical=numerical,
            nprocs=self._settings.nprocs,
            maxcore=self._settings.maxcore,
        )
        job = "EnGrad" + ((" NumFreq" if numerical else " Freq") if hessian else "")
        title = "freq" if hessian else "engrad"
        context = f"{level.key()}: ORCA {job} for {orca.label(mol)}"
        result = orca.run_job(self._settings, deck, orca.job_name(title, mol))
        if result.failure is not None:
            raise EngineError(result.describe(f"{context}: {result.failure}"))
        needed = [orca.ENGRAD_NAME] + ([orca.HESS_NAME] if hessian else [])
        missing = [name for name in needed if name not in result.files]
        if missing:
            raise EngineError(
                result.describe(f"{context}: ORCA wrote no {', '.join(missing)}")
            )
        try:
            energy, gradient = orca.engrad_results(
                orca.parse_engrad(result.files[orca.ENGRAD_NAME]), mol
            )
            full = (
                orca.hess_result(orca.parse_hess(result.files[orca.HESS_NAME]), mol)
                if hessian
                else None
            )
        except ValueError as exc:
            raise EngineError(f"{context}: {exc}") from None
        return EngineResult(energy, gradient, full)
