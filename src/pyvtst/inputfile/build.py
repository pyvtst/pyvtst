"""Run a validated input: build the adapters and the Reaction, then each job."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime

from pyvtst.core.arrhenius import ArrheniusFit
from pyvtst.core.mep import ReactionPath
from pyvtst.inputfile.registry import Registry
from pyvtst.inputfile.schema import InputSpec
from pyvtst.reaction import RateResult, Reaction, StationaryPoints


@dataclass
class RunLog:
    """What a run produced, filled in as each step finishes.

    Attributes:
        spec: The input.
        registry: The registry the adapters were built from.
        started: When the run started, local time.
        points: The stationary points, once optimized.
        path: The reaction path, once computed (when the run needs one).
        results: One rate result per finished job, in file order.
        fits: One entry per finished job: its Arrhenius fit, the fit's error
            message, or ``None`` when the job asks for no fit.
        warnings: The text of every warning issued during the run.
        exception: The exception that stopped the run, or ``None``.
    """

    spec: InputSpec
    registry: Registry
    started: datetime
    points: StationaryPoints | None = None
    path: ReactionPath | None = None
    results: list[RateResult] = field(default_factory=list)
    fits: list[ArrheniusFit | str | None] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exception: BaseException | None = None


def execute(spec: InputSpec, registry: Registry) -> RunLog:
    """Run every step of ``spec``, recording results, warnings and any error.

    Exceptions (and Ctrl-C) are recorded in :attr:`RunLog.exception` rather
    than raised, so the caller can report what finished before them.
    """
    log = RunLog(spec, registry, datetime.now().astimezone())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _steps(spec, registry, log)
        except (Exception, KeyboardInterrupt) as exc:
            log.exception = exc
    log.warnings = [str(w.message) for w in caught]
    return log


def _steps(spec: InputSpec, registry: Registry, log: RunLog) -> None:
    engine = registry.engines[spec.engine.type].build(spec.engine.kwargs)
    optimizer = None
    if spec.optimizer is not None:
        entry = registry.optimizers[spec.optimizer.type]
        optimizer = entry.build(spec.optimizer.kwargs)
    rxn = Reaction(
        spec.reactants,
        spec.ts,
        engine,
        products=spec.products,
        optimizer=optimizer,
        store=spec.store,
        freq_scale=spec.freq_scale,
    )
    log.points = rxn.optimize()
    if spec.needs_path:
        kwargs = {} if spec.mep is None else spec.mep.kwargs
        log.path = rxn.compute_mep(**kwargs)  # type: ignore[arg-type]
    for job in spec.rates:
        path = log.path if job.needs_path else None
        result = rxn.rates(job.T, job.theory, job.tunneling, path=path)
        log.results.append(result)
        log.fits.append(_fit(result, job.arrhenius))


def _fit(result: RateResult, model: str | None) -> ArrheniusFit | str | None:
    """Fit ``result``, or return the fit's error message so later jobs still run."""
    if model is None:
        return None
    try:
        return result.fit_arrhenius(model)
    except ValueError as exc:
        return str(exc)
