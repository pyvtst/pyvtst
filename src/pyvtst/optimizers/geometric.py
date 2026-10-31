"""geomeTRIC optimizer for any pyvtst engine. geomeTRIC is imported lazily.

geomeTRIC behaviours this adapter relies on (checked against 1.1.1):

- A custom engine's ``calc_new(coords, dirname)`` receives flat (3N,)
  coordinates in bohr and returns ``{"energy", "gradient"}`` in hartree and
  hartree/bohr, with a flat gradient.
- ``run_optimizer`` returns a progress object whose ``xyzs`` frames are in
  ångström and whose ``qm_energies[-1]`` belongs to the last frame.
- ``hess_data`` must be a list: geomeTRIC tests ``if kwargs.get("hess_data")``,
  which fails for a numpy array.
- Non-convergence raises ``geometric.errors.GeomOptNotConvergedError``, which
  is not a ``RuntimeError``. Exceptions raised in ``calc_new`` propagate.
- Every ``run_optimizer`` call looks up ``logging.config.fileConfig`` at call
  time and passes the log path as ``defaults["logfilename"]``. ``fileConfig``
  closes every handler in the process, so it is swapped out during a run.
- All output goes through the ``geometric.nifty`` logger.
- Convergence set names are upper-cased and coordinate systems lower-cased
  before lookup; the accepted names are listed below.
- With an absolute ``input`` prefix nothing is written to the cwd.
- Its step code can emit numpy ``RuntimeWarning``s (0/0 on an exactly
  quadratic surface); they are silenced unless ``verbose=True``.
"""

from __future__ import annotations

import contextlib
import logging
import logging.config
import os
import sys
import tempfile
import warnings
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pyvtst.core.level import render_options
from pyvtst.core.molecule import Molecule
from pyvtst.core.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from pyvtst.ports.engine import Engine
from pyvtst.ports.errors import ConvergenceError
from pyvtst.ports.optimizer import OptimizationResult

FloatArray = npt.NDArray[np.float64]

CONVERGENCE_SETS = frozenset(
    {
        "GAU",
        "GAU_LOOSE",
        "GAU_TIGHT",
        "GAU_VERYTIGHT",
        "INTERFRAG_TIGHT",
        "NWCHEM_LOOSE",
        "TURBOMOLE",
    }
)
COORDINATE_SYSTEMS = frozenset({"cart", "prim", "dlc", "hdlc", "tric", "tric-p"})


def _import_geometric() -> Any:
    # geomeTRIC is optional, so it is imported on first use rather than at load time.
    try:
        import geometric  # noqa: PLC0415
        import geometric.engine  # noqa: PLC0415
        import geometric.errors  # noqa: PLC0415
        import geometric.molecule  # noqa: PLC0415
        import geometric.nifty  # noqa: PLC0415
        import geometric.optimize  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'GeometricOptimizer needs geomeTRIC: pip install "pyvtst[geometric]"'
        ) from exc
    return geometric


class GeometricOptimizer:
    """Minima and transition states with geomeTRIC, driving any :class:`Engine`.

    Args:
        convergence: geomeTRIC convergence set, one of
            :data:`CONVERGENCE_SETS`, case-insensitively.
        max_iterations: Maximum number of optimization steps.
        coordsys: geomeTRIC coordinate system, one of
            :data:`COORDINATE_SYSTEMS`, case-insensitively.
        workdir: Directory for geomeTRIC's logs and trajectories. By default a
            temporary directory is used and removed afterwards.
        verbose: Print geomeTRIC's progress output to stderr and its log file.

    The caller's logging handlers are never touched. While a run is in progress
    ``logging.config.fileConfig`` is replaced process-wide, so do not run
    optimizers from several threads at once.
    """

    def __init__(
        self,
        convergence: str = "GAU_TIGHT",
        max_iterations: int = 300,
        coordsys: str = "tric",
        workdir: str | os.PathLike[str] | None = None,
        verbose: bool = False,
    ) -> None:
        self._geometric = _import_geometric()
        self.convergence = convergence.upper()
        if self.convergence not in CONVERGENCE_SETS:
            raise ValueError(f"convergence: unknown convergence set {convergence!r}")
        self.coordsys = coordsys.lower()
        if self.coordsys not in COORDINATE_SYSTEMS:
            raise ValueError(f"coordsys: unknown coordinate system {coordsys!r}")
        self.max_iterations = max_iterations
        self.workdir = workdir
        self.verbose = verbose

    def key(self) -> str:
        """Identify the settings that can change a converged result.

        ``max_iterations``, ``workdir`` and ``verbose`` are not part of it.
        """
        options = (("convergence", self.convergence), ("coordsys", self.coordsys))
        return "geometric" + render_options(options)

    def minimize(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        return self._run(mol, engine, hessian="never")

    def saddle(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        hess = engine.hessian(mol)
        return self._run(mol, engine, transition=True, hess_data=hess.tolist())

    def _run(self, mol: Molecule, engine: Engine, **kwargs: Any) -> OptimizationResult:
        geometric = self._geometric
        bridge = _bridge(geometric, mol, engine)
        # geomeTRIC's step code can emit numpy RuntimeWarnings (e.g. 0/0 on an
        # exactly quadratic surface); they are not actionable.
        quiet = (
            contextlib.nullcontext()
            if self.verbose
            else warnings.catch_warnings(action="ignore")
        )
        with (
            quiet,
            _workdir(self.workdir) as workdir,
            _geometric_logging(geometric, self.verbose),
        ):
            try:
                progress = geometric.optimize.run_optimizer(
                    customengine=bridge,
                    input=str(workdir / "pyvtst"),
                    convergence_set=self.convergence,
                    maxiter=self.max_iterations,
                    coordsys=self.coordsys,
                    **kwargs,
                )
            except geometric.errors.GeomOptNotConvergedError as exc:
                raise ConvergenceError(
                    f"geomeTRIC did not converge in {self.max_iterations} "
                    f"iterations ({exc})",
                    bridge.last_molecule,
                ) from exc
        final = np.asarray(progress.xyzs[-1], dtype=np.float64) * ANGSTROM_TO_BOHR
        return OptimizationResult(
            molecule=mol.replace(coords=final),
            energy=float(progress.qm_energies[-1]),
            n_iterations=len(progress.xyzs) - 1,
        )


def _bridge(geometric: Any, mol: Molecule, engine: Engine) -> Any:
    """Wrap ``engine`` as a geomeTRIC custom engine for ``mol``."""

    class Bridge(geometric.engine.Engine):  # type: ignore[misc]
        def __init__(self) -> None:
            gmol = geometric.molecule.Molecule()
            gmol.elem = list(mol.symbols)
            gmol.xyzs = [mol.coords * BOHR_TO_ANGSTROM]
            super().__init__(gmol)
            self.last_molecule: Molecule | None = None

        def calc_new(self, coords: FloatArray, dirname: str) -> dict[str, Any]:
            m = mol.replace(coords=np.reshape(coords, (-1, 3)))
            self.last_molecule = m
            return {
                "energy": engine.energy(m),
                "gradient": np.ravel(engine.gradient(m)),
            }

    return Bridge()


@contextlib.contextmanager
def _workdir(workdir: str | os.PathLike[str] | None) -> Generator[Path]:
    """Yield an absolute working directory; a temporary one is removed on exit."""
    if workdir is not None:
        path = Path(workdir).absolute()
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="pyvtst-geometric-") as tmp:
        yield Path(tmp)


@contextlib.contextmanager
def _geometric_logging(geometric: Any, verbose: bool) -> Generator[None]:
    """Confine geomeTRIC's logging to its own logger for the duration of a run.

    ``logging.config.fileConfig`` is replaced by a function that attaches
    handlers to the ``geometric`` logger only: stderr and the run's log file
    when verbose. The logger is detached from the root so the caller's handlers
    see none of its output, and holds a ``NullHandler`` so that, when quiet,
    its warnings do not reach :data:`logging.lastResort` either.
    """
    logger = logging.getLogger("geometric")
    propagate = logger.propagate
    added: list[logging.Handler] = [logging.NullHandler()]
    logger.addHandler(added[0])

    def configure(
        fname: Any,
        defaults: Mapping[str, str] | None = None,
        disable_existing_loggers: bool = True,
        encoding: str | None = None,
    ) -> None:
        if not verbose:
            return
        handlers = [geometric.nifty.RawStreamHandler(sys.stderr)]
        if defaults and "logfilename" in defaults:
            handlers.append(geometric.nifty.RawFileHandler(defaults["logfilename"]))
        for handler in handlers:
            logger.addHandler(handler)
        added.extend(handlers)

    file_config = logging.config.fileConfig
    logging.config.fileConfig = configure
    logger.propagate = False
    try:
        yield
    finally:
        logging.config.fileConfig = file_config
        logger.propagate = propagate
        for handler in added:
            logger.removeHandler(handler)
            handler.close()
