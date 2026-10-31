"""PySCF engine: HF, DFT and MP2. PySCF is imported when an engine is built."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyvtst.core.finite_difference import fd_hessian
from pyvtst.core.level import LevelOfTheory, OptionValue
from pyvtst.core.molecule import Molecule
from pyvtst.ports.errors import EngineError

FloatArray = npt.NDArray[np.float64]
Reference = Literal["auto", "restricted", "unrestricted"]
HessianMode = Literal["auto", "analytic", "fd"]


def _import_pyscf() -> Any:
    # PySCF is optional, so it is imported on first use rather than at load time.
    try:
        import pyscf  # noqa: PLC0415
        import pyscf.dft.libxc  # noqa: PLC0415
        import pyscf.mp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'PySCFEngine needs PySCF: pip install "pyvtst[pyscf]"'
        ) from exc
    return pyscf


class PySCFEngine:
    """Energies, gradients and Hessians from PySCF.

    Args:
        method: ``"HF"``, ``"MP2"`` or any exchange-correlation functional PySCF
            recognises, case-insensitively.
        basis: Basis-set name.
        reference: ``"auto"`` uses a restricted reference for singlets and an
            unrestricted one otherwise; ``"restricted"`` or ``"unrestricted"``
            force the choice.
        hessian: ``"auto"`` uses analytic Hessians for HF and DFT and finite
            differences of analytic gradients for MP2; ``"analytic"`` and
            ``"fd"`` force the choice.
        fd_step: Finite-difference step in bohr (finite-difference Hessians
            only).
        conv_tol: SCF energy convergence threshold in hartree.
        grid_level: DFT integration grid level (DFT only).
        max_cycle: Maximum number of SCF iterations.
        max_memory_mb: Memory budget for PySCF in MB. PySCF keeps integrals in
            memory when they fit in this budget, so set it no higher than the
            RAM available. ``None`` keeps PySCF's own default, which honours
            the ``PYSCF_MAX_MEMORY`` environment variable.
        verbose: Print PySCF's output.

    Each call runs a fresh SCF, so results never depend on earlier calls.
    """

    def __init__(
        self,
        method: str,
        basis: str,
        *,
        reference: Reference = "auto",
        hessian: HessianMode = "auto",
        fd_step: float = 5e-3,
        conv_tol: float = 1e-10,
        grid_level: int = 3,
        max_cycle: int = 50,
        max_memory_mb: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._pyscf = _import_pyscf()
        name = method.strip().lower()
        if name in ("hf", "mp2"):
            self._kind = name
        else:
            try:
                self._pyscf.dft.libxc.parse_xc(name)
            except Exception as exc:
                raise ValueError(f"method: unknown method {method!r}") from exc
            self._kind = "dft"
        if reference not in ("auto", "restricted", "unrestricted"):
            raise ValueError(f"reference: unknown reference {reference!r}")
        if hessian not in ("auto", "analytic", "fd"):
            raise ValueError(f"hessian: unknown Hessian mode {hessian!r}")
        if hessian == "analytic" and self._kind == "mp2":
            raise ValueError(f"hessian: no analytic Hessian for method {method!r}")
        if not fd_step > 0:
            raise ValueError(f"fd_step: must be positive, got {fd_step}")
        if max_cycle < 1:
            raise ValueError(f"max_cycle: must be at least 1, got {max_cycle}")
        if max_memory_mb is not None and max_memory_mb < 1:
            raise ValueError(f"max_memory_mb: must be at least 1, got {max_memory_mb}")

        self._reference = reference
        self._analytic_hessian = hessian != "fd" and self._kind != "mp2"
        self._fd_step = float(fd_step)
        self._conv_tol = float(conv_tol)
        self._grid_level = int(grid_level)
        self._max_cycle = int(max_cycle)
        self._max_memory_mb = None if max_memory_mb is None else int(max_memory_mb)
        self._verbose = verbose

        # The key records the Hessian mode actually used, so "auto" and the
        # mode it resolves to share a key; fd_step only matters for "fd".
        options: list[tuple[str, OptionValue]] = [("reference", reference)]
        if self._analytic_hessian:
            options.append(("hessian", "analytic"))
        else:
            options += [("hessian", "fd"), ("fd_step", self._fd_step)]
        options.append(("conv_tol", self._conv_tol))
        if self._kind == "dft":
            options.append(("grid_level", self._grid_level))
        self._level = LevelOfTheory(method, basis, tuple(options))

    @property
    def level(self) -> LevelOfTheory:
        return self._level

    def energy(self, mol: Molecule) -> float:
        mf, post = self._run(mol)
        return float(post.e_tot if post is not None else mf.e_tot)

    def gradient(self, mol: Molecule) -> FloatArray:
        mf, post = self._run(mol)
        solver = post if post is not None else mf
        grad = solver.nuc_grad_method().kernel()
        return np.array(grad, dtype=np.float64).reshape(mol.n_atoms, 3)

    def hessian(self, mol: Molecule) -> FloatArray:
        if not self._analytic_hessian:
            return fd_hessian(
                lambda x: self.gradient(mol.replace(coords=x)),
                mol.coords,
                self._fd_step,
            )
        mf, _ = self._run(mol)
        n = mol.n_atoms
        h = np.asarray(mf.Hessian().kernel(), dtype=np.float64)
        return h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n).copy()

    def _restricted(self, mol: Molecule) -> bool:
        if self._reference == "restricted" and mol.mult != 1:
            raise ValueError(
                f"reference: a restricted reference needs multiplicity 1, "
                f"got {mol.mult}"
            )
        if self._reference == "auto":
            return mol.mult == 1
        return self._reference == "restricted"

    def _build(self, mol: Molecule) -> Any:
        try:
            return self._pyscf.gto.M(
                atom=list(zip(mol.symbols, mol.coords.tolist(), strict=True)),
                unit="Bohr",
                basis=self._level.basis,
                charge=mol.charge,
                spin=mol.mult - 1,
                verbose=4 if self._verbose else 0,
                max_memory=self._max_memory_mb,
            )
        except RuntimeError as exc:
            # PySCF raises RuntimeError (BasisNotFoundError among others) for
            # missing basis functions and inconsistent charge/spin.
            raise ValueError(
                f"{self._level.key()}: cannot describe {_label(mol)} ({exc})"
            ) from exc

    def _run(self, mol: Molecule) -> tuple[Any, Any]:
        """Run the SCF (and MP2), returning the SCF object and the MP2 object."""
        pyscf = self._pyscf
        restricted = self._restricted(mol)
        pmol = self._build(mol)
        if self._kind == "dft":
            mf = pyscf.dft.RKS(pmol) if restricted else pyscf.dft.UKS(pmol)
            mf.xc = self._level.method
            mf.grids.level = self._grid_level
        else:
            mf = pyscf.scf.RHF(pmol) if restricted else pyscf.scf.UHF(pmol)
        mf.conv_tol = self._conv_tol
        mf.max_cycle = self._max_cycle
        mf.kernel()
        if not mf.converged:
            raise EngineError(
                f"{self._level.key()}: SCF did not converge for {_label(mol)}"
            )
        if self._kind != "mp2":
            return mf, None
        post = pyscf.mp.MP2(mf)
        post.kernel()
        return mf, post


def _label(mol: Molecule) -> str:
    """The molecule's name, or its formula when it has none."""
    if mol.name:
        return mol.name
    counts = Counter(mol.symbols)
    return "".join(f"{s}{n if n > 1 else ''}" for s, n in counts.items())
