# Quantum chemistry engines

An engine computes energies, gradients and Hessians for a `Molecule`, at one
level of theory. `Reaction` calls it for every optimization step, stationary
point and path point. PyVTST ships an engine for PySCF, and any other program
can be connected through the `Engine` protocol.

## PySCF

`PySCFEngine` computes energies, gradients and Hessians with HF, MP2 or any
DFT functional PySCF knows. It needs the `pyscf` extra. All results are in
atomic units.

```python
from pyvtst import Molecule
from pyvtst.engines import PySCFEngine

water = Molecule(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],  # bohr
)
engine = PySCFEngine("B3LYP", "def2-SVP")

engine.energy(water)  # hartree
engine.gradient(water)  # hartree/bohr, shape (3, 3)
engine.hessian(water)  # hartree/bohr², shape (9, 9), not mass-weighted
```

The molecule's charge and multiplicity are used as given. Singlets get a
restricted reference and everything else an unrestricted one:

```python
hydroxyl = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], mult=2)
PySCFEngine("MP2", "cc-pVDZ").energy(hydroxyl)  # UHF-based MP2
```

Keyword options:

| Option | Default | Meaning |
| --- | --- | --- |
| `reference` | `"auto"` | `"restricted"` or `"unrestricted"` to override the choice above |
| `hessian` | `"auto"` | Analytic for HF and DFT, finite differences of gradients for MP2; `"analytic"` or `"fd"` to force |
| `fd_step` | `5e-3` | Finite-difference step in bohr |
| `conv_tol` | `1e-10` | SCF energy convergence threshold in hartree |
| `grid_level` | `3` | DFT integration grid level |
| `max_cycle` | `50` | Maximum SCF iterations |
| `max_memory_mb` | `None` | Memory budget for PySCF; `None` uses PySCF's default |
| `verbose` | `False` | Print PySCF's output |

PySCF keeps integrals in memory when they fit its budget, so keep it below the
RAM you have. Set it once with the `PYSCF_MAX_MEMORY` environment variable,
or per engine with `max_memory_mb`.

Each call runs a fresh SCF. A failed SCF raises `pyvtst.ports.EngineError`; a
basis set that lacks an element raises `ValueError`.

`engine.level.key()` is a string identifying the settings that change a
result, for use as a cache key. `max_cycle`, `max_memory_mb` and `verbose` are
not part of it.

## Other programs

`Reaction` accepts any object with the methods of the `Engine` protocol in
`pyvtst.ports`. All values are in atomic units:

```python
import numpy as np
from pyvtst import Molecule
from pyvtst.ports import LevelOfTheory


class MyEngine:
    @property
    def level(self) -> LevelOfTheory:
        # Every setting that can change a result belongs here: it is the cache key.
        return LevelOfTheory("ccsd(t)", "cc-pvtz")

    def energy(self, mol: Molecule) -> float:
        raise NotImplementedError  # hartree

    def gradient(self, mol: Molecule) -> np.ndarray:
        raise NotImplementedError  # hartree/bohr, shape (N, 3)

    def hessian(self, mol: Molecule) -> np.ndarray:
        # hartree/bohr², shape (3N, 3N), not mass-weighted
        raise NotImplementedError
```

Use the molecule's `charge` and `mult`, leave the molecule unchanged, and
raise `pyvtst.ports.EngineError` when a calculation cannot give a trustworthy
result. Results are cached under the level of theory and the engine's class
name; see [Caching and restarts](caching.md).
