# Quantum chemistry engines

An engine computes energies, gradients and Hessians for a `Molecule`, at one
level of theory. `Reaction` calls it for every optimization step, stationary
point and path point. PyVTST ships engines for PySCF and Gaussian 16, and any
other program can be connected through the `Engine` protocol.

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

## Gaussian

`GaussianEngine` runs Gaussian 16 for each geometry, converts the checkpoint
with `formchk` and reads the energy, gradient and Hessian from the formatted
checkpoint. Gaussian is licensed separately; PyVTST neither ships nor
downloads it. Importing `pyvtst.engines` works without Gaussian; only
constructing a `GaussianEngine` needs it.

```python
from pyvtst.engines import GaussianEngine

engine = GaussianEngine(
    "B3LYP",
    "def2SVP",
    keywords="EmpiricalDispersion=GD3BJ Int=UltraFine",
    nprocs=8,
    memory="16GB",
)
engine.hessian(water)  # one Freq job
engine.energy(water)  # served by the same job
```

Method, basis and `keywords` take any Gaussian names, written as in a route
section. Case and repeated spaces in `keywords` do not matter. Keywords that
name a job type or that the engine sets itself are rejected: `Opt`, `Freq`,
`Force`, `SP`, `IRC`, `Scan`, `Geom`, `Guess`, `Units`, `NoSymm` and
`Symmetry`. `Gen` and `GenECP` basis sets are not supported, since they need
extra input sections.

Keyword options:

| Option | Default | Meaning |
| --- | --- | --- |
| `keywords` | `""` | Extra route keywords, e.g. `"SCF=Tight Int=UltraFine"` |
| `reference` | `"auto"` | `"auto"` leaves it to Gaussian (restricted for singlets, unrestricted otherwise); `"restricted"` or `"unrestricted"` force it |
| `nprocs` | `1` | Processors per job (`%NProcShared`) |
| `memory` | `"2GB"` | Memory per job (`%Mem`) |
| `executable` | `None` | The Gaussian executable; see the lookup order below |
| `formchk` | `None` | The `formchk` utility; see the lookup order below |
| `workdir` | `None` | Keep each job's input, log and formatted checkpoint in a new subdirectory here; by default a temporary directory, removed afterwards |
| `timeout` | `None` | Seconds after which a job is stopped and `EngineError` raised |
| `verbose` | `False` | Pass Gaussian's own output through |

The executables are found in this order, separately for each:

1. The `executable` or `formchk` argument.
2. The environment variable `PYVTST_G16` or `PYVTST_FORMCHK`.
3. `g16` or `formchk` on `PATH`.

An argument or a set variable that does not name an executable file is an
error; PyVTST does not fall back to `PATH`, so it never runs a different
Gaussian install than the one you chose. An empty variable counts as unset.

Gaussian runs with your environment unchanged, so source `g16.profile` (or
set `g16root`) as you would to run Gaussian by hand. Scratch files go to
`GAUSS_SCRDIR` when it is set, and to the job's directory otherwise.

Each geometry costs one Gaussian run. A `Force` job serves both the energy
and the gradient, and a `Freq` job serves the Hessian together with the
energy and the gradient. The engine keeps the last 8 results, and
`compute_mep` asks for the Hessian first at each path point, so a path point
is one `Freq` job. Jobs use `NoSymm` and coordinates in bohr, so results are
in the molecule's own orientation.

A job that ends without normal termination, a failed `formchk`, a missing
result or a timeout raises `pyvtst.ports.EngineError`, with the end of the
Gaussian log in the message. Results never depend on earlier calls: each job
starts from Gaussian's default SCF guess.

`engine.level` holds the method, basis, `keywords` and `reference`. The other
options are not part of it, so they do not change cache keys.

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
