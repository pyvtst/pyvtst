<p align="center">
  <img src="https://raw.githubusercontent.com/pyvtst/pyvtst/main/docs/logo.svg" alt="PyVTST" width="400">
</p>

<p align="center">
  <a href="https://github.com/pyvtst/pyvtst/actions/workflows/tests.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/pyvtst/pyvtst/tests.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://codecov.io/gh/pyvtst/pyvtst"><img src="https://img.shields.io/codecov/c/github/pyvtst/pyvtst?logo=codecov&label=coverage" alt="Coverage"></a>
  <a href="https://pypi.org/project/pyvtst/"><img src="https://img.shields.io/pypi/v/pyvtst.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/pyvtst"><img src="https://static.pepy.tech/badge/pyvtst/month" alt="Downloads"></a>
  <a href="https://github.com/pyvtst/pyvtst"><img src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpyvtst%2Fpyvtst%2Fmain%2Fpyproject.toml" alt="Python versions"></a>
  <a href="https://github.com/pyvtst/pyvtst/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pyvtst/pyvtst.svg" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/types-mypy%20strict-blue" alt="mypy strict"></a>
</p>

Thermal rate constants for gas-phase chemical reactions from transition state
theory, with tunneling corrections.

> **Status: pre-alpha.** Conventional TST rate constants with Wigner and
> Eckart tunneling, and minimum-energy paths, are available; CVT and
> zero-curvature tunneling are not yet.

## Installation

```bash
pip install pyvtst              # numpy and scipy only
pip install "pyvtst[ase]"       # + conversion to and from ase.Atoms
pip install "pyvtst[rdkit]"     # + 3D structures from SMILES
pip install "pyvtst[pyscf]"     # + energies, gradients and Hessians from PySCF
pip install "pyvtst[geometric]" # + geometry optimization with geomeTRIC
pip install "pyvtst[hdf5]"      # + result caching in HDF5 files
pip install "pyvtst[plot]"      # + reaction path plots with matplotlib
```

## Quick start

`Reaction` takes geometry guesses for the reactants and the transition state,
re-optimizes them at the engine's level of theory, checks that the transition
state has exactly one imaginary frequency, and computes conventional TST rate
constants. This example needs the `pyscf`, `geometric` and `hdf5` extras.

```python
from pyvtst import Molecule, Reaction
from pyvtst.engines import PySCFEngine

h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], symmetry_number=2)
h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
h3_guess = Molecule(
    ["H", "H", "H"],
    [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]],  # bohr
    mult=2,
    symmetry_number=2,
)

rxn = Reaction([h2, h], h3_guess, PySCFEngine("B3LYP", "def2-SVP"), store="h3.h5")
result = rxn.rates([300, 500, 1000], tunneling="eckart")
print(result.table())
```

`rates(T, theory="tst", tunneling=None)` accepts `tunneling=None`, `"wigner"`
or `"eckart"`. Eckart uses an asymmetric barrier when `products=` are given to
`Reaction` and a symmetric one otherwise. Unimolecular rate constants are in
s⁻¹ and bimolecular ones in cm³ molecule⁻¹ s⁻¹. `result.k`, `result.k_tst` and
`result.kappa` are numpy arrays, one value per temperature.

Reaction-path degeneracy comes only from each molecule's `symmetry_number`,
so set it on reactants and transition state. Partition functions use the rigid
rotor and harmonic oscillator; thermochemistry from
`pyvtst.core.partition.thermo` uses the 1 bar standard state.

## Usage

### Molecules

pyvtst stores geometries in bohr and masses in amu (most abundant isotope by
default).

```python
from pyvtst import Molecule
from pyvtst.io import from_ase, from_smiles, read_xyz, to_ase, write_xyz

water = read_xyz("water.xyz")
methyl = from_smiles("[CH3]")  # multiplicity 2 inferred
h_atom = read_xyz("h.xyz", mult=2)
write_xyz(water, "water_out.xyz")
atoms = to_ase(water)  # and from_ase(atoms)

h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])  # bohr
```

Unit conversion factors live in `pyvtst.units`, derived from the CODATA
values shipped with SciPy:

```python
from pyvtst.units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_PER_MOL
```

### Energies with PySCF

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

### Caching

Optimizations, energies, gradients and Hessians are cached, so a repeated
call costs nothing. With `Reaction`, pass `store=`:

- omitted: results are kept in memory for the life of the `Reaction`;
- a path such as `"h3.h5"`: results are kept in an HDF5 file (needs the
  `hdf5` extra), and a later session with the same inputs reuses all of them;
- a store object from `pyvtst.stores` (`MemoryStore`, `HDF5Store`).

`CachedEngine` is for scripts that call engines or optimizers directly:

```python
from pyvtst.engines import CachedEngine, PySCFEngine
from pyvtst.stores import HDF5Store

engine = CachedEngine(PySCFEngine("B3LYP", "def2-SVP"), HDF5Store("results.h5"))
engine.hessian(h2)  # computed once, then read from results.h5
```

A `CachedEngine` passed to `Reaction` is used as it is: its results stay in
its own store, and the `Reaction`'s optimized geometries go to `store=`.

### Reaction path

`compute_mep()` traces the minimum-energy path (MEP) through the transition
state, optimizing first if needed. With `rxn` from the quick start:

```python
path = rxn.compute_mep(s_min=-0.5, s_max=0.5)  # bohr
print(path.table())
```

s is the arc length in mass-scaled coordinates with a scaling mass of 1 amu,
so it is in bohr, with s = 0 at the transition state and s < 0 on the reactant
side. `path.s`, `path.v_mep`, `path.v_ag` and `path.zpe` are numpy arrays, one
value per point. V_MEP is the energy relative to the reactants and V_a^G adds
the generalized zero-point energy, relative to the reactants' zero-point
level, both in hartree. `path.frequencies` holds the generalized frequencies
(cm⁻¹) at each point with a Hessian: translations, rotations and the path
direction are projected out.

| `method` | Default `ds` | Hessians |
| --- | --- | --- |
| `"lqa"` (default) | 0.01 bohr | At every point (Page–McIver local quadratic approximation) |
| `"euler"` | 0.001 bohr | Every `hessian_every` points, default 10 (steepest descent from gradients) |

With PySCF a Hessian costs far more than a gradient. The default LQA path
from −0.5 to 0.5 bohr needs 100 Hessians. Euler at its defaults also computes
one Hessian every 0.01 bohr, plus ten times as many gradients; raise
`hessian_every` when Hessians are expensive. V_a^G and the frequencies exist
only at points with a Hessian.

Every point goes to the `Reaction`'s store as soon as it is computed. An
interrupted `compute_mep()` resumes from the last stored point, wider limits
compute only the new points, and a repeated call is free. A different
`method`, `ds` or `hessian_every` starts a new path.

Which side is s < 0 is decided from geometry, so list the transition state's
atoms in the same order as the reactants' atoms concatenated (and the
products', if given), with the same elements and masses. Otherwise the path
is still computed, `path.oriented` is false, and a warning says the sign of s
is arbitrary. A side that stops before its limit, at a minimum or because the
next step would go uphill (try a smaller `ds`), gives a warning;
`path.stop_reactant` and `path.stop_product` say why each side stopped.

`plot_path` and `plot_paths` (with the `plot` extra) draw V_MEP as a solid
line and V_a^G as a dashed one, mark the transition state and return the
matplotlib `Axes`. Paths from different methods, step sizes or engines share
one axis, one color per path:

```python
from pyvtst import plot_path, plot_paths

ax = plot_path(path)
ax.figure.savefig("h3_mep.png")

euler = rxn.compute_mep(method="euler")
ax = plot_paths([path, euler], labels=["LQA", "Euler"], units="kJ/mol")
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. In short, set up a
development environment with [uv](https://docs.astral.sh/uv/) and run the same
checks as CI:

```bash
uv sync --all-extras                    # dev tools + all optional adapters
uv run pytest
uv run ruff format --check && uv run ruff check
uv run mypy                             # strict
uv run lint-imports                     # architecture contracts
uv run bandit -c pyproject.toml -r src
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
