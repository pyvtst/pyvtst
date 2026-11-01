<p align="center">
  <img src="https://raw.githubusercontent.com/pyvtst/pyvtst/main/docs/logo.svg" alt="PyVTST" width="400">
</p>

<p align="center">
  <a href="https://github.com/pyvtst/pyvtst/actions/workflows/tests.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/pyvtst/pyvtst/tests.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://codecov.io/gh/pyvtst/pyvtst"><img src="https://img.shields.io/codecov/c/github/pyvtst/pyvtst?logo=codecov&label=coverage" alt="Coverage"></a>
  <a href="https://pypi.org/project/pyvtst/"><img src="https://img.shields.io/pypi/v/pyvtst.svg" alt="PyPI"></a>
  <a href="https://github.com/pyvtst/pyvtst"><img src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpyvtst%2Fpyvtst%2Fmain%2Fpyproject.toml" alt="Python versions"></a>
  <a href="https://github.com/pyvtst/pyvtst/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pyvtst/pyvtst.svg" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/types-mypy%20strict-blue" alt="mypy strict"></a>
</p>

Thermal rate constants for gas-phase chemical reactions from conventional and
canonical variational transition state theory, with tunneling corrections.

> **Status: pre-alpha.** Conventional and canonical variational TST rate
> constants with Wigner and Eckart tunneling, and minimum-energy paths, are
> available; zero-curvature tunneling is not yet.

## Installation

```bash
pip install pyvtst              # numpy and scipy only
pip install "pyvtst[ase]"       # + conversion to and from ase.Atoms
pip install "pyvtst[rdkit]"     # + 3D structures from SMILES
pip install "pyvtst[pyscf]"     # + energies, gradients and Hessians from PySCF
pip install "pyvtst[geometric]" # + geometry optimization with geomeTRIC
pip install "pyvtst[hdf5]"      # + result caching in HDF5 files
pip install "pyvtst[plot]"      # + path and free-energy plots with matplotlib
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

`rates(T, theory="tst", tunneling=None)` accepts `theory="tst"` or `"cvt"`
(see [Variational TST](#variational-tst)) and `tunneling=None`, `"wigner"`
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

Results are keyed by the level of theory and by the engine, so two programs
that name the same method and basis never share optimizations, path points or
energies. The engine is identified by its class name; pass
`CachedEngine(engine, store, engine_id=...)` to keep a cache across a class
rename.

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
direction are projected out. They are curvilinear by default; see
[Coordinates of the generalized frequencies](#coordinates-of-the-generalized-frequencies).

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
`method`, `ds`, `hessian_every` or engine starts a new path.

Which side is s < 0 is decided from geometry, so list the transition state's
atoms in the same order as the reactants' atoms concatenated (and the
products', if given), with the same elements and masses. Otherwise the path
is still computed, `path.oriented` is false, and a warning says the sign of s
is arbitrary. A side that stops before its limit, at a minimum or because the
next step would go uphill (try a smaller `ds`), gives a warning;
`path.stop_reactant` and `path.stop_product` say why each side stopped.

#### Coordinates of the generalized frequencies

`coordinates=` sets how the generalized frequencies are computed:

| `coordinates` | Generalized frequencies |
| --- | --- |
| `"redundant"` (default) | Curvilinear, in redundant internal coordinates (Jackels, Gu and Truhlar, *J. Chem. Phys.* 102, 3188 (1995)) |
| `"cartesian"` | Rectilinear, from the projected Cartesian Hessian |

Away from the transition state the Cartesian Hessian includes the gradient
times the curvature of straight-line motion. On curved paths that term turns
real bends imaginary, and V_a^G gets kinks where they drop out of the
zero-point energy. Internal coordinates bend with the molecule, so the term is
removed. Both give the same frequencies at the transition state. Away from
it, curvilinear frequencies depend on the set of internal coordinates, since
the set decides which motions count as curvilinear. When a Cartesian path has
large imaginary frequencies, the warning that lists them suggests
`coordinates="redundant"`.

By default the internal coordinates are generated from the optimized
transition state. Atoms closer than 1.3 times the sum of their covalent radii
are bonded, and separate fragments are bonded through their closest atoms.
Every two bonds on one atom give an angle, or a linear bend from 170°, and
every chain of three bonds gives a torsion. `path.internals` records the set,
and the same set is used at every point. To choose your own, pass tuples with
atom indices in the transition state's order:

```python
path = rxn.compute_mep(
    internals=[
        ("bond", 0, 1),
        ("bond", 1, 2),
        ("linear_bend", 0, 1, 2),  # two components, smooth through 180°
    ]
)
```

The kinds are `("bond", i, j)`, `("angle", i, j, k)` with apex j,
`("linear_bend", i, j, k)` with middle atom j and `("torsion", i, j, k, l)`
about the j–k bond. Redundant sets are fine. The set must span every internal
motion, which is checked at the transition state before tracing and again at
every point with a Hessian. When it does not, `compute_mep()` raises a
`ValueError` that says how many directions are missing, where, and which atoms
move most in them:

```text
internal coordinates span 2 of 4 internal directions at the transition state;
2 directions are missing, moving mostly atoms 1 (H), 2 (H), 0 (H); add
coordinates that involve them with internals=
```

Add a coordinate on those atoms: an angle, a linear bend for a nearly
collinear group, or a torsion. pyvtst never falls back to Cartesian
coordinates on its own.

The coordinates do not change the path, only the frequencies computed from
its stored Hessians, so switching makes no engine calls:

```python
cartesian = rxn.compute_mep(coordinates="cartesian")
```

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

### Variational TST

`theory="cvt"` gives canonical variational TST rate constants. At each
temperature the dividing surface is placed where the generalized TST rate
k^GT(T, s) along the MEP is lowest, which is where the generalized free energy
of activation is highest, at the variational transition state s*(T):

```python
result = rxn.rates([300, 500, 1000], theory="cvt", tunneling="wigner")
print(result.table())  # T, s*, k_TST, k_CVT, kappa, k
print(result.cvt.s_star, result.cvt.k / result.k_tst)
```

Without `path=`, `rates` runs `compute_mep()` with its defaults (s from −0.5
to 0.5 bohr), which is free once that path is stored. Pass a path for a wider
range, another step size or other coordinates:

```python
path = rxn.compute_mep(s_min=-1.0, s_max=1.0)
result = rxn.rates([300, 500, 1000], theory="cvt", path=path)
```

`result.k_tst` stays the conventional TST rate, so `result.cvt.k /
result.k_tst` is the variational effect, at most 1. `result.cvt` also holds
the path points used (`s`), k^GT(T, s) (`k_gt`) and ΔG^GT,°(T, s) at the 1 bar
standard state (`delta_g`, hartree). Between path points s* is refined with a
parabola through the three points around the lowest k^GT. The symmetry number
of the transition state is used at every s. Wigner and Eckart κ(T) are the
saddle-point values in both theories.

When s* falls on the first or last usable point of the path at some
temperature, a warning names the temperatures and the side: the bottleneck may
lie beyond the path, so compute a wider one and pass it with `path=`. Points
with imaginary generalized frequencies above the transition state's threshold
are left out of the search, with a warning.

`plot_free_energy` (with the `plot` extra) draws ΔG^GT,°(T, s) against s, one
curve per temperature, with s*(T) marked:

```python
from pyvtst import plot_free_energy

ax = plot_free_energy(result, units="kJ/mol")
ax.figure.savefig("h3_cvt.png")
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
