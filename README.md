# pyvtst

pyvtst is a Python library for thermal rate constants of gas-phase chemical
reactions. It is planned to cover conventional transition state theory (TST),
canonical variational TST (CVT) along the minimum-energy path, and Wigner,
Eckart and zero-curvature tunneling corrections.

> **Status: pre-alpha.** The `Molecule` structure type, units, structure
> readers and the PySCF engine are in place; rate calculations are not
> available yet.

## Install

```bash
pip install pyvtst              # numpy and scipy only
pip install "pyvtst[ase]"       # + conversion to and from ase.Atoms
pip install "pyvtst[rdkit]"     # + 3D structures from SMILES
pip install "pyvtst[pyscf]"     # + energies, gradients and Hessians from PySCF
```

## Molecules

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

## Energies with PySCF

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

## Development

```bash
uv sync --all-extras                    # dev tools + all optional adapters
uv run pytest
uv run ruff format --check && uv run ruff check
uv run mypy                             # strict
uv run lint-imports                     # architecture contracts
uv run bandit -c pyproject.toml -r src
```

## License

BSD-3-Clause.
