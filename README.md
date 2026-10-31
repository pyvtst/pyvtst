# pyvtst

pyvtst is a Python library for thermal rate constants of gas-phase chemical
reactions. It is planned to cover conventional transition state theory (TST),
canonical variational TST (CVT) along the minimum-energy path, and Wigner,
Eckart and zero-curvature tunneling corrections.

> **Status: pre-alpha.** The `Molecule` structure type, units and structure
> readers are in place; rate calculations are not available yet.

## Install

```bash
pip install pyvtst              # numpy and scipy only
pip install "pyvtst[ase]"       # + conversion to and from ase.Atoms
pip install "pyvtst[rdkit]"     # + 3D structures from SMILES
```

## Molecules

pyvtst stores geometries in bohr and masses in amu (most abundant isotope by
default).

```python
import pyvtst

water = pyvtst.read_xyz("water.xyz")
methyl = pyvtst.from_smiles("[CH3]")  # multiplicity 2 inferred
h_atom = pyvtst.read_xyz("h.xyz", mult=2)
pyvtst.write_xyz(water, "water_out.xyz")

atoms = pyvtst.to_ase(water)  # and pyvtst.from_ase(atoms)
```

You can also build a `pyvtst.Molecule` directly from symbols and bohr
coordinates.

Unit conversion factors live in `pyvtst.units`, derived from the CODATA
values shipped with SciPy:

```python
from pyvtst.units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_PER_MOL
```

## Development

```bash
uv sync --all-extras                    # dev tools + ASE + RDKit
uv run pytest
uv run ruff format --check && uv run ruff check
uv run mypy                             # strict
uv run lint-imports                     # architecture contracts
uv run bandit -c pyproject.toml -r src
```

## License

BSD-3-Clause.
