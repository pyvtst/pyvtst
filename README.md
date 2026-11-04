<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pyvtst/pyvtst/main/docs/_static/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pyvtst/pyvtst/main/docs/_static/logo-light.svg">
    <img src="https://raw.githubusercontent.com/pyvtst/pyvtst/main/docs/logo.svg" alt="PyVTST" width="400">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/pyvtst/pyvtst/actions/workflows/tests.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/pyvtst/pyvtst/tests.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://codecov.io/gh/pyvtst/pyvtst"><img src="https://img.shields.io/codecov/c/github/pyvtst/pyvtst?logo=codecov&label=coverage" alt="Coverage"></a>
  <a href="https://pypi.org/project/pyvtst/"><img src="https://img.shields.io/pypi/v/pyvtst.svg" alt="PyPI"></a>
  <a href="https://github.com/pyvtst/pyvtst"><img src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpyvtst%2Fpyvtst%2Fmain%2Fpyproject.toml" alt="Python versions"></a>
  <a href="https://github.com/pyvtst/pyvtst/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pyvtst/pyvtst.svg" alt="License"></a>
  <br>
  <a href="https://doi.org/10.5281/zenodo.23946059"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23946059.svg" alt="DOI"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/types-mypy%20strict-blue" alt="mypy strict"></a>
</p>

PyVTST computes thermal rate constants for gas-phase chemical reactions from
conventional and variational transition state theory: TST, and canonical,
improved canonical and microcanonical variational TST (CVT, ICVT and µVT),
with Wigner, Eckart, zero-curvature and small-curvature tunneling corrections.
It optimizes the stationary points, traces the minimum-energy path and
evaluates the rate constants from energies, gradients and Hessians computed
with PySCF.

> **Status: pre-alpha.** The API may change between releases.

## Getting started

### Installing PyVTST

```bash
pip install pyvtst              # numpy and scipy only
pip install "pyvtst[ase]"       # + conversion to and from ase.Atoms
pip install "pyvtst[rdkit]"     # + 3D structures from SMILES
pip install "pyvtst[pyscf]"     # + energies, gradients and Hessians from PySCF
pip install "pyvtst[geometric]" # + geometry optimization with geomeTRIC
pip install "pyvtst[hdf5]"      # + result caching in HDF5 files
pip install "pyvtst[plot]"      # + path and free-energy plots with matplotlib
```

### Using PyVTST

`Reaction` takes geometry guesses for the reactants and the transition state,
re-optimizes them at the engine's level of theory, checks that the transition
state has exactly one imaginary frequency, and computes rate constants. This
example needs the `pyscf`, `geometric` and `hdf5` extras.

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

`theory` is `"tst"` or `"cvt"`; `tunneling` is `None`, `"wigner"`, `"eckart"`,
`"zct"` or `"sct"`. Geometries are in bohr. The
[user guide](https://github.com/pyvtst/pyvtst/blob/main/docs/user/index.md) covers
molecules, engines, caching, reaction paths, variational TST, tunneling and
frequency scale factors.

## Citation

If you use PyVTST in your research, please cite it:

```bibtex
@software{rosca_pyvtst,
  author  = {Roșca, Emilian},
  title   = {{PyVTST}: variational transition state theory in Python},
  url     = {https://github.com/pyvtst/pyvtst},
  version = {0.1.0},
  year    = {2026},
}
```

The same metadata is in [CITATION.cff](https://github.com/pyvtst/pyvtst/blob/main/CITATION.cff).
