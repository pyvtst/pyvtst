# How to install PyVTST

PyVTST needs Python 3.11 or later. The core package depends only on numpy and
scipy:

```bash
pip install pyvtst
```

Quantum chemistry, file formats, caching and plotting come as optional extras:

| Extra | Adds |
| --- | --- |
| `ase` | Conversion to and from `ase.Atoms` |
| `rdkit` | 3D structures from SMILES |
| `pyscf` | Energies, gradients and Hessians from PySCF |
| `geometric` | Geometry optimization with geomeTRIC |
| `hdf5` | Result caching in HDF5 files |
| `plot` | Path and free-energy plots with matplotlib |

Combine them as needed. The examples in this guide use:

```bash
pip install "pyvtst[pyscf,geometric,hdf5,plot]"
```

Gaussian 16 is not a Python package: `GaussianEngine` and `GaussianOptimizer`
run an existing Gaussian install, which you license and install separately.
PyVTST finds it through the `executable` argument, the `PYVTST_G16`
environment variable or `g16` on `PATH` (and likewise `formchk`); see
[Engines](engines.md#gaussian).
