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

ORCA 6 is not a Python package either: `OrcaEngine` and `OrcaOptimizer` run
an existing ORCA install. ORCA is free for academic use after registration
with its developers, and you install it separately. For runs with more than
one process, ORCA also needs the OpenMPI version that its release names.
PyVTST finds ORCA through the `executable` argument, the `PYVTST_ORCA`
environment variable or `orca` on `PATH`; see [Engines](engines.md#orca).
