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
