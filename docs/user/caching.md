# Caching and restarts

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

## Restarts

Every optimized geometry and every path point goes to the store as soon as it
is computed, so with an HDF5 store an interrupted script loses at most the
calculation in progress. Run it again with the same inputs:

- optimizations already finished are read back, not repeated;
- an interrupted `compute_mep()` resumes from the last stored point;
- wider `s_min` or `s_max` compute only the new points;
- a repeated call is free.

A different `method`, `ds`, `hessian_every` or engine starts a new path; see
[Reaction path](mep.md).
