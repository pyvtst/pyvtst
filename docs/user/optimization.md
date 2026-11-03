# Geometry optimizers

An optimizer locates minima and first-order saddle points, calling an engine
for the energies, gradients and Hessians it needs. `Reaction` uses it to
optimize the reactants, products and transition state. PyVTST ships an
optimizer for geomeTRIC, and any other can be connected through the
`Optimizer` protocol.

## Optimizing the stationary points

`Reaction` optimizes every species before it computes anything else:
multi-atom reactants and products to minima, the transition-state guess to a
first-order saddle point. It then computes a Hessian at each optimized
geometry to get the harmonic frequencies. Single atoms get only an energy.

`rates()` and `compute_mep()` do this on first use. Call `optimize()` to run
it on its own and inspect the result before computing a path:

```python
points = rxn.optimize()
print(points.ts.energy)  # hartree
print(points.ts.imaginary_frequencies)  # cm⁻¹, as negative values
print(points.reactants[0].frequencies)
```

`points.reactants`, `points.ts` and `points.products` (or `None`) are
`StationaryPoint` objects with the optimized `molecule`, its `energy` and its
`frequencies`. Optimized geometries are stored, so a later session with the
same inputs skips the optimization.

## geomeTRIC

`GeometricOptimizer` is the default optimizer and needs the `geometric`
extra. Pass one to `Reaction` to change its settings:

```python
from pyvtst.engines import PySCFEngine
from pyvtst.optimizers import GeometricOptimizer

engine = PySCFEngine("B3LYP", "def2-SVP")
optimizer = GeometricOptimizer(convergence="GAU_VERYTIGHT", max_iterations=500)
rxn = Reaction([h2, h], h3_guess, engine, optimizer=optimizer, store="h3.h5")
```

| Option | Default | Meaning |
| --- | --- | --- |
| `convergence` | `"GAU_TIGHT"` | geomeTRIC convergence set: `GAU`, `GAU_LOOSE`, `GAU_TIGHT`, `GAU_VERYTIGHT`, `INTERFRAG_TIGHT`, `NWCHEM_LOOSE` or `TURBOMOLE` |
| `coordsys` | `"tric"` | geomeTRIC coordinate system: `cart`, `prim`, `dlc`, `hdlc`, `tric` or `tric-p` |
| `max_iterations` | `300` | Maximum number of optimization steps |
| `workdir` | `None` | Directory for geomeTRIC's logs and trajectories; by default a temporary one, removed afterwards |
| `verbose` | `False` | Print geomeTRIC's progress |

`convergence` and `coordsys` are part of the cache key, so changing them
optimizes again. The saddle-point search starts with a Hessian at the guess,
which costs one engine Hessian per transition state.

geomeTRIC changes logging configuration process-wide while it runs, so do not
run optimizers from several threads at once.

## Checking the stationary points

After optimization each species is checked by its imaginary frequencies.
A reactant or product must have none and the transition state exactly one.
Otherwise `StationaryPointError` (from `pyvtst.core.stationary`) is raised,
naming the species and listing the imaginary frequencies:

```text
transition state has 2 imaginary frequencies, expected 1 ([-1520.3, -310.8] cm^-1)
```

A transition state with two or more imaginary frequencies is a higher-order
saddle point: start from a guess closer to the transition state. A minimum
with an imaginary frequency is usually a symmetric structure that should
distort: break the symmetry of the guess.

Imaginary frequencies of 50 cm⁻¹ or less are numerical noise. They are not
counted, enter the partition functions and zero-point energies as real
frequencies, and each gives a warning.

## When an optimization fails

An optimization that does not converge raises `ConvergenceError` (from
`pyvtst.ports`). The message starts with the species, and `last_molecule`
holds the last geometry evaluated:

```python
from pyvtst.ports import ConvergenceError

try:
    rxn.optimize()
except ConvergenceError as exc:
    print(exc)
    last = exc.last_molecule
```

`last_molecule` can be a rejected trial step rather than the best point
found, so check it before using it as a new guess. With geomeTRIC, raising
`max_iterations` or switching `coordsys` often helps. A failed SCF in the engine raises
`EngineError` instead; see [Quantum chemistry engines](engines.md).

## Other optimizers

`Reaction` accepts any object with the methods of the `Optimizer` protocol in
`pyvtst.ports`:

```python
from pyvtst import Molecule
from pyvtst.ports import Engine, OptimizationResult


class MyOptimizer:
    def minimize(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        # The minimum nearest to mol.
        raise NotImplementedError

    def saddle(self, mol: Molecule, engine: Engine) -> OptimizationResult:
        # A first-order saddle point, starting from the guess mol.
        raise NotImplementedError

    def key(self) -> str:
        # Every setting that can change a converged result: it is the cache key.
        return "myoptimizer?tol=1e-6"
```

Get energies, gradients and Hessians only from `engine`, so they are cached
with everything else. Return `OptimizationResult(molecule, energy,
n_iterations)`, where `molecule` is the input with new coordinates and
`energy` is in hartree. Raise `pyvtst.ports.ConvergenceError(message,
last_molecule)` when the optimization does not converge, and let
`EngineError` from the engine propagate. Counting imaginary frequencies is
not the optimizer's job: `Reaction` checks every result as described above.
Optimized geometries are cached under `key()`, the level of theory and the
engine; see [Caching and restarts](caching.md).
