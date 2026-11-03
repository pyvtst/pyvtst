# Defining a reaction

A `Reaction` ties together the species, the engine that computes their
energies, and the store that caches the results:

```python
from pyvtst import Reaction
from pyvtst.engines import PySCFEngine

rxn = Reaction(
    [h2, h],  # reactants
    h3_guess,  # transition-state guess
    PySCFEngine("B3LYP", "def2-SVP"),
    products=[h, h2],  # optional
    store="h3.h5",
)
```

Building a `Reaction` makes no engine calls. The geometries are optimized the
first time a result needs them; see [Geometry optimizers](optimization.md).

## Reactants, transition state and products

`reactants` is one molecule (unimolecular) or two (bimolecular). This sets the
units of the rate constants: s⁻¹ for one reactant and cm³ molecule⁻¹ s⁻¹ for
two.

`ts` is a guess for the transition state. It does not need to be accurate,
but it should be closer to the saddle point than to either minimum.

`products` is optional, but several results depend on it:

- Eckart tunneling uses an asymmetric barrier with products and a symmetric
  one without.
- Zero- and small-curvature tunneling start at the higher of the reactants'
  and products' zero-point levels. Without products, the product side has no
  zero-point level to reach; see [Multidimensional tunneling](tunneling.md).

## Atom order

List the transition state's atoms in the same order as the reactants' atoms
concatenated, and the products' atoms in that order too. Above, the reactants
`[h2, h]` give the atom order H, H, H, and `h3_guess` follows it.
`compute_mep()` uses this order to decide which side of the path is the
reactant side. With another order the path is still computed, but the sign of
s is arbitrary and a warning says so; see [Reaction path](mep.md).

## Checks at construction

`Reaction` raises `ValueError` when the reactants and the transition state, or
the reactants and the products, differ in element composition or in total
charge:

```text
element composition differs: reactants H3, transition state H2
```

It warns when the transition state's multiplicity cannot be reached by
coupling the reactants' (or products') spins, for example a triplet
transition state from two singlets.

## Other arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `optimizer` | `GeometricOptimizer()` | Locates minima and the saddle point; see [Geometry optimizers](optimization.md) |
| `store` | in memory | Where results are cached; see [Caching and restarts](caching.md) |
| `freq_scale` | `1.0` | Frequency scale factor; see [Frequency scale factors](scaling.md) |
