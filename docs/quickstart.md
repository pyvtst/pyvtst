# Quickstart

This page walks through one reaction, H₂ + H → H + H₂, from geometry guesses to
variational rate constants with tunneling. Each step is covered in detail in
the [User Guide](user/index.md). Install PyVTST with the extras used here:

```bash
pip install "pyvtst[pyscf,geometric,hdf5,plot]"
```

## Molecules

A `Molecule` holds elements and coordinates in bohr. Set `mult` for open-shell
species and `symmetry_number` for the rotational symmetry number, which also
gives the reaction-path degeneracy:

```python
from pyvtst import Molecule

h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], symmetry_number=2)
h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
h3_guess = Molecule(
    ["H", "H", "H"],
    [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]],  # bohr
    mult=2,
    symmetry_number=2,
)
```

Geometries can also be read from XYZ files or built from SMILES; see
[Molecules](user/molecules.md).

## Conventional TST

`Reaction` re-optimizes the reactants and the transition-state guess with the
engine, checks that the transition state has exactly one imaginary frequency,
and computes rate constants. `store=` caches every result in an HDF5 file, so
running the script again makes no new engine calls.

```python
from pyvtst import Reaction
from pyvtst.engines import PySCFEngine

rxn = Reaction([h2, h], h3_guess, PySCFEngine("B3LYP", "def2-SVP"), store="h3.h5")
result = rxn.rates([300, 500, 1000], tunneling="eckart")
print(result.table())
```

Bimolecular rate constants are in cm³ molecule⁻¹ s⁻¹. See
[Transition state theory](user/tst.md), [Quantum chemistry engines](user/engines.md) and
[Caching and restarts](user/caching.md).

## Minimum-energy path

`compute_mep()` traces the path through the transition state. s is in bohr,
with s = 0 at the transition state and s < 0 on the reactant side:

```python
from pyvtst import plot_path

path = rxn.compute_mep(s_min=-2.0, s_max=2.0)
print(path.table())

ax = plot_path(path)
ax.figure.savefig("h3_mep.png")
```

See [Reaction path](user/mep.md).

## Variational TST and tunneling

`theory="cvt"` places the dividing surface where the rate is lowest at each
temperature. `tunneling="sct"` computes small-curvature tunneling along the
path, which must reach down to the reactants' zero-point level on both sides:

```python
result = rxn.rates([200, 300, 500], theory="cvt", tunneling="sct", path=path)
print(result.table())
```

See [Variational TST](user/cvt.md),
[Multidimensional tunneling](user/tunneling.md) and
[Frequency scale factors](user/scaling.md).
