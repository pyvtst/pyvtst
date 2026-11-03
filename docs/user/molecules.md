# Molecules

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

## Charge and multiplicity

`charge` (default 0) and `mult`, the spin multiplicity 2S + 1 (default 1), are
passed to the engine as given. Set `mult=2` for radicals such as H or OH.
`Molecule` raises `ValueError` when the multiplicity does not fit the number
of electrons.

## Isotopes

Masses default to the most abundant isotope of each element. Pass `masses=`
(amu, one per atom) for isotopic substitution, for example D₂:

```python
d2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], masses=[2.014, 2.014])
```

## Symmetry numbers

`symmetry_number` is the rotational symmetry number σ: 2 for H₂ and linear
H–H–H, 3 for NH₃, 12 for CH₄. It enters the rotational partition function,
and it is the only source of the reaction-path degeneracy, so set it on every
reactant and on the transition state. A chiral transition state needs an
extra factor of 2 for its two mirror-image paths, which cannot be expressed
yet: its rate constant comes out a factor of 2 too low.

