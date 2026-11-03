# Variational TST

## Background

In [Transition state theory](tst.md), recrossing can only make the true rate *lower* than the TST count, so TST
gives an upper bound to the classical rate constant. The saddle point is not
always the best place for the dividing surface: the bottleneck of the
reaction is where the *free energy*, not the potential energy, is highest.
Along the path, vibrations stiffen or loosen and the zero-point energy and
entropy change with them, so the free-energy maximum can lie away from the
saddle point, and at a different place at each temperature.

**Canonical variational transition state theory** (CVT) moves the dividing
surface along the MEP to the position that gives the lowest rate,

$$
k^\mathrm{CVT}(T) = \min_s\, k^\mathrm{GT}(T, s),
$$

where $k^\mathrm{GT}(T, s)$ is the generalized TST rate constant for a
dividing surface at $s$. Minimizing the rate is the same as maximizing the
generalized free energy of activation $\Delta G^\mathrm{GT}(T, s)$. The
position of the maximum, $s^*(T)$, is the **variational transition state**:

```{raw} html
:file: ../_generated/figures/free_energy.html
```

*Generalized free energy of activation along $s$ for a model reaction with a
transverse vibration that stiffens past the saddle point. The maximum, the
variational transition state, moves away from the saddle point as the
temperature rises.*

Both are upper bounds to the classical rate constant and
$k^\mathrm{CVT} \le k^\mathrm{TST}$ at every temperature, so CVT is never
further from it than TST. The ratio $k^\mathrm{CVT}/k^\mathrm{TST}$
measures the variational effect. It matters most for reactions with low or
no barriers and at high temperature. The rest of this page
shows how to compute it with PyVTST.

## Computing CVT rate constants

`theory="cvt"` gives canonical variational TST rate constants. At each
temperature the dividing surface is placed where the generalized TST rate
k^GT(T, s) along the MEP is lowest, which is where the generalized free energy
of activation is highest, at the variational transition state s*(T):

```python
result = rxn.rates([300, 500, 1000], theory="cvt", tunneling="wigner")
print(result.table())  # T, s*, k_TST, k_CVT, kappa, k
print(result.cvt.s_star, result.cvt.k / result.k_tst)
```

Without `path=`, `rates` runs `compute_mep()` with its defaults (s from −0.5
to 0.5 bohr), which is free once that path is stored. Pass a path for a wider
range, another step size or other coordinates:

```python
path = rxn.compute_mep(s_min=-1.0, s_max=1.0)
result = rxn.rates([300, 500, 1000], theory="cvt", path=path)
```

`result.k_tst` stays the conventional TST rate, so `result.cvt.k /
result.k_tst` is the variational effect, at most 1. `result.cvt` also holds
the path points used (`s`), k^GT(T, s) (`k_gt`) and ΔG^GT,°(T, s) at the 1 bar
standard state (`delta_g`, hartree). Between path points s* is refined with a
parabola through the three points around the lowest k^GT. The symmetry number
of the transition state is used at every s. Wigner and Eckart κ(T) are the
saddle-point values in every theory; ZCT and SCT follow the path.

When s* falls on the first or last usable point of the path at some
temperature, a warning names the temperatures and the side: the bottleneck may
lie beyond the path, so compute a wider one and pass it with `path=`. Points
with imaginary generalized frequencies above the transition state's threshold
are left out of the search, with a warning.

`plot_free_energy` (with the `plot` extra) draws ΔG^GT,°(T, s) against s, one
curve per temperature, with s*(T) marked:

```python
from pyvtst import plot_free_energy

ax = plot_free_energy(result, units="kJ/mol")
ax.figure.savefig("h3_cvt.png")
```

## ICVT and µVT

CVT counts every state of the generalized transition state, including those
below the top V^AG of the vibrationally adiabatic barrier, which cannot react.
Two theories leave them out. Both start from the threshold energy E_th, which
is V^AG or, if higher, the reactants' or products' zero-point level:

- `theory="icvt"` (improved CVT) multiplies k^GT(T, s) by the Boltzmann-weighted
  fraction F(T, s) of the point's states at or above E_th, then takes the
  minimum over s as CVT does.
- `theory="muvt"` (microcanonical variational TST) finds the bottleneck
  separately at each energy E ≥ E_th, where the sum of states
  N^GT(E − V_a^G(s), s) is lowest, then averages over a Boltzmann distribution.

```python
icvt = rxn.rates([200, 300, 500], theory="icvt", path=path)
muvt = rxn.rates([200, 300, 500], theory="muvt", path=path)
print(icvt.table())  # T, s*, k_TST, k_ICVT, kappa, k
print(muvt.table())  # T, k_TST, k_muVT, kappa, k
print(icvt.icvt.f, muvt.muvt.s_star)  # F(T, s); s*(E) on muvt.muvt.energies
```

k^µVT ≤ k^ICVT ≤ k^CVT. The difference matters at low temperature and when s*
moves away from the V_a^G maximum. Sums of states count the harmonic
vibrations exactly on a 1 cm⁻¹ grid, convolved with the same classical rigid
rotor as the partition functions, so ICVT equals CVT wherever V_a^G(s*) = E_th.
µVT warns when more than 1 % of its rate comes from energies whose bottleneck
is an end of the path.

References: B. C. Garrett and D. G. Truhlar, *J. Phys. Chem.* 83, 1079 (1979);
B. C. Garrett, D. G. Truhlar, R. S. Grev and A. W. Magnuson, *J. Phys. Chem.*
84, 1730 (1980).
