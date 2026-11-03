# Variational transition state theory

**Variational transition state theory** (VTST) moves the dividing surface of
[Transition state theory](tst.md) off the saddle point, to wherever the
reaction's real bottleneck is. This page shows how to use it, then explains the theory, then how PyVTST
computes it.

## Using it

Pick the theory with `theory=`:

```python
cvt = rxn.rates([300, 500, 1000], theory="cvt")
print(cvt.table())  # T, s*, k_TST, k_CVT, kappa, k
print(cvt.cvt.s_star, cvt.cvt.k / cvt.k_tst)  # s*(T); the variational effect

icvt = rxn.rates([300, 500, 1000], theory="icvt")
muvt = rxn.rates([300, 500, 1000], theory="muvt")
```

Without `path=`, `rates()` traces the default path from −0.5 to 0.5 bohr; see
[Reaction path](mep.md). A warning says when the bottleneck may lie beyond
the path; then compute a wider one and pass it with `path=`. Tunneling
combines with any theory through `tunneling=`; see
[Multidimensional tunneling](tunneling.md).

## Theory

### Why the saddle point is not enough

TST counts every system that crosses the dividing surface at the saddle point
as reacting. Some of them turn back and recross, so the true rate can only be
*lower* than the TST count: TST gives an upper bound to the classical rate
constant. That bound is the tightest when the dividing surface sits where the
fewest systems cross, and that is not always the saddle point.

The bottleneck of a reaction is where the *free energy*, not the potential
energy, is highest. Along the [minimum-energy path](mep.md) the vibrations
across the path stiffen or loosen, and the zero-point energy and the entropy
change with them. The free-energy maximum can therefore lie away from the
saddle point, and at a different place at each temperature.

### Generalized transition states

A dividing surface at right angles to the path at any $s$ defines a
**generalized transition state**. Its rate constant has the TST form, with
the properties of the path point at $s$ in place of those of the saddle
point:

$$
k^\mathrm{GT}(T, s) = \frac{k_B T}{h}\,
\frac{Q^\mathrm{GT}(T, s)}{\prod_\mathrm{R} Q_\mathrm{R}(T)}\,
e^{-V_a^G(s) / k_B T}.
$$

- $Q^\mathrm{GT}(T, s)$ is the partition function of the geometry at $s$,
  with its vibrations given by the
  [generalized frequencies](mep.md#along-the-path) and counted from their
  zero-point level.
- $V_a^G(s)$ is the vibrationally adiabatic ground-state potential, measured
  from the reactants' zero-point level. At $s = 0$ it equals
  $\Delta E_0^\ddagger$, so $k^\mathrm{GT}(T, 0) = k^\mathrm{TST}(T)$.

In thermodynamic form, $k^\mathrm{GT}$ is set by the **generalized free
energy of activation** $\Delta G^\mathrm{GT,\circ}(T, s)$, as
$k^\mathrm{TST}$ is set by $\Delta G^{\ddagger,\circ}$.

### Canonical variational TST

Every generalized transition state gives an upper bound, so the best one is
the lowest. **Canonical variational TST** (CVT) takes, at each temperature,

$$
k^\mathrm{CVT}(T) = \min_s\, k^\mathrm{GT}(T, s).
$$

Minimizing the rate is the same as maximizing
$\Delta G^\mathrm{GT,\circ}(T, s)$. The position of the maximum, $s^*(T)$, is
the **variational transition state**:

```{raw} html
:file: ../_generated/figures/free_energy.html
```

*Generalized free energy of activation along $s$ for a model reaction with a
transverse vibration that stiffens past the saddle point. The maximum, the
variational transition state, moves away from the saddle point as the
temperature rises.*

$k^\mathrm{CVT} \le k^\mathrm{TST}$ at every temperature, so CVT is never
further from the classical rate constant than TST. The ratio
$k^\mathrm{CVT}/k^\mathrm{TST}$ measures the **variational effect**. It
matters most for reactions with low or no barriers and at high temperature.

### Improved and microcanonical VTST

$Q^\mathrm{GT}$ counts every state of the generalized transition state,
including those whose total energy lies below the top of the adiabatic
barrier, $V^\mathrm{AG} = \max_s V_a^G(s)$. A system in such a state cannot
cross that barrier classically, so it should not count:

```{raw} html
:file: ../_generated/figures/threshold.html
```

*Vibrational levels of two generalized transition states on a model
barrier, with one 400 cm⁻¹ vibration across the path standing in for all
their states. Each ladder starts at $V_a^G(s)$, which already holds the
zero-point energy. At the top of the barrier every level lies at or above
$V^\mathrm{AG}$. Further down the path the lowest levels lie below it: CVT
counts them in $Q^\mathrm{GT}$, although a system in them cannot cross the
barrier.*

Both refinements
below start from the **threshold energy** $E_\mathrm{th}$: $V^\mathrm{AG}$,
or the reactants' or products' zero-point level if that is higher.

Written with the sum of states $N^\mathrm{GT}(\varepsilon, s)$, the number of
rovibrational states of the generalized transition state up to an energy
$\varepsilon$ above its zero-point level, the generalized rate constant is a
Boltzmann average over the total energy $E$:

$$
k^\mathrm{GT}(T, s) = \frac{1}{h\,\Phi_\mathrm{R}(T)}
\int_{V_a^G(s)}^{\infty} N^\mathrm{GT}\!\left(E - V_a^G(s), s\right)
e^{-E/k_B T}\,\mathrm{d}E,
$$

with $\Phi_\mathrm{R}$ the reactants' partition function per unit volume.

**Improved CVT** (ICVT) starts the integral at $E_\mathrm{th}$ instead of
$V_a^G(s)$, then minimizes over $s$ as CVT does:

$$
k^\mathrm{ICVT}(T) = \min_s\, k^\mathrm{GT}(T, s)\, F(T, s),
$$

where $F(T, s) \le 1$ is the Boltzmann-weighted fraction of the point's
states at or above $E_\mathrm{th}$.

**Microcanonical VTST** (µVT) goes one step further and finds the bottleneck
separately at each energy, where the fewest states are open:

$$
k^\mathrm{\mu VT}(T) = \frac{1}{h\,\Phi_\mathrm{R}(T)}
\int_{E_\mathrm{th}}^{\infty} \min_s N^\mathrm{GT}\!\left(E - V_a^G(s), s\right)
e^{-E/k_B T}\,\mathrm{d}E.
$$

Each step can only lower the rate:
$k^\mathrm{\mu VT} \le k^\mathrm{ICVT} \le k^\mathrm{CVT} \le k^\mathrm{TST}$.
The differences matter at low temperature and when $s^*$ lies away from the
maximum of $V_a^G$.

VTST relaxes the no-recrossing assumption of TST, but motion along the path
stays classical; [Multidimensional tunneling](tunneling.md) adds the quantum
correction.

## How PyVTST computes it

`rxn.rates(T, theory="cvt")`, `"icvt"` or `"muvt"` evaluates the equations
above in five steps.

### 1. The path

VTST needs the reaction path. Without `path=`, `rates()` runs
`compute_mep()` with its defaults (−0.5 to 0.5 bohr), which costs nothing
once that path is stored; see [Reaction path](mep.md).

### 2. Usable points

Only path points with generalized frequencies, the ones with a Hessian, can
be used. Imaginary generalized frequencies of up to the transition state's
threshold (`imag_threshold`, 50 cm⁻¹ by default) count with their
magnitude. A point with a larger one is left out of the search, with a
warning that lists it.

### 3. Generalized rate constants

At each usable point PyVTST builds $\ln Q^\mathrm{GT}(T, s)$ from the same
four RRHO factors as for
[TST](tst.md#3-partition-functions), with the geometry at $s$ for the
translation and rotation and its generalized frequencies for the vibration.
$V_a^G(s)$ is the point's electronic energy plus its generalized ZPE, minus
the reactants' energies and ZPEs. Two choices keep CVT consistent with TST:

- The symmetry number is the transition state's at every $s$, so the
  reaction-path degeneracy does not change along the path.
- $V_a^G$ is measured from the reactants' zero-point level, not from the
  path's own reference, so one path serves the forward and the reverse
  reaction.

$\ln k^\mathrm{GT}$ is then assembled from logarithms, as in TST. At the
saddle point it reproduces $k^\mathrm{TST}$ exactly.

### 4. The variational minimum

At each temperature PyVTST takes the lowest $\ln k^\mathrm{GT}$ over the
usable points. When that point has a neighbour on each side, it fits a
parabola in $s$ through the three values of $\ln k^\mathrm{GT}$ and moves
$s^*$ and $k^\mathrm{CVT}$ to its vertex, if the vertex is a minimum between
the neighbours. This gives a smooth $s^*(T)$ between path points.

When the minimum falls on the first or last usable point, the bottleneck may
lie beyond the path. A warning names the temperatures and the side; compute
a wider path and pass it with `path=`.

### 5. ICVT and µVT

Both need $E_\mathrm{th}$ and the sums of states.

- **Threshold.** $V^\mathrm{AG}$ is the highest $V_a^G$ over the usable
  points. Energies are measured from the reactants' zero-point level, so
  that level is $E = 0$. $E_\mathrm{th} = \max(V^\mathrm{AG}, 0)$, or the
  products' zero-point level if `products=` was given and it is higher. The
  0 matters for a submerged barrier, $V^\mathrm{AG} < 0$: no state below the
  reactants' zero-point level can react, whatever the barrier. A maximum at an end of the path gives a warning,
  because the barrier top is then outside the path and $E_\mathrm{th}$ may be
  too low.
- **Sums of states.** $N^\mathrm{GT}(\varepsilon, s)$ is counted exactly for
  the harmonic generalized frequencies on a 1 cm⁻¹ grid (Beyer–Swinehart),
  starting from the classical rigid-rotor sum of states, which convolves the
  rotation in (Stein–Rabinovitch). The rotor is the one whose partition
  function TST uses. The grid reaches $40\,k_B T_\mathrm{max}$, plus the
  number of degrees of freedom in units of $k_B T_\mathrm{max}$, above the
  threshold.
- **Normalization.** Each point's discrete sum of states is normalized to its
  own $k^\mathrm{GT}(T, s)$. The counting error of the grid then cancels, so
  $k^\mathrm{ICVT} \le k^\mathrm{CVT}$ and $k^\mathrm{\mu VT} \le
  k^\mathrm{ICVT}$ hold exactly, and ICVT equals CVT wherever
  $V_a^G(s^*) = E_\mathrm{th}$.

ICVT computes $F(T, s)$ as the ratio of two trapezoid integrals of
$N^\mathrm{GT} e^{-E/k_B T}$ over the same grid, one from $E_\mathrm{th}$ and
one from $V_a^G(s)$, then takes the minimum over $s$ as in step 4.

µVT takes, at each energy of the grid, the lowest normalized sum of states
over the usable points, refined by the same parabola in $s$. It then
integrates over energy with the trapezoid rule. When more than 1 % of the
rate at some temperature comes from energies whose bottleneck is an end of
the path, a warning names the temperatures, the side and the energy range.

## More options

Pass a path for a wider range, another step size or other coordinates:

```python
path = rxn.compute_mep(s_min=-1.0, s_max=1.0)
result = rxn.rates([300, 500, 1000], theory="cvt", path=path)
```

`result.k_tst` stays the conventional TST rate, so `result.cvt.k /
result.k_tst` is the variational effect, at most 1. `result.cvt` also holds
the path points used (`s`), $k^\mathrm{GT}(T, s)$ (`k_gt`) and
$\Delta G^\mathrm{GT,\circ}(T, s)$ at the 1 bar standard state (`delta_g`,
hartree). Wigner and Eckart κ(T) are the saddle-point values in every
theory; ZCT and SCT follow the path.

ICVT and µVT work the same way:

```python
icvt = rxn.rates([200, 300, 500], theory="icvt", path=path)
muvt = rxn.rates([200, 300, 500], theory="muvt", path=path)
print(icvt.table())  # T, s*, k_TST, k_ICVT, kappa, k
print(muvt.table())  # T, k_TST, k_muVT, kappa, k
print(icvt.icvt.f, muvt.muvt.s_star)  # F(T, s); s*(E) on muvt.muvt.energies
```

`icvt.icvt` also holds $k^\mathrm{IGT}(T, s)$ (`k_igt`), and both hold
$E_\mathrm{th}$ (`e_th`), $V^\mathrm{AG}$ (`v_max`) and its position
(`s_max`). `muvt.muvt.n_min` is the lowest sum of states at each energy.
Give `products=` to `Reaction` so the products' zero-point level bounds
$E_\mathrm{th}$.

`plot_free_energy` (with the `plot` extra) draws
$\Delta G^\mathrm{GT,\circ}(T, s)$ against $s$, one curve per temperature,
with $s^*(T)$ marked:

```python
from pyvtst import plot_free_energy

ax = plot_free_energy(result, units="kJ/mol")
ax.figure.savefig("h3_cvt.png")
```

## Limits of the approximations

- **Dividing surfaces at right angles to the path.** The search covers one
  family of surfaces; a better one of another shape could give a lower rate.
- **Harmonic generalized frequencies.** As for TST, low-frequency torsions are
  vibrations, not hindered rotors. This matters more along the path, where
  modes loosen into rotations of the separating fragments.
- **Finite path.** $s^*$ can only be found inside the path. The end-of-path
  warnings say when it may lie beyond.
- **Classical motion along the path.** No tunneling; see
  [Multidimensional tunneling](tunneling.md).

References: B. C. Garrett and D. G. Truhlar, *J. Phys. Chem.* 83, 1079 (1979);
B. C. Garrett, D. G. Truhlar, R. S. Grev and A. W. Magnuson, *J. Phys. Chem.*
84, 1730 (1980).
