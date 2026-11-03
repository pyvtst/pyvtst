# Transition state theory

**Transition state theory** (TST) turns the
[potential energy surface](introduction.md#the-potential-energy-surface) into a
rate constant. This page explains the theory, then how PyVTST computes each
part of it.

## The barrier

The distance along the minimum-energy path is the **reaction coordinate**
$s$. PyVTST measures it in bohr, with $s = 0$ at the saddle point and
$s < 0$ on the reactant side. The energy along the path,
$V_\mathrm{MEP}(s)$, is the familiar energy profile:

```{raw} html
:file: ../_generated/figures/profile.html
```

*Energy along the reaction coordinate for an exothermic reaction. The barrier
is the energy of the transition state above the reactants.*

Molecules are never at rest at the bottom of the valley: even at 0 K each
vibration keeps its zero-point energy (ZPE), $\tfrac{1}{2}h\nu$. The barrier
that matters is therefore between zero-point levels, the **ZPE-corrected
barrier**

$$
\Delta E_0^\ddagger = \left(E^\ddagger + \mathrm{ZPE}^\ddagger\right)
- \sum_\mathrm{R} \left(E_\mathrm{R} + \mathrm{ZPE}_\mathrm{R}\right),
$$

where $E$ are electronic energies and the sum runs over the reactants. The
transition state has one vibration fewer than a minimum of the same atoms: its motion along the
reaction coordinate is not a vibration, so it has no zero-point energy.

## The TST rate constant

TST places a *dividing surface* across the valley at the saddle point, and
counts how fast systems in thermal equilibrium cross it from the reactant
side. The result is

$$
k^\mathrm{TST}(T) = \frac{k_B T}{h}\,
\frac{Q^\ddagger(T)}{\prod_\mathrm{R} Q_\mathrm{R}(T)}\,
e^{-\Delta E_0^\ddagger / k_B T}.
$$

- $k_B T/h$ is a universal frequency, about 6 × 10¹² s⁻¹ at 300 K: how
  often a system at the dividing surface moves across it.
- $Q_\mathrm{R}$ are the partition functions of the reactants and
  $Q^\ddagger$ that of the transition state *without* its motion along the
  reaction coordinate. Their ratio counts how many states are available at
  the transition state compared with the reactants.
- $e^{-\Delta E_0^\ddagger/k_B T}$ is the Boltzmann factor for reaching the
  barrier.

The equation explains the Arrhenius form $k = A\,e^{-E_a/RT}$:
$\Delta E_0^\ddagger$ largely sets the activation energy, and the ratio of
partition functions sets the pre-exponential factor.

The same equation in thermodynamic form, with $m$ reactants and the standard
concentration $c^\circ$, is

$$
k^\mathrm{TST}(T) = \frac{k_B T}{h}\,(c^\circ)^{1-m}\,
e^{-\Delta G^{\ddagger,\circ}/RT},
$$

so the rate is controlled by the **free energy of activation**
$\Delta G^{\ddagger,\circ}$: enthalpy *and* entropy. A tight transition state,
with stiff vibrations, has low entropy and reacts slowly even over a low
barrier.

TST rests on three assumptions:

1. Reactants and transition state are in thermal equilibrium.
2. A system that crosses the dividing surface never comes back
   (no recrossing).
3. Motion along the reaction coordinate is classical: no tunneling.

[Variational TST](cvt.md) relaxes the second, and the tunneling corrections
of [Multidimensional tunneling](tunneling.md) the third.

## How PyVTST computes it

`rxn.rates(T)` with the default `theory="tst"` evaluates the equation above
in four steps.

### 1. Stationary points and frequencies

The reactants are optimized to minima and the transition-state guess to a
saddle point; see [Geometry optimizers](optimization.md). At each optimized
geometry the engine computes the Cartesian Hessian, the matrix of second
derivatives of the energy. PyVTST then:

1. divides it by the square roots of the atomic masses (mass weighting);
2. projects out the three translations and the three rotations (two for a
   linear molecule), which leaves $3N - 6$ (or $3N - 5$) internal motions;
3. diagonalizes it. The eigenvalues give the harmonic frequencies, in cm⁻¹;
   a negative eigenvalue gives an imaginary frequency, stored as a negative
   number.

A reactant must have no imaginary frequency and the transition state exactly
one. Imaginary frequencies of 50 cm⁻¹ or less are numerical noise and count
as real. With `freq_scale`, every frequency is multiplied by the
[scale factor](scaling.md) at this point.

### 2. ZPE-corrected barrier

$\mathrm{ZPE} = \tfrac{1}{2} \sum_i h c \tilde\nu_i$ over the real
frequencies, which leaves out the transition state's imaginary one.
$\Delta E_0^\ddagger$ follows from the formula above and is reported as
`result.barrier_kcal`. A barrier at or below zero gives a warning, because
conventional TST is unreliable then; [Variational TST](cvt.md) is the
better choice.

### 3. Partition functions

Each partition function is a product of four factors, in the rigid-rotor,
harmonic-oscillator (RRHO) approximation:

| Factor | Formula | Notes |
| --- | --- | --- |
| Translation | $\left(2\pi m k_B T/h^2\right)^{3/2}$ | Per unit volume (m⁻³); $m$ is the total mass |
| Rotation, linear | $T/(\sigma\,\Theta)$ | $\Theta = h^2/(8\pi^2 I k_B)$ from the moment of inertia $I$ |
| Rotation, non-linear | $\dfrac{\sqrt{\pi}}{\sigma}\left(\dfrac{T^3}{\Theta_A \Theta_B \Theta_C}\right)^{1/2}$ | From the three principal moments of inertia; 1 for an atom |
| Vibration | $\prod_i \left(1 - e^{-hc\tilde\nu_i/k_B T}\right)^{-1}$ | Real frequencies only, counted from the zero-point level |
| Electronic | $2S + 1$ | The spin multiplicity of the ground state |

Counting vibrations from the zero-point level is what moves the zero-point
energies out of the partition functions and into $\Delta E_0^\ddagger$.

$\sigma$ is each molecule's `symmetry_number`. It is the only place where
the **reaction-path degeneracy** enters, through the ratio of the reactants'
symmetry numbers to the transition state's. For H + H₂, H₂ and the linear
H–H–H transition state both have $\sigma = 2$. Set it on every reactant and
on the transition state; see
[Symmetry numbers](molecules.md#symmetry-numbers).

### 4. Assembly

The rate constant is assembled from logarithms, which cannot overflow even
when the partition functions are huge:

$$
\ln k = \ln\frac{k_B T}{h} + \ln Q^\ddagger - \sum_\mathrm{R} \ln Q_\mathrm{R}
- \frac{\Delta E_0^\ddagger}{k_B T}.
$$

With translational partition functions per unit volume, $k$ comes out in s⁻¹
for one reactant and m³ molecule⁻¹ s⁻¹ for two; PyVTST reports the latter in
cm³ molecule⁻¹ s⁻¹. All constants come from `scipy.constants` (CODATA).

## Using it

`rates()` returns the TST rate constants, one per temperature:

```python
result = rxn.rates([300, 500, 1000])
print(result.table())
print(result.k)  # cm³ molecule⁻¹ s⁻¹ for a bimolecular reaction
print(result.barrier_kcal)  # ΔE₀‡ in kcal/mol
```

`result.k`, `result.k_tst` and `result.kappa` are numpy arrays, one value per
temperature. `rates()` also takes `theory="cvt"` for
[Variational TST](cvt.md) and `tunneling=` for a tunneling correction:
`"wigner"`, `"eckart"`, `"zct"` or `"sct"`; see
[Multidimensional tunneling](tunneling.md).

The same calculation runs without an engine from energies and frequencies
you already have, for example from a paper or another program, with
`StationaryPoint.from_frequencies` and `tst_rate`:

```python
from pyvtst import Molecule
from pyvtst.core.rates import tst_rate
from pyvtst.core.stationary import StationaryPoint

h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.40]], symmetry_number=2)
h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
h3 = Molecule(
    ["H", "H", "H"],
    [[0.0, 0.0, -1.76], [0.0, 0.0, 0.0], [0.0, 0.0, 1.76]],  # bohr
    mult=2,
    symmetry_number=2,
)

# Energies in hartree, frequencies in cm⁻¹; the imaginary one is negative.
reactants = [
    StationaryPoint.from_frequencies(h2, -1.1744, [4401.0]),
    StationaryPoint.from_frequencies(h, -0.5, []),
]
ts = StationaryPoint.from_frequencies(h3, -1.6590, [-1500.0, 910.0, 910.0, 2060.0])

result = tst_rate(reactants, ts, [300, 500, 1000])
print(result.k, result.units)  # 2.4e-17 cm³ molecule⁻¹ s⁻¹ at 300 K
```

The geometries are used for the moments of inertia and the masses, so they
must be the optimized ones. The numbers above are illustrative, not a
published H + H₂ surface.

## Limits of the approximations

- **Harmonic vibrations.** Low-frequency torsions are treated as
  vibrations, not hindered rotors, which can misestimate their entropy.
- **Rigid rotation.** Centrifugal distortion and rotation–vibration coupling
  are ignored.
- **Ground electronic state.** Low-lying excited states and spin–orbit
  splitting (for example in OH or Cl) are not included.
- **Chiral transition states.** Two mirror-image paths need an extra factor
  of 2 that cannot be set yet, so the rate comes out a factor of 2 too low.
