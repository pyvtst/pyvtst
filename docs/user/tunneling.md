# Multidimensional tunneling

**Tunneling** lets a reaction pass *through* its barrier instead of over it.
This page shows how to use it, then explains the theory, then how PyVTST
computes the four tunneling corrections it offers.

## Using it

Pick the correction with `tunneling=`, with any theory:

```python
eckart = rxn.rates([200, 300, 500], tunneling="eckart")
print(eckart.kappa)  # one κ per temperature

path = rxn.compute_mep(s_min=-2.0, s_max=2.0)
cvt_sct = rxn.rates([200, 300, 500], theory="cvt", tunneling="sct", path=path)
print(cvt_sct.table())  # T, s*, k_TST, k_CVT, kappa, k
```

`"wigner"` and `"eckart"` use the saddle point only. `"zct"` and `"sct"`
tunnel along the reaction path, which must reach down to the reactants' and
products' zero-point levels on both sides: the default ±0.5 bohr rarely does,
so compute a wider path as above. Give `products=` to `Reaction` so the
product side has a zero-point level to reach.

## Theory

### Tunneling through the barrier

Nuclei are quantum particles. Light ones, hydrogen above all, can cross a
barrier with less energy than its top. This makes reactions faster than any
classical theory predicts, by a **transmission coefficient** $\kappa(T)$:

$$
k(T) = \kappa(T)\, k^\mathrm{X}(T),
$$

where X is the classical rate theory it corrects: TST or one of the
[variational theories](cvt.md).

Tunneling matters most at low temperature, where few molecules have enough
energy to go over the barrier, and for narrow barriers, which have a large
imaginary frequency at the saddle point. It shows up as an Arrhenius plot
that curves upward at low temperature:

```{raw} html
:file: ../_generated/figures/arrhenius.html
```

*Arrhenius plot of a model hydrogen-transfer reaction (40 kJ/mol barrier,
1500i cm⁻¹ imaginary frequency), without tunneling and with the Eckart
transmission coefficient computed by PyVTST. At 200 K tunneling makes the
reaction about 800 times faster.*

#### The transmission probability

A classical system with total energy $E$ crosses a barrier of height
$V^\mathrm{AG}$ if $E > V^\mathrm{AG}$ and is reflected otherwise. A quantum
one crosses with a **transmission probability** $P(E)$ that rises smoothly
from 0 to 1: above 0 below the barrier top (tunneling) and below 1 above it
(non-classical reflection). $\kappa$ is the ratio of the Boltzmann averages of
the quantum and the classical probability:

$$
\kappa(T) = \frac{e^{V^\mathrm{AG}/k_B T}}{k_B T}
\int_{E_0}^{\infty} P(E)\, e^{-E/k_B T}\,\mathrm{d}E,
$$

with energies measured from the reactants' zero-point level. $E_0$ is the
lowest energy at which tunneling can happen: the higher of the reactants' and
the products' zero-point levels, because below it the system has nowhere to
tunnel to.

#### One-dimensional corrections

The simplest corrections use only the saddle point.

- **Wigner** expands the quantum correction to lowest order in $\hbar$, with
  the imaginary frequency $\tilde\nu^\ddagger$:

  $$
  \kappa^\mathrm{W}(T) = 1 + \frac{1}{24}
  \left(\frac{h c\, |\tilde\nu^\ddagger|}{k_B T}\right)^2.
  $$

  It is valid only when tunneling is small, $\kappa$ close to 1.
- **Eckart** replaces the barrier with an Eckart function, a smooth
  one-dimensional barrier whose $P(E)$ is known exactly. It is fitted to the
  forward and reverse ZPE-corrected barriers and to the curvature at the top,
  which the imaginary frequency gives.

Both see the barrier only through the saddle point. A real barrier is
usually thicker or thinner than either assumes.

#### Tunneling along the reaction path

Multidimensional methods tunnel through the real barrier. A system moving
along the [minimum-energy path](mep.md) in its vibrational ground state feels
the adiabatic potential $V_a^G(s)$, and its transmission probability below
the barrier top is the **uniform semiclassical** one,

$$
P(E) = \frac{1}{1 + e^{2\theta(E)}}, \qquad
\theta(E) = \frac{1}{\hbar}\int \sqrt{2\,\mu_\mathrm{eff}(s)
\left[V_a^G(s) - E\right]}\;\mathrm{d}s,
$$

with the integral over the region where $V_a^G(s) > E$. $\theta$ is the
imaginary action: a wider or higher barrier, or a heavier mass, gives a
larger $\theta$ and less tunneling. Above the top,
$P(E) = 1 - P(2V^\mathrm{AG} - E)$.

- **Zero-curvature tunneling** (ZCT) tunnels along the path itself, with
  $\mu_\mathrm{eff} = \mu$, the scaling mass of the coordinates.
- **Small-curvature tunneling** (SCT) lets the system cut the corner of a
  curved path. Tunneling on the inside of the bend shortens the route, which
  the method expresses as an effective mass $\mu_\mathrm{eff}(s) \le \mu$
  along the path. In the centrifugal-dominant form,

  $$
  \frac{\mu_\mathrm{eff}(s)}{\mu} = \min\left\{
  e^{-2\bar a - \bar a^2 + (\mathrm{d}\bar t/\mathrm{d}s)^2},\ 1 \right\},
  \qquad \bar a = |\kappa\,\bar t|,
  $$

  where $\kappa(s)$ is the total curvature of the path (not the transmission coefficient) and $\bar t(s)$ the
  ground-state turning point of the vibrations along which the path bends:
  $\bar t = \big[\sum_m (\kappa_m/\kappa)^2\, t_m^{-4}\big]^{-1/4}$ with
  $t_m = (\hbar/\mu\omega_m)^{1/2}$. Therefore $\kappa^\mathrm{SCT} \ge
  \kappa^\mathrm{ZCT}$.

#### Matching the rate theory

$\kappa$ above refers to a classical threshold at $V^\mathrm{AG}$. The rate
theory it multiplies may have its classical threshold elsewhere: TST at
$V_a^G(0)$, CVT at $V_a^G(s^*(T))$. The **classical adiabatic ground-state
factor**

$$
\kappa^\mathrm{CAG}(T) = e^{-\left[V^\mathrm{AG} - V_a^G(s_\mathrm{ref})\right]/k_B T}
$$

corrects for that, with $s_\mathrm{ref} = 0$ for TST and $s^*(T)$ for CVT.
ICVT and µVT already use $V^\mathrm{AG}$ as their threshold, so for them
$\kappa^\mathrm{CAG} = 1$. The combinations reported in the literature, such
as TST/ZCT, CVT/SCT or µVT/SCT, all include it.

## How PyVTST computes it

`rxn.rates(T, tunneling=...)` computes κ(T) after the rate theory, in the
steps below. Wigner and Eckart use the saddle point only and need no path;
ZCT and SCT need one.

### 1. Wigner

$\kappa^\mathrm{W}$ is evaluated from the transition state's imaginary
frequency, after any [frequency scale factor](scaling.md).

### 2. Eckart

The forward barrier $V_1$ is $\Delta E_0^\ddagger$. The reverse barrier $V_2$
is the transition state's zero-point level above the products', when
`products=` was given to `Reaction`; without products the barrier is taken as
symmetric, $V_2 = V_1$. Both must be positive. $P(E)$ is the analytic Eckart
probability (Johnston and Heicklen), evaluated in logarithms so it cannot
overflow. The Boltzmann average is integrated with `scipy.integrate.quad` from
$\max(0, V_1 - V_2)$ to $50\,k_B T$ above the barrier top, plus the
analytic tail where $P = 1$. The result is the same for $(V_1, V_2)$ and
$(V_2, V_1)$, so forward and reverse rate constants keep detailed balance.

### 3. Adiabatic profile

For ZCT and SCT, $V_a^G$ is computed at the usable path points, exactly as
for [VTST](cvt.md#2-usable-points), and joined by a not-a-knot cubic spline.
At least three usable points are needed. $V^\mathrm{AG}$ is the maximum of
the spline next to the highest point, so it can lie between path points. A
maximum at an end of the path gives a warning that the barrier top is outside
it.

### 4. Lowest tunneling energy

$E_0$ is the higher of the reactants' and the products' zero-point levels.
When $V_a^G$ at an end of the path is above $E_0$, the path does not reach
down far enough: $E_0$ is raised to that end and a warning names the side and
the gap. κ is then too low at low temperature. If $V^\mathrm{AG}$ is not above
$E_0$, there is no barrier to tunnel through and `rates()` raises a
`ValueError`.

### 5. Effective mass

ZCT uses $\mu_\mathrm{eff}/\mu = 1$. SCT computes $\bar a$ and $\bar t$ from
the generalized frequencies and the curvature components (`path.curvature`,
from `compute_mep()`) at each usable point that has them, joins each by a
cubic spline, and evaluates the formula above between and beyond the points.
A path without curvature raises a `ValueError` for SCT.

### 6. Imaginary action and P(E)

$\theta(E)$ is computed at 64 Chebyshev points between $E_0$ and
$V^\mathrm{AG}$ and interpolated between them. Each integral runs with
`scipy.integrate.quad` over the stretches where $V_a^G > E$, split at the
path points. At a turning point the substitution $s = s_0 \pm u^2$ removes the
square-root singularity. $P(E)$ follows from the uniform semiclassical
formula below $V^\mathrm{AG}$, from $1 - P(2V^\mathrm{AG} - E)$ up to
$2V^\mathrm{AG} - E_0$, and is 1 above.

### 7. Boltzmann average

κ is integrated with `quad` from $E_0$ to $2V^\mathrm{AG} - E_0$, plus the
analytic tail where $P = 1$, and multiplied by $\kappa^\mathrm{CAG}$ with
$s_\mathrm{ref}$ set by the theory: 0 for TST, $s^*(T)$ for CVT, and none
($\kappa^\mathrm{CAG} = 1$) for ICVT and µVT. Swapping reactants and products
gives the same κ, so detailed balance holds here too.

## More options

Wider path limits reuse the stored points, so extending the path for
tunneling costs only the new points. Tunneling and the rate theory share the
path:

```python
tst_sct = rxn.rates([200, 300, 500], tunneling="sct", path=path)
print(cvt_sct.path_tunneling.e0, cvt_sct.path_tunneling.kappa_cag)
```

`result.kappa` includes $\kappa^\mathrm{CAG}$. `result.path_tunneling` holds
$E_0$ (`e0`), $V^\mathrm{AG}$ (`v_max`) and its position (`s_max`), the CAG
factor (`kappa_cag`), $V_a^G$ (`v_ag`) and $\mu_\mathrm{eff}/\mu$ (`mu_eff`) at
the path points, and $P(E)$ (`energies`, `probability`). Give `products=` to
`Reaction` so the product side has a zero-point level to reach and Eckart an
asymmetric barrier.

## Limits of the approximations

- **Ground-state tunneling.** Only the vibrational ground state of the modes
  across the path tunnels; excited states are not included.
- **Small curvature.** SCT assumes the path is only mildly curved. For
  heavy–light–heavy reactions with large curvature, large-curvature tunneling
  (LCT), not available in PyVTST, can give more.
- **Path range.** ZCT and SCT need the path to reach $E_0$ on both sides; the
  $E_0$ warning says when it does not.
- **Saddle-point corrections.** Wigner fails once κ is far from 1; Eckart
  assumes the barrier has the Eckart shape.

References: E. Wigner, *Z. Phys. Chem. B* 19, 203 (1932); C. Eckart,
*Phys. Rev.* 35, 1303 (1930); H. S. Johnston and J. Heicklen, *J. Phys.
Chem.* 66, 532 (1962); A. Fernández-Ramos, B. A. Ellingson, B. C. Garrett and
D. G. Truhlar, *Rev. Comput. Chem.* 23, 125 (2007); for the SCT effective
mass, Y.-P. Liu, G. C. Lynch, T. N. Truong, D.-h. Lu, D. G. Truhlar and
B. C. Garrett, *J. Am. Chem. Soc.* 115, 2408 (1993).
