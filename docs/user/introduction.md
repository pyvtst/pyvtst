# Introduction

This page introduces the ideas PyVTST builds on, starting from what a
chemical reaction is. It assumes general chemistry and some physical chemistry: energy,
the Boltzmann distribution, and the idea of a partition function. The figures
are interactive: hover to read values, drag to zoom, and rotate the 3D surface.

## What is a chemical reaction?

A chemical reaction rearranges atoms: some bonds break and others form. In

$$
\mathrm{H_A{-}H_B} + \mathrm{H_C} \longrightarrow \mathrm{H_A} + \mathrm{H_B{-}H_C}
$$

the hydrogen atom H{sub}`C` approaches the molecule H{sub}`A`–H{sub}`B`, the
B–C bond forms as the A–B bond breaks, and H{sub}`A` leaves. This is the
simplest chemical reaction, and the one the [Quickstart](../quickstart.md)
uses.

Most reactions written in a textbook, such as the combustion of methane, are
*overall* reactions: they happen through a sequence of *elementary* steps,
each a single collision or a single molecule falling apart. PyVTST computes
rate constants of elementary gas-phase reactions with one reactant
(*unimolecular*, A → products) or two (*bimolecular*, A + B → products).

## How fast? Rate laws and rate constants

The rate of an elementary reaction is proportional to the concentration of
each reactant. For a bimolecular reaction A + B → products,

$$
-\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = k\,[\mathrm{A}][\mathrm{B}].
$$

The **rate constant** $k$ carries all the chemistry: it does not depend on
the concentrations, only on the temperature (and, for some reactions, on the
pressure). It has units of s⁻¹ for a unimolecular reaction and
cm³ molecule⁻¹ s⁻¹ for a bimolecular one. Rate constants are what kinetic
models of combustion, atmospheric chemistry or astrochemistry are built from,
and what PyVTST computes from first principles.

## Why temperature matters

Experimentally, rate constants usually follow the **Arrhenius equation**,

$$
k(T) = A\,e^{-E_a/RT},
$$

where $E_a$ is the activation energy and $A$ the pre-exponential factor. A
reaction with $E_a$ = 50 kJ/mol runs about 150 times faster at 400 K than at
300 K.

The exponential comes from the Boltzmann distribution. Molecules in a gas
have a spread of energies, and only those with enough energy to cross a
barrier react. The fraction above the barrier grows roughly as
$e^{-E_a/RT}$, so a modest rise in temperature multiplies it:

```{raw} html
:file: ../_generated/figures/boltzmann.html
```

*Maxwell–Boltzmann distribution of the kinetic energy of gas molecules at
300 K and 600 K. The shaded areas are the fractions with more than 10 kJ/mol;
doubling the temperature raises it from 5 % to 26 %.*

To predict $A$ and $E_a$ instead of measuring them, we need to know what the
barrier is.

## The potential energy surface

Electrons are thousands of times lighter than nuclei and adjust almost
instantly to where the nuclei are (the Born–Oppenheimer approximation). For
each arrangement of the nuclei there is therefore one electronic energy $V$,
which a quantum chemistry program such as PySCF computes. As a function of
all nuclear positions this is the **potential energy surface** (PES). A
molecule with $N$ atoms has $3N - 6$ internal coordinates ($3N - 5$ if it is
linear), so the PES is usually a high-dimensional landscape.

For three hydrogen atoms kept on a line there are only two coordinates, the
distances $r_{AB}$ and $r_{BC}$, and the PES can be drawn:

```{raw} html
:file: ../_generated/figures/pes.html
```

*Potential energy surface of collinear H{sub}`A`–H{sub}`B`–H{sub}`C`, from
the London–Eyring–Polanyi–Sato (LEPS) model, an analytic approximation used
here for illustration. Energies above 200 kJ/mol are cut off. Rotate the
surface to look down the valleys.*

The landscape has the features every reaction has:

- **Minima** are stable species. The valley along large $r_{BC}$ is H₂ + H
  (the reactants), the valley along large $r_{AB}$ is H + H₂ (the products).
  All vibrational frequencies at a minimum are real.
- The **saddle point** is the top of the barrier: the highest point along
  the reaction path, but the lowest point of the ridge that separates the two
  valleys, like a mountain pass. It is a maximum in one direction and a
  minimum in all others, so it has exactly one imaginary vibrational
  frequency. It is the **transition state** of the reaction.
- The **minimum-energy path** (MEP) runs from the saddle point down to the
  reactants and products by steepest descent, in mass-weighted coordinates.
  It is the route that requires the least energy.

The next page, [Transition state theory](tst.md), turns this landscape into a
rate constant.
