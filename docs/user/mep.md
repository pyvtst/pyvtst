# Reaction path

[Transition state theory](tst.md) needs only the reactants and the saddle
point. [Variational TST](cvt.md) and
[multidimensional tunneling](tunneling.md) need more: the whole valley that
leads over the barrier. This page explains the reaction path PyVTST traces,
then how it computes it.

## The minimum-energy path

The **minimum-energy path** (MEP) leaves the saddle point downhill on both
sides, always in the direction of steepest descent, until it reaches the
reactants and the products. Picture a ball rolling down from the pass with so
much friction that it never builds up speed: it follows the valley floor.

"Steepest" depends on how distance is measured. PyVTST measures it in
**mass-scaled coordinates**,

$$
x_i = \sqrt{m_i / \mu}\; R_i ,
$$

where $R_i$ are the Cartesian coordinates of the atoms, $m_i$ their masses
and $\mu$ = 1 amu a scaling mass. In these coordinates the kinetic energy is
$\tfrac{1}{2}\mu\,|\dot{x}|^2$, the same for every direction, so a heavy atom
moving 1 bohr counts as more motion than a light one. That makes the path a
property of the dynamics, not only of the surface: H + H₂ and D + H₂ share a
surface but not a path.

The path satisfies

$$
\frac{\mathrm{d}x}{\mathrm{d}s} = -\frac{g(x)}{|g(x)|},
$$

with $g$ the gradient in mass-scaled coordinates. The arc length $s$ is the
**reaction coordinate**, in bohr, with $s = 0$ at the saddle point and
$s < 0$ on the reactant side. At the saddle point the gradient vanishes, so
the path starts along the **transition vector**, the normal mode of the
imaginary frequency.

## Along the path

At each point the energy is $V_\mathrm{MEP}(s)$, measured from the
reactants. The motions at right angles to the path are vibrations, like those
of a molecule, and their frequencies are the **generalized frequencies**: the
normal-mode frequencies with the overall translations, the rotations and the
motion along the path removed. A molecule of $N$ atoms has $3N - 7$ of them
($3N - 6$ if it is linear).

Each generalized vibration keeps its zero-point energy, so a molecule moving
along the path in its vibrational ground state feels the
**vibrationally adiabatic ground-state potential**

$$
V_a^G(s) = V_\mathrm{MEP}(s) + \mathrm{ZPE}^G(s),
$$

measured from the reactants' zero-point level. At the saddle point,
$V_a^G(0) = \Delta E_0^\ddagger$, the barrier of TST. Away from it the two
curves can differ a lot:

```{raw} html
:file: ../_generated/figures/path_energies.html
```

*$V_\mathrm{MEP}$ and $V_a^G$ along the minimum-energy path of collinear
H + H₂ on the LEPS model surface of the [Introduction](introduction.md).
Hover to read the frequency of the vibration across the path: 4395 cm⁻¹ (the
H₂ stretch) far out, falling to 2161 cm⁻¹ at the saddle point, where the bond
being broken and the one being formed share it. The lost zero-point energy
lowers the barrier, and $V_a^G$ peaks on either side of the saddle point, not
at it: one reason the bottleneck of a reaction can lie away from the saddle
point, which [Variational TST](cvt.md) accounts for. Only the collinear
vibration is included; the two bends of the real three-dimensional system
would add zero-point energy near the saddle point.*

The path also bends. Its **curvature** $\kappa(s)$ measures how fast the
direction of the path turns, split into components along the generalized
modes. A strongly curved path lets a tunneling system cut the corner, which
small-curvature tunneling accounts for; see
[Multidimensional tunneling](tunneling.md).

## How PyVTST computes it

`rxn.compute_mep()` traces the path in six steps.

### 1. Starting point and direction

The transition state is optimized and characterized first; see
[Geometry optimizers](optimization.md). Its imaginary normal mode, in
mass-scaled coordinates and normalized, is the transition vector. Its sign is
fixed by making the largest component positive, so the same saddle point
always gives the same vector.

The first point on each side is one step along $\pm$ the transition vector.
Which side is the reactant side is decided from the geometry: a small step
along each side is compared with the reactants' (and products')
intramolecular distances. This needs the transition state's atoms in the same
order as the reactants' atoms; see [Atom order](reaction.md#atom-order).
When they do not match, the path is still traced, `path.oriented` is false,
and a warning says the sign of $s$ is arbitrary.

### 2. Steps

Two methods take the path from one point to the next:

| `method` | Step | Engine calls per point | Default `ds` |
| --- | --- | --- | --- |
| `"lqa"` (default) | Local quadratic approximation (Page and McIver, *J. Chem. Phys.* 88, 922 (1988)) | Energy, gradient, Hessian | 0.01 bohr |
| `"euler"` | Steepest descent: $x_{n+1} = x_n - \mathrm{d}s\; g/\lvert g\rvert$ | Energy, gradient; a Hessian every `hessian_every` points (default 10) | 0.001 bohr |

The **LQA** step models the surface around the current point as a quadratic,
with the current gradient $g$ and Hessian $H$. On that model the
steepest-descent path is known exactly: in the eigenvectors $u_i$ of $H$, with
eigenvalues $\lambda_i$,

$$
x(t) = x_n - \sum_i u_i\,(u_i^\top g)\,\frac{1 - e^{-\lambda_i t}}{\lambda_i}.
$$

PyVTST finds the $t$ at which the arc length along $x(t)$ equals `ds`, by
numerical integration and root finding, and moves there. The step follows
the curve of the valley instead of a straight line, so it stays on the
valley floor with large steps. Translations and rotations are removed from
the step, so the molecule does not drift or turn.

The **Euler** step goes straight along the current gradient. It needs no
Hessian, but on a curved valley it overshoots onto the walls and zigzags
across the floor, so it needs a much smaller `ds` for the same accuracy.

### 3. Stopping

Each side stops at the first of:

- its limit, `s_min` on the reactant side or `s_max` on the product side;
- a minimum: the mass-scaled gradient falls below 10⁻⁵ hartree/bohr, or the LQA model's
  minimum is closer than one step;
- an energy rise: the next point would be higher than the current one, which
  means the step is too large; that point is discarded.

A side that stops before its limit gives a warning, and `path.stop_reactant`
and `path.stop_product` record why.

### 4. Generalized frequencies

At every point with a Hessian, PyVTST mass-weights the Hessian and projects
out the translations, the rotations and the path direction (the gradient, or
the transition vector at $s = 0$). Diagonalizing what is left gives the
generalized frequencies. The frequency scale factor, if any, is applied to
them; see [Frequency scale factors](scaling.md).

How the projection is done is set by `coordinates=`:

| `coordinates` | Generalized frequencies |
| --- | --- |
| `"redundant"` (default) | Curvilinear, in redundant internal coordinates (Jackels, Gu and Truhlar, *J. Chem. Phys.* 102, 3188 (1995); Chuang and Truhlar, *J. Phys. Chem. A* 102, 242 (1998)) |
| `"cartesian"` | Rectilinear, from the projected Cartesian Hessian |

Away from the saddle point the Cartesian Hessian contains a term from the
gradient times the curvature of straight-line motion. On curved paths that
term turns real bends imaginary, and $V_a^G$ gets kinks where they drop out
of the zero-point energy. Internal coordinates (bonds, angles, torsions) bend
with the molecule, so in them the term is removed. Both give the same
frequencies at the saddle point. Away from it, curvilinear frequencies depend
on the set of internal coordinates, which decides what counts as a curved
motion. When a Cartesian path has large imaginary frequencies, the warning
that lists them suggests `coordinates="redundant"`.

By default the internal coordinates are generated from the optimized
transition state:

- atoms closer than 1.3 times the sum of their covalent radii are bonded, and
  separate fragments are bonded through their closest atoms;
- every two bonds on one atom give an angle, or a linear bend above 170°;
- every chain of three bonds gives a torsion.

`path.internals` records the set, and the same set is used at every point.
The set must span every internal motion. This is checked at the transition
state before tracing and at every point with a Hessian; when it fails,
`compute_mep()` raises a `ValueError` that says how many directions are
missing and which atoms move most in them:

```text
internal coordinates span 2 of 4 internal directions at the transition state;
2 directions are missing, moving mostly atoms 1 (H), 2 (H), 0 (H); add
coordinates that involve them with internals=
```

Add a coordinate on those atoms with `internals=`; see
[Using it](#using-it). PyVTST never falls back to Cartesian coordinates on
its own.

### 5. Zero-point energy and curvature

$\mathrm{ZPE}^G(s) = \tfrac{1}{2}\sum_m h c\,\tilde\nu_m(s)$ over the real
generalized frequencies. Imaginary ones of 50 cm⁻¹ or less count with their
magnitude; larger ones are left out, with a warning that lists them.

The curvature components follow from the gradient and Hessian at each point
(Page and McIver). With $F$ the mass-weighted Hessian and $v = g/|g|$ the
path direction,

$$
\frac{\mathrm{d}v}{\mathrm{d}s} = -\frac{(I - v v^\top)\,F\,v}{|g|},
\qquad
\kappa_m = L_m^\top \frac{\mathrm{d}v}{\mathrm{d}s},
$$

where $L_m$ are the generalized normal modes.

### 6. Storage

Every point goes to the `Reaction`'s store as soon as it is computed. An
interrupted `compute_mep()` resumes where it stopped, wider limits compute
only the new points, and a repeated call is free. A different `method`, `ds`,
`hessian_every` or engine starts a new path. Changing `coordinates` or
`internals` makes no engine calls: the frequencies are recomputed from the
stored Hessians. See [Caching and restarts](caching.md#restarts).

## Using it

With `rxn` from the [Quickstart](../quickstart.md):

```python
path = rxn.compute_mep(s_min=-0.5, s_max=0.5)  # bohr
print(path.table())
```

`path.s`, `path.v_mep`, `path.v_ag` and `path.zpe` are numpy arrays, one value
per point, in bohr and hartree. `path.frequencies` holds the generalized
frequencies (cm⁻¹) and `path.curvature` the curvature components at each
point with a Hessian, and `None` elsewhere. The curvature is also `None` at
$s = 0$, where it would need third derivatives.

With PySCF a Hessian costs far more than a gradient. The default LQA path
from −0.5 to 0.5 bohr needs 100 Hessians. Euler at its defaults also computes
one Hessian every 0.01 bohr, plus ten times as many gradients; raise
`hessian_every` when Hessians are expensive. $V_a^G$ and the frequencies exist
only at points with a Hessian.

To choose the internal coordinates, pass tuples with atom indices in the
transition state's order:

```python
path = rxn.compute_mep(
    internals=[
        ("bond", 0, 1),
        ("bond", 1, 2),
        ("linear_bend", 0, 1, 2),  # two components, smooth through 180°
    ]
)
```

The kinds are `("bond", i, j)`, `("angle", i, j, k)` with apex j,
`("linear_bend", i, j, k)` with middle atom j and `("torsion", i, j, k, l)`
about the j–k bond. Redundant sets are fine.

The coordinates do not change the path, only the frequencies, so switching
makes no engine calls:

```python
cartesian = rxn.compute_mep(coordinates="cartesian")
```

`plot_path` and `plot_paths` (with the `plot` extra) draw V_MEP as a solid
line and V_a^G as a dashed one, mark the transition state and return the
matplotlib `Axes`. Paths from different methods, step sizes or engines share
one axis, one color per path:

```python
from pyvtst import plot_path, plot_paths

ax = plot_path(path)
ax.figure.savefig("h3_mep.png")

euler = rxn.compute_mep(method="euler")
ax = plot_paths([path, euler], labels=["LQA", "Euler"], units="kJ/mol")
```

## Limits of the approximations

- **Harmonic generalized frequencies.** Each vibration across the path is a
  harmonic oscillator in its ground state; anharmonicity and hindered
  rotation are not included.
- **Choice of internal coordinates.** Away from the saddle point, curvilinear
  frequencies depend on the coordinate set. Check that $V_a^G$ is smooth, and
  compare sets if it is not.
- **Step size.** Both methods converge as `ds` shrinks. Compare two step
  sizes, or LQA with Euler as in the `plot_paths` example above, before
  trusting a path.
- **Finite range.** The path covers only `s_min` to `s_max`. Variational TST
  and tunneling warn when they need points beyond it.
