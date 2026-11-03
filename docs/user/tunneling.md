# Multidimensional tunneling

## Background

Nuclei are quantum particles. Light ones, hydrogen above all, can pass
*through* the barrier instead of over it. This **tunneling** makes reactions
faster than any classical theory predicts, by a transmission coefficient
$\kappa(T) \ge 1$:

$$
k(T) = \kappa(T)\, k^\mathrm{CVT}(T).
$$

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

PyVTST offers four tunneling corrections, from simplest to most accurate:

| Method | Uses |
| --- | --- |
| Wigner | The imaginary frequency only; a first-order correction, valid when tunneling is small |
| Eckart | A one-dimensional barrier fitted to the barrier height and imaginary frequency |
| Zero-curvature (ZCT) | The real $V_a^G(s)$ along the path |
| Small-curvature (SCT) | $V_a^G(s)$ and the curvature of the path, which lets the system cut the corner |

## Zero- and small-curvature tunneling

`tunneling="zct"` (zero-curvature) and `"sct"` (small-curvature) compute κ(T)
along the minimum-energy path rather than from the saddle point alone. The
barrier is the vibrationally adiabatic ground-state curve V_a^G(s), and the
transmission probability P(E) is the uniform semiclassical one through it. SCT
also lets the tunneling path cut the corner of a curved reaction path, through
an effective mass μ_eff(s) ≤ μ computed from the path's curvature
(`path.curvature`, from `compute_mep()`), so κ^SCT ≥ κ^ZCT. Both work with
any theory:

```python
path = rxn.compute_mep(s_min=-2.0, s_max=2.0)
tst_sct = rxn.rates([200, 300, 500], tunneling="sct", path=path)
cvt_sct = rxn.rates([200, 300, 500], theory="cvt", tunneling="sct", path=path)
print(cvt_sct.table())
print(cvt_sct.path_tunneling.e0, cvt_sct.path_tunneling.kappa_cag)
```

κ includes the classical adiabatic ground-state (CAG) factor
exp(−[V^AG − V_a^G(s_ref)]/k_B T), which refers it to the rate theory it
multiplies: s_ref = 0 for TST and s*(T) for CVT. ICVT and µVT already use
V^AG as their threshold, so for them the CAG factor is 1. That gives TST/ZCT,
CVT/SCT, ICVT/SCT, µVT/SCT and so on, as reported in the literature.
`result.path_tunneling`
holds E₀, the maximum V^AG and its s, the CAG factor, V_a^G and μ_eff/μ at the
path points, and P(E).

Tunneling starts at E₀, the higher of the reactants' and products' zero-point
levels, so the path must reach down to it on both sides. The default path
(±0.5 bohr) rarely does. When an end of the path lies above E₀, E₀ is raised to
it and a warning names the side and the gap: κ is then too low at low
temperature, so compute a wider path, as above, and pass it with `path=`.
Wider limits reuse the stored points. Give `products=` to `Reaction` so the
product side has a zero-point level to reach.

References: A. Fernández-Ramos, B. A. Ellingson, B. C. Garrett and
D. G. Truhlar, *Rev. Comput. Chem.* 23, 125 (2007); for the SCT effective
mass, Y.-P. Liu, G. C. Lynch, T. N. Truong, D.-h. Lu, D. G. Truhlar and
B. C. Garrett, *J. Am. Chem. Soc.* 115, 2408 (1993).
