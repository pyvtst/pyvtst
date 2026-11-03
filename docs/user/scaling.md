# Frequency scale factors

A **frequency scale factor** corrects the systematic error of computed
harmonic frequencies. This page shows how to use it, then explains the
theory, then how PyVTST applies it.

## Using it

Pass `freq_scale` to `Reaction`:

```python
engine = PySCFEngine("B3LYP", "def2-SVP")
# Look up the ZPE factor for your method and basis; 0.97 is only a placeholder.
rxn = Reaction([h2, h], h3_guess, engine, store="h3.h5", freq_scale=0.97)
result = rxn.rates([300, 500, 1000], theory="cvt", tunneling="eckart")
```

Use a **ZPE** scale factor for your method and basis set, for example from
the [NIST CCCBDB vibrational scaling factors](https://cccbdb.nist.gov/vsfx.asp),
which list ZPE factors for many methods and basis sets. Changing the factor
makes no engine calls: every geometry, Hessian and path point in the store is
reused.

## Theory

### Why computed frequencies are off

PyVTST computes **harmonic** frequencies: it treats each vibration as a
spring with the curvature of the potential energy surface at the bottom of
the well. Two errors make them systematically too high:

- **Anharmonicity.** Real wells widen above the bottom, so real vibrational
  levels lie closer together than harmonic ones. The true zero-point energy
  and fundamentals are lower than the harmonic values.
- **Electronic-structure method.** HF misses electron correlation and
  overestimates force constants; with anharmonicity, its frequencies come out
  roughly 10 % above the observed fundamentals. DFT and MP2 errors are
  smaller and depend on the functional and the basis set.

Both errors are similar for most vibrations of a given method and basis, so a
single factor $\lambda$ removes most of them:

$$
\tilde\nu_\mathrm{scaled} = \lambda\,\tilde\nu_\mathrm{harmonic}.
$$

$\lambda$ is fitted to reference data for a set of molecules, which makes it
specific to the method, the basis set and the property it was fitted to.

### Which factor to use

Published factors are fitted to different properties: fundamental
frequencies, zero-point energies, or thermal enthalpies and entropies. They
differ: ZPE factors are usually somewhat larger than those for fundamentals,
because anharmonicity lowers the zero-point level less than the
fundamentals.

For rate constants, use a ZPE factor. The ZPE enters
$\Delta E_0^\ddagger$, and with it the exponent of the rate constant: an
error of 1 kcal/mol in the barrier changes $k$ by a factor of about 5 at
300 K. The frequencies also enter the vibrational partition functions, where
the low ones matter most, and the imaginary frequency sets the Wigner and
Eckart tunneling corrections.

## How PyVTST applies it

### What is scaled

The factor multiplies every frequency right after the Hessian is
diagonalized: the harmonic frequencies of every species and the generalized
frequencies at every path point, imaginary ones included. Everything computed
from frequencies therefore uses scaled values:

- ZPEs, $\Delta E_0^\ddagger$ and $V_a^G(s)$;
- the vibrational partition functions, and so the TST and variational rate
  constants;
- the imaginary frequency used by Wigner and Eckart.

Normal modes, geometries and the path itself do not change.

### Where it is recorded

The factor is recorded as `freq_scale` on each `StationaryPoint` and
`ReactionPath`. `rates(..., path=p)` rejects a path computed with another
factor, so scaled and unscaled frequencies are never mixed.

The factor is not part of any cache key. A `Reaction` with another factor on
the same store reuses every optimized geometry, Hessian and path point and
makes no engine calls.

### The imaginary-frequency threshold

The threshold (`imag_threshold`, 50 cm⁻¹ by default) applies to the scaled
frequencies, so a factor above 1 can turn a −48 cm⁻¹ noise mode into a
counted imaginary frequency. If that happens, check the geometry or raise the
threshold.

## More options

`StationaryPoint.from_hessian` and `from_frequencies` take the same
`freq_scale` argument. Keep the default 1.0 for published frequencies that are
already scaled.
