# Frequency scale factors

Harmonic frequencies from HF, DFT and MP2 are systematically too high, and the
error reaches the rate constant through the ZPE-corrected barrier, in the
exponent. `freq_scale` multiplies every frequency by one empirical factor:

```python
engine = PySCFEngine("B3LYP", "def2-SVP")
# Look up the ZPE factor for your method and basis; 0.97 is only a placeholder.
rxn = Reaction([h2, h], h3_guess, engine, store="h3.h5", freq_scale=0.97)
result = rxn.rates([300, 500, 1000], theory="cvt", tunneling="eckart")
```

The factor scales the harmonic frequencies of every species and the
generalized frequencies along the path, imaginary ones included, after
diagonalization. ZPEs, partition functions, V_a^G(s), the TST and CVT rate
constants, and the imaginary frequency used by Wigner and Eckart therefore
all use scaled values. Normal modes, geometries and the path itself do not
change. The factor is recorded as `freq_scale` on each `StationaryPoint` and
`ReactionPath`. `rates(..., path=p)` rejects a path computed with another
factor.

Published factors depend on the property they were fitted to (ZPE,
fundamentals, enthalpy or entropy). For rate constants use a ZPE scale factor
for your method and basis, for example from I. M. Alecu, J. Zheng, Y. Zhao and
D. G. Truhlar, *J. Chem. Theory Comput.* 6, 2872 (2010).

The factor is not part of any cache key. A `Reaction` with another factor on
the same store reuses every optimized geometry, Hessian and path point and
makes no engine calls. The imaginary-frequency threshold (`imag_threshold`,
50 cm⁻¹ by default) applies to the scaled frequencies, so a factor above 1 can
turn a −48 cm⁻¹ noise mode into a counted imaginary frequency. If that
happens, check the geometry or raise the threshold.

`StationaryPoint.from_hessian` and `from_frequencies` take the same
`freq_scale` argument. Keep the default 1.0 for published frequencies that are
already scaled.
