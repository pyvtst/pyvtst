# Arrhenius fits

Kinetic models (Chemkin, Cantera, master-equation codes) take rate constants
as a few parameters, not as a table. PyVTST fits k(T) to the Arrhenius and
modified Arrhenius forms. This page shows how to use them, then explains the
forms, then how PyVTST fits them.

## Using it

Fit any `RateResult`:

```python
result = rxn.rates([300, 400, 500, 700, 1000, 1500, 2000], theory="cvt")
fit = result.fit_arrhenius()       # the modified Arrhenius form
print(fit)                         # model, parameters with units, max deviation
fit.A, fit.n, fit.Ea_kcal          # A in result.units, Ea in kcal/mol
fit([350, 450])                    # k from the fit
fit.activation_energy([300, 1000]) # Ea(T) = Ea + nRT, in kcal/mol
```

`print(fit)` gives:

```text
Arrhenius fit (modified): k = A (T/T_ref)^n exp(-Ea/RT)
  A     = … cm^3 molecule^-1 s^-1
  n     = …
  Ea    = … kcal/mol
  T_ref = 300 K
  fitted over 300-2000 K, max deviation …%
```

Rate constants from anywhere else (experiment, another code) fit the same
way:

```python
from pyvtst import fit_arrhenius

fit = fit_arrhenius(T, k, "arrhenius", units="s^-1")
```

Each form needs at least as many distinct temperatures as it has parameters:
2 and 3. Use more than that, spread over the range you need: a fit is only
good inside the temperatures it was fitted to.

## The two forms

`"arrhenius"`, two parameters ($A$, $E_a$):

$$
k = A \exp\!\left(-\frac{E_a}{RT}\right)
$$

`"modified"` (the default), three parameters ($A$, $n$, $E_a$):

$$
k = A \left(\frac{T}{T_\text{ref}}\right)^{n} \exp\!\left(-\frac{E_a}{RT}\right)
$$

$T_\text{ref}$ is 300 K.

### Which one to use

An Arrhenius plot (ln k against 1/T) is a straight line only if the
activation energy does not depend on T. The **local activation energy**

$$
E_a(T) = -R\,\frac{d\ln k}{d(1/T)}
$$

measures its slope at each temperature; `fit.activation_energy(T)` returns it
in kcal/mol. It is not the same as the fitted parameter `fit.Ea_kcal`: the two
are equal for `"arrhenius"`, while for `"modified"` E_a(T) = `Ea_kcal` + nRT.
The two forms differ in how E_a(T) may change:

- `"arrhenius"`: E_a(T) = Ea, constant. Fine over a narrow range.
- `"modified"`: E_a(T) = Ea + nRT, linear in T. This covers the upward
  curvature at high T from the partition functions, and fits TST and CVT
  rate constants without tunneling to about 1–2 % over wide ranges.

Neither form follows strong tunneling. At low T a ZCT or SCT reaction goes
through the barrier rather than over it, k stops depending strongly on T, and
E_a(T) falls towards zero, which E_a + nRT cannot do. When the fit warns for
such rate constants, fit a narrower temperature range, or fit the low- and
high-temperature parts separately.

$n$ and $E_a$ are correlated: over a narrow range, a larger $n$ with a smaller
$E_a$ fits almost as well. Read the curve and `max_rel_dev`, not $n$ alone.

### When the fit is poor

`max_rel_dev` is the largest relative deviation of the fit from the data, over
the fitted temperatures $T_i$:

$$
\mathtt{max\_rel\_dev} = \max_i \left| \frac{k_\text{fit}(T_i)}{k_i} - 1 \right|
$$

A value of 0.02 means the fit is within 2 % of k at every fitted temperature.
Above 5 % the fit warns: for `"arrhenius"` it suggests `"modified"`, for
`"modified"` a narrower temperature range. The fit is still returned.

## How PyVTST fits

### Least squares in ln k

Rate constants over a wide temperature range span many orders of magnitude.
A fit to k itself would follow the largest values and ignore the low-T ones.
PyVTST minimises the squared error in ln k instead, so every temperature
counts the same in relative terms.

### Linear least squares

ln k = ln A + n ln(T/T_ref) − Ea/RT is linear in ln A, n and Ea, so both fits
are exact linear least squares: no starting guess, one answer. The modified
form contains the Arrhenius form (n = 0), so its fit is never worse.

### Reference temperature and units

A is in the unit of k (`result.units`) and Ea in kcal/mol, like
`barrier_kcal`. Codes that write the modified form as k = A′ Tⁿ e^(−Ea/RT),
with T in K, use

$$
A' = A \cdot 300^{-n}.
$$

For Chemkin, which takes Ea in cal/mol by default, multiply `Ea_kcal` by
1000.

## In input files

Add `arrhenius` to a `[[rates]]` job (see [Input files](inputfile.md)):

```text
[[rates]]
T = { start = 200, stop = 1000, step = 100 }
theory = "cvt"
tunneling = "sct"
arrhenius = "modified"
```

The report prints the fitted parameters under that job's rate table. A job
with fewer distinct temperatures than the form needs is an input error.
