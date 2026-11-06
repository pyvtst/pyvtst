# Input files

A calculation can also be described in a TOML input file and run from the
shell, as with ORCA or Gaussian. The input file drives the same `Reaction` as
the Python API, so both give the same rate constants.

```toml
title = "H2 + H -> H + H2, B3LYP/def2-SVP"
store = "h3.h5"          # HDF5 cache next to this file; omit to keep results in memory
freq_scale = 1.0

[engine]
type = "pyscf"
method = "B3LYP"
basis = "def2-SVP"

[optimizer]              # optional; geomeTRIC with its defaults when omitted
type = "geometric"
convergence = "GAU_TIGHT"

[reactants.r1]
name = "H2"
symmetry_number = 2
xyz = """
H 0.00 0.0 0.0
H 0.74 0.0 0.0
"""

[reactants.r2]
name = "H"
mult = 2
xyz = "H 0.0 0.0 0.0"

[ts]
name = "H3"
mult = 2
symmetry_number = 2
xyz = """
H -0.85 0.0  0.0
H  0.00 0.0  0.0
H  1.00 0.04 0.0
"""

[mep]                    # optional; compute_mep() defaults when omitted
s_min = -1.0
s_max = 1.0

[[rates]]
T = [300, 500, 1000]
tunneling = "eckart"

[[rates]]
T = { start = 200, stop = 1000, step = 100 }
theory = "cvt"
tunneling = "sct"
```

Run it with the `pyscf`, `geometric` and `hdf5` extras installed:

```bash
pyvtst h3.toml           # writes h3.out next to h3.toml
pyvtst h3.toml -o runs/h3-b3lyp.out
python -m pyvtst h3.toml # the same
```

All `[[rates]]` jobs share one `Reaction`: the stationary points and the path
are computed once, and a rerun with the same `store` costs nothing.

## Keys

Every key not listed here is an error, so a typo stops the run before any
calculation starts. All errors in a file are reported together, each with the
file name and the key path, e.g. `h3.toml: reactants.r2.multt: unknown key,
did you mean 'mult'?`. Array indices (in `T` or `internals`) count from 0.

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `title` | string | none | Free text, printed in the report |
| `store` | string | in memory | Path of the HDF5 result store (needs `hdf5`) |
| `freq_scale` | number | 1.0 | Frequency scale factor; see [Frequency scaling](scaling.md) |

### `[reactants.r1]`, `[ts]`, `[products.p1]`

Reactants are numbered tables `[reactants.r1]` and, for a bimolecular
reaction, `[reactants.r2]`. One `[ts]` is required. Products are optional,
numbered `[products.p1]`, `[products.p2]`, …; they make Eckart tunneling
asymmetric and bound the ICVT and µVT threshold.

Numbers start at 1 without gaps, and the molecules are taken in number order,
whatever their order in the file. That order matters for the reaction path:
the transition state's atoms should be the reactants' atoms in order, r1's
first.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | the key (`r1`, `p1`) or `ts` | Name in the report and in messages |
| `charge` | integer | 0 | Total charge |
| `mult` | integer | 1 | Spin multiplicity |
| `symmetry_number` | integer | 1 | Rotational symmetry number |
| `masses` | array of numbers | standard isotopes | Atomic masses in amu, one per atom |
| `xyz` | string | | Inline geometry: one `symbol x y z` line per atom |
| `units` | string | `"angstrom"` | Unit of `xyz`: `"angstrom"` or `"bohr"` |
| `file` | string | | Path of an XYZ file (always in Ångström) |

Give exactly one of `xyz` and `file`. Inline blocks have no atom-count or
comment line. A molecule read from a file takes the file's comment line as its
name unless `name` is set.

### `[engine]` and `[optimizer]`

`type` selects the adapter; the other keys are its options, with the defaults
of the Python class.

| `type` | Required | Optional |
|---|---|---|
| engine `"pyscf"` | `method`, `basis` | `reference`, `hessian`, `fd_step`, `conv_tol`, `grid_level`, `max_cycle`, `max_memory_mb`, `verbose` |
| optimizer `"geometric"` | | `convergence`, `max_iterations`, `coordsys`, `workdir`, `verbose` |

See [Engines](engines.md) and [Geometry optimizers](optimization.md) for what
each option does.

### `[mep]`

The options of `Reaction.compute_mep()`: `s_min`, `s_max`, `ds` (bohr),
`method` (`"lqa"` or `"euler"`), `hessian_every`, `coordinates`
(`"redundant"` or `"cartesian"`) and `internals`, written as arrays such as
`[["bond", 0, 1], ["linear_bend", 0, 1, 2]]` with 0-based atom indices. See
[Reaction path](mep.md).

The path is computed when `[mep]` is present or a job needs one (CVT, ICVT,
µVT, ZCT or SCT), once, and used by every job that needs it.

### `[[rates]]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `T` | number, array, or `{ start, stop, step }` | required | Temperatures in K; a range includes `stop` |
| `theory` | string | `"tst"` | `"tst"`, `"cvt"`, `"icvt"` or `"muvt"` |
| `tunneling` | string | `"none"` | `"none"`, `"wigner"`, `"eckart"`, `"zct"` or `"sct"` |

## Paths

Relative paths (`store`, `file`, `workdir`) are relative to the input file's
directory, not to the directory `pyvtst` is run from. `~` is the home
directory.

## The report

`pyvtst h3.toml` writes `h3.out`, replacing an existing one. It has, in order:
a header with the pyvtst version and date, the input file verbatim, the
settings with every default filled in, the stationary points (energy in
hartree, geometry in Ångström, frequencies in cm⁻¹ with imaginary ones marked
`i`), the ZPE-corrected barrier, the reaction path, one rate table per job,
the warnings, and a last line `pyvtst finished normally`.

When the run computes a reaction path, its geometries go to `h3_mep.xyz` next
to the report (`run1_mep.xyz` for `-o run1.out`): one frame per path point in
order of s, in Ångström. Each comment line reads
`index=… s_bohr=… energy_hartree=… v_mep_kcal_mol=…`, with V_MEP relative to
the reactants, so the file opens as an animation in Avogadro, VMD or Jmol, and
`ase.io.read("h3_mep.xyz", index=":")` puts the values in each frame's
`atoms.info`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Every job finished |
| 1 | A calculation failed (no convergence, wrong number of imaginary frequencies, missing extra); the report ends with the error |
| 2 | The input could not be loaded; nothing was computed and no report written |
| 130 | Interrupted with Ctrl-C; the report has what finished |

## From Python

```python
from pyvtst import inputfile

spec = inputfile.load("h3.toml")  # validate only; raises InputError
results = inputfile.run("h3.toml", output="h3.out")
print(results[1].table())
```
