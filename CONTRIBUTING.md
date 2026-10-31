# Contributing to pyvtst

Thanks for your interest in PyVTST. Bug reports, fixes, documentation and new
features are all welcome.

## Reporting bugs and requesting features

Open an issue on [GitHub](https://github.com/pyvtst/pyvtst/issues). For bugs,
include:

- the pyvtst version (`python -c "import pyvtst; print(pyvtst.__version__)"`
  or `pip show pyvtst`), the Python version and your operating system;
- a minimal script that reproduces the problem, with the molecule geometries
  it uses;
- what you expected and what happened, including the full traceback.

For new methods, link the paper that defines them. A published reference
value makes the feature much easier to test.

## Development setup

pyvtst uses [uv](https://docs.astral.sh/uv/) to manage its environment.

```bash
git clone git@github.com:pyvtst/pyvtst.git
cd pyvtst
uv sync --all-extras    # dev tools + all optional adapters
```

## Making a change

1. Branch from `main`, named after the change type:
   `feat/eckart-asymmetric`, `fix/hessian-units`, `docs/caching`.
2. Make the change, with tests.
3. Run the checks below.
4. Open a pull request against `main` and describe what changed and why.

## Checks

CI runs these on Python 3.11 to 3.14. Run them locally before you push:

```bash
uv run ruff format --check && uv run ruff check
uv run mypy                             # strict
uv run lint-imports                     # architecture contracts
uv run bandit -c pyproject.toml -r src
uv run pytest                           # coverage must stay at or above 90%
```

`uv run ruff format` fixes formatting. Tests marked `slow` validate against
published results and are skipped by default. Run them with
`uv run pytest -m slow`.

## Code guidelines

- **Architecture.** The package is layered: `core` is pure numerics, `ports`
  defines interfaces, and the adapters (`io`, `engines`, `optimizers`,
  `stores`) implement them. `core` and `ports` may import only numpy and
  scipy, and adapters do not import each other. `lint-imports` enforces this.
- **Numerics.** Use numpy for arrays and scipy for linear algebra,
  integration, interpolation, optimization and physical constants.
- **Units.** Work in atomic units (bohr, hartree, amu) internally, and convert
  with the factors in `pyvtst.units`.
- **Optional dependencies.** ASE, RDKit, PySCF, geomeTRIC and h5py are
  extras. Import them lazily, inside the adapter function that needs them,
  and mark the import with `# noqa: PLC0415`.
- **Types.** All code is type-checked with `mypy --strict`.
- **Tests.** Every change needs tests. Compare numerical results with a
  stated tolerance, and cite the source of any reference value.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): summary in the imperative
```

| Type | Use for |
| --- | --- |
| `feat` | a new feature (minor version bump) |
| `fix` | a bug fix (patch version bump) |
| `docs` | documentation only |
| `test` | tests only |
| `refactor` | code changes that neither fix a bug nor add a feature |
| `ci` | CI configuration |
| `chore` | maintenance, such as dependencies or packaging |

The scope is optional and names the package touched, for example
`feat(core): add RRHO partition functions` or `fix(engines): ...`. Mark a
breaking change with `!` (`feat(core)!: ...`) and explain it in the body.

## Versioning

pyvtst follows [Semantic Versioning](https://semver.org/), with version
strings that are valid under [PEP 440](https://peps.python.org/pep-0440/).

## License

By contributing, you agree that your contributions are licensed under the
[BSD-3-Clause License](LICENSE).
