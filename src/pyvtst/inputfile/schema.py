"""Parse and validate pyvtst TOML input files.

:func:`load` reads a file, checks every key against the format and returns an
immutable :class:`InputSpec`. It collects every error in the file and raises
them together as one :class:`InputError`, before any engine is built.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pyvtst.core.arrhenius import MODELS, check_count
from pyvtst.core.internals import Internal, validate_internals
from pyvtst.core.mep import COORDINATES, METHODS
from pyvtst.core.molecule import Molecule
from pyvtst.core.stationary import check_freq_scale
from pyvtst.core.units import ANGSTROM_TO_BOHR
from pyvtst.inputfile.registry import DEFAULT_REGISTRY, AdapterEntry, Field, Registry
from pyvtst.io.xyz import read_xyz
from pyvtst.reaction import (
    PATH_TUNNELING,
    THEORIES,
    TUNNELING,
    VARIATIONAL,
    _check_balance,
    _coordinate_parameters,
    _mep_parameters,
)

# Reaction takes one or two reactants.
MAX_REACTANTS = 2

# Relative tolerance on the last temperature of a { start, stop, step } range.
T_RANGE_RTOL = 1e-9


class InputError(ValueError):
    """An input file could not be loaded.

    Attributes:
        errors: Every error found, each ``"<file>: <key path>: <message>"``.
    """

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class AdapterSpec:
    """A selected engine or optimizer with all of its options.

    Attributes:
        type: The registry key, lower case.
        options: ``(name, value)`` pairs for every declared option, absent ones
            at their default.
    """

    type: str
    options: tuple[tuple[str, object], ...]

    @property
    def kwargs(self) -> dict[str, object]:
        return dict(self.options)


@dataclass(frozen=True)
class MEPSpec:
    """The ``[mep]`` settings, with defaults filled in."""

    s_min: float
    s_max: float
    ds: float
    method: str
    hessian_every: int
    coordinates: str
    internals: tuple[Internal, ...] | None

    @property
    def kwargs(self) -> dict[str, object]:
        """Keyword arguments for :meth:`pyvtst.Reaction.compute_mep`."""
        return {
            "s_min": self.s_min,
            "s_max": self.s_max,
            "ds": self.ds,
            "method": self.method,
            "hessian_every": self.hessian_every,
            "coordinates": self.coordinates,
            "internals": self.internals,
        }


@dataclass(frozen=True)
class RateJob:
    """One ``[[rates]]`` job.

    ``arrhenius`` is the Arrhenius fit model, or ``None`` for no fit.
    """

    T: tuple[float, ...]
    theory: str
    tunneling: str | None
    arrhenius: str | None = None

    @property
    def needs_path(self) -> bool:
        return self.theory in VARIATIONAL or self.tunneling in PATH_TUNNELING


@dataclass(frozen=True, eq=False)
class InputSpec:
    """A validated input file.

    Attributes:
        path: The input file, as given.
        text: The file's text.
        title: The ``title``, or ``None``.
        store: The store path, resolved, or ``None`` for an in-memory store.
        freq_scale: The frequency scale factor.
        reactants: One or two reactant molecules.
        ts: The transition-state guess.
        products: Product molecules, or ``None``.
        engine: The selected engine.
        optimizer: The selected optimizer, or ``None`` for the default.
        mep: The ``[mep]`` settings, or ``None`` when the section is absent.
        rates: The rate jobs, in file order.
    """

    path: Path
    text: str
    title: str | None
    store: Path | None
    freq_scale: float
    reactants: tuple[Molecule, ...]
    ts: Molecule
    products: tuple[Molecule, ...] | None
    engine: AdapterSpec
    optimizer: AdapterSpec | None
    mep: MEPSpec | None
    rates: tuple[RateJob, ...]

    @property
    def needs_path(self) -> bool:
        """Whether the run computes a reaction path."""
        return self.mep is not None or any(job.needs_path for job in self.rates)


TOP_FIELDS = (
    Field("title", "str"),
    Field("store", "str"),
    Field("freq_scale", "float", default=1.0),
    Field("reactants", "any", required=True),
    Field("ts", "table", required=True),
    Field("products", "any"),
    Field("engine", "table", required=True),
    Field("optimizer", "table"),
    Field("mep", "table"),
    Field("rates", "any", required=True),
)

MOLECULE_FIELDS = (
    Field("name", "str"),
    Field("charge", "int", default=0),
    Field("mult", "int", default=1),
    Field("symmetry_number", "int", default=1),
    Field("masses", "floats"),
    Field("xyz", "str"),
    Field("file", "str"),
    Field("units", "str", default="angstrom", choices=("angstrom", "bohr")),
)

MEP_FIELDS = (
    Field("s_min", "float", default=-0.5),
    Field("s_max", "float", default=0.5),
    Field("ds", "float"),
    Field("method", "str", default="lqa", choices=METHODS),
    Field("hessian_every", "int"),
    Field("coordinates", "str", default="redundant", choices=COORDINATES),
    Field("internals", "list"),
)

RATE_FIELDS = (
    Field("T", "any", required=True),
    Field("theory", "str", default="tst", choices=THEORIES),
    Field(
        "tunneling",
        "str",
        default="none",
        choices=("none", *(t for t in TUNNELING if t is not None)),
    ),
    Field("arrhenius", "str", default="none", choices=("none", *MODELS)),
)

T_RANGE_FIELDS = (
    Field("start", "float", required=True),
    Field("stop", "float", required=True),
    Field("step", "float", required=True),
)

_EXPECTED = {
    "str": "a string",
    "int": "an integer",
    "float": "a number",
    "bool": "a boolean",
    "floats": "an array of numbers",
    "list": "an array",
    "table": "a table",
}


def load(path: str | os.PathLike[str], registry: Registry | None = None) -> InputSpec:
    """Read and validate an input file.

    Args:
        path: The TOML input file.
        registry: The engines and optimizers ``type`` may select. ``None``
            means :data:`~pyvtst.inputfile.registry.DEFAULT_REGISTRY`.

    Raises:
        InputError: The file is missing or unreadable, is not valid TOML, or
            breaks the format. It carries every error found.
    """
    file = Path(path)
    name = os.fspath(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise InputError([f"{name}: no such file"]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError([f"{name}: cannot read the file: {exc}"]) from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InputError([f"{name}: invalid TOML: {exc}"]) from None
    ctx = _Context(name, file.resolve().parent)
    spec = _input(
        ctx, data, file, text, DEFAULT_REGISTRY if registry is None else registry
    )
    if ctx.errors or spec is None:
        raise InputError(ctx.errors)
    return spec


class _Context:
    """The file name, its directory and the errors collected so far."""

    def __init__(self, file: str, base_dir: Path) -> None:
        self.file = file
        self.base_dir = base_dir
        self.errors: list[str] = []

    def error(self, path: str, message: str) -> None:
        where = f"{self.file}: {path}" if path else self.file
        self.errors.append(f"{where}: {message}")

    def resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.base_dir / p

    def keyed_error(
        self, path: str, exc: Exception, keys: Mapping[str, str], fallback: str = ""
    ) -> None:
        """Report a ``"<key>: <message>"`` error at ``path.<keys[key]>``.

        An empty ``keys[key]`` reports it at ``path`` itself.
        """
        message = str(exc)
        head, sep, rest = message.partition(": ")
        if sep and head in keys:
            self.error(_key(path, keys[head]) if keys[head] else path, rest)
        else:
            self.error(_key(path, fallback) if fallback else path, message)


def _key(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _toml_type(value: object) -> str:
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a float"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, dict):
        return "a table"
    return f"a {type(value).__name__}"


def _number(value: object) -> float | None:
    """``value`` as a float if it is a TOML integer or float, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _convert(value: object, f: Field) -> tuple[object, str | None]:
    """Return ``(converted value, None)`` or ``(None, error message)``."""
    wrong = f"expected {_EXPECTED.get(f.kind, f.kind)}, found {_toml_type(value)}"
    if f.kind == "any":
        return value, None
    if f.kind == "str":
        if not isinstance(value, str):
            return None, wrong
        if f.choices is not None:
            for choice in f.choices:
                if value.lower() == choice.lower():
                    return choice, None
            listed = ", ".join(repr(c) for c in f.choices)
            return None, f"expected one of {listed}, got {value!r}"
        return value, None
    if f.kind == "bool":
        return (value, None) if isinstance(value, bool) else (None, wrong)
    if f.kind == "int":
        if isinstance(value, bool):
            return None, wrong
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        return None, wrong
    if f.kind == "float":
        number = _number(value)
        if number is None:
            return None, wrong
        if not math.isfinite(number):
            return None, f"must be finite, got {number}"
        return number, None
    if f.kind == "floats":
        if not isinstance(value, list):
            return None, wrong
        numbers = [_number(v) for v in value]
        bad = [i for i, n in enumerate(numbers) if n is None or not math.isfinite(n)]
        if bad:
            return (
                None,
                f"{wrong.split(',')[0]}; element {bad[0]} is not a finite number",
            )
        return tuple(n for n in numbers if n is not None), None
    if f.kind == "list":
        return (value, None) if isinstance(value, list) else (None, wrong)
    # "table"
    return (value, None) if isinstance(value, dict) else (None, wrong)


def _read_table(
    ctx: _Context, data: object, fields: Sequence[Field], path: str
) -> dict[str, Any] | None:
    """:func:`_read_fields` for a value that may not be a table.

    Returns ``None``, after reporting it, when ``data`` is not a table.
    """
    if not isinstance(data, dict):
        ctx.error(path, f"expected a table, found {_toml_type(data)}")
        return None
    return _read_fields(ctx, data, fields, path)


def _read_fields(
    ctx: _Context, data: Mapping[str, object], fields: Sequence[Field], path: str
) -> dict[str, Any]:
    """Check a table against ``fields``; return every field's value or default.

    A field whose value is wrong is left out of the result.
    """
    names = [f.name for f in fields]
    for key in data:
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f", did you mean {close[0]!r}?" if close else ""
            allowed = ", ".join(names) if names else "none"
            ctx.error(_key(path, key), f"unknown key{hint} (allowed keys: {allowed})")
    values: dict[str, Any] = {}
    for f in fields:
        if f.name not in data:
            if f.required:
                ctx.error(_key(path, f.name), "missing required key")
            else:
                values[f.name] = f.default
            continue
        value, problem = _convert(data[f.name], f)
        if problem is not None:
            ctx.error(_key(path, f.name), problem)
        else:
            values[f.name] = value
    return values


def _input(
    ctx: _Context, data: dict[str, Any], file: Path, text: str, registry: Registry
) -> InputSpec | None:
    top = _read_fields(ctx, data, TOP_FIELDS, "")
    freq_scale = 1.0
    if "freq_scale" in top:
        try:
            freq_scale = check_freq_scale(top["freq_scale"])
        except ValueError as exc:
            ctx.keyed_error("", exc, {"freq_scale": "freq_scale"})
    store = None if top.get("store") is None else ctx.resolve(top["store"])

    reactants = None
    if top.get("reactants") is not None:
        reactants = _molecules(ctx, top["reactants"], "reactants", "r", MAX_REACTANTS)
    ts = _molecule(ctx, top["ts"], "ts") if "ts" in top else None
    products = None
    if top.get("products") is not None:
        products = _molecules(ctx, top["products"], "products", "p", None)

    engine = None
    if "engine" in top:
        engine = _adapter(ctx, top["engine"], "engine", registry.engines)
    optimizer = None
    if top.get("optimizer") is not None:
        optimizer = _adapter(ctx, top["optimizer"], "optimizer", registry.optimizers)
    mep = None
    if top.get("mep") is not None:
        mep = _mep(ctx, top["mep"], ts)
    rates = None
    if "rates" in top:
        rates = _rate_jobs(ctx, top["rates"])

    if reactants is not None and all(m is not None for m in reactants):
        reacts = tuple(m for m in reactants if m is not None)
        if ts is not None:
            _balance(ctx, reacts, ts, "transition state", "ts")
        if products is not None and all(m is not None for m in products):
            prods = tuple(m for m in products if m is not None)
            _balance(ctx, reacts, prods, "products", "products")

    if (
        ctx.errors
        or reactants is None
        or ts is None
        or engine is None
        or rates is None
        or (top.get("optimizer") is not None and optimizer is None)
        or (top.get("mep") is not None and mep is None)
    ):
        return None
    return InputSpec(
        path=file,
        text=text,
        title=top.get("title"),
        store=store,
        freq_scale=freq_scale,
        reactants=tuple(m for m in reactants if m is not None),
        ts=ts,
        products=None
        if products is None
        else tuple(m for m in products if m is not None),
        engine=engine,
        optimizer=optimizer,
        mep=mep,
        rates=rates,
    )


def _balance(
    ctx: _Context,
    reactants: tuple[Molecule, ...],
    other: Molecule | tuple[Molecule, ...],
    name: str,
    path: str,
) -> None:
    try:
        _check_balance(reactants, other, name)
    except ValueError as exc:
        ctx.error(path, str(exc))


def _tables(ctx: _Context, value: object, path: str) -> list[object] | None:
    """The items of an array of tables; ``None`` (with an error) otherwise."""
    if isinstance(value, dict):
        ctx.error(path, f"expected an array of tables: write [[{path}]], not [{path}]")
        return None
    if not isinstance(value, list):
        ctx.error(path, f"expected an array of tables, found {_toml_type(value)}")
        return None
    if not value:
        ctx.error(path, "expected at least one table")
        return None
    return value


def _molecules(
    ctx: _Context, value: object, path: str, prefix: str, limit: int | None
) -> list[Molecule | None] | None:
    """The numbered molecule tables ``[<path>.<prefix>1]``, ``[<path>.<prefix>2]``,
    ..., in number order; ``None`` after reporting a structural error.

    Numbers start at 1 without gaps; ``limit`` caps the count.
    """
    example = f"[{path}.{prefix}1], [{path}.{prefix}2]"
    if isinstance(value, list):
        ctx.error(path, f"write numbered tables {example}, not [[{path}]]")
        return None
    if not isinstance(value, dict):
        ctx.error(path, f"expected tables {example}, found {_toml_type(value)}")
        return None
    if not value:
        ctx.error(path, f"expected at least one molecule: {example}")
        return None
    n_errors = len(ctx.errors)
    numbered: dict[int, object] = {}
    for key, item in value.items():
        match = re.fullmatch(prefix + r"([1-9][0-9]*)", key)
        if match is None:
            ctx.error(
                _key(path, key),
                f"unknown key; {path} are numbered tables {example}, ...",
            )
            continue
        number = int(match.group(1))
        if limit is not None and number > limit:
            allowed = " and ".join(f"{prefix}{i}" for i in range(1, limit + 1))
            ctx.error(
                _key(path, key),
                f"a reaction has one or two {path} ({allowed}), got {key}",
            )
            continue
        numbered[number] = item
    if numbered:
        missing = sorted(set(range(1, max(numbered) + 1)) - set(numbered))
        for number in missing:
            ctx.error(
                _key(path, f"{prefix}{number}"),
                f"missing: number the {path} from {prefix}1 without gaps",
            )
    # Molecules are still checked after a numbering error, so every error in
    # the file is reported, but none is returned: the set is incomplete.
    misnumbered = len(ctx.errors) > n_errors
    molecules: list[Molecule | None] = []
    for number in sorted(numbered):
        key = f"{prefix}{number}"
        item = numbered[number]
        if not isinstance(item, dict):
            ctx.error(
                _key(path, key),
                f"expected a table [{path}.{key}], found {_toml_type(item)}",
            )
            molecules.append(None)
            continue
        molecules.append(_molecule(ctx, item, _key(path, key), default_name=key))
    return None if misnumbered else molecules


_MOLECULE_KEYS = {
    "charge": "charge",
    "mult": "mult",
    "symmetry_number": "symmetry_number",
    "masses": "masses",
}


def _molecule(
    ctx: _Context, data: dict[str, Any], path: str, default_name: str | None = None
) -> Molecule | None:
    """Build one molecule table's molecule; ``None`` after reporting errors.

    The name is ``name``, else the XYZ file's comment line, else
    ``default_name`` (or ``path``).
    """
    n_errors = len(ctx.errors)
    values = _read_fields(ctx, data, MOLECULE_FIELDS, path)
    xyz, file = values.get("xyz"), values.get("file")
    if ("xyz" in data) == ("file" in data):
        ctx.error(path, "exactly one of 'xyz' and 'file' is required")
    elif "file" in data and "units" in data:
        ctx.error(
            _key(path, "units"),
            "XYZ files are always in angstrom; 'units' applies only to inline 'xyz'",
        )
    if len(ctx.errors) > n_errors:
        return None

    name = values["name"]
    masses = values["masses"]
    common = {
        "charge": values["charge"],
        "mult": values["mult"],
        "symmetry_number": values["symmetry_number"],
    }
    if xyz is not None:
        source = _key(path, "xyz")
        parsed = _parse_xyz(ctx, xyz, source)
        if parsed is None:
            return None
        symbols, coords = parsed
        if not _mass_count(ctx, masses, len(symbols), path):
            return None
        factor = ANGSTROM_TO_BOHR if values["units"] == "angstrom" else 1.0
        try:
            return Molecule(
                symbols,
                [[c * factor for c in row] for row in coords],
                masses=masses,
                name=name or default_name or path,
                **common,
            )
        except ValueError as exc:
            keys = {**_MOLECULE_KEYS, "symbols": "xyz", "coords": "xyz"}
            ctx.keyed_error(path, exc, keys, fallback="xyz")
            return None

    resolved = ctx.resolve(cast(str, file))  # exactly one source, and not xyz
    if not resolved.is_file():
        ctx.error(_key(path, "file"), f"no such file: {resolved}")
        return None
    try:
        mol = read_xyz(resolved, name=name, **common)
    except ValueError as exc:
        keys = {**_MOLECULE_KEYS, "symbols": "file", "coords": "file"}
        ctx.keyed_error(path, exc, keys, fallback="file")
        return None
    if masses is not None:
        if not _mass_count(ctx, masses, mol.n_atoms, path):
            return None
        try:
            mol = mol.replace(masses=masses)
        except ValueError as exc:
            ctx.keyed_error(path, exc, _MOLECULE_KEYS)
            return None
    if not mol.name:
        mol = mol.replace(name=default_name or path)
    return mol


def _mass_count(
    ctx: _Context, masses: tuple[float, ...] | None, n_atoms: int, path: str
) -> bool:
    if masses is None or len(masses) == n_atoms:
        return True
    ctx.error(
        _key(path, "masses"),
        f"the molecule has {n_atoms} atoms but {len(masses)} masses are given",
    )
    return False


def _parse_xyz(
    ctx: _Context, text: str, path: str
) -> tuple[list[str], list[list[float]]] | None:
    """Parse inline ``symbol x y z`` lines; ``None`` after reporting errors."""
    symbols: list[str] = []
    coords: list[list[float]] = []
    ok = True
    for lineno, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        shown = line.strip()
        if len(fields) != 4:
            ctx.error(path, f"line {lineno}: expected 'symbol x y z', got {shown!r}")
            ok = False
            continue
        try:
            xyz = [float(v) for v in fields[1:]]
        except ValueError:
            ctx.error(path, f"line {lineno}: non-numeric coordinate in {shown!r}")
            ok = False
            continue
        symbols.append(fields[0])
        coords.append(xyz)
    if ok and not symbols:
        ctx.error(path, "no atoms: expected one 'symbol x y z' line per atom")
        return None
    return (symbols, coords) if ok else None


def _adapter(
    ctx: _Context,
    data: dict[str, Any],
    path: str,
    entries: Mapping[str, AdapterEntry],
) -> AdapterSpec | None:
    kind = path  # "engine" or "optimizer"
    registered = ", ".join(repr(k) for k in sorted(entries))
    if "type" not in data:
        ctx.error(_key(path, "type"), f"missing required key (one of {registered})")
        return None
    name = data["type"]
    if not isinstance(name, str):
        ctx.error(_key(path, "type"), f"expected a string, found {_toml_type(name)}")
        return None
    entry = entries.get(name.lower())
    if entry is None:
        ctx.error(
            _key(path, "type"),
            f"unknown {kind} type {name!r}; registered {kind}s: {registered}",
        )
        return None
    options = {k: v for k, v in data.items() if k != "type"}
    n_errors = len(ctx.errors)
    values = _read_fields(ctx, options, entry.fields, path)
    if len(ctx.errors) > n_errors:
        return None
    for key in entry.path_keys:
        if values.get(key) is not None:
            values[key] = str(ctx.resolve(values[key]))
    return AdapterSpec(
        type=name.lower(),
        options=tuple((f.name, values[f.name]) for f in entry.fields),
    )


_MEP_KEYS = {k: k for k in ("s_min", "s_max", "ds", "method", "hessian_every")}


def _mep(ctx: _Context, data: dict[str, Any], ts: Molecule | None) -> MEPSpec | None:
    n_errors = len(ctx.errors)
    values = _read_fields(ctx, data, MEP_FIELDS, "mep")
    if len(ctx.errors) > n_errors:
        return None
    try:
        s_min, s_max, ds, every, method = _mep_parameters(
            values["s_min"],
            values["s_max"],
            values["ds"],
            values["method"],
            values["hessian_every"],
        )
    except ValueError as exc:
        ctx.keyed_error("mep", exc, _MEP_KEYS)
        return None

    raw = values["internals"]
    internals: tuple[Internal, ...] | None = None
    if raw is not None:
        if ts is None:
            return None  # the transition state's errors are already reported
        entries: list[tuple[object, ...]] = []
        for i, entry in enumerate(raw):
            item = tuple(entry) if isinstance(entry, list) else entry
            try:
                validate_internals([item], ts.n_atoms)
            except ValueError as exc:
                ctx.keyed_error(f"mep.internals[{i}]", exc, {"internals": ""})
                continue
            entries.append(item)
        if len(ctx.errors) > n_errors:
            return None
        raw = entries
    try:
        coordinates, internals = _coordinate_parameters(
            values["coordinates"], raw, 1 if ts is None else ts.n_atoms
        )
    except ValueError as exc:
        ctx.keyed_error(
            "mep", exc, {"coordinates": "coordinates", "internals": "internals"}
        )
        return None
    return MEPSpec(s_min, s_max, ds, method, every, coordinates, internals)


def _rate_jobs(ctx: _Context, value: object) -> tuple[RateJob, ...] | None:
    items = _tables(ctx, value, "rates")
    if items is None:
        return None
    jobs = []
    for i, item in enumerate(items):
        path = f"rates[{i}]"
        n_errors = len(ctx.errors)
        values = _read_table(ctx, item, RATE_FIELDS, path)
        if values is None:
            continue
        temps = (
            _temperatures(ctx, values["T"], _key(path, "T")) if "T" in values else None
        )
        if temps is None or len(ctx.errors) > n_errors:
            continue
        tunneling = None if values["tunneling"] == "none" else values["tunneling"]
        model = None if values["arrhenius"] == "none" else values["arrhenius"]
        if model is not None:
            try:
                check_count(model, len(set(temps)))
            except ValueError as exc:
                ctx.error(_key(path, "arrhenius"), str(exc))
                continue
        jobs.append(RateJob(temps, values["theory"], tunneling, model))
    return tuple(jobs) if len(jobs) == len(items) else None


_T_FORMS = "a number, an array of numbers or a table { start, stop, step }"


def _temperatures(ctx: _Context, value: object, path: str) -> tuple[float, ...] | None:
    if isinstance(value, dict):
        n_errors = len(ctx.errors)
        r = _read_fields(ctx, value, T_RANGE_FIELDS, path)
        if len(ctx.errors) > n_errors:
            return None
        start, stop, step = r["start"], r["stop"], r["step"]
        if not start > 0:
            ctx.error(_key(path, "start"), f"temperatures must be > 0 K, got {start:g}")
        if not step > 0:
            ctx.error(_key(path, "step"), f"must be > 0, got {step:g}")
        if not stop >= start:
            ctx.error(_key(path, "stop"), f"must be >= start ({start:g}), got {stop:g}")
        if len(ctx.errors) > n_errors:
            return None
        n = math.floor((stop * (1 + T_RANGE_RTOL) - start) / step)
        return tuple(start + i * step for i in range(n + 1))
    if isinstance(value, list):
        if not value:
            ctx.error(path, "expected at least one temperature")
            return None
        temps: list[float] = []
        for i, v in enumerate(value):
            t = _checked_temperature(ctx, v, f"{path}[{i}]")
            if t is not None:
                temps.append(t)
        return tuple(temps) if len(temps) == len(value) else None
    if _number(value) is None:
        ctx.error(path, f"expected {_T_FORMS}, found {_toml_type(value)}")
        return None
    t = _checked_temperature(ctx, value, path)
    return None if t is None else (t,)


def _checked_temperature(ctx: _Context, value: object, path: str) -> float | None:
    t = _number(value)
    if t is None:
        ctx.error(path, f"expected a number, found {_toml_type(value)}")
        return None
    if not (math.isfinite(t) and t > 0):
        ctx.error(path, f"temperatures must be finite and > 0 K, got {t:g}")
        return None
    return t
