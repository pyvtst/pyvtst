"""Adapters an input file can select by ``type``, and the fields they accept.

Each :class:`AdapterEntry` declares the options of one engine or optimizer
and how to build it. Supporting a new adapter in input files means adding one
entry; the parser does not change. ``build`` imports the adapter's class when
called, so importing this module imports no third-party tool.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, get_args

FieldKind = Literal["str", "int", "float", "bool", "floats", "list", "table", "any"]


@dataclass(frozen=True)
class Field:
    """One key of an input-file table.

    Attributes:
        name: The key.
        kind: The TOML type accepted: ``"str"``, ``"int"``, ``"float"`` (an
            integer or a float, finite), ``"bool"``, ``"floats"`` (an array of
            numbers), ``"list"`` (any array), ``"table"`` or ``"any"``.
        required: Whether the key must be present.
        default: The value used when the key is absent and not required.
        choices: For ``"str"``, the accepted values, matched
            case-insensitively; the value is replaced by the matching choice.
    """

    name: str
    kind: FieldKind
    required: bool = False
    default: object = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AdapterEntry:
    """An engine or optimizer that input files can select.

    Attributes:
        fields: The adapter's options, without ``type``.
        build: Builds the adapter from the validated options, every field
            present (absent ones at their default).
        path_keys: Options that are paths, resolved against the input file's
            directory.
    """

    fields: tuple[Field, ...]
    build: Callable[[Mapping[str, Any]], Any]
    path_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Registry:
    """The engines and optimizers available to input files, by lower-case type."""

    engines: Mapping[str, AdapterEntry]
    optimizers: Mapping[str, AdapterEntry]

    def __post_init__(self) -> None:
        for name in ("engines", "optimizers"):
            entries = {k.lower(): v for k, v in getattr(self, name).items()}
            object.__setattr__(self, name, MappingProxyType(entries))

    def with_engine(self, name: str, entry: AdapterEntry) -> Registry:
        """Return a copy with the engine ``name`` added or replaced."""
        return replace(self, engines={**self.engines, name.lower(): entry})

    def with_optimizer(self, name: str, entry: AdapterEntry) -> Registry:
        """Return a copy with the optimizer ``name`` added or replaced."""
        return replace(self, optimizers={**self.optimizers, name.lower(): entry})


def _build_pyscf(options: Mapping[str, Any]) -> Any:
    from pyvtst.engines import PySCFEngine  # noqa: PLC0415

    return PySCFEngine(**options)


def _build_geometric(options: Mapping[str, Any]) -> Any:
    from pyvtst.optimizers import GeometricOptimizer  # noqa: PLC0415

    return GeometricOptimizer(**options)


def _build_gaussian_engine(options: Mapping[str, Any]) -> Any:
    from pyvtst.engines import GaussianEngine  # noqa: PLC0415

    return GaussianEngine(**options)


def _build_gaussian_optimizer(options: Mapping[str, Any]) -> Any:
    from pyvtst.optimizers import GaussianOptimizer  # noqa: PLC0415

    return GaussianOptimizer(**options)


def _pyscf_entry() -> AdapterEntry:
    # The PySCF module imports PySCF itself only when an engine is built.
    from pyvtst.engines.pyscf import HessianMode, Reference  # noqa: PLC0415

    return AdapterEntry(
        fields=(
            Field("method", "str", required=True),
            Field("basis", "str", required=True),
            Field("reference", "str", default="auto", choices=get_args(Reference)),
            Field("hessian", "str", default="auto", choices=get_args(HessianMode)),
            Field("fd_step", "float", default=5e-3),
            Field("conv_tol", "float", default=1e-10),
            Field("grid_level", "int", default=3),
            Field("max_cycle", "int", default=50),
            Field("max_memory_mb", "int", default=None),
            Field("verbose", "bool", default=False),
        ),
        build=_build_pyscf,
    )


def _geometric_entry() -> AdapterEntry:
    from pyvtst.optimizers.geometric import (  # noqa: PLC0415
        CONVERGENCE_SETS,
        COORDINATE_SYSTEMS,
    )

    return AdapterEntry(
        fields=(
            Field(
                "convergence",
                "str",
                default="GAU_TIGHT",
                choices=tuple(sorted(CONVERGENCE_SETS)),
            ),
            Field("max_iterations", "int", default=300),
            Field(
                "coordsys",
                "str",
                default="tric",
                choices=tuple(sorted(COORDINATE_SYSTEMS)),
            ),
            Field("workdir", "str", default=None),
            Field("verbose", "bool", default=False),
        ),
        build=_build_geometric,
        path_keys=frozenset({"workdir"}),
    )


# Gaussian optimizer options taken from the [engine] table, so that a
# Gaussian engine and optimizer run with the same programs and resources.
GAUSSIAN_SHARED_KEYS = ("nprocs", "memory", "executable", "formchk")


def _gaussian_engine_entry() -> AdapterEntry:
    from pyvtst.programs.gaussian import REFERENCES  # noqa: PLC0415

    return AdapterEntry(
        fields=(
            Field("method", "str", required=True),
            Field("basis", "str", required=True),
            Field("keywords", "str", default=""),
            Field("reference", "str", default="auto", choices=REFERENCES),
            Field("nprocs", "int", default=1),
            Field("memory", "str", default="2GB"),
            Field("executable", "str", default=None),
            Field("formchk", "str", default=None),
            Field("workdir", "str", default=None),
            Field("timeout", "float", default=None),
            Field("verbose", "bool", default=False),
        ),
        build=_build_gaussian_engine,
        path_keys=frozenset({"executable", "formchk", "workdir"}),
    )


def _gaussian_optimizer_entry() -> AdapterEntry:
    """The optimizer's own keys; the schema adds :data:`GAUSSIAN_SHARED_KEYS`
    from the Gaussian engine."""
    from pyvtst.optimizers.gaussian import CONVERGENCE  # noqa: PLC0415

    return AdapterEntry(
        fields=(
            Field("convergence", "str", default="tight", choices=CONVERGENCE),
            Field("max_cycles", "int", default=100),
            Field("workdir", "str", default=None),
            Field("timeout", "float", default=None),
            Field("verbose", "bool", default=False),
        ),
        build=_build_gaussian_optimizer,
        path_keys=frozenset({"workdir"}),
    )


DEFAULT_REGISTRY = Registry(
    engines={"gaussian": _gaussian_engine_entry(), "pyscf": _pyscf_entry()},
    optimizers={
        "gaussian": _gaussian_optimizer_entry(),
        "geometric": _geometric_entry(),
    },
)
"""The built-in adapters: engines ``"gaussian"`` and ``"pyscf"``, optimizers
``"gaussian"`` and ``"geometric"``."""
