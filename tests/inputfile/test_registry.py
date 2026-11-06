"""Adapter sections and the registry."""

import inspect
from dataclasses import replace

import pytest

from pyvtst.engines import PySCFEngine
from pyvtst.inputfile import DEFAULT_REGISTRY, Field, InputError, load
from pyvtst.optimizers import GeometricOptimizer
from tests.inputfile.conftest import H3_INPUT, PYSCF_INPUT

ADAPTERS = [
    ("pyscf", DEFAULT_REGISTRY.engines["pyscf"], PySCFEngine),
    ("geometric", DEFAULT_REGISTRY.optimizers["geometric"], GeometricOptimizer),
]


def engine_section(body):
    return PYSCF_INPUT.replace(
        '[engine]\ntype = "pyscf"\nmethod = "B3LYP"\nbasis = "def2-SVP"\n', body
    )


def load_errors(write, text):
    with pytest.raises(InputError) as info:
        load(write(text))
    return info.value.errors


def signature_mismatches(entry, cls):
    """Declared fields that are not keyword parameters of ``cls`` with the
    same default (required fields must have no default)."""
    params = inspect.signature(cls).parameters
    problems = []
    for f in entry.fields:
        p = params.get(f.name)
        if p is None:
            problems.append(f"{f.name}: not a parameter")
        elif f.required and p.default is not inspect.Parameter.empty:
            problems.append(f"{f.name}: required but has a default")
        elif not f.required and p.default != f.default:
            problems.append(f"{f.name}: default {f.default!r} != {p.default!r}")
    return problems


@pytest.mark.parametrize(
    ("name", "entry", "cls"), ADAPTERS, ids=lambda a: a if isinstance(a, str) else ""
)
def test_declared_fields_match_the_constructor(name, entry, cls):
    assert signature_mismatches(entry, cls) == []


def test_renamed_field_is_detected():
    entry = DEFAULT_REGISTRY.engines["pyscf"]
    renamed = replace(
        entry, fields=(*entry.fields[:-1], Field("verbosity", "bool", default=False))
    )
    assert signature_mismatches(renamed, PySCFEngine) == ["verbosity: not a parameter"]


def test_pyscf_engine_options(write, monkeypatch):
    built: dict[str, object] = {}

    class Recorder:
        def __init__(self, method, basis, **kwargs):
            built.update(method=method, basis=basis, **kwargs)

    monkeypatch.setattr("pyvtst.engines.PySCFEngine", Recorder)
    text = engine_section(
        '[engine]\ntype = "PySCF"\nmethod = "B3LYP"\nbasis = "def2-SVP"\n'
        'hessian = "fd"\n'
    )
    spec = load(write(text))
    assert spec.engine.type == "pyscf"
    DEFAULT_REGISTRY.engines["pyscf"].build(spec.engine.kwargs)
    expected = dict(spec.engine.kwargs)
    assert built == expected
    assert built["hessian"] == "fd"
    assert built["grid_level"] == 3


def test_unknown_adapter_type(write):
    errors = load_errors(write, engine_section('[engine]\ntype = "orca"\n'))
    assert len(errors) == 1
    assert (
        "engine.type: unknown engine type 'orca'; registered engines: 'pyscf'"
        in errors[0]
    )


def test_option_of_another_adapter(write):
    text = engine_section(
        '[engine]\ntype = "pyscf"\nmethod = "HF"\nbasis = "sto-3g"\n'
        'convergence = "GAU"\n'
    )
    errors = load_errors(write, text)
    assert len(errors) == 1
    assert "engine.convergence: unknown key" in errors[0]
    assert "method, basis, reference" in errors[0]


def test_missing_required_option(write):
    errors = load_errors(
        write, engine_section('[engine]\ntype = "pyscf"\nmethod = "HF"\n')
    )
    assert len(errors) == 1
    assert "engine.basis: missing required key" in errors[0]


def test_missing_type(write):
    errors = load_errors(write, engine_section('[engine]\nmethod = "HF"\n'))
    assert "engine.type: missing required key (one of 'pyscf')" in errors[0]


def test_bad_choice(write):
    text = engine_section(
        '[engine]\ntype = "pyscf"\nmethod = "HF"\nbasis = "sto-3g"\n'
        'reference = "open"\n'
    )
    errors = load_errors(write, text)
    assert (
        "engine.reference: expected one of 'auto', 'restricted', 'unrestricted'"
        in errors[0]
    )


def test_geometric_options_and_workdir(write, tmp_path):
    text = PYSCF_INPUT.replace(
        "[reactants.r1]",
        '[optimizer]\ntype = "geometric"\nconvergence = "gau"\n'
        'workdir = "logs"\n\n[reactants.r1]',
        1,
    )
    spec = load(write(text))
    assert spec.optimizer is not None
    options = spec.optimizer.kwargs
    assert spec.optimizer.type == "geometric"
    assert options["convergence"] == "GAU"
    assert options["workdir"] == str(tmp_path.resolve() / "logs")
    assert options["max_iterations"] == 300


def test_registered_test_adapter_loads(write, registry):
    spec = load(write(H3_INPUT), registry)
    assert spec.engine.type == "leps"
    assert spec.optimizer is not None
    assert spec.optimizer.type == "exact"


def test_test_adapter_unknown_to_default_registry(write):
    errors = load_errors(write, H3_INPUT)
    assert any("engine.type: unknown engine type 'leps'" in e for e in errors)
    assert any("optimizer.type: unknown optimizer type 'exact'" in e for e in errors)


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        DEFAULT_REGISTRY.engines["x"] = DEFAULT_REGISTRY.engines["pyscf"]  # type: ignore[index]
