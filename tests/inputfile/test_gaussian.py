"""Gaussian engine and optimizer in input files, run against the fake programs."""

import pytest

from pyvtst.engines import GaussianEngine
from pyvtst.inputfile import DEFAULT_REGISTRY, InputError, load, run
from pyvtst.optimizers import GaussianOptimizer
from tests.inputfile.conftest import PYSCF_INPUT

GAUSSIAN_INPUT = '''\
title = "H2 + H with Gaussian"

[engine]
type = "gaussian"
method = "HF"
basis = "3-21G"
nprocs = 4

[optimizer]
type = "gaussian"
convergence = "verytight"

[reactants.r1]
name = "H2"
units = "bohr"
xyz = """
H 0.0 0.0 0.0
H 1.8 0.0 0.0
"""

[reactants.r2]
name = "H"
mult = 2
xyz = "H 0 0 0"

[ts]
name = "H3"
mult = 2
units = "bohr"
xyz = """
H -1.6 0.0 0.0
H  0.0 0.0 0.0
H  1.9 0.0 0.0
"""

[mep]
s_min = -0.04
s_max = 0.04
ds = 0.02

[[rates]]
T = [300, 1000]
theory = "cvt"
'''


def load_errors(write, text):
    with pytest.raises(InputError) as info:
        load(write(text))
    return info.value.errors


def test_both_adapters_build_from_validated_options(write, fake_gaussian):
    spec = load(write(GAUSSIAN_INPUT))
    assert spec.engine.type == "gaussian"
    assert spec.optimizer is not None
    engine = DEFAULT_REGISTRY.engines["gaussian"].build(spec.engine.kwargs)
    optimizer = DEFAULT_REGISTRY.optimizers["gaussian"].build(spec.optimizer.kwargs)
    assert isinstance(engine, GaussianEngine)
    assert isinstance(optimizer, GaussianOptimizer)
    assert engine.level == GaussianEngine("HF", "3-21G").level
    assert optimizer.key() == GaussianOptimizer(convergence="verytight").key()


def test_optimizer_takes_the_engine_run_settings(write, tmp_path):
    text = GAUSSIAN_INPUT.replace(
        "nprocs = 4\n",
        'nprocs = 4\nmemory = "8GB"\nexecutable = "bin/g16"\n'
        'formchk = "/opt/g16/formchk"\n',
    )
    spec = load(write(text))
    assert spec.optimizer is not None
    options = spec.optimizer.kwargs
    assert options["nprocs"] == 4
    assert options["memory"] == "8GB"
    assert options["executable"] == str(tmp_path.resolve() / "bin" / "g16")
    assert options["formchk"] == "/opt/g16/formchk"
    assert spec.engine.kwargs["executable"] == options["executable"]


def test_unset_executables_follow_the_lookup(write):
    spec = load(write(GAUSSIAN_INPUT))
    assert spec.optimizer is not None
    for kwargs in (spec.engine.kwargs, spec.optimizer.kwargs):
        assert kwargs["executable"] is None
        assert kwargs["formchk"] is None


def test_gaussian_optimizer_with_another_engine(write):
    text = PYSCF_INPUT.replace(
        "[reactants.r1]", '[optimizer]\ntype = "gaussian"\n\n[reactants.r1]', 1
    )
    errors = load_errors(write, text)
    assert len(errors) == 1
    expected = "optimizer.type: the 'gaussian' optimizer needs the 'gaussian' engine"
    assert expected in errors[0]


def test_gaussian_optimizer_with_a_broken_engine_table(write):
    text = GAUSSIAN_INPUT.replace('basis = "3-21G"\n', "")
    errors = load_errors(write, text)
    assert len(errors) == 1
    assert "engine.basis: missing required key" in errors[0]


def test_unknown_engine_key_reported(write):
    text = GAUSSIAN_INPUT.replace("nprocs = 4\n", "nprocs = 4\ngrid_level = 3\n")
    errors = load_errors(write, text)
    assert len(errors) == 1
    assert "engine.grid_level: unknown key" in errors[0]


def test_engine_keys_are_not_optimizer_keys(write):
    text = GAUSSIAN_INPUT.replace(
        'convergence = "verytight"\n', 'convergence = "verytight"\nnprocs = 2\n'
    )
    errors = load_errors(write, text)
    assert "optimizer.nprocs: unknown key" in errors[0]


def test_complete_run(write, fake_gaussian, tmp_path):
    report = tmp_path / "h3.out"
    (result,) = run(write(GAUSSIAN_INPUT), output=report)
    assert result.theory == "cvt"
    assert report.is_file()
    assert "H2 + H with Gaussian" in report.read_text()
    decks = fake_gaussian.decks()
    assert all(deck.startswith("%NProcShared=4\n") for deck in decks)
    opt_routes = [r for r in fake_gaussian.routes() if " Opt=(" in r]
    assert len(opt_routes) == 2
    assert all("VeryTight" in r for r in opt_routes)
