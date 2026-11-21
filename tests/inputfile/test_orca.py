"""ORCA engine and optimizer in input files, run against the fake program."""

import pytest

from pyvtst.engines import OrcaEngine
from pyvtst.inputfile import DEFAULT_REGISTRY, InputError, load, run
from pyvtst.optimizers import OrcaOptimizer
from tests.inputfile.test_gaussian import GAUSSIAN_INPUT

ORCA_INPUT = (
    GAUSSIAN_INPUT.replace(
        'title = "H2 + H with Gaussian"', 'title = "H2 + H with ORCA"'
    )
    .replace(
        '[engine]\ntype = "gaussian"\nmethod = "HF"\nbasis = "3-21G"\nnprocs = 4\n',
        '[engine]\ntype = "orca"\nmethod = "HF"\nbasis = "3-21G"\nnprocs = 4\n'
        "maxcore = 3000\n",
    )
    .replace('[optimizer]\ntype = "gaussian"', '[optimizer]\ntype = "orca"')
)


def load_errors(write, text):
    with pytest.raises(InputError) as info:
        load(write(text))
    return info.value.errors


def test_input_is_an_orca_input():
    assert 'type = "gaussian"' not in ORCA_INPUT
    assert ORCA_INPUT.count('type = "orca"') == 2


def test_both_adapters_build_from_validated_options(write, fake_orca):
    spec = load(write(ORCA_INPUT))
    assert spec.engine.type == "orca"
    assert spec.optimizer is not None
    engine = DEFAULT_REGISTRY.engines["orca"].build(spec.engine.kwargs)
    optimizer = DEFAULT_REGISTRY.optimizers["orca"].build(spec.optimizer.kwargs)
    assert isinstance(engine, OrcaEngine)
    assert isinstance(optimizer, OrcaOptimizer)
    assert engine.level == OrcaEngine("HF", "3-21G").level
    assert optimizer.key() == OrcaOptimizer(convergence="verytight").key()
    assert engine._settings.nprocs == 4
    assert engine._settings.maxcore == 3000


def test_all_engine_keys(write, fake_orca):
    text = ORCA_INPUT.replace(
        "maxcore = 3000\n",
        'maxcore = 3000\nkeywords = "D3BJ"\n'
        'blocks = """\n%scf\n  MaxIter 300\nend\n"""\n'
        'reference = "unrestricted"\nhessian = "numerical"\ntimeout = 60\n'
        "verbose = false\n",
    )
    spec = load(write(text))
    engine = DEFAULT_REGISTRY.engines["orca"].build(spec.engine.kwargs)
    expected = OrcaEngine(
        "HF",
        "3-21G",
        keywords="d3bj",
        blocks="%scf\nMaxIter 300\nend",
        reference="unrestricted",
        hessian="numerical",
    )
    assert engine.level == expected.level


def test_composite_method_without_a_basis(write, fake_orca):
    text = ORCA_INPUT.replace(
        'method = "HF"\nbasis = "3-21G"\n', 'method = "r2SCAN-3c"\n'
    )
    spec = load(write(text))
    assert spec.engine.kwargs["basis"] is None
    engine = DEFAULT_REGISTRY.engines["orca"].build(spec.engine.kwargs)
    assert engine.level.basis == "-"


def test_optimizer_takes_the_engine_run_settings(write, tmp_path):
    text = ORCA_INPUT.replace(
        "maxcore = 3000\n", 'maxcore = 3000\nexecutable = "bin/orca"\n'
    )
    spec = load(write(text))
    assert spec.optimizer is not None
    options = spec.optimizer.kwargs
    assert options["nprocs"] == 4
    assert options["maxcore"] == 3000
    assert options["executable"] == str(tmp_path.resolve() / "bin" / "orca")
    assert spec.engine.kwargs["executable"] == options["executable"]


def test_unset_executable_follows_the_lookup(write):
    spec = load(write(ORCA_INPUT))
    assert spec.optimizer is not None
    assert spec.engine.kwargs["executable"] is None
    assert spec.optimizer.kwargs["executable"] is None


def test_workdir_resolves_against_the_input_file(write, tmp_path):
    text = ORCA_INPUT.replace("maxcore = 3000\n", 'maxcore = 3000\nworkdir = "runs"\n')
    spec = load(write(text))
    assert spec.engine.kwargs["workdir"] == str(tmp_path.resolve() / "runs")


def test_memory_is_not_an_orca_key(write):
    text = ORCA_INPUT.replace("maxcore = 3000\n", 'maxcore = 3000\nmemory = "4GB"\n')
    errors = load_errors(write, text)
    assert len(errors) == 1
    assert "engine.memory: unknown key" in errors[0]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("hessian", '"fd"', "engine.hessian: expected one of 'analytic', 'numerical'"),
        ("reference", '"open"', "engine.reference: expected one of"),
    ],
)
def test_bad_engine_choices(write, key, value, message):
    text = ORCA_INPUT.replace("maxcore = 3000\n", f"maxcore = 3000\n{key} = {value}\n")
    errors = load_errors(write, text)
    assert message in errors[0]


def test_bad_optimizer_convergence(write):
    text = ORCA_INPUT.replace('convergence = "verytight"', 'convergence = "loose"')
    errors = load_errors(write, text)
    assert (
        "optimizer.convergence: expected one of 'normal', 'tight', 'verytight'"
        in (errors[0])
    )


def test_orca_optimizer_with_another_engine(write):
    text = GAUSSIAN_INPUT.replace(
        '[optimizer]\ntype = "gaussian"', '[optimizer]\ntype = "orca"'
    )
    errors = load_errors(write, text)
    assert len(errors) == 1
    expected = (
        "optimizer.type: the 'orca' optimizer needs the 'orca' engine, "
        "got engine type 'gaussian'"
    )
    assert expected in errors[0]


def test_gaussian_optimizer_with_the_orca_engine(write):
    text = ORCA_INPUT.replace(
        '[optimizer]\ntype = "orca"', '[optimizer]\ntype = "gaussian"'
    )
    errors = load_errors(write, text)
    assert "the 'gaussian' optimizer needs the 'gaussian' engine" in errors[0]


def test_engine_keys_are_not_optimizer_keys(write):
    text = ORCA_INPUT.replace(
        'convergence = "verytight"\n', 'convergence = "verytight"\nmaxcore = 2\n'
    )
    errors = load_errors(write, text)
    assert "optimizer.maxcore: unknown key" in errors[0]


def test_complete_run(write, fake_orca, tmp_path):
    report = tmp_path / "h3.out"
    (result,) = run(write(ORCA_INPUT), output=report)
    assert result.theory == "cvt"
    assert report.is_file()
    assert "H2 + H with ORCA" in report.read_text()
    decks = fake_orca.decks()
    assert all("%pal nprocs 4 end\n%maxcore 3000\n" in deck for deck in decks)
    opt_lines = [line for line in fake_orca.keyword_lines() if "Opt" in line]
    assert len(opt_lines) == 2
    assert all("VeryTightOpt" in line for line in opt_lines)
    hessian_lines = [
        line for line in fake_orca.keyword_lines() if "EnGrad Freq" in line
    ]
    assert hessian_lines
