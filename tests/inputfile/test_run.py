"""Running input files on the LEPS engine."""

import numpy as np
import pytest

from pyvtst import Molecule, Reaction
from pyvtst.inputfile import load, run
from pyvtst.inputfile.build import execute
from pyvtst.ports import ConvergenceError
from tests.inputfile.conftest import (
    H3_INPUT,
    CountingLEPS,
    ExactOptimizer,
    leps_registry,
)


def with_rates(rates):
    return H3_INPUT.replace('[[rates]]\nT = [300, 1000]\ntunneling = "eckart"\n', rates)


def direct_reaction(engine):
    """The H3_INPUT reaction built with the Python API."""
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], name="H2")
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, name="H")
    ts = Molecule(
        ["H", "H", "H"],
        [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
        mult=2,
        name="H3",
    )
    return Reaction([h2, h], ts, engine, optimizer=ExactOptimizer())


def test_registered_test_adapter_runs(write, registry, leps):
    results = run(write(H3_INPUT), registry=registry)
    assert len(results) == 1
    assert list(results[0].T) == [300.0, 1000.0]
    assert results[0].tunneling == "eckart"
    assert leps.calls > 0


def test_same_numbers_as_the_python_api(write, registry):
    (result,) = run(write(H3_INPUT), registry=registry)
    direct = direct_reaction(CountingLEPS()).rates([300, 1000], tunneling="eckart")
    np.testing.assert_array_equal(result.k, direct.k)
    np.testing.assert_array_equal(result.kappa, direct.kappa)


def test_several_jobs_share_the_work(write, registry, leps):
    rates = (
        '[[rates]]\nT = [300, 1000]\ntheory = "tst"\ntunneling = "eckart"\n\n'
        '[[rates]]\nT = [300, 1000]\ntheory = "cvt"\ntunneling = "sct"\n'
    )
    with pytest.warns(UserWarning, match="the path ends"):
        results = run(write(with_rates(rates)), registry=registry)
    assert [(r.theory, r.tunneling) for r in results] == [
        ("tst", "eckart"),
        ("cvt", "sct"),
    ]

    engine = CountingLEPS()
    rxn = direct_reaction(engine)
    rxn.optimize()
    path = rxn.compute_mep()
    rxn.rates([300, 1000], "tst", "eckart")
    with pytest.warns(UserWarning, match="the path ends"):
        direct = rxn.rates([300, 1000], "cvt", "sct", path=path)
    assert leps.calls == engine.calls
    # Path tunneling sums in BLAS, which can differ in the last bit between runs.
    np.testing.assert_allclose(results[1].k, direct.k, rtol=1e-12)


def test_wider_path_for_cvt(write, registry):
    text = with_rates(
        "[mep]\ns_min = -1.0\ns_max = 1.0\nds = 0.02\n\n"
        '[[rates]]\nT = 300\ntheory = "cvt"\n'
    )
    log = execute(load(write(text), registry), registry)
    assert log.exception is None
    assert log.path is not None
    assert log.path.s[0] == pytest.approx(-1.0)
    assert log.path.s[-1] == pytest.approx(1.0)
    cvt = log.results[0].cvt
    assert cvt is not None
    assert cvt.s[0] == pytest.approx(-1.0)
    assert cvt.s[-1] == pytest.approx(1.0)


def test_path_defaults_without_the_section(write, registry):
    log = execute(
        load(write(with_rates('[[rates]]\nT = 300\ntunneling = "zct"\n')), registry),
        registry,
    )
    assert log.exception is None
    assert log.path is not None
    assert log.path.s[0] == pytest.approx(-0.5)
    assert log.path.s[-1] == pytest.approx(0.5)
    assert (log.path.method, log.path.ds) == ("lqa", 0.01)


def test_internal_coordinates_and_path_without_path_jobs(write, registry):
    body = 'internals = [["bond", 0, 1], ["bond", 1, 2], ["linear_bend", 0, 1, 2]]'
    text = with_rates(f"[mep]\nds = 0.05\n{body}\n\n[[rates]]\nT = 300\n")
    log = execute(load(write(text), registry), registry)
    assert log.exception is None
    assert log.path is not None
    assert log.path.internals == (
        ("bond", 0, 1),
        ("bond", 1, 2),
        ("linear_bend", 0, 1, 2),
    )
    assert log.results[0].theory == "tst"


def test_explicit_no_tunneling(write, registry):
    (result,) = run(
        write(with_rates('[[rates]]\nT = [300, 600]\ntunneling = "none"\n')),
        registry=registry,
    )
    assert result.tunneling is None
    np.testing.assert_array_equal(result.kappa, [1.0, 1.0])


def test_calculation_error_is_raised_after_the_report(write, tmp_path, leps):
    registry = leps_registry(leps)
    text = H3_INPUT.replace('type = "exact"', 'type = "exact"\nfail_saddle = true')
    output = tmp_path / "out" / "h3.out"
    with pytest.raises(ConvergenceError, match="H3"):
        run(write(text), output=output, registry=registry)
    assert (
        output.read_text()
        .rstrip()
        .splitlines()[-1]
        .startswith("pyvtst stopped with an error:")
    )


def test_warnings_are_recorded(write, registry):
    text = H3_INPUT.replace('name = "H3"\nmult = 2', 'name = "H3"\nmult = 4')
    log = execute(load(write(text), registry), registry)
    assert log.exception is None
    assert any("total spin is not conserved" in w for w in log.warnings)


def test_default_optimizer(write, registry):
    pytest.importorskip("geometric")
    text = H3_INPUT.replace('[optimizer]\ntype = "exact"\n', "")
    (result,) = run(write(text), registry=registry)
    direct = direct_reaction(CountingLEPS()).rates([300, 1000], tunneling="eckart")
    np.testing.assert_allclose(result.k, direct.k, rtol=1e-3)
