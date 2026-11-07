"""The pyvtst command and its report."""

import runpy
import subprocess
import sys
from datetime import datetime

import pytest

import pyvtst
from pyvtst.inputfile import load, run
from pyvtst.inputfile.build import RunLog
from pyvtst.inputfile.cli import main
from pyvtst.inputfile.report import render
from tests.inputfile.conftest import H3_INPUT, PYSCF_INPUT, CountingLEPS, leps_registry


def masked(report):
    """The report without its date line, the only line that changes per run."""
    return [line for line in report.splitlines() if not line.startswith("date:")]


def section(report, title):
    """The lines of one report section."""
    lines = report.splitlines()
    start = lines.index(title) + 2
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("====="))
    return lines[start:end]


def test_default_report_path(write, registry, tmp_path, capsys):
    path = write(H3_INPUT, "calc/h3.toml")
    assert main([str(path)], registry=registry) == 0
    report = tmp_path / "calc" / "h3.out"
    assert report.is_file()
    assert str(report) in capsys.readouterr().out


def test_custom_report_path(write, registry, tmp_path):
    path = write(H3_INPUT)
    output = tmp_path / "results" / "run1.out"
    assert main([str(path), "-o", str(output)], registry=registry) == 0
    assert output.is_file()
    assert not (tmp_path / "h3.out").exists()


def test_existing_report_is_overwritten(write, registry, tmp_path):
    path = write(H3_INPUT)
    (tmp_path / "h3.out").write_text("old")
    assert main([str(path)], registry=registry) == 0
    assert (tmp_path / "h3.out").read_text().startswith("=")


def test_report_prints_the_arrhenius_fit(write, registry, tmp_path):
    rates = (
        "[[rates]]\nT = { start = 300, stop = 1000, step = 100 }\n"
        'arrhenius = "modified"\n'
        '\n[[rates]]\nT = [2, 3, 4, 300]\narrhenius = "arrhenius"\n'
    )
    text = H3_INPUT.replace('[[rates]]\nT = [300, 1000]\ntunneling = "eckart"\n', rates)
    assert main([str(write(text))], registry=registry) == 0
    lines = section((tmp_path / "h3.out").read_text(), "RATE CONSTANTS")
    fit = lines.index("Arrhenius fit (modified): k = A (T/T_ref)^n exp(-Ea/RT)")
    assert lines[fit - 2].startswith("  1000.00")
    assert lines[fit + 1].startswith("  A     = ")
    assert lines[fit + 1].endswith("cm^3 molecule^-1 s^-1")
    assert lines[fit + 5].startswith("  fitted over 300-1000 K")
    assert "Arrhenius fit failed: k: values must be finite and > 0, got 0, 0" in lines


def test_module_invocation():
    result = subprocess.run(
        [sys.executable, "-m", "pyvtst", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"pyvtst {pyvtst.__version__}"


def test_help(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "INPUT" in capsys.readouterr().out


def test_missing_input_file(tmp_path, capsys):
    assert main([str(tmp_path / "nothere.toml")]) == 2
    assert "nothere.toml" in capsys.readouterr().err


def test_validation_errors(write, tmp_path, capsys):
    text = PYSCF_INPUT.replace("T = 300", "T = 0\ncolour = 1")
    path = write(text, "bad.toml")
    assert main([str(path)]) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert any("rates[0].T" in line for line in err)
    assert any("rates[0].colour" in line for line in err)
    assert not (tmp_path / "bad.out").exists()


def test_optimizer_fails(write, registry, tmp_path, capsys):
    text = H3_INPUT.replace('type = "exact"', 'type = "exact"\nfail_saddle = true')
    path = write(text)
    assert main([str(path)], registry=registry) == 1
    err = capsys.readouterr().err
    assert "pyvtst: error:" in err
    assert "H3" in err
    assert "Traceback" not in err
    report = (tmp_path / "h3.out").read_text()
    assert "\n".join(section(report, "INPUT")).strip() == text.strip()
    assert report.rstrip().splitlines()[-1].startswith("pyvtst stopped with an error:")


def test_missing_extra(write, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pyscf", None)
    path = write(PYSCF_INPUT)
    assert main([str(path)]) == 1
    assert 'pip install "pyvtst[pyscf]"' in capsys.readouterr().err


def test_interrupt(write, tmp_path, capsys):
    registry = leps_registry(CountingLEPS(error=KeyboardInterrupt()))
    path = write(H3_INPUT)
    assert main([str(path)], registry=registry) == 130
    assert "interrupted" in capsys.readouterr().err
    assert (tmp_path / "h3.out").read_text().rstrip().endswith("interrupted")


def test_unexpected_errors_keep_their_traceback(write):
    registry = leps_registry(CountingLEPS(error=ZeroDivisionError("bug")))
    with pytest.raises(ZeroDivisionError):
        main([str(write(H3_INPUT))], registry=registry)


def test_successful_run_report(write, registry, tmp_path):
    path = write(H3_INPUT)
    (result,) = run(path, output=tmp_path / "h3.out", registry=registry)
    report = (tmp_path / "h3.out").read_text()
    titles = [
        "PYVTST",
        "INPUT",
        "SETTINGS",
        "STATIONARY POINTS",
        "BARRIER",
        "RATE CONSTANTS",
        "WARNINGS",
    ]
    positions = [report.splitlines().index(t) for t in titles]
    assert positions == sorted(positions)
    assert f"pyvtst {pyvtst.__version__}" in report
    assert "title: H2 + H on LEPS" in report
    assert "\n".join(section(report, "INPUT")).strip() == H3_INPUT.strip()

    points = "\n".join(section(report, "STATIONARY POINTS"))
    for name in ("reactant 'H2'", "reactant 'H'", "transition state 'H3'"):
        assert name in points
    assert "energy:" in points
    ts_part = points.split("transition state 'H3'")[1]
    frequencies = ts_part.split("frequencies")[1]
    assert frequencies.count("i") == 1

    assert f"{result.barrier_kcal:.4f} kcal/mol" in report
    assert result.table() in report
    assert section(report, "WARNINGS") == ["none", ""]
    assert report.rstrip().splitlines()[-1] == "pyvtst finished normally"


def test_report_settings_with_defaults(write, registry):
    text = H3_INPUT.replace('[optimizer]\ntype = "exact"\n', "").replace(
        'tunneling = "eckart"', 'tunneling = "eckart"\n\n[mep]\nds = 0.05'
    )
    spec = load(write(text), registry)
    report = render(RunLog(spec, registry, started=datetime.now()))
    assert "optimizer: geometric (default)" in report
    assert "  ds = 0.05" in report
    assert "store: in memory" in report


def test_warnings_are_recorded_in_report_and_stderr(write, registry, tmp_path, capsys):
    text = H3_INPUT.replace('name = "H3"\nmult = 2', 'name = "H3"\nmult = 4')
    assert main([str(write(text))], registry=registry) == 0
    assert "pyvtst: warning: total spin is not conserved" in capsys.readouterr().err
    warnings = section((tmp_path / "h3.out").read_text(), "WARNINGS")
    assert any("total spin is not conserved" in w for w in warnings)


def test_report_matches_returned_results(write, tmp_path):
    path = write(H3_INPUT)
    run(path, output=tmp_path / "api.out", registry=leps_registry(CountingLEPS()))
    assert main([str(path)], registry=leps_registry(CountingLEPS())) == 0
    api = (tmp_path / "api.out").read_text()
    cli = (tmp_path / "h3.out").read_text()
    assert masked(api) == masked(cli)


def test_main_module_in_process(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["pyvtst", str(tmp_path / "nothere.toml")])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("pyvtst", run_name="__main__")
    assert info.value.code == 2
    assert "nothere.toml" in capsys.readouterr().err
    # Imported under its own name, the module runs nothing.
    runpy.run_module("pyvtst.__main__")
