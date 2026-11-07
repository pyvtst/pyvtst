"""GaussianEngine against fake g16 and formchk programs on the LEPS surface."""

import subprocess
import sys
import tempfile

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.engines import CachedEngine, GaussianEngine
from pyvtst.ports import Engine, EngineError
from pyvtst.stores import MemoryStore
from tests.fakes.gaussian import tilted_h3
from tests.fakes.leps import LEPSEngine
from tests.test_reaction import mep_reaction


@pytest.fixture
def engine(fake_gaussian):
    return GaussianEngine("HF", "3-21G")


# --- construction and level --------------------------------------------------


def test_level_reflects_result_affecting_settings(fake_gaussian):
    engine = GaussianEngine(
        "B3LYP", "def2SVP", keywords="EmpiricalDispersion=GD3BJ Int=UltraFine"
    )
    level = engine.level
    options = dict(level.options)
    assert level.method == "b3lyp"
    assert level.basis == "def2svp"
    assert options["program"] == "gaussian"
    assert options["reference"] == "auto"
    assert options["keywords"] == "empiricaldispersion=gd3bj int=ultrafine"


def test_keyword_spelling_does_not_change_the_level(fake_gaussian):
    a = GaussianEngine("HF", "3-21G", keywords="SCF=Tight  Int=UltraFine")
    b = GaussianEngine("HF", "3-21G", keywords="scf=tight int=ultrafine")
    assert a.level == b.level


def test_run_settings_are_excluded(fake_gaussian, tmp_path):
    plain = GaussianEngine("HF", "3-21G")
    tuned = GaussianEngine(
        "HF",
        "3-21G",
        nprocs=8,
        memory="16GB",
        executable=fake_gaussian.g16,
        formchk=fake_gaussian.formchk,
        workdir=tmp_path,
        timeout=60,
        verbose=True,
    )
    assert plain.level == tuned.level


def test_reference_is_part_of_the_level(fake_gaussian):
    assert (
        GaussianEngine("HF", "3-21G").level
        != GaussianEngine("HF", "3-21G", reference="unrestricted").level
    )


def test_job_keywords_rejected(fake_gaussian):
    with pytest.raises(ValueError, match="'Opt'"):
        GaussianEngine("HF", "3-21G", keywords="Opt Freq")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": ""}, "method"),
        ({"basis": "  "}, "basis"),
        ({"reference": "open"}, "reference: unknown reference 'open'"),
        ({"nprocs": 0}, "nprocs"),
        ({"nprocs": True}, "nprocs"),
        ({"memory": ""}, "memory"),
        ({"memory": "4 GB"}, "memory"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": "60"}, "timeout"),
    ],
)
def test_invalid_settings(fake_gaussian, kwargs, match):
    args = {"method": "HF", "basis": "3-21G"} | kwargs
    method, basis = args.pop("method"), args.pop("basis")
    with pytest.raises(ValueError, match=match):
        GaussianEngine(method, basis, **args)


def test_missing_gaussian_reported_at_construction(monkeypatch, tmp_path):
    monkeypatch.delenv("PYVTST_G16", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PYVTST_G16"):
        GaussianEngine("HF", "3-21G")


def test_import_without_gaussian():
    script = "from pyvtst.engines import GaussianEngine; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": ""},
    )
    assert result.stdout.strip() == "ok"


def test_satisfies_the_engine_port(engine):
    assert isinstance(engine, Engine)


# --- results -----------------------------------------------------------------


def test_results_match_the_reference_surface(engine):
    mol = tilted_h3()
    leps = LEPSEngine()
    assert engine.energy(mol) == pytest.approx(leps.energy(mol), abs=1e-8)
    np.testing.assert_allclose(engine.gradient(mol), leps.gradient(mol), atol=1e-8)
    hessian = engine.hessian(mol)
    assert hessian.shape == (9, 9)
    np.testing.assert_allclose(hessian, leps.hessian(mol), atol=1e-8)
    np.testing.assert_array_equal(hessian, hessian.T)


def test_returned_arrays_are_copies(engine, fake_gaussian):
    mol = tilted_h3()
    engine.hessian(mol)
    engine.gradient(mol)[:] = 0.0
    engine.hessian(mol)[:] = 0.0
    assert np.any(engine.gradient(mol) != 0.0)
    assert np.any(engine.hessian(mol) != 0.0)
    assert fake_gaussian.count == 1


def test_energy_and_gradient_share_a_run(engine, fake_gaussian):
    mol = tilted_h3()
    engine.energy(mol)
    engine.gradient(mol)
    assert fake_gaussian.count == 1
    assert " Force " in fake_gaussian.routes()[0]


def test_hessian_serves_energy_and_gradient(engine, fake_gaussian):
    mol = tilted_h3()
    engine.hessian(mol)
    engine.energy(mol)
    engine.gradient(mol)
    assert fake_gaussian.count == 1
    assert " Freq=NoRaman " in fake_gaussian.routes()[0]


def test_hessian_after_gradient_runs_freq(engine, fake_gaussian):
    mol = tilted_h3()
    engine.gradient(mol)
    engine.hessian(mol)
    engine.energy(mol)
    assert fake_gaussian.count == 2


def test_results_do_not_depend_on_the_order_of_requests(fake_gaussian):
    mol = tilted_h3()
    first, second = GaussianEngine("HF", "3-21G"), GaussianEngine("HF", "3-21G")
    e1 = first.energy(mol)
    h1 = first.hessian(mol)
    h2 = second.hessian(mol)
    assert second.energy(mol) == e1
    np.testing.assert_array_equal(h1, h2)


def test_charge_and_multiplicity_are_part_of_the_memo(engine, fake_gaussian):
    mol = tilted_h3()
    engine.energy(mol)
    engine.energy(mol.replace(charge=1, mult=1))
    assert fake_gaussian.count == 2


def test_old_runs_are_dropped(engine, fake_gaussian):
    first = tilted_h3()
    engine.energy(first)
    for k in range(8):
        engine.energy(first.replace(coords=first.coords + 0.01 * (k + 1)))
    engine.energy(first)
    assert fake_gaussian.count == 10


def test_one_run_per_geometry_through_a_cached_engine(engine, fake_gaussian):
    cached = CachedEngine(engine, MemoryStore())
    mol = tilted_h3()
    cached.hessian(mol)
    cached.energy(mol)
    cached.gradient(mol)
    assert fake_gaussian.count == 1


# --- reference ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "mult", "method"),
    [
        ("auto", 2, "hf"),
        ("unrestricted", 1, "Uhf"),
        ("unrestricted", 2, "Uhf"),
        ("restricted", 1, "Rhf"),
    ],
)
def test_reference_prefix(fake_gaussian, reference, mult, method):
    engine = GaussianEngine("HF", "3-21G", reference=reference)
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    mol = h2 if mult == 1 else tilted_h3()
    engine.energy(mol)
    assert fake_gaussian.routes()[0].startswith(f"#P {method}/3-21g Force ")


def test_restricted_open_shell_rejected_before_running(fake_gaussian):
    engine = GaussianEngine("HF", "3-21G", reference="restricted")
    with pytest.raises(ValueError, match="multiplicity 1"):
        engine.energy(tilted_h3())
    assert fake_gaussian.count == 0


# --- failures ----------------------------------------------------------------


def test_scf_failure(engine, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "error")
    with pytest.raises(EngineError) as info:
        engine.energy(tilted_h3())
    message = str(info.value)
    assert message.startswith("hf/3-21g?")
    assert "Force for H3: Gaussian reported an error termination" in message
    assert "Convergence failure -- run terminated." in message


def test_timeout(fake_gaussian, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "hang")
    engine = GaussianEngine("HF", "3-21G", timeout=1)
    with pytest.raises(EngineError, match="did not finish within 1 s"):
        engine.hessian(tilted_h3())


def test_reoriented_output(engine, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "reoriented")
    with pytest.raises(EngineError, match=r"job\.fchk: .*reoriented"):
        engine.gradient(tilted_h3())


def test_formchk_failure(fake_gaussian, tmp_path):
    broken = tmp_path / "formchk"
    broken.write_text("#!/bin/sh\necho 'cannot open job.chk'\nexit 1\n")
    broken.chmod(0o755)
    engine = GaussianEngine("HF", "3-21G", formchk=broken)
    with pytest.raises(EngineError, match=r"formchk failed: cannot open job\.chk"):
        engine.energy(tilted_h3())


# --- working files -----------------------------------------------------------


@pytest.fixture
def private_tmp(tmp_path, monkeypatch):
    """An empty current directory and an empty temporary-file directory."""
    cwd = tmp_path / "cwd"
    temp = tmp_path / "temp"
    cwd.mkdir()
    temp.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return cwd, temp


@pytest.mark.parametrize("mode", ["", "error"])
def test_clean_working_directory(engine, private_tmp, monkeypatch, mode):
    monkeypatch.setenv("FAKE_G16_MODE", mode)
    if mode:
        with pytest.raises(EngineError):
            engine.hessian(tilted_h3())
    else:
        engine.hessian(tilted_h3())
    cwd, temp = private_tmp
    assert list(cwd.iterdir()) == []
    assert list(temp.iterdir()) == []


def test_kept_files(fake_gaussian, tmp_path):
    workdir = tmp_path / "runs"
    engine = GaussianEngine("HF", "3-21G", workdir=workdir)
    mol = tilted_h3()
    engine.energy(mol)
    engine.energy(mol.replace(coords=mol.coords * 1.01))
    jobs = sorted(workdir.iterdir())
    assert len(jobs) == 2
    for job in jobs:
        assert job.name.startswith("force-H3-")
        assert {p.name for p in job.iterdir()} >= {"job.gjf", "job.log", "job.fchk"}


def test_quiet_by_default(engine, capfd):
    engine.hessian(tilted_h3())
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


# --- reaction path -----------------------------------------------------------


def test_one_run_per_mep_point(engine, fake_gaussian):
    rxn = mep_reaction(engine)
    rxn.optimize()
    fake_gaussian.reset()
    path = rxn.compute_mep(s_min=-0.06, s_max=0.06, ds=0.02, method="lqa")
    new_points = len(path.molecules) - 1  # every point but the saddle point
    routes = fake_gaussian.routes()
    assert sum(" Freq=NoRaman " in r for r in routes) == new_points
    assert len(routes) == new_points
