"""OrcaEngine against a fake orca program on the LEPS surface."""

import subprocess
import sys
import tempfile

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.engines import CachedEngine, OrcaEngine
from pyvtst.ports import Engine, EngineError
from pyvtst.stores import MemoryStore
from tests.fakes.gaussian import tilted_h3
from tests.fakes.leps import LEPSEngine
from tests.test_reaction import mep_reaction


@pytest.fixture
def engine(fake_orca):
    return OrcaEngine("HF", "3-21G")


# --- construction and level --------------------------------------------------


def test_level_reflects_result_affecting_settings(fake_orca):
    engine = OrcaEngine(
        "B3LYP",
        "def2-SVP",
        keywords="D3BJ  DefGrid3",
        blocks="%scf MaxIter 300 end",
    )
    level = engine.level
    options = dict(level.options)
    assert level.method == "b3lyp"
    assert level.basis == "def2-svp"
    assert options["program"] == "orca"
    assert options["keywords"] == "d3bj defgrid3"
    assert options["blocks"] == "%scf MaxIter 300 end"
    assert options["reference"] == "auto"
    assert options["hessian"] == "analytic"


def test_composite_method_without_a_basis(fake_orca):
    engine = OrcaEngine("r2SCAN-3c")
    assert (engine.level.method, engine.level.basis) == ("r2scan-3c", "-")
    engine.energy(tilted_h3())
    assert fake_orca.keyword_lines() == ["! r2scan-3c EnGrad Bohrs"]


def test_keyword_spelling_does_not_change_the_level(fake_orca):
    a = OrcaEngine("HF", "3-21G", keywords="TightSCF  RIJCOSX")
    b = OrcaEngine("HF", "3-21G", keywords="tightscf rijcosx")
    assert a.level == b.level


def test_block_layout_does_not_change_the_level(fake_orca):
    a = OrcaEngine("HF", "3-21G", blocks="%scf\n  MaxIter   300\nend\n")
    b = OrcaEngine("HF", "3-21G", blocks="%scf\nMaxIter 300\nend")
    assert a.level == b.level
    a.energy(tilted_h3())
    assert "%scf\nMaxIter 300\nend\n" in fake_orca.decks()[0]


def test_run_settings_are_excluded(fake_orca, tmp_path):
    plain = OrcaEngine("HF", "3-21G")
    tuned = OrcaEngine(
        "HF",
        "3-21G",
        nprocs=8,
        maxcore=4000,
        executable=fake_orca.orca,
        workdir=tmp_path,
        timeout=60,
        verbose=True,
    )
    assert plain.level == tuned.level


@pytest.mark.parametrize(
    "changes", [{"reference": "unrestricted"}, {"hessian": "numerical"}]
)
def test_reference_and_hessian_are_part_of_the_level(fake_orca, changes):
    assert OrcaEngine("HF", "3-21G").level != OrcaEngine("HF", "3-21G", **changes).level


def test_job_keywords_rejected(fake_orca):
    with pytest.raises(ValueError, match="'TightOpt'"):
        OrcaEngine("HF", "3-21G", keywords="TightOpt Freq")


def test_reserved_block_rejected(fake_orca):
    with pytest.raises(ValueError, match=r"%pal.*nprocs"):
        OrcaEngine("HF", "3-21G", blocks="%pal nprocs 4 end")


def test_comment_character_rejected(fake_orca):
    with pytest.raises(ValueError, match="#"):
        OrcaEngine("HF", "3-21G", keywords="TightSCF # tight")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": ""}, "method"),
        ({"basis": ""}, "basis"),
        ({"basis": "  "}, "basis"),
        ({"reference": "open"}, "reference: unknown reference 'open'"),
        ({"hessian": "fd"}, "hessian: unknown Hessian mode 'fd'"),
        ({"nprocs": 0}, "nprocs"),
        ({"nprocs": True}, "nprocs"),
        ({"maxcore": 0}, "maxcore"),
        ({"maxcore": "4GB"}, "maxcore"),
        ({"timeout": 0}, "timeout"),
    ],
)
def test_invalid_settings(fake_orca, kwargs, match):
    args = {"method": "HF", "basis": "3-21G"} | kwargs
    method, basis = args.pop("method"), args.pop("basis")
    with pytest.raises(ValueError, match=match):
        OrcaEngine(method, basis, **args)


def test_missing_orca_reported_at_construction(monkeypatch, tmp_path):
    monkeypatch.delenv("PYVTST_ORCA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PYVTST_ORCA"):
        OrcaEngine("HF", "3-21G")


def test_symbolic_link_resolved(fake_orca, tmp_path, monkeypatch):
    link = tmp_path / "bin" / "orca6"
    link.parent.mkdir()
    link.symlink_to(fake_orca.orca)
    monkeypatch.setenv("PYVTST_ORCA", str(link))
    engine = OrcaEngine("HF", "3-21G")
    assert engine._settings.orca == str(fake_orca.orca.resolve())
    engine.energy(tilted_h3())
    assert fake_orca.count == 1


def test_import_without_orca():
    script = "from pyvtst.engines import OrcaEngine; print('ok')"
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


def test_gradient_job_gives_energy_and_gradient(engine, fake_orca):
    result = engine.compute(tilted_h3(), hessian=False)
    assert fake_orca.keyword_lines() == ["! hf 3-21g EnGrad Bohrs"]
    assert result.hessian is None


def test_hessian_job_gives_all_three(engine, fake_orca):
    mol = tilted_h3()
    result = engine.compute(mol, hessian=True)
    assert fake_orca.keyword_lines() == ["! hf 3-21g EnGrad Freq Bohrs"]
    leps = LEPSEngine()
    assert result.energy == pytest.approx(leps.energy(mol), abs=1e-8)
    np.testing.assert_allclose(result.gradient, leps.gradient(mol), atol=1e-8)
    np.testing.assert_allclose(result.hessian, leps.hessian(mol), atol=1e-8)


def test_numerical_hessian(fake_orca):
    engine = OrcaEngine("MP2", "3-21G", hessian="numerical")
    mol = tilted_h3()
    result = engine.compute(mol, hessian=True)
    assert fake_orca.keyword_lines() == ["! mp2 3-21g EnGrad NumFreq Bohrs"]
    np.testing.assert_allclose(result.gradient, LEPSEngine().gradient(mol), atol=1e-8)


def test_returned_arrays_are_new(engine):
    mol = tilted_h3()
    first = engine.compute(mol, hessian=True)
    first.gradient[:] = 0.0
    assert first.hessian is not None
    first.hessian[:] = 0.0
    second = engine.compute(mol, hessian=True)
    assert np.any(second.gradient != 0.0)
    assert second.hessian is not None
    assert np.any(second.hessian != 0.0)


def test_each_call_runs_one_job(engine, fake_orca):
    mol = tilted_h3()
    engine.energy(mol)
    engine.gradient(mol)
    engine.hessian(mol)
    assert ["Freq" in line for line in fake_orca.keyword_lines()] == [
        False,
        False,
        True,
    ]


def test_one_run_per_geometry_through_a_cached_engine(engine, fake_orca):
    cached = CachedEngine(engine, MemoryStore())
    mol = tilted_h3()
    cached.hessian(mol)
    cached.energy(mol)
    cached.gradient(mol)
    assert fake_orca.count == 1


# --- reference ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "mult", "line"),
    [
        ("auto", 2, "! hf 3-21g EnGrad Bohrs"),
        ("unrestricted", 1, "! UHF hf 3-21g EnGrad Bohrs"),
        ("unrestricted", 2, "! UHF hf 3-21g EnGrad Bohrs"),
        ("restricted", 1, "! RHF hf 3-21g EnGrad Bohrs"),
    ],
)
def test_reference_keyword(fake_orca, reference, mult, line):
    engine = OrcaEngine("HF", "3-21G", reference=reference)
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    engine.energy(h2 if mult == 1 else tilted_h3())
    assert fake_orca.keyword_lines() == [line]


def test_restricted_open_shell_rejected_before_running(fake_orca):
    engine = OrcaEngine("HF", "3-21G", reference="restricted")
    with pytest.raises(ValueError, match="multiplicity 1"):
        engine.energy(tilted_h3())
    assert fake_orca.count == 0


# --- failures ----------------------------------------------------------------


def test_scf_failure_with_exit_status_zero(engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "scf_error")
    with pytest.raises(EngineError) as info:
        engine.energy(tilted_h3())
    message = str(info.value)
    assert message.startswith("hf/3-21g?")
    assert "EnGrad for H3: ORCA reported an error termination" in message
    assert "SCF NOT CONVERGED" in message
    assert "UCX" not in message


def test_input_error(engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "input_error")
    with pytest.raises(EngineError, match=r"(?s)status 4.*UNRECOGNIZED"):
        engine.energy(tilted_h3())


def test_method_without_analytic_hessian(engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "no_analytic_hessian")
    with pytest.raises(EngineError, match='hessian="numerical"'):
        engine.hessian(tilted_h3())


def test_timeout(fake_orca, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "hang")
    engine = OrcaEngine("HF", "3-21G", timeout=1)
    with pytest.raises(EngineError, match="did not finish within 1 s"):
        engine.hessian(tilted_h3())


def test_rotated_output(engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "rotated")
    with pytest.raises(EngineError, match=r"job\.engrad: .*reoriented"):
        engine.gradient(tilted_h3())


def test_not_orca(engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "not_orca")
    with pytest.raises(EngineError, match="PYVTST_ORCA"):
        engine.energy(tilted_h3())


def test_missing_result_file(fake_orca, tmp_path):
    silent = tmp_path / "orca"
    silent.write_text(
        f"#!{sys.executable}\n"
        "print('Program Version 6.0.1')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n"
    )
    silent.chmod(0o755)
    engine = OrcaEngine("HF", "3-21G", executable=silent)
    with pytest.raises(EngineError, match=r"wrote no job\.engrad, job\.hess"):
        engine.hessian(tilted_h3())


def test_unreadable_result_file(fake_orca, tmp_path):
    broken = tmp_path / "orca"
    broken.write_text(
        f"#!{sys.executable}\n"
        "open('job.engrad', 'w').write('# nothing\\n')\n"
        "print('Program Version 6.0.1')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n"
    )
    broken.chmod(0o755)
    engine = OrcaEngine("HF", "3-21G", executable=broken)
    with pytest.raises(EngineError, match=r"job\.engrad: too short"):
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


@pytest.mark.parametrize("mode", ["", "scf_error"])
def test_clean_working_directory(engine, private_tmp, monkeypatch, mode):
    monkeypatch.setenv("FAKE_ORCA_MODE", mode)
    if mode:
        with pytest.raises(EngineError):
            engine.hessian(tilted_h3())
    else:
        engine.hessian(tilted_h3())
    cwd, temp = private_tmp
    assert list(cwd.iterdir()) == []
    assert list(temp.iterdir()) == []


def test_kept_files(fake_orca, tmp_path):
    workdir = tmp_path / "runs"
    engine = OrcaEngine("HF", "3-21G", workdir=workdir)
    mol = tilted_h3()
    engine.energy(mol)
    engine.hessian(mol.replace(coords=mol.coords * 1.01))
    jobs = sorted(workdir.iterdir())
    assert [job.name for job in jobs] == ["0001-engrad-H3", "0002-freq-H3"]
    assert {p.name for p in jobs[0].iterdir()} >= {"job.inp", "job.out", "job.engrad"}
    assert {p.name for p in jobs[1].iterdir()} >= {
        "job.inp",
        "job.out",
        "job.engrad",
        "job.hess",
    }


def test_quiet_by_default(engine, capfd):
    engine.hessian(tilted_h3())
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_verbose_says_where_the_log_is(fake_orca, capfd):
    OrcaEngine("HF", "3-21G", verbose=True).energy(tilted_h3())
    _, err = capfd.readouterr()
    assert "running ORCA (engrad-H3)" in err
    assert "job.out" in err


# --- reaction path -----------------------------------------------------------


def test_one_run_per_mep_point(engine, fake_orca):
    rxn = mep_reaction(engine)
    rxn.optimize()
    fake_orca.reset()
    path = rxn.compute_mep(s_min=-0.06, s_max=0.06, ds=0.02, method="lqa")
    new_points = len(path.molecules) - 1  # every point but the saddle point
    lines = fake_orca.keyword_lines()
    assert sum("EnGrad Freq" in line for line in lines) == new_points
    assert len(lines) == new_points
