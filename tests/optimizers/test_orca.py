"""OrcaOptimizer against a fake orca program on the LEPS surface."""

import subprocess
import sys
import tempfile

import numpy as np
import pytest

from pyvtst import Molecule, Reaction
from pyvtst.engines import CachedEngine, GaussianEngine, OrcaEngine
from pyvtst.optimizers import OrcaOptimizer
from pyvtst.ports import ConvergenceError, EngineError, Optimizer
from pyvtst.stores import MemoryStore
from tests.fakes.leps import LEPSEngine, collinear_h3, saddle_distance


@pytest.fixture
def engine(fake_orca):
    return OrcaEngine("HF", "3-21G")


@pytest.fixture
def optimizer(fake_orca):
    return OrcaOptimizer()


def h2(r=1.8):
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], name="H2")


# --- construction and key ----------------------------------------------------


def test_satisfies_the_optimizer_port(optimizer):
    assert isinstance(optimizer, Optimizer)


def test_convergence_changes_the_key(fake_orca):
    assert OrcaOptimizer().key() != OrcaOptimizer(convergence="verytight").key()


def test_case_does_not_change_the_key(fake_orca):
    assert OrcaOptimizer(convergence="TIGHT").key() == OrcaOptimizer().key()


def test_run_settings_do_not_change_the_key(fake_orca, tmp_path):
    tuned = OrcaOptimizer(
        max_cycles=20,
        nprocs=8,
        maxcore=4000,
        executable=fake_orca.orca,
        workdir=tmp_path,
        timeout=10,
        verbose=True,
    )
    assert tuned.key() == OrcaOptimizer().key()


def test_key_is_stable_across_processes(fake_orca):
    script = (
        "from pyvtst.optimizers import OrcaOptimizer\n"
        "print(OrcaOptimizer(convergence='VeryTight').key())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert outputs == {"orca?convergence=s:verytight"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"convergence": "loose"}, "convergence: unknown convergence 'loose'"),
        ({"convergence": 1}, "convergence"),
        ({"max_cycles": 0}, "max_cycles: must be at least 1"),
        ({"max_cycles": 2.5}, "max_cycles: expected an integer"),
        ({"nprocs": 0}, "nprocs"),
        ({"maxcore": 0}, "maxcore"),
        ({"timeout": -1}, "timeout"),
    ],
)
def test_invalid_settings(fake_orca, kwargs, match):
    with pytest.raises(ValueError, match=match):
        OrcaOptimizer(**kwargs)


def test_environment_variable_takes_precedence_over_path(
    fake_orca, tmp_path, monkeypatch
):
    other = tmp_path / "path" / "orca"
    other.parent.mkdir()
    other.write_text("#!/bin/sh\nexit 1\n")
    other.chmod(0o755)
    monkeypatch.setenv("PATH", str(other.parent))
    assert OrcaOptimizer()._settings.orca == str(fake_orca.orca.resolve())


def test_missing_orca_reported_at_construction(monkeypatch, tmp_path):
    monkeypatch.delenv("PYVTST_ORCA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PYVTST_ORCA"):
        OrcaOptimizer()


# --- level check -------------------------------------------------------------


def test_other_engines_rejected(optimizer, fake_orca, fake_gaussian):
    with pytest.raises(TypeError, match="needs the ORCA engine"):
        optimizer.minimize(h2(), LEPSEngine())
    with pytest.raises(TypeError, match="needs the ORCA engine"):
        optimizer.saddle(
            collinear_h3(1.6, 1.9), CachedEngine(LEPSEngine(), MemoryStore())
        )
    with pytest.raises(TypeError, match="needs the ORCA engine"):
        optimizer.minimize(h2(), GaussianEngine("HF", "3-21G"))
    assert fake_orca.count == 0


def test_cached_orca_engine_accepted(optimizer, engine, fake_orca):
    cached = CachedEngine(engine, MemoryStore())
    optimizer.minimize(h2(), cached)
    assert fake_orca.keyword_lines() == ["! hf 3-21g TightOpt Bohrs"]


def test_deck_follows_the_level(fake_orca):
    engine = OrcaEngine(
        "B3LYP",
        "def2-SVP",
        keywords="D3BJ",
        blocks="%scf\nMaxIter 300\nend",
        reference="unrestricted",
        hessian="numerical",
    )
    OrcaOptimizer(convergence="verytight", max_cycles=30, nprocs=4).saddle(
        collinear_h3(1.6, 1.9), engine
    )
    (deck,) = fake_orca.decks()
    assert "! UHF b3lyp def2-svp d3bj OptTS VeryTightOpt Bohrs\n" in deck
    assert "%pal nprocs 4 end\n%maxcore 2000\n" in deck
    assert (
        "%geom\n  MaxIter 30\n  Calc_Hess true\n  NumHess true\nend\n"
        "%scf\nMaxIter 300\nend\n"
    ) in deck


def test_composite_method_without_a_basis(fake_orca):
    OrcaOptimizer().minimize(h2(), OrcaEngine("r2SCAN-3c"))
    assert fake_orca.keyword_lines() == ["! r2scan-3c TightOpt Bohrs"]


def test_restricted_open_shell_rejected(fake_orca):
    engine = OrcaEngine("HF", "3-21G", reference="restricted")
    with pytest.raises(ValueError, match="multiplicity 1"):
        OrcaOptimizer().saddle(collinear_h3(1.6, 1.9), engine)
    assert fake_orca.count == 0


def test_reaction_optimizes_with_orca(engine, optimizer, fake_orca):
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    rxn = Reaction([h2(), h], collinear_h3(1.6, 1.9), engine, optimizer=optimizer)
    points = rxn.optimize()
    r = saddle_distance()
    ts = points.ts.molecule.coords
    assert np.linalg.norm(ts[1] - ts[0]) == pytest.approx(r, abs=1e-6)
    # H2 and the saddle point; a single atom needs no optimization.
    assert sum("Opt" in line for line in fake_orca.keyword_lines()) == 2


# --- results -----------------------------------------------------------------


def test_identity_fields_preserved(optimizer, engine):
    guess = Molecule(
        ["H", "H", "H"],
        collinear_h3(1.6, 1.9).coords,
        mult=2,
        masses=[1.0, 2.0, 3.0],
        symmetry_number=3,
        name="H3 guess",
    )
    mol = optimizer.saddle(guess, engine).molecule
    assert mol.mult == 2
    np.testing.assert_array_equal(mol.masses, [1.0, 2.0, 3.0])
    assert mol.symmetry_number == 3
    assert mol.name == "H3 guess"


def test_saddle_found(optimizer, engine):
    result = optimizer.saddle(collinear_h3(1.6, 1.9), engine)
    coords = result.molecule.coords
    r = saddle_distance()
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(r, abs=1e-6)
    assert np.linalg.norm(coords[2] - coords[1]) == pytest.approx(r, abs=1e-6)
    assert result.energy == pytest.approx(
        LEPSEngine().energy(result.molecule), abs=1e-9
    )
    assert result.n_iterations > 1


def test_minimum_found(optimizer, engine):
    result = optimizer.minimize(h2(), engine)
    coords = result.molecule.coords
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(1.40142, abs=1e-4)
    assert result.energy == pytest.approx(engine.energy(result.molecule), abs=1e-8)


# --- failures ----------------------------------------------------------------


def test_cycle_limit_reached(engine, fake_orca, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "maxiter")
    with pytest.raises(ConvergenceError, match="did not converge in 2 cycles") as info:
        OrcaOptimizer(max_cycles=2).minimize(h2(3.0), engine)
    last = info.value.last_molecule
    assert isinstance(last, Molecule)
    assert last.symbols == ("H", "H")
    assert last.name == "H2"


def test_cycle_limit_geometry_is_the_last_cycle(engine, fake_orca, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "maxiter")
    with pytest.raises(ConvergenceError) as info:
        OrcaOptimizer(max_cycles=1).minimize(h2(3.0), engine)
    # One cycle: the only geometry evaluated is the input.
    last = info.value.last_molecule
    assert last is not None
    np.testing.assert_allclose(last.coords, h2(3.0).coords, atol=1e-10)


def test_cycle_limit_without_a_readable_geometry(engine, tmp_path):
    orca = fake_script(
        tmp_path,
        "print('Program Version 6.0.1')\n"
        "print('The optimization did not converge but reached the maximum number of')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
    )
    with pytest.raises(ConvergenceError) as info:
        OrcaOptimizer(executable=orca).minimize(h2(), engine)
    assert info.value.last_molecule is None


def test_cycle_limit_with_unreadable_geometry(engine, tmp_path):
    orca = fake_script(
        tmp_path,
        "print('Program Version 6.0.1')\n"
        "open('job.xyz', 'w').write('2\\nt\\n O 0 0 0\\n O 0 0 1\\n')\n"
        "print('The optimization did not converge but reached the maximum number of')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
    )
    with pytest.raises(ConvergenceError) as info:
        OrcaOptimizer(executable=orca).minimize(h2(), engine)
    assert info.value.last_molecule is None


def test_scf_failure_with_exit_status_zero(optimizer, engine, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "scf_error")
    with pytest.raises(EngineError, match="SCF NOT CONVERGED"):
        optimizer.minimize(h2(), engine)


def test_timeout(engine, monkeypatch, fake_orca):
    monkeypatch.setenv("FAKE_ORCA_MODE", "hang")
    with pytest.raises(EngineError, match="did not finish within 1 s"):
        OrcaOptimizer(timeout=1).minimize(h2(), engine)


def fake_script(tmp_path, body):
    path = tmp_path / "orca"
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


NORMAL = "print('Program Version 6.0.1')\n"


def test_normal_termination_without_convergence(engine, tmp_path):
    orca = fake_script(tmp_path, NORMAL + "print('****ORCA TERMINATED NORMALLY****')\n")
    with pytest.raises(EngineError, match="no converged optimization"):
        OrcaOptimizer(executable=orca).minimize(h2(), engine)


def test_missing_result_files(engine, tmp_path):
    orca = fake_script(
        tmp_path,
        NORMAL + "print('THE OPTIMIZATION HAS CONVERGED')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
    )
    with pytest.raises(EngineError, match=r"wrote no job\.xyz, job\.engrad"):
        OrcaOptimizer(executable=orca).minimize(h2(), engine)


def test_unreadable_result(engine, tmp_path):
    orca = fake_script(
        tmp_path,
        NORMAL + "open('job.xyz', 'w').write('2\\nt\\n O 0 0 0\\n O 0 0 1\\n')\n"
        "open('job.engrad', 'w').write('1\\n')\n"
        "print('THE OPTIMIZATION HAS CONVERGED')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
    )
    with pytest.raises(EngineError, match=r"job\.xyz: atomic numbers"):
        OrcaOptimizer(executable=orca).minimize(h2(), engine)


def test_final_geometry_must_match_engrad(engine, tmp_path):
    engrad = "2\n-1.0\n" + "0.0\n" * 6 + "1 0.0 0.0 0.0\n1 1.5 0.0 0.0\n"
    orca = fake_script(
        tmp_path,
        NORMAL + "open('job.xyz', 'w').write('2\\nt\\n H 0 0 0\\n H 0.74 0 0\\n')\n"
        f"open('job.engrad', 'w').write({engrad!r})\n"
        "print('THE OPTIMIZATION HAS CONVERGED')\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
    )
    with pytest.raises(EngineError, match=r"job\.engrad: the geometry differs"):
        OrcaOptimizer(executable=orca).minimize(h2(), engine)


# --- working files -----------------------------------------------------------


def test_clean_working_directory(optimizer, engine, tmp_path, monkeypatch):
    cwd, temp = tmp_path / "cwd", tmp_path / "temp"
    cwd.mkdir()
    temp.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    optimizer.minimize(h2(), engine)
    assert list(cwd.iterdir()) == []
    assert list(temp.iterdir()) == []


def test_kept_files(engine, fake_orca, tmp_path, capfd):
    workdir = tmp_path / "opt"
    OrcaOptimizer(workdir=workdir).minimize(h2(), engine)
    (job,) = workdir.iterdir()
    assert job.name == "0001-opt-H2"
    assert {p.name for p in job.iterdir()} == {
        "job.inp",
        "job.out",
        "job.err",
        "job.engrad",
        "job.xyz",
    }
    assert capfd.readouterr() == ("", "")
