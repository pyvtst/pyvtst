"""GaussianOptimizer against fake g16 and formchk programs on the LEPS surface."""

import subprocess
import sys
import tempfile

import numpy as np
import pytest

from pyvtst import Molecule, Reaction
from pyvtst.engines import CachedEngine, GaussianEngine
from pyvtst.optimizers import GaussianOptimizer
from pyvtst.ports import ConvergenceError, EngineError, Optimizer
from pyvtst.stores import MemoryStore
from tests.fakes.leps import LEPSEngine, collinear_h3, saddle_distance


@pytest.fixture
def engine(fake_gaussian):
    return GaussianEngine("HF", "3-21G")


@pytest.fixture
def optimizer(fake_gaussian):
    return GaussianOptimizer()


def h2(r=1.8):
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], name="H2")


# --- construction and key ----------------------------------------------------


def test_satisfies_the_optimizer_port(optimizer):
    assert isinstance(optimizer, Optimizer)


def test_convergence_changes_the_key(fake_gaussian):
    assert GaussianOptimizer().key() != GaussianOptimizer(convergence="verytight").key()


def test_case_does_not_change_the_key(fake_gaussian):
    assert GaussianOptimizer(convergence="TIGHT").key() == GaussianOptimizer().key()


def test_run_settings_do_not_change_the_key(fake_gaussian, tmp_path):
    tuned = GaussianOptimizer(
        max_cycles=20,
        nprocs=8,
        memory="8GB",
        executable=fake_gaussian.g16,
        formchk=fake_gaussian.formchk,
        workdir=tmp_path,
        timeout=10,
        verbose=True,
    )
    assert tuned.key() == GaussianOptimizer().key()


def test_key_is_stable_across_processes(fake_gaussian):
    script = (
        "from pyvtst.optimizers import GaussianOptimizer\n"
        "print(GaussianOptimizer(convergence='VeryTight').key())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert outputs == {"gaussian?convergence=s:verytight"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"convergence": "loose"}, "convergence: unknown convergence 'loose'"),
        ({"convergence": 1}, "convergence"),
        ({"max_cycles": 0}, "max_cycles: must be at least 1"),
        ({"max_cycles": 2.5}, "max_cycles: expected an integer"),
        ({"nprocs": 0}, "nprocs"),
        ({"timeout": -1}, "timeout"),
    ],
)
def test_invalid_settings(fake_gaussian, kwargs, match):
    with pytest.raises(ValueError, match=match):
        GaussianOptimizer(**kwargs)


def test_missing_gaussian_reported_at_construction(monkeypatch, tmp_path):
    monkeypatch.delenv("PYVTST_G16", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="PYVTST_G16"):
        GaussianOptimizer()


# --- level check -------------------------------------------------------------


def test_other_engines_rejected(optimizer, fake_gaussian):
    with pytest.raises(TypeError, match="needs the Gaussian engine"):
        optimizer.minimize(h2(), LEPSEngine())
    with pytest.raises(TypeError, match="needs the Gaussian engine"):
        optimizer.saddle(
            collinear_h3(1.6, 1.9), CachedEngine(LEPSEngine(), MemoryStore())
        )
    assert fake_gaussian.count == 0


def test_cached_gaussian_engine_accepted(optimizer, engine, fake_gaussian):
    cached = CachedEngine(engine, MemoryStore())
    optimizer.minimize(h2(), cached)
    assert fake_gaussian.routes() == [
        "#P hf/3-21g Opt=(MaxCycles=100,Tight) NoSymm Units=AU"
    ]


def test_route_follows_the_level(fake_gaussian):
    engine = GaussianEngine(
        "B3LYP", "def2SVP", keywords="Int=UltraFine", reference="unrestricted"
    )
    GaussianOptimizer(convergence="verytight", max_cycles=30, nprocs=4).saddle(
        collinear_h3(1.6, 1.9), engine
    )
    (deck,) = fake_gaussian.decks()
    assert deck.startswith("%NProcShared=4\n%Mem=2GB\n")
    assert (
        "#P Ub3lyp/def2svp Opt=(TS,CalcFC,NoEigenTest,MaxCycles=30,VeryTight) "
        "NoSymm Units=AU int=ultrafine\n"
    ) in deck


def test_restricted_open_shell_rejected(fake_gaussian):
    engine = GaussianEngine("HF", "3-21G", reference="restricted")
    with pytest.raises(ValueError, match="multiplicity 1"):
        GaussianOptimizer().saddle(collinear_h3(1.6, 1.9), engine)
    assert fake_gaussian.count == 0


def test_reaction_optimizes_with_gaussian(engine, optimizer, fake_gaussian):
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2)
    rxn = Reaction([h2(), h], collinear_h3(1.6, 1.9), engine, optimizer=optimizer)
    points = rxn.optimize()
    r = saddle_distance()
    ts = points.ts.molecule.coords
    assert np.linalg.norm(ts[1] - ts[0]) == pytest.approx(r, abs=1e-6)
    # H2 and the saddle point; a single atom needs no optimization.
    assert sum(" Opt=(" in route for route in fake_gaussian.routes()) == 2


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
    result = optimizer.saddle(guess, engine)
    mol = result.molecule
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
        LEPSEngine().energy(result.molecule), abs=1e-12
    )
    assert result.n_iterations > 1


def test_minimum_found(optimizer, engine):
    result = optimizer.minimize(h2(), engine)
    coords = result.molecule.coords
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(1.40142, abs=1e-4)
    assert result.energy == pytest.approx(engine.energy(result.molecule), abs=1e-8)


# --- failures ----------------------------------------------------------------


def test_cycle_limit_reached(engine, fake_gaussian):
    with pytest.raises(ConvergenceError, match="did not converge in 1 steps") as info:
        GaussianOptimizer(max_cycles=1).minimize(h2(3.0), engine)
    last = info.value.last_molecule
    assert isinstance(last, Molecule)
    assert last.symbols == ("H", "H")
    assert last.name == "H2"


def test_steps_exceeded_mode(optimizer, engine, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "steps_exceeded")
    with pytest.raises(ConvergenceError, match="Number of steps exceeded") as info:
        GaussianOptimizer(max_cycles=3).saddle(collinear_h3(1.6, 1.9), engine)
    assert info.value.last_molecule is not None


def test_not_converged_without_a_readable_geometry(engine, fake_gaussian, tmp_path):
    broken = tmp_path / "formchk"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    optimizer = GaussianOptimizer(max_cycles=1, formchk=broken)
    with pytest.raises(ConvergenceError) as info:
        optimizer.minimize(h2(3.0), engine)
    assert info.value.last_molecule is None


def test_scf_failure(optimizer, engine, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "error")
    with pytest.raises(EngineError, match="Convergence failure"):
        optimizer.minimize(h2(), engine)


def test_timeout(engine, monkeypatch, fake_gaussian):
    monkeypatch.setenv("FAKE_G16_MODE", "hang")
    with pytest.raises(EngineError, match="did not finish within 1 s"):
        GaussianOptimizer(timeout=1).minimize(h2(), engine)


def fake_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_normal_termination_without_stationary_point(engine, tmp_path):
    g16 = fake_script(tmp_path, "g16", "echo ' Normal termination' > job.log\n")
    optimizer = GaussianOptimizer(executable=g16)
    with pytest.raises(EngineError, match="no stationary point"):
        optimizer.minimize(h2(), engine)


def test_missing_checkpoint(engine, tmp_path):
    g16 = fake_script(
        tmp_path,
        "g16",
        "printf ' Stationary point found.\\n Normal termination\\n' > job.log\n",
    )
    optimizer = GaussianOptimizer(executable=g16)
    with pytest.raises(EngineError, match=r"no job\.chk"):
        optimizer.minimize(h2(), engine)


def test_unreadable_result(engine, tmp_path):
    g16 = fake_script(
        tmp_path,
        "g16",
        "printf ' Stationary point found.\\n Normal termination\\n' > job.log\n"
        "touch job.chk\n",
    )
    header = f"{'Atomic numbers':<40}   I   N=           2"
    formchk = fake_script(
        tmp_path, "formchk", f"printf 't\\nt\\n{header}\\n 8 8\\n' > job.fchk\n"
    )
    optimizer = GaussianOptimizer(executable=g16, formchk=formchk)
    with pytest.raises(EngineError, match=r"job\.fchk: atomic numbers"):
        optimizer.minimize(h2(), engine)


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


def test_kept_files(engine, fake_gaussian, tmp_path, capfd):
    workdir = tmp_path / "opt"
    GaussianOptimizer(workdir=workdir).minimize(h2(), engine)
    (job,) = workdir.iterdir()
    assert job.name == "0001-opt-H2"
    assert {p.name for p in job.iterdir()} == {"job.gjf", "job.log", "job.fchk"}
    assert capfd.readouterr() == ("", "")
