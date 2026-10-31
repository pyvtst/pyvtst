import logging
import logging.config
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg

pytest.importorskip("geometric")

from pyvtst.optimizers import GeometricOptimizer
from pyvtst.ports import (
    ConvergenceError,
    EngineError,
    LevelOfTheory,
    Optimizer,
)
from tests.fakes.leps import R0

GAU_TIGHT_MAX_GRADIENT = 1.5e-5


def bond(mol, i, j):
    return float(np.linalg.norm(mol.coords[i] - mol.coords[j]))


def angle_deg(mol):
    a = mol.coords[0] - mol.coords[1]
    b = mol.coords[2] - mol.coords[1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class FailingEngine:
    """Delegates to LEPS, then raises a fixed EngineError on the third energy."""

    def __init__(self, inner):
        self.inner = inner
        self.error = EngineError("boom")
        self.calls = 0

    @property
    def level(self):
        return LevelOfTheory("failing", "none")

    def energy(self, mol):
        self.calls += 1
        if self.calls == 3:
            raise self.error
        return self.inner.energy(mol)

    def gradient(self, mol):
        return self.inner.gradient(mol)

    def hessian(self, mol):
        return self.inner.hessian(mol)


def test_satisfies_protocol():
    assert isinstance(GeometricOptimizer(), Optimizer)


def test_minimum_with_analytic_engine(leps_engine, h2_stretched):
    result = GeometricOptimizer().minimize(h2_stretched, leps_engine)
    assert bond(result.molecule, 0, 1) == pytest.approx(R0, abs=1e-3)
    assert np.abs(leps_engine.gradient(result.molecule)).max() <= GAU_TIGHT_MAX_GRADIENT
    assert result.n_iterations >= 1


def test_energy_matches_returned_geometry(leps_engine, h2_stretched):
    result = GeometricOptimizer().minimize(h2_stretched, leps_engine)
    assert result.energy == pytest.approx(leps_engine.energy(result.molecule), abs=1e-8)


def test_input_left_unchanged(leps_engine, h2_stretched):
    before = h2_stretched.replace()
    GeometricOptimizer().minimize(h2_stretched, leps_engine)
    assert h2_stretched == before


def test_result_preserves_identity_fields(leps_engine, h3_ts_guess):
    guess = h3_ts_guess.replace(
        masses=[2.014, 1.008, 2.014], symmetry_number=3, name="DHD"
    )
    mol = GeometricOptimizer().saddle(guess, leps_engine).molecule
    assert mol.symbols == guess.symbols
    assert (mol.charge, mol.mult) == (0, 2)
    np.testing.assert_array_equal(mol.masses, guess.masses)
    assert mol.symmetry_number == 3
    assert mol.name == "DHD"


def test_iteration_limit(leps_engine, h2_stretched):
    far = h2_stretched.replace(coords=[[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    with pytest.raises(ConvergenceError) as excinfo:
        GeometricOptimizer(max_iterations=2).minimize(far, leps_engine)
    last = excinfo.value.last_molecule
    assert last is not None
    assert last.symbols == far.symbols


def test_saddle_iteration_limit(leps_engine, h3_ts_guess):
    with pytest.raises(ConvergenceError) as excinfo:
        GeometricOptimizer(max_iterations=1).saddle(h3_ts_guess, leps_engine)
    assert excinfo.value.last_molecule is not None


def test_engine_error_propagates(leps_engine, h2_stretched):
    engine = FailingEngine(leps_engine)
    with pytest.raises(EngineError) as excinfo:
        GeometricOptimizer().minimize(h2_stretched, engine)
    assert excinfo.value is engine.error


def test_clean_working_directory(leps_engine, h2_stretched, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    GeometricOptimizer().minimize(h2_stretched, leps_engine)
    assert os.listdir(tmp_path) == []


def test_workdir_keeps_files(leps_engine, h2_stretched, tmp_path):
    workdir = tmp_path / "run"
    GeometricOptimizer(workdir=workdir).minimize(h2_stretched, leps_engine)
    assert any(workdir.iterdir())


def test_different_convergence_sets_give_different_keys():
    assert (
        GeometricOptimizer(convergence="GAU_TIGHT").key()
        != GeometricOptimizer(convergence="GAU").key()
    )
    assert (
        GeometricOptimizer(coordsys="cart").key()
        != GeometricOptimizer(coordsys="tric").key()
    )


def test_non_result_settings_do_not_change_the_key(tmp_path):
    assert (
        GeometricOptimizer(max_iterations=50, verbose=True, workdir=tmp_path).key()
        == GeometricOptimizer().key()
    )


def test_key_format():
    assert (
        GeometricOptimizer().key()
        == "geometric?convergence=s:GAU_TIGHT&coordsys=s:tric"
    )


def test_key_is_stable_across_processes():
    script = (
        "from pyvtst.optimizers import GeometricOptimizer\n"
        "print(GeometricOptimizer().key())"
    )
    keys = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert keys == {GeometricOptimizer().key()}


def test_convergence_and_coordsys_are_case_insensitive():
    assert (
        GeometricOptimizer(convergence="gau_tight", coordsys="TRIC").key()
        == GeometricOptimizer().key()
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"convergence": "nope"}, "convergence: unknown convergence set 'nope'"),
        ({"coordsys": "polar"}, "coordsys: unknown coordinate system 'polar'"),
    ],
)
def test_unknown_settings_rejected_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeometricOptimizer(**kwargs)


@pytest.fixture
def user_logging(tmp_path):
    """File handlers on the root logger and on a named logger; undo afterwards."""
    root = logging.getLogger()
    app = logging.getLogger("pyvtst_test_app")
    saved = (list(root.handlers), root.level)
    root_handler = logging.FileHandler(tmp_path / "root.log", mode="w")
    app_handler = logging.FileHandler(tmp_path / "app.log", mode="w")
    root.addHandler(root_handler)
    root.setLevel(logging.DEBUG)
    app.addHandler(app_handler)
    yield root, app, tmp_path
    for handler in (root_handler, app_handler):
        handler.close()
    app.removeHandler(app_handler)
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def assert_user_logging_intact(root, app, logs):
    before = (list(root.handlers), root.level)
    app.warning("after")
    assert (list(root.handlers), root.level) == before
    assert (logs / "app.log").read_text() == "before\nafter\n"
    assert "geomeTRIC" not in (logs / "root.log").read_text()


def test_user_log_handlers_keep_writing(user_logging, leps_engine, h2_stretched):
    root, app, logs = user_logging
    app.warning("before")
    GeometricOptimizer().minimize(h2_stretched, leps_engine)
    assert_user_logging_intact(root, app, logs)


def test_user_log_handlers_keep_writing_when_verbose(
    capfd, user_logging, leps_engine, h2_stretched
):
    root, app, logs = user_logging
    app.warning("before")
    GeometricOptimizer(verbose=True).minimize(h2_stretched, leps_engine)
    assert_user_logging_intact(root, app, logs)


def test_logging_restored_after_failure(user_logging, leps_engine, h2_stretched):
    root, app, logs = user_logging
    app.warning("before")
    far = h2_stretched.replace(coords=[[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    with pytest.raises(ConvergenceError):
        GeometricOptimizer(max_iterations=2).minimize(far, leps_engine)
    assert_user_logging_intact(root, app, logs)


def test_geometric_logger_and_file_config_restored(capfd, leps_engine, h2_stretched):
    logger = logging.getLogger("geometric")
    before = (list(logger.handlers), logger.propagate, logging.config.fileConfig)
    GeometricOptimizer(verbose=True).minimize(h2_stretched, leps_engine)
    assert (list(logger.handlers), logger.propagate, logging.config.fileConfig) == (
        before
    )


def test_silent_by_default(capfd, leps_engine, h2_stretched, h3_ts_guess):
    GeometricOptimizer().minimize(h2_stretched, leps_engine)
    GeometricOptimizer().saddle(h3_ts_guess, leps_engine)
    out, err = capfd.readouterr()
    assert (out, err) == ("", "")


def test_verbose_prints_progress_and_writes_log(
    capfd, leps_engine, h2_stretched, tmp_path
):
    GeometricOptimizer(verbose=True, workdir=tmp_path).minimize(
        h2_stretched, leps_engine
    )
    out, err = capfd.readouterr()
    assert "geomeTRIC" in out + err
    assert "geomeTRIC" in (tmp_path / "pyvtst.log").read_text()


def test_h3_saddle_on_leps(leps_engine, h3_ts_guess):
    """Acceptance test: the collinear symmetric H + H2 saddle, from 10 guesses."""
    rng = np.random.default_rng(42)
    optimizer = GeometricOptimizer()
    for _ in range(10):
        guess = h3_ts_guess.replace(
            coords=h3_ts_guess.coords + rng.uniform(-0.05, 0.05, size=(3, 3))
        )
        ts = optimizer.saddle(guess, leps_engine).molecule

        assert angle_deg(ts) == pytest.approx(180.0, abs=1.0)
        assert bond(ts, 0, 1) == pytest.approx(bond(ts, 1, 2), abs=1e-3)
        assert np.abs(leps_engine.gradient(ts)).max() <= GAU_TIGHT_MAX_GRADIENT

        inv_sqrt_m = 1.0 / np.sqrt(np.repeat(ts.masses, 3))
        mass_weighted = leps_engine.hessian(ts) * np.outer(inv_sqrt_m, inv_sqrt_m)
        eigenvalues = scipy.linalg.eigh(mass_weighted, eigvals_only=True)
        assert np.count_nonzero(eigenvalues < -1e-6) == 1
