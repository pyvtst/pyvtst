"""The fake Gaussian programs reproduce the LEPS engine through the real parser."""

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.programs.gaussian import (
    FORCE_JOB,
    FREQ_JOB,
    RunSettings,
    build_deck,
    fchk_coordinates,
    fchk_results,
    opt_job,
    run_job,
)
from tests.fakes.gaussian import tilted_h3
from tests.fakes.leps import LEPSEngine, collinear_h3, saddle_distance


@pytest.fixture
def settings(fake_gaussian):
    return RunSettings(str(fake_gaussian.g16), str(fake_gaussian.formchk))


def run(settings, mol, job, **kwargs):
    deck = build_deck(
        mol,
        method="hf",
        basis="3-21g",
        reference="auto",
        keywords="",
        job=job,
        title="test",
        nprocs=1,
        memory="2GB",
    )
    return run_job(settings, deck, "test", **kwargs)


def test_freq_matches_leps(settings):
    mol = tilted_h3()
    result = run(settings, mol, FREQ_JOB)
    assert result.fchk is not None
    energy, gradient, hessian = fchk_results(result.fchk, mol, hessian=True)
    leps = LEPSEngine()
    assert energy == pytest.approx(leps.energy(mol), abs=1e-8)
    np.testing.assert_allclose(gradient, leps.gradient(mol), atol=1e-8)
    assert hessian is not None
    np.testing.assert_allclose(hessian, leps.hessian(mol), atol=1e-8)


def test_force_has_no_hessian(settings):
    mol = tilted_h3()
    result = run(settings, mol, FORCE_JOB)
    assert result.fchk is not None
    assert "Cartesian Force Constants" not in result.fchk.arrays


def test_calls_are_recorded(settings, fake_gaussian):
    run(settings, tilted_h3(), FORCE_JOB)
    run(settings, tilted_h3(), FREQ_JOB)
    assert fake_gaussian.count == 2
    assert [" Force " in r for r in fake_gaussian.routes()] == [True, False]
    fake_gaussian.reset()
    assert fake_gaussian.decks() == []


def test_opt_minimum(settings):
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    result = run(
        settings, h2, opt_job(saddle=False, max_cycles=50, convergence="tight")
    )
    assert result.failure is None
    assert "Stationary point found" in result.log
    assert "Step number   1 out of a maximum of  50" in result.log
    assert result.fchk is not None
    coords = fchk_coordinates(result.fchk, h2)
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(1.4014, abs=1e-3)


def test_opt_saddle(settings):
    guess = Molecule(
        ["H"] * 3, [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]], mult=2
    )
    job = opt_job(saddle=True, max_cycles=50, convergence="tight")
    result = run(settings, guess, job)
    assert result.failure is None
    assert result.fchk is not None
    coords = fchk_coordinates(result.fchk, guess)
    r = saddle_distance()
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(r, abs=1e-6)
    assert np.linalg.norm(coords[2] - coords[1]) == pytest.approx(r, abs=1e-6)


def test_steps_exceeded_mode(settings, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "steps_exceeded")
    job = opt_job(saddle=False, max_cycles=3, convergence="tight")
    result = run(settings, collinear_h3(1.5, 4.0), job, formchk_on_failure=True)
    assert result.failure == "Gaussian reported an error termination"
    assert "Number of steps exceeded" in result.log
    assert "Step number   3 out of a maximum of   3" in result.log
    assert result.fchk is not None


def test_reoriented_mode(settings, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "reoriented")
    mol = tilted_h3()
    result = run(settings, mol, FORCE_JOB)
    assert result.fchk is not None
    np.testing.assert_allclose(
        fchk_coordinates(result.fchk, mol), mol.coords[:, [1, 0, 2]], atol=1e-7
    )
