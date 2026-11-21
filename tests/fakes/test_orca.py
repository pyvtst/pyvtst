"""The fake ORCA program reproduces the LEPS engine through the real readers."""

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.programs import orca
from pyvtst.programs.orca import RunSettings, run_job
from tests.fakes.gaussian import tilted_h3
from tests.fakes.leps import LEPSEngine, collinear_h3, saddle_distance


@pytest.fixture
def settings(fake_orca):
    return RunSettings(str(fake_orca.orca))


def engine_run(settings, mol, *, hessian, numerical=False):
    deck = orca.engine_deck(
        mol,
        method="hf",
        basis="3-21g",
        reference="auto",
        keywords="",
        blocks="",
        hessian=hessian,
        numerical=numerical,
        nprocs=2,
        maxcore=500,
    )
    return run_job(settings, deck, "test")


def opt_run(settings, mol, *, saddle, max_cycles=50):
    deck = orca.opt_deck(
        mol,
        method="hf",
        basis="3-21g",
        reference="auto",
        keywords="",
        blocks="",
        saddle=saddle,
        convergence="tight",
        max_cycles=max_cycles,
        numerical=False,
        nprocs=1,
        maxcore=500,
    )
    return run_job(settings, deck, "test")


@pytest.mark.parametrize("numerical", [False, True])
def test_hessian_run_matches_leps(settings, numerical):
    mol = tilted_h3()
    result = engine_run(settings, mol, hessian=True, numerical=numerical)
    assert result.failure is None
    energy, gradient = orca.engrad_results(
        orca.parse_engrad(result.files["job.engrad"]), mol
    )
    hessian = orca.hess_result(orca.parse_hess(result.files["job.hess"]), mol)
    leps = LEPSEngine()
    assert energy == pytest.approx(leps.energy(mol), abs=1e-8)
    np.testing.assert_allclose(gradient, leps.gradient(mol), atol=1e-8)
    np.testing.assert_allclose(hessian, leps.hessian(mol), atol=1e-8)


def test_hessian_geometry_is_moved_to_the_centre_of_mass(settings):
    mol = tilted_h3()
    result = engine_run(settings, mol, hessian=True)
    atoms = orca.parse_hess(result.files["job.hess"]).coords
    np.testing.assert_allclose(atoms.mean(axis=0), 0.0, atol=1e-10)
    assert np.max(np.abs(atoms - mol.coords)) > 0.1


def test_gradient_run_writes_no_hessian(settings):
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert result.failure is None
    assert set(result.files) == {"job.engrad"}


def test_stderr_noise_stays_out_of_the_log(settings):
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert "UCX" in result.err
    assert "UCX" not in result.log


def test_calls_are_recorded(settings, fake_orca):
    engine_run(settings, tilted_h3(), hessian=False)
    engine_run(settings, tilted_h3(), hessian=True)
    assert fake_orca.count == 2
    assert fake_orca.keyword_lines() == [
        "! hf 3-21g EnGrad Bohrs",
        "! hf 3-21g EnGrad Freq Bohrs",
    ]
    fake_orca.reset()
    assert fake_orca.count == 0
    assert fake_orca.decks() == []


def test_minimum_is_found(settings):
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    result = opt_run(settings, mol, saddle=False)
    assert result.failure is None
    assert "THE OPTIMIZATION HAS CONVERGED" in result.log
    coords = orca.xyz_coordinates(result.files["job.xyz"], mol)
    engrad = orca.parse_engrad(result.files["job.engrad"])
    np.testing.assert_allclose(engrad.coords, coords, atol=1e-6)
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(1.4014, abs=1e-3)
    assert orca.cycles(result.log)


def test_saddle_is_found(settings):
    mol = collinear_h3(1.6, 1.9)
    result = opt_run(settings, mol, saddle=True)
    assert result.failure is None
    coords = orca.xyz_coordinates(result.files["job.xyz"], mol)
    r = np.linalg.norm(coords[1] - coords[0]), np.linalg.norm(coords[2] - coords[1])
    assert r == pytest.approx((saddle_distance(),) * 2, abs=1e-4)


# --- failure modes -----------------------------------------------------------


def test_scf_error_exits_zero_with_an_error_termination(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "scf_error")
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert "error termination" in result.log
    assert result.failure is not None
    assert "error termination" in result.failure


def test_input_error_exits_four(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "input_error")
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert result.failure is not None
    assert "status 4" in result.failure


def test_no_analytic_hessian(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "no_analytic_hessian")
    result = engine_run(settings, tilted_h3(), hessian=True)
    assert result.failure is not None
    assert 'hessian="numerical"' in result.failure
    numerical = engine_run(settings, tilted_h3(), hessian=True, numerical=True)
    assert numerical.failure is None


def test_maxiter_leaves_a_further_step_in_engrad(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "maxiter")
    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    result = opt_run(settings, mol, saddle=False, max_cycles=3)
    assert result.failure is None
    assert "did not converge but reached the maximum number of" in result.log
    assert orca.cycles(result.log) == [1, 2, 3]
    xyz = orca.xyz_coordinates(result.files["job.xyz"], mol)
    engrad = orca.parse_engrad(result.files["job.engrad"])
    assert np.max(np.abs(engrad.coords - xyz)) > 1e-3


def test_rotated_results(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "rotated")
    mol = tilted_h3()
    result = engine_run(settings, mol, hessian=True)
    with pytest.raises(ValueError, match="reoriented"):
        orca.engrad_results(orca.parse_engrad(result.files["job.engrad"]), mol)
    with pytest.raises(ValueError, match="translation"):
        orca.hess_result(orca.parse_hess(result.files["job.hess"]), mol)


def test_hang_is_stopped_by_the_timeout(fake_orca, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "hang")
    settings = RunSettings(str(fake_orca.orca), timeout=1.0)
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert result.failure is not None
    assert "did not finish" in result.failure


def test_not_orca(settings, monkeypatch):
    monkeypatch.setenv("FAKE_ORCA_MODE", "not_orca")
    result = engine_run(settings, tilted_h3(), hessian=False)
    assert result.failure is not None
    assert "PYVTST_ORCA" in result.failure


def test_symbolic_link_to_the_fake_is_resolved(fake_orca, tmp_path, monkeypatch):
    link = tmp_path / "bin" / "orca6"
    link.parent.mkdir()
    link.symlink_to(fake_orca.orca)
    monkeypatch.setenv("PYVTST_ORCA", str(link))
    assert orca.find_orca(None) == str(fake_orca.orca.resolve())
