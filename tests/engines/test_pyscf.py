import re

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, scf  # noqa: E402

from pyvtst import Molecule  # noqa: E402
from pyvtst.core.finite_difference import fd_hessian  # noqa: E402
from pyvtst.engines import PySCFEngine  # noqa: E402
from pyvtst.ports import EngineError  # noqa: E402


@pytest.fixture
def water():
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
        name="water",
    )


@pytest.fixture
def hydroxyl():
    return Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], mult=2)


def test_level_reflects_result_affecting_settings():
    level = PySCFEngine("B3LYP", "def2-SVP", grid_level=4).level
    assert level.method == "b3lyp"
    assert level.basis == "def2-svp"
    assert dict(level.options)["grid_level"] == 4
    assert set(dict(level.options)) == {
        "reference",
        "hessian",
        "conv_tol",
        "grid_level",
    }


def test_grid_level_only_for_dft():
    assert "grid_level" not in dict(PySCFEngine("HF", "sto-3g").level.options)


def test_level_records_resolved_hessian_mode():
    auto = PySCFEngine("HF", "sto-3g")
    assert auto.level == PySCFEngine("HF", "sto-3g", hessian="analytic").level
    assert dict(auto.level.options)["hessian"] == "analytic"
    mp2 = PySCFEngine("MP2", "sto-3g")
    assert mp2.level == PySCFEngine("MP2", "sto-3g", hessian="fd").level
    assert dict(mp2.level.options)["hessian"] == "fd"


def test_fd_step_only_in_level_for_fd_hessians():
    assert (
        PySCFEngine("HF", "sto-3g", fd_step=1e-3).level
        == PySCFEngine("HF", "sto-3g").level
    )
    assert (
        PySCFEngine("HF", "sto-3g", hessian="fd", fd_step=1e-3).level
        != PySCFEngine("HF", "sto-3g", hessian="fd").level
    )


def test_performance_settings_excluded():
    a = PySCFEngine("HF", "sto-3g", max_memory_mb=1000, max_cycle=200, verbose=True)
    b = PySCFEngine("HF", "sto-3g", max_memory_mb=8000)
    assert a.level == b.level


def test_unknown_method():
    with pytest.raises(ValueError, match="NOT-A-FUNCTIONAL"):
        PySCFEngine("NOT-A-FUNCTIONAL", "sto-3g")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reference": "rohf"}, "reference"),
        ({"hessian": "magic"}, "hessian"),
        ({"fd_step": 0.0}, "fd_step"),
        ({"max_cycle": 0}, "max_cycle"),
        ({"max_memory_mb": 0}, "max_memory_mb"),
    ],
)
def test_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PySCFEngine("HF", "sto-3g", **kwargs)


def test_memory_defaults_to_pyscf_setting(water, monkeypatch):
    # Mole.max_memory holds PySCF's default, read from PYSCF_MAX_MEMORY.
    monkeypatch.setattr(pyscf.gto.Mole, "max_memory", 1234)
    assert PySCFEngine("HF", "sto-3g")._build(water).max_memory == 1234


def test_explicit_memory_overrides_pyscf_setting(water):
    engine = PySCFEngine("HF", "sto-3g", max_memory_mb=1500)
    assert engine._build(water).max_memory == 1500


def test_closed_shell_uses_restricted(water):
    mf, _ = PySCFEngine("HF", "sto-3g")._run(water)
    assert isinstance(mf, scf.hf.RHF)


def test_open_shell_uses_unrestricted(hydroxyl):
    mf, _ = PySCFEngine("HF", "sto-3g")._run(hydroxyl)
    assert isinstance(mf, scf.uhf.UHF)


def test_dft_reference_follows_multiplicity(water, hydroxyl):
    engine = PySCFEngine("PBE", "sto-3g")
    assert isinstance(engine._run(water)[0], dft.rks.RKS)
    assert isinstance(engine._run(hydroxyl)[0], dft.uks.UKS)


def test_forced_unrestricted_singlet(water):
    mf, _ = PySCFEngine("HF", "sto-3g", reference="unrestricted")._run(water)
    assert isinstance(mf, scf.uhf.UHF)


def test_restricted_doublet_rejected(hydroxyl):
    with pytest.raises(ValueError, match="multiplicity"):
        PySCFEngine("HF", "sto-3g", reference="restricted").energy(hydroxyl)


def test_analytic_hessian_for_hf(water):
    engine = PySCFEngine("HF", "sto-3g")
    numeric = fd_hessian(
        lambda x: engine.gradient(water.replace(coords=x)), water.coords
    )
    np.testing.assert_allclose(engine.hessian(water), numeric, rtol=0, atol=1e-4)


@pytest.mark.parametrize("method", ["HF", "PBE"])
def test_analytic_hessian_for_open_shell(method, h3_ts_guess):
    analytic = PySCFEngine(method, "sto-3g").hessian(h3_ts_guess)
    fd = PySCFEngine(method, "sto-3g", hessian="fd").hessian(h3_ts_guess)
    np.testing.assert_allclose(analytic, fd, rtol=0, atol=1e-4)


def test_finite_difference_fallback_for_mp2(water):
    hess = PySCFEngine("MP2", "sto-3g").hessian(water)
    assert hess.shape == (9, 9)
    np.testing.assert_allclose(hess, hess.T, rtol=0, atol=1e-12)


def test_mp2_energy_below_hf(water):
    assert PySCFEngine("MP2", "sto-3g").energy(water) < PySCFEngine(
        "HF", "sto-3g"
    ).energy(water)


def test_analytic_requested_but_unavailable():
    with pytest.raises(ValueError, match="MP2"):
        PySCFEngine("MP2", "sto-3g", hessian="analytic")


def test_gradient_points_uphill():
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    grad = PySCFEngine("HF", "sto-3g").gradient(h2)
    assert grad[1, 0] > 0
    assert grad[0, 0] == pytest.approx(-grad[1, 0])


def test_non_converged_scf_is_an_error(water):
    engine = PySCFEngine("HF", "sto-3g", max_cycle=1)
    with pytest.raises(EngineError, match=re.escape(engine.level.key())) as excinfo:
        engine.energy(water)
    assert "water" in str(excinfo.value)


@pytest.mark.filterwarnings("ignore:Basis may be available:UserWarning")
def test_element_missing_from_basis():
    potassium = Molecule(["K"], [[0.0, 0.0, 0.0]], mult=2)
    with pytest.raises(ValueError, match="cannot describe K"):
        PySCFEngine("HF", "cc-pvdz").energy(potassium)
