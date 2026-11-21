"""Shared fixtures: the analytic LEPS engine, fake Gaussian and ORCA programs
and H / H2 / H3 structures."""

import pytest

from pyvtst import Molecule
from tests.fakes import orca as fake_orca_module
from tests.fakes.gaussian import install
from tests.fakes.leps import LEPSEngine


@pytest.fixture
def leps_engine():
    return LEPSEngine()


@pytest.fixture
def fake_gaussian(tmp_path, monkeypatch):
    """Fake ``g16`` and ``formchk`` on the LEPS surface, selected through
    ``PYVTST_G16`` and ``PYVTST_FORMCHK``; set ``FAKE_G16_MODE`` for failures."""
    fake = install(tmp_path / "fake-gaussian")
    monkeypatch.setenv("PYVTST_G16", str(fake.g16))
    monkeypatch.setenv("PYVTST_FORMCHK", str(fake.formchk))
    monkeypatch.delenv("FAKE_G16_MODE", raising=False)
    return fake


@pytest.fixture
def fake_orca(tmp_path, monkeypatch):
    """A fake ``orca`` on the LEPS surface, selected through ``PYVTST_ORCA``;
    set ``FAKE_ORCA_MODE`` for failures."""
    fake = fake_orca_module.install(tmp_path / "fake-orca")
    monkeypatch.setenv("PYVTST_ORCA", str(fake.orca))
    monkeypatch.delenv("FAKE_ORCA_MODE", raising=False)
    return fake


@pytest.fixture
def h_atom():
    return Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, name="H")


@pytest.fixture
def h2_stretched():
    """H2 at 1.8 bohr, beyond the 1.401 bohr equilibrium."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], name="H2")


@pytest.fixture
def h3_ts_guess():
    """Slightly bent H3 with unequal distances (1.6 and 1.9 bohr)."""
    return Molecule(
        ["H", "H", "H"],
        [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]],
        mult=2,
        name="H3",
    )
