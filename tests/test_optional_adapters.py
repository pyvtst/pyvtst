import sys

import pytest

from pyvtst.engines import PySCFEngine
from pyvtst.optimizers import GeometricOptimizer


def test_missing_pyscf(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscf", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[pyscf\]"') as excinfo:
        PySCFEngine("HF", "sto-3g")
    assert excinfo.value.__cause__ is not None


def test_missing_geometric(monkeypatch):
    monkeypatch.setitem(sys.modules, "geometric", None)
    with pytest.raises(
        ImportError, match=r'pip install "pyvtst\[geometric\]"'
    ) as excinfo:
        GeometricOptimizer()
    assert excinfo.value.__cause__ is not None
