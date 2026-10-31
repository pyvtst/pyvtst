import sys

import pytest

from pyvtst.engines import PySCFEngine
from pyvtst.optimizers import GeometricOptimizer
from pyvtst.stores import HDF5Store, open_store


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


def test_missing_h5py(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r'pip install "pyvtst\[hdf5\]"') as excinfo:
        HDF5Store(tmp_path / "cache.h5")
    assert excinfo.value.__cause__ is not None
    assert not (tmp_path / "cache.h5").exists()


def test_memory_store_works_without_h5py(monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    store = open_store(None)
    store.put("k", {"x": 1.0})
    assert store.get("k") == {"x": 1.0}
