import pytest

from pyvtst.stores import HDF5Store, MemoryStore, open_store


def test_none_becomes_an_in_memory_store():
    a, b = open_store(None), open_store(None)
    assert isinstance(a, MemoryStore)
    assert a is not b


@pytest.mark.parametrize("as_str", [True, False], ids=["str", "pathlike"])
def test_path_becomes_an_hdf5_store(tmp_path, as_str):
    pytest.importorskip("h5py")
    path = tmp_path / "ch4_h.h5"
    store = open_store(str(path) if as_str else path)
    assert isinstance(store, HDF5Store)
    assert store.path == str(path)
    assert store.mode == "a"
    assert path.exists()


def test_no_suffix_is_enforced(tmp_path):
    pytest.importorskip("h5py")
    assert isinstance(open_store(tmp_path / "cache.db"), HDF5Store)


def test_existing_store_passes_through():
    s = MemoryStore()
    assert open_store(s) is s


def test_duck_typed_store_passes_through():
    class Custom:
        def get(self, key):
            return None

        def put(self, key, record):
            pass

        def keys(self, prefix=""):
            return iter(())

        def __contains__(self, key):
            return False

    s = Custom()
    assert open_store(s) is s


@pytest.mark.parametrize("value", [42, {}, [], b"cache.h5", object()], ids=repr)
def test_wrong_type(value):
    with pytest.raises(TypeError, match="store"):
        open_store(value)
