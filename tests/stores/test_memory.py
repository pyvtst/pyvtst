import pytest

from pyvtst.stores import MemoryStore
from tests.stores.contract import StoreContract


class TestMemoryStore(StoreContract):
    @pytest.fixture
    def store(self):
        return MemoryStore()


def test_stores_are_independent():
    a, b = MemoryStore(), MemoryStore()
    a.put("k", {"x": 1})
    assert "k" not in b
