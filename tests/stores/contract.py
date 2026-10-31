"""Behaviour every result store must have.

Subclass ``StoreContract`` in a test module and give it a ``store`` fixture
that returns a new, empty store.
"""

import numpy as np
import pytest

from pyvtst.ports import Store

INVALID_KEYS = ["", "a//b", "/a", "a/", "a/.tmp", "a b"]


def full_record():
    return {
        "e": -76.4,
        "n": 3,
        "label": "ts",
        "h": np.arange(81, dtype=np.float64).reshape(9, 9),
        "idx": np.arange(3, dtype=np.int64),
    }


def assert_records_equal(got, expected):
    assert got is not None
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            assert isinstance(got[name], np.ndarray), name
            assert got[name].dtype == value.dtype, name
            assert got[name].shape == value.shape, name
            np.testing.assert_array_equal(got[name], value, err_msg=name)
        else:
            assert type(got[name]) is type(value), name
            assert got[name] == value, name


class StoreContract:
    def test_satisfies_protocol(self, store):
        assert isinstance(store, Store)

    def test_valid_hierarchical_key(self, store):
        store.put("mep/3f2a/forward/000012", {"x": 1.0})
        assert "mep/3f2a/forward/000012" in store

    @pytest.mark.parametrize("key", INVALID_KEYS)
    def test_invalid_keys_are_rejected(self, store, key):
        with pytest.raises(ValueError, match="invalid store key"):
            store.get(key)
        with pytest.raises(ValueError, match="invalid store key"):
            store.put(key, {"x": 1.0})
        with pytest.raises(ValueError, match="invalid store key"):
            key in store  # noqa: B015
        assert list(store.keys()) == []

    def test_round_trip_of_all_value_types(self, store):
        record = full_record()
        store.put("r", record)
        assert_records_equal(store.get("r"), record)

    @pytest.mark.parametrize(
        "array",
        [
            np.array(2.5),
            np.zeros((0, 3)),
            np.array([True, False]),
            np.arange(4, dtype=np.uint8),
            np.arange(4, dtype=np.int32),
            np.arange(4, dtype=np.float32),
            np.array([1 + 2j, -3j]),
            np.array([np.nan, np.inf, -0.0]),
            np.arange(24.0).reshape(2, 3, 4),
        ],
        ids=lambda a: f"{a.dtype}{a.shape}",
    )
    def test_array_dtypes_and_shapes(self, store, array):
        store.put("a", {"v": array})
        assert_records_equal(store.get("a"), {"v": array})

    def test_scalar_edge_values(self, store):
        record = {
            "nan": float("nan"),
            "big": 2**63 - 1,
            "neg": -(2**63),
            "empty": "",
            "text": "héllo ✓",
        }
        store.put("s", record)
        got = store.get("s")
        assert got is not None
        assert np.isnan(got.pop("nan"))
        record.pop("nan")
        assert_records_equal(got, record)

    def test_empty_record(self, store):
        store.put("empty", {})
        assert store.get("empty") == {}
        assert "empty" in store
        assert list(store.keys()) == ["empty"]

    def test_unsupported_value_stores_nothing(self, store):
        with pytest.raises(TypeError, match="'bad'"):
            store.put("k", {"ok": 1.0, "bad": [1, 2]})
        assert store.get("k") is None
        assert list(store.keys()) == []

    def test_failed_put_keeps_previous_record(self, store):
        store.put("k", {"a": 1})
        with pytest.raises(TypeError):
            store.put("k", {"a": 2, "bad": None})
        assert store.get("k") == {"a": 1}

    def test_returned_data_is_a_copy(self, store):
        store.put("k", {"h": np.zeros(3)})
        got = store.get("k")
        assert got is not None
        got["h"][0] = 5.0
        got["extra"] = 1
        assert_records_equal(store.get("k"), {"h": np.zeros(3)})

    def test_put_copies_its_input(self, store):
        arr = np.zeros(3)
        store.put("k", {"h": arr})
        arr[0] = 5.0
        assert_records_equal(store.get("k"), {"h": np.zeros(3)})

    def test_missing_key(self, store):
        assert store.get("never/put") is None
        assert "never/put" not in store

    def test_overwrite_replaces_the_record(self, store):
        store.put("k", {"a": 1, "b": 2})
        store.put("k", {"a": 3})
        assert store.get("k") == {"a": 3}

    def test_overwrite_can_change_field_kinds(self, store):
        store.put("k", {"a": np.zeros(2)})
        store.put("k", {"a": "now a string"})
        assert store.get("k") == {"a": "now a string"}

    def test_nested_records_coexist(self, store):
        store.put("a/b", {"x": 1})
        store.put("a/b/c", {"x": 2})
        assert store.get("a/b") == {"x": 1}
        assert store.get("a/b/c") == {"x": 2}
        store.put("a/b", {"x": 3})
        assert store.get("a/b/c") == {"x": 2}

    def test_ordered_mep_points(self, store):
        for key in [
            "mep/x/forward/000002",
            "mep/x/forward/000000",
            "mep/x/reverse/000000",
            "mep/x/forward/000001",
        ]:
            store.put(key, {"s": 0.0})
        assert list(store.keys("mep/x/forward/")) == [
            "mep/x/forward/000000",
            "mep/x/forward/000001",
            "mep/x/forward/000002",
        ]

    def test_only_records_are_listed(self, store):
        store.put("a/b/c", {})
        assert list(store.keys()) == ["a/b/c"]
        assert "a/b" not in store
        assert store.get("a") is None

    def test_keys_use_plain_string_prefix(self, store):
        for key in ["ab", "a/b", "abc/d", "b"]:
            store.put(key, {})
        assert list(store.keys("ab")) == ["ab", "abc/d"]
        assert list(store.keys()) == ["a/b", "ab", "abc/d", "b"]
        assert list(store.keys("zzz")) == []

    def test_keys_in_code_point_order(self, store):
        for key in ["a", "B", "_", "-", "0", "a.b", "a-b"]:
            store.put(key, {})
        keys = list(store.keys())
        assert keys == sorted(keys)
        assert len(keys) == 7
