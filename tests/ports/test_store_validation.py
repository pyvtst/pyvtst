import numpy as np
import pytest

from pyvtst.ports import Store, StoreError, StoreFormatError, StoreLockedError
from pyvtst.ports.store import MAX_KEY_LENGTH, validate_key, validate_record


@pytest.mark.parametrize(
    "key",
    [
        "a",
        "mep/3f2a/forward/000012",
        "A-z_0.9/x",
        "a/b./c",
        "x" * MAX_KEY_LENGTH,
    ],
)
def test_valid_keys(key):
    assert validate_key(key) == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        "a//b",
        "/a",
        "a/",
        "a/.tmp",
        ".a",
        "a b",
        "a\\b",
        "é",
        "a/b\n",
        "x" * (MAX_KEY_LENGTH + 1),
        None,
        b"a",
    ],
)
def test_invalid_keys(key):
    with pytest.raises(ValueError, match="invalid store key"):
        validate_key(key)


def test_all_value_types_validate():
    record = {
        "e": -76.4,
        "n": 3,
        "label": "ts",
        "h": np.zeros((9, 9)),
        "idx": np.arange(3),
        "flags": np.array([True, False]),
        "u": np.arange(2, dtype=np.uint8),
        "c": np.array([1 + 2j]),
    }
    out = validate_record(record)
    assert out.keys() == record.keys()
    assert type(out["e"]) is float
    assert type(out["n"]) is int
    assert type(out["label"]) is str


def test_validated_arrays_are_copies():
    arr = np.zeros(3)
    out = validate_record({"a": arr})
    arr[0] = 1.0
    copy = out["a"]
    assert isinstance(copy, np.ndarray)
    assert copy[0] == 0.0


def test_subclasses_become_exact_types():
    out = validate_record({"f": np.float64(1.5), "s": np.str_("x")})
    assert type(out["f"]) is float
    assert type(out["s"]) is str


def test_empty_record_is_valid():
    assert validate_record({}) == {}


@pytest.mark.parametrize(
    "value",
    [
        [1, 2],
        (1, 2),
        None,
        True,
        np.bool_(True),
        np.int64(3),
        {"a": 1},
        b"bytes",
        np.array(["a", "b"]),
        np.array([b"a"]),
        np.array([1, None], dtype=object),
        np.array(["2020-01-01"], dtype="datetime64[D]"),
    ],
    ids=repr,
)
def test_unsupported_values_name_the_field(value):
    with pytest.raises(TypeError, match="'bad'"):
        validate_record({"ok": 1.0, "bad": value})


@pytest.mark.parametrize("name", ["", "a-b", "a.b", "a/b", "é", 3])
def test_invalid_field_names(name):
    with pytest.raises(ValueError, match="invalid field name"):
        validate_record({name: 1.0})


def test_int_out_of_64_bit_range():
    validate_record({"n": 2**63 - 1})
    with pytest.raises(ValueError, match="64 bits"):
        validate_record({"n": 2**63})


def test_record_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        validate_record([("a", 1.0)])  # type: ignore[arg-type]


def test_error_hierarchy():
    assert issubclass(StoreFormatError, StoreError)
    assert issubclass(StoreLockedError, StoreError)
    assert issubclass(StoreError, RuntimeError)
    err = StoreLockedError("/tmp/x.h5")
    assert err.path == "/tmp/x.h5"
    assert "/tmp/x.h5" in str(err)


def test_dict_is_not_a_store():
    assert not isinstance({}, Store)
