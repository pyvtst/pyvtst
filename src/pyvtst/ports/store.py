"""The store port: a key/value store for result records."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

Value: TypeAlias = npt.NDArray[np.generic] | float | int | str
Record: TypeAlias = dict[str, Value]

MAX_KEY_LENGTH = 512

_SEGMENT = re.compile(r"[A-Za-z0-9_\-][A-Za-z0-9_.\-]*")
_FIELD = re.compile(r"[A-Za-z0-9_]+")
# bool, signed and unsigned integer, floating and complex arrays.
_ARRAY_KINDS = "biufc"
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


class StoreError(RuntimeError):
    """A result store could not be used."""


class StoreFormatError(StoreError):
    """A store file is not a pyvtst store, is too new, or is damaged."""


class StoreLockedError(StoreError):
    """A store file stayed locked by another process for too long.

    Attributes:
        path: The locked file.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"{path}: store is locked by another process")
        self.path = path


@runtime_checkable
class Store(Protocol):
    """A key/value store for result records.

    Keys are one or more ``/``-separated segments of ``A-Z a-z 0-9 _ . -``,
    none empty or starting with ``.``, at most 512 characters in total. A
    record maps field names (``A-Z a-z 0-9 _``) to numpy arrays (bool,
    integer, floating or complex dtype), floats, ints or strs.
    """

    def get(self, key: str) -> Record | None:
        """Return a new copy of the record under ``key``, or ``None``."""
        ...

    def put(self, key: str, record: Mapping[str, Value]) -> None:
        """Store ``record`` under ``key``, replacing any earlier record whole.

        If ``put`` raises, the store is left as it was before the call.
        """
        ...

    def __contains__(self, key: object) -> bool:
        """Whether a record exists under ``key``."""
        ...

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Yield record keys starting with ``prefix``, in code-point order."""
        ...


def validate_key(key: object) -> str:
    """Return ``key`` if it is a valid store key, else raise ``ValueError``."""
    if (
        not isinstance(key, str)
        or len(key) > MAX_KEY_LENGTH
        or not all(_SEGMENT.fullmatch(segment) for segment in key.split("/"))
    ):
        raise ValueError(f"invalid store key {key!r}")
    return key


def validate_record(record: Mapping[str, object]) -> Record:
    """Return a validated copy of ``record`` with values in their exact types.

    Raises:
        TypeError: ``record`` is not a mapping, or a field has an unsupported
            type (Python ``bool``, any other Python type, or an array of object
            or string dtype).
        ValueError: A field name is invalid, or an int does not fit in 64 bits.
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"record: expected a mapping, got {type(record).__name__}")
    out: Record = {}
    for name, value in record.items():
        if not isinstance(name, str) or not _FIELD.fullmatch(name):
            raise ValueError(f"record: invalid field name {name!r}")
        out[name] = _value(name, value)
    return out


def _value(name: str, value: object) -> Value:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(
                f"record: field {name!r} has unsupported dtype {value.dtype}"
            )
        return value.copy()
    if isinstance(value, bool):
        raise TypeError(f"record: field {name!r} is a bool; store it as an int")
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"record: field {name!r} does not fit in 64 bits")
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    raise TypeError(
        f"record: field {name!r} has unsupported type {type(value).__name__}"
    )
