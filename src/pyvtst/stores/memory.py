"""In-process result store."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from pyvtst.ports.store import Record, Value, validate_key, validate_record


class MemoryStore:
    """A result store held in process memory and lost with the object.

    Used for tests and whenever no store file is given.
    """

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def get(self, key: str) -> Record | None:
        record = self._records.get(validate_key(key))
        return None if record is None else _copy(record)

    def put(self, key: str, record: Mapping[str, Value]) -> None:
        validate_key(key)
        self._records[key] = validate_record(record)

    def __contains__(self, key: object) -> bool:
        return validate_key(key) in self._records

    def keys(self, prefix: str = "") -> Iterator[str]:
        return iter(sorted(k for k in self._records if k.startswith(prefix)))

    def __repr__(self) -> str:
        return f"MemoryStore(<{len(self._records)} records>)"


def _copy(record: Record) -> Record:
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in record.items()
    }
