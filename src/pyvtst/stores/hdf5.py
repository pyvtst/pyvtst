"""HDF5 result store in a single file. h5py is imported when a store is built.

File layout::

    /                       attrs: pyvtst_store, format_version, created_by
    /records/<key>/.data    one record: a dataset per array field, an
                            attribute per scalar field, and ".types"
    /.staging/<id>          a record being written (invisible to readers)
    /.trash/<id>            a replaced record awaiting deletion
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Any, Literal

import numpy as np

from pyvtst import __version__
from pyvtst.ports.store import (
    Record,
    StoreFormatError,
    StoreLockedError,
    Value,
    validate_key,
    validate_record,
)

FORMAT_VERSION = 1

_MARKER = "pyvtst-result-store"
_RECORDS = "records"
_DATA = ".data"
_STAGING = ".staging"
_TRASH = ".trash"
_TYPES = ".types"

_MIN_BACKOFF = 0.01  # seconds
_MAX_BACKOFF = 1.0

# (pid, real path) of every store file this process has already backed up.
_backed_up: set[tuple[int, str]] = set()


def _import_h5py() -> Any:
    # h5py is optional, so it is imported on first use rather than at load time.
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError('HDF5Store needs h5py: pip install "pyvtst[hdf5]"') from exc
    return h5py


class HDF5Store:
    """A result store in one HDF5 file.

    Args:
        path: The store file. In mode ``"a"`` it is created if missing (its
            directory must exist); in mode ``"r"`` it must exist.
        mode: ``"a"`` to read and write, ``"r"`` to read only (every ``put``
            raises ``PermissionError``).
        lock_timeout: Seconds to keep retrying while another process holds
            the file, before raising :class:`~pyvtst.ports.StoreLockedError`.
        backup: Copy the file to ``<path>.bak`` before this process first
            writes to it, so a file damaged mid-write loses at most one
            session of results.

    The file is opened for each operation and closed again, so it is
    consistent on disk between operations and a killed process loses nothing
    already stored. Every write is staged and then linked into place, so an
    exception during a ``put`` leaves the key's previous record (or none).

    Several processes may share one file on a local file system: HDF5's file
    locking lets one writer in at a time. Concurrent use is unsupported where
    locking is disabled (``HDF5_USE_FILE_LOCKING=FALSE``, common on NFS and
    Lustre), and can corrupt the file there.

    Raises:
        ImportError: h5py is not installed.
        FileNotFoundError: The file (mode ``"r"``) or its directory is missing.
        StoreFormatError: The file is not a pyvtst store, has a newer format,
            or cannot be read as HDF5.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        mode: Literal["a", "r"] = "a",
        lock_timeout: float = 30.0,
        backup: bool = True,
    ) -> None:
        self._h5py = _import_h5py()
        if mode not in ("a", "r"):
            raise ValueError(f"mode: expected 'a' or 'r', got {mode!r}")
        if not lock_timeout >= 0:
            raise ValueError(f"lock_timeout: must be >= 0, got {lock_timeout}")
        self.path = os.fspath(path)
        self.mode = mode
        self.lock_timeout = float(lock_timeout)
        self.backup = bool(backup)

        if os.path.exists(self.path):
            with self._open("r") as f:
                needs_init = self._check(f)
        elif mode == "r":
            raise FileNotFoundError(f"{self.path}: store file not found")
        else:
            parent = os.path.dirname(os.path.abspath(self.path))
            if not os.path.isdir(parent):
                raise FileNotFoundError(f"{parent}: directory not found")
            needs_init = True
        if needs_init and mode == "a":
            with self._open("a") as f:
                if self._check(f):
                    _init(f)

    def get(self, key: str) -> Record | None:
        path = _data_path(validate_key(key))
        with self._open("r") as f:
            self._check(f)
            data = f.get(path)
            return None if data is None else _read(data)

    def put(self, key: str, record: Mapping[str, Value]) -> None:
        path = _data_path(validate_key(key))
        fields = validate_record(record)
        if self.mode == "r":
            raise PermissionError(f"{self.path}: store is open read-only")
        with self._open("a") as f:
            if self._check(f):
                _init(f)
            self._backup_once()
            _clear(f, _STAGING)
            _clear(f, _TRASH)
            uid = uuid.uuid4().hex
            staged = f.require_group(_STAGING).create_group(uid)
            _write(staged, fields)
            f.require_group(path.rsplit("/", 1)[0])
            trashed = None
            if path in f:
                trashed = f"{_TRASH}/{uid}"
                f.require_group(_TRASH)
                f.move(path, trashed)
            f.move(staged.name, path)
            if trashed is not None:
                del f[trashed]
            f.flush()

    def __contains__(self, key: object) -> bool:
        path = _data_path(validate_key(key))
        with self._open("r") as f:
            self._check(f)
            return path in f

    def keys(self, prefix: str = "") -> Iterator[str]:
        found: list[str] = []
        suffix = "/" + _DATA

        def visit(name: str) -> None:
            if name.endswith(suffix):
                key = name[: -len(suffix)]
                if key.startswith(prefix):
                    found.append(key)

        with self._open("r") as f:
            self._check(f)
            records = f.get(_RECORDS)
            if records is not None:
                records.visit(visit)
        return iter(sorted(found))

    def __repr__(self) -> str:
        return f"HDF5Store({self.path!r}, mode={self.mode!r})"

    def _open(self, mode: Literal["a", "r"]) -> Any:
        """Open the file, retrying with backoff while another process holds it."""
        deadline = time.monotonic() + self.lock_timeout
        delay = _MIN_BACKOFF
        while True:
            try:
                return self._h5py.File(self.path, mode)
            except OSError as exc:
                if not _is_busy(exc):
                    if isinstance(exc, FileNotFoundError | PermissionError):
                        raise
                    raise StoreFormatError(self._unreadable(exc)) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StoreLockedError(self.path) from exc
                time.sleep(min(delay, remaining))
                delay = min(2 * delay, _MAX_BACKOFF)

    def _check(self, f: Any) -> bool:
        """Check the root marker; return True for an empty file to initialize."""
        marker = f.attrs.get("pyvtst_store")
        if marker is None:
            if len(f) == 0 and len(f.attrs) == 0:
                return True
            raise StoreFormatError(
                f"{self.path}: not a pyvtst result store; it was left unchanged"
            )
        if isinstance(marker, bytes):
            marker = marker.decode(errors="replace")
        version = f.attrs.get("format_version")
        if marker != _MARKER or not isinstance(version, int | np.integer):
            raise StoreFormatError(f"{self.path}: unrecognized pyvtst store header")
        if version > FORMAT_VERSION:
            raise StoreFormatError(
                f"{self.path}: store format version {version} is newer than this "
                f"pyvtst supports ({FORMAT_VERSION}); upgrade pyvtst to read it"
            )
        return False

    def _unreadable(self, exc: OSError) -> str:
        message = f"{self.path}: cannot be read as an HDF5 file ({exc})"
        backup = self.path + ".bak"
        if os.path.exists(backup):
            message += (
                f". A copy from before an earlier session's writes is at {backup};"
                f" to restore it, copy it over {self.path}"
            )
        return message

    def _backup_once(self) -> None:
        # Called with the file open for writing, so this process holds the
        # exclusive lock and no other writer is mid-write: the copy is whole.
        if not self.backup:
            return
        tag = (os.getpid(), os.path.realpath(self.path))
        if tag in _backed_up:
            return
        shutil.copy2(self.path, self.path + ".bak")
        _backed_up.add(tag)


def _is_busy(exc: OSError) -> bool:
    """Whether opening failed because another process holds the file.

    FileExistsError covers two processes racing to create a missing file.
    """
    return isinstance(exc, BlockingIOError | FileExistsError) or (
        "unable to lock file" in str(exc)
    )


def _data_path(key: str) -> str:
    return f"{_RECORDS}/{key}/{_DATA}"


def _init(f: Any) -> None:
    f.attrs["pyvtst_store"] = _MARKER
    f.attrs["format_version"] = np.int64(FORMAT_VERSION)
    f.attrs["created_by"] = f"pyvtst {__version__}"
    f.require_group(_RECORDS)


def _clear(f: Any, name: str) -> None:
    """Delete everything left in a scratch group by an earlier, failed write."""
    group = f.get(name)
    if group is not None:
        for child in list(group):
            del group[child]


def _write(group: Any, fields: Record) -> None:
    types: dict[str, str] = {}
    for name, value in fields.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(name, data=value)
            types[name] = "array"
        elif isinstance(value, float):
            group.attrs[name] = np.float64(value)
            types[name] = "float"
        elif isinstance(value, int):
            group.attrs[name] = np.int64(value)
            types[name] = "int"
        else:
            group.attrs[name] = value
            types[name] = "str"
    group.attrs[_TYPES] = json.dumps(types)


def _read(group: Any) -> Record:
    record: Record = {}
    for name, kind in json.loads(group.attrs[_TYPES]).items():
        if kind == "array":
            record[name] = np.array(group[name][()])
        elif kind == "float":
            record[name] = float(group.attrs[name])
        elif kind == "int":
            record[name] = int(group.attrs[name])
        else:
            record[name] = str(group.attrs[name])
    return record
