"""Turn a user's ``store=`` argument into a store."""

from __future__ import annotations

import os

from pyvtst.ports.store import Store
from pyvtst.stores.hdf5 import HDF5Store
from pyvtst.stores.memory import MemoryStore

_METHODS = ("get", "put", "keys", "__contains__")


def open_store(store: Store | str | os.PathLike[str] | None) -> Store:
    """Return a store for ``store``.

    ``None`` gives a new :class:`MemoryStore`; a path gives an
    :class:`HDF5Store` on that file with default settings; any object with
    ``get``, ``put``, ``keys`` and ``__contains__`` methods is returned as is.

    Raises:
        TypeError: ``store`` is none of these.
    """
    if store is None:
        return MemoryStore()
    if isinstance(store, str | os.PathLike):
        return HDF5Store(os.fspath(store))
    if all(callable(getattr(store, name, None)) for name in _METHODS):
        return store
    raise TypeError(
        f"store: expected None, a path or a store, got {type(store).__name__}"
    )
