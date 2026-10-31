"""Deterministic cache keys for results computed on a molecule.

Keys depend only on the bytes hashed here, never on Python's hash
randomization, object identity or platform byte order.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pyvtst.core.molecule import Molecule

# Coordinates are rounded to this many bohr before hashing, so float noise from
# file round trips or other platforms still hits. A hit can return results for
# a geometry up to half a step away per coordinate. Anything that takes finite
# differences through a cache must use steps much larger than this.
KEY_COORD_RESOLUTION = 1e-6
_MASS_RESOLUTION = 1e-6  # amu
_DIGEST_CHARS = 32  # 128 bits
_MAX_KEY_LENGTH = 512

# Bumping a tag changes every key, so entries from an old scheme are never read.
_MOLECULE_TAG = b"pyvtst-molecule/v1\0"
_LEVEL_TAG = b"pyvtst-level/v1\0"
_PARAMS_TAG = b"pyvtst-params/v1\0"

_SEGMENT = re.compile(r"[A-Za-z0-9_\-][A-Za-z0-9_.\-]*")

ParamValue = str | int | float | bool | None


class _Keyed(Protocol):
    def key(self) -> str: ...


def molecule_key(mol: Molecule, *, include_masses: bool = True) -> str:
    """Return a 32-character hex digest of a molecule's identity.

    It covers the element symbols in order, the coordinates (rounded to
    :data:`KEY_COORD_RESOLUTION` bohr), the charge, the multiplicity and, when
    ``include_masses`` is true, the masses (rounded to 1e-6 amu). The name and
    symmetry number are ignored.
    """
    h = hashlib.sha256(_MOLECULE_TAG)
    h.update(",".join(mol.symbols).encode())
    h.update(b"\0")
    h.update(_quantize(mol.coords, KEY_COORD_RESOLUTION))
    h.update(np.array([mol.charge, mol.mult], dtype="<i8").tobytes())
    if include_masses:
        h.update(_quantize(mol.masses, _MASS_RESOLUTION))
    return h.hexdigest()[:_DIGEST_CHARS]


def cache_key(
    kind: str,
    mol: Molecule,
    level: _Keyed,
    *,
    include_masses: bool = True,
    **params: ParamValue | np.generic,
) -> str:
    """Return a store key for a result of ``kind`` on ``mol`` at ``level``.

    The key is ``"<kind>/<level digest>/<molecule key>"``, followed by
    ``"/<params digest>"`` when parameters are given, so all keys for one kind
    and level share a prefix. ``level`` is any object with a ``key() -> str``
    method. Parameter values are str, int, float, bool or None (numpy scalars
    are converted); ``1`` and ``1.0`` give different keys.

    Raises:
        ValueError: ``kind`` is not one or more valid key segments, or a float
            parameter is not finite.
        TypeError: A parameter value has an unsupported type.
    """
    if not isinstance(kind, str) or not all(
        _SEGMENT.fullmatch(segment) for segment in kind.split("/")
    ):
        raise ValueError(f"kind: invalid key segments {kind!r}")
    level_digest = _digest(_LEVEL_TAG + level.key().encode())
    parts = [kind, level_digest, molecule_key(mol, include_masses=include_masses)]
    if params:
        parts.append(_params_digest(params))
    key = "/".join(parts)
    if len(key) > _MAX_KEY_LENGTH:
        raise ValueError(f"kind: too long for a {_MAX_KEY_LENGTH}-character key")
    return key


def _quantize(values: npt.NDArray[np.float64], step: float) -> bytes:
    # rint maps -0.0 and +0.0 to the same integer.
    return np.rint(values / step).astype("<i8").tobytes()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_DIGEST_CHARS]


def _params_digest(params: dict[str, ParamValue | np.generic]) -> str:
    plain = {name: _param(name, value) for name, value in params.items()}
    text = json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _digest(_PARAMS_TAG + text.encode())


def _param(name: str, value: object) -> ParamValue:
    if isinstance(value, np.generic):
        value = value.item()
    # Exact builtin types, so subclasses serialize like the values they hold.
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"params: {name!r} must be finite, got {value!r}")
        return float(value)
    if isinstance(value, str):
        return str(value)
    raise TypeError(
        f"params: {name!r} must be str, int, float, bool or None, "
        f"got {type(value).__name__}"
    )
