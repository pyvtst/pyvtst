"""Level of theory: the identity of a calculation, used in cache keys."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote

OptionValue = str | int | float | bool

# Characters that are part of the key syntax (or the escape character) get
# percent-encoded, so that no two different levels can render the same key.
# Everything else that is common in method and basis names stays readable.
_SAFE = "-+*()[],_.'~"


@dataclass(frozen=True, eq=False)
class LevelOfTheory:
    """A method, a basis set and the options that can change a result.

    Attributes:
        method: Method name, lower-cased (e.g. ``"b3lyp"``).
        basis: Basis-set name, lower-cased (e.g. ``"def2-svp"``).
        options: ``(name, value)`` pairs sorted by name. Values are ``str``,
            ``int``, ``float`` or ``bool``.

    Two levels are equal, and hash equally, exactly when their :meth:`key`
    strings are equal. ``1``, ``1.0`` and ``True`` are different option values.
    """

    method: str
    basis: str
    options: tuple[tuple[str, OptionValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _name("method", self.method))
        object.__setattr__(self, "basis", _name("basis", self.basis))
        object.__setattr__(self, "options", _options(self.options))

    def key(self) -> str:
        """Return a string identifying this level, stable across processes.

        Example: ``b3lyp/def2-svp?conv_tol=f:1e-10&grid_level=i:3``.
        """
        head = f"{_quote(self.method)}/{_quote(self.basis)}"
        return head + render_options(self.options)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LevelOfTheory):
            return NotImplemented
        return self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


def render_options(options: Iterable[tuple[str, OptionValue]]) -> str:
    """Render validated options as ``"?name=tag:value&..."``, or ``""`` if none.

    The tag is ``s``, ``i``, ``f`` or ``b``, so ``1``, ``1.0`` and ``True``
    render differently. Floats use ``repr`` (shortest round-trip form).
    """
    parts = [f"{_quote(name)}={_render_value(value)}" for name, value in options]
    return "?" + "&".join(parts) if parts else ""


def _render_value(value: OptionValue) -> str:
    if isinstance(value, bool):
        return "b:true" if value else "b:false"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value!r}"
    return f"s:{_quote(value)}"


def _quote(text: str) -> str:
    return quote(text, safe=_SAFE)


def _name(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a non-empty string, got {value!r}")
    return value.strip().lower()


def _options(
    options: Iterable[tuple[str, OptionValue]],
) -> tuple[tuple[str, OptionValue], ...]:
    seen: dict[str, OptionValue] = {}
    for item in options:
        try:
            name, value = item
        except (TypeError, ValueError):
            raise ValueError(
                f"options: expected (name, value) pairs, got {item!r}"
            ) from None
        if not isinstance(name, str) or not name:
            raise ValueError(f"options: name must be a non-empty string, got {name!r}")
        if name in seen:
            raise ValueError(f"options: duplicate option {name!r}")
        seen[name] = _value(name, value)
    return tuple(sorted(seen.items()))


def _value(name: str, value: object) -> OptionValue:
    # Values are converted to the exact builtin type, so that e.g. numpy
    # scalars render the same as the Python numbers they stand for.
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"options: {name!r} must be finite, got {value!r}")
        return float(value)
    if isinstance(value, str):
        return str(value)
    raise ValueError(
        f"options: {name!r} must be str, int, float or bool, got {type(value).__name__}"
    )
