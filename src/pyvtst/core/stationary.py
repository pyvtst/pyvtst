"""Characterized stationary points: geometry, energy and harmonic frequencies."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyvtst.core.elements import formula
from pyvtst.core.molecule import Molecule
from pyvtst.core.units import CM1_TO_HARTREE
from pyvtst.core.vib import is_linear, n_vibrations, normal_modes

FloatArray = npt.NDArray[np.float64]
Kind = Literal["minimum", "saddle", "higher-order"]

DEFAULT_IMAG_THRESHOLD = 50.0  # cm⁻¹


class StationaryPointError(ValueError):
    """A stationary point does not have the character a calculation needs."""


@dataclass(frozen=True, eq=False)
class StationaryPoint:
    """A molecule at a stationary point, with its energy and harmonic frequencies.

    Attributes:
        molecule: The structure.
        energy: Electronic energy in hartree.
        frequencies: Harmonic frequencies in cm⁻¹, ascending, read-only, with
            imaginary modes as negative values.
        modes: Mass-weighted normal modes as rows, shape (n, 3N), or ``None``
            when built from frequencies.
        imag_threshold: Imaginary frequencies with a magnitude at or below
            this (cm⁻¹) are numerical noise: they are not counted as imaginary
            and enter the vibrational partition function and the ZPE with
            their magnitude. A warning is emitted for each at construction.

    Build instances with :meth:`from_hessian` or :meth:`from_frequencies`.
    Instances compare by value.
    """

    molecule: Molecule
    energy: float
    frequencies: FloatArray
    modes: FloatArray | None = None
    imag_threshold: float = DEFAULT_IMAG_THRESHOLD

    def __post_init__(self) -> None:
        energy = float(self.energy)
        if not np.isfinite(energy):
            raise ValueError(f"energy: must be finite, got {energy}")
        threshold = float(self.imag_threshold)
        if not threshold >= 0:
            raise ValueError(f"imag_threshold: must be >= 0, got {threshold}")
        freqs = np.array(self.frequencies, dtype=np.float64).reshape(-1)
        expected = n_vibrations(self.molecule)
        if freqs.size != expected:
            raise ValueError(
                f"frequencies: expected {expected} for "
                f"{_describe(self.molecule)}, got {freqs.size}"
            )
        if not np.all(np.isfinite(freqs)):
            raise ValueError("frequencies: all frequencies must be finite")
        order = np.argsort(freqs, kind="stable")
        freqs = freqs[order]
        freqs.flags.writeable = False
        modes = self.modes
        if modes is not None:
            modes = np.array(modes, dtype=np.float64)
            shape = (expected, 3 * self.molecule.n_atoms)
            if modes.shape != shape:
                raise ValueError(f"modes: expected shape {shape}, got {modes.shape}")
            modes = modes[order]
            modes.flags.writeable = False
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "imag_threshold", threshold)
        object.__setattr__(self, "frequencies", freqs)
        object.__setattr__(self, "modes", modes)

        noise = freqs[(freqs < 0) & (freqs >= -threshold)]
        if noise.size:
            listed = ", ".join(f"{f:.1f}" for f in noise)
            warnings.warn(
                f"{species_label(self.molecule)}: imaginary frequencies "
                f"[{listed}] cm^-1 are below the {threshold:g} cm^-1 threshold; "
                "treated as numerical noise and used as real frequencies",
                stacklevel=3,
            )

    @classmethod
    def from_hessian(
        cls,
        mol: Molecule,
        energy: float,
        hessian: npt.ArrayLike,
        imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
    ) -> StationaryPoint:
        """Characterize ``mol`` from its Cartesian Hessian in hartree/bohr²."""
        freqs, modes = normal_modes(mol, hessian)
        return cls(mol, energy, freqs, modes, imag_threshold)

    @classmethod
    def from_frequencies(
        cls,
        mol: Molecule,
        energy: float,
        frequencies: Sequence[float] | npt.ArrayLike,
        imag_threshold: float = DEFAULT_IMAG_THRESHOLD,
    ) -> StationaryPoint:
        """Build a stationary point from frequencies in cm⁻¹ (e.g. published).

        Imaginary frequencies are given as negative values. The count must be
        3N-6, 3N-5 for a linear molecule, or 0 for a single atom.
        """
        return cls(
            mol, energy, np.asarray(frequencies, dtype=np.float64), None, imag_threshold
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StationaryPoint):
            return NotImplemented
        return (
            self.molecule == other.molecule
            and self.energy == other.energy
            and np.array_equal(self.frequencies, other.frequencies)
            and _same_modes(self.modes, other.modes)
            and self.imag_threshold == other.imag_threshold
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def imaginary_frequencies(self) -> FloatArray:
        """Counted imaginary frequencies (below ``-imag_threshold``), in cm⁻¹."""
        f = self.frequencies
        return f[f < -self.imag_threshold]

    @property
    def n_imaginary(self) -> int:
        """Number of imaginary frequencies with magnitude above the threshold."""
        return int(self.imaginary_frequencies.size)

    @property
    def kind(self) -> Kind:
        """``"minimum"``, ``"saddle"`` (first order) or ``"higher-order"``."""
        n = self.n_imaginary
        return "minimum" if n == 0 else "saddle" if n == 1 else "higher-order"

    @property
    def imaginary_frequency(self) -> float | None:
        """Magnitude in cm⁻¹ of a first-order saddle's imaginary mode, else None."""
        if self.kind != "saddle":
            return None
        return float(-self.imaginary_frequencies[0])

    @property
    def real_frequencies(self) -> FloatArray:
        """Frequencies used for the vibrational partition function and ZPE.

        Counted imaginary modes are excluded; imaginary modes below the
        threshold are included with their magnitude. Ascending, cm⁻¹.
        """
        f = self.frequencies
        return np.sort(np.abs(f[f >= -self.imag_threshold]))

    @property
    def zpe(self) -> float:
        """Zero-point energy ½ Σ h c nu over the real frequencies, in hartree."""
        return float(0.5 * np.sum(self.real_frequencies) * CM1_TO_HARTREE)

    @property
    def is_linear(self) -> bool:
        return is_linear(self.molecule)


def species_label(mol: Molecule, role: str = "") -> str:
    """Name a species in messages: ``"<role> '<name>'"``, the role or the formula."""
    if mol.name:
        return f"{role} {mol.name!r}" if role else repr(mol.name)
    return role or formula(mol.symbols)


def check_character(sp: StationaryPoint, kind: Kind, label: str) -> None:
    """Raise :class:`StationaryPointError` unless ``sp.kind == kind``.

    The message names the species (``label``), gives its number of imaginary
    frequencies and the number expected, and lists them in cm⁻¹.
    """
    if sp.kind == kind:
        return
    expected = 1 if kind == "saddle" else 0
    message = f"{label} has {sp.n_imaginary} imaginary frequencies, expected {expected}"
    if sp.n_imaginary:
        listed = ", ".join(f"{f:.1f}" for f in sp.imaginary_frequencies)
        message += f" ([{listed}] cm^-1)"
    raise StationaryPointError(message)


def _describe(mol: Molecule) -> str:
    if mol.n_atoms == 1:
        return "a single atom"
    shape = "linear" if is_linear(mol) else "non-linear"
    return f"a {shape} {mol.n_atoms}-atom molecule"


def _same_modes(a: FloatArray | None, b: FloatArray | None) -> bool:
    if a is None or b is None:
        return a is b
    return np.array_equal(a, b)
