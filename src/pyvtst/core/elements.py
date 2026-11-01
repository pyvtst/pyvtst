"""Element table for H through Kr (Z = 1-36).

Each entry gives the element symbol, atomic number, the mass of its most
abundant isotope in amu and its covalent radius in ångström. Masses are from
the NIST "Atomic Weights and Isotopic Compositions for All Elements" tables
(J. S. Coursey et al., NIST Standard Reference Database 144, https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses),
which take relative atomic masses from the Atomic Mass Evaluation.

Covalent radii in ångström are from B. Cordero et al., "Covalent radii
revisited", Dalton Trans. 2008, 2832-2838, using the sp³ value for carbon and
the low-spin values for Mn, Fe and Co.

Masses and radii are tabulated data, not physical constants, so they are not
taken from ``scipy.constants``.
"""

from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple


class Element(NamedTuple):
    symbol: str
    atomic_number: int
    mass: float  # most abundant isotope, amu
    covalent_radius: float  # Å


_TABLE = (
    Element("H", 1, 1.00782503223, 0.31),  # 1H
    Element("He", 2, 4.00260325413, 0.28),  # 4He
    Element("Li", 3, 7.0160034366, 1.28),  # 7Li
    Element("Be", 4, 9.012183065, 0.96),  # 9Be
    Element("B", 5, 11.00930536, 0.84),  # 11B
    Element("C", 6, 12.0, 0.76),  # 12C
    Element("N", 7, 14.00307400443, 0.71),  # 14N
    Element("O", 8, 15.99491461957, 0.66),  # 16O
    Element("F", 9, 18.99840316273, 0.57),  # 19F
    Element("Ne", 10, 19.9924401762, 0.58),  # 20Ne
    Element("Na", 11, 22.989769282, 1.66),  # 23Na
    Element("Mg", 12, 23.985041697, 1.41),  # 24Mg
    Element("Al", 13, 26.98153853, 1.21),  # 27Al
    Element("Si", 14, 27.97692653465, 1.11),  # 28Si
    Element("P", 15, 30.97376199842, 1.07),  # 31P
    Element("S", 16, 31.9720711744, 1.05),  # 32S
    Element("Cl", 17, 34.968852682, 1.02),  # 35Cl
    Element("Ar", 18, 39.9623831237, 1.06),  # 40Ar
    Element("K", 19, 38.9637064864, 2.03),  # 39K
    Element("Ca", 20, 39.962590863, 1.76),  # 40Ca
    Element("Sc", 21, 44.95590828, 1.70),  # 45Sc
    Element("Ti", 22, 47.94794198, 1.60),  # 48Ti
    Element("V", 23, 50.94395704, 1.53),  # 51V
    Element("Cr", 24, 51.94050623, 1.39),  # 52Cr
    Element("Mn", 25, 54.93804391, 1.39),  # 55Mn
    Element("Fe", 26, 55.93493633, 1.32),  # 56Fe
    Element("Co", 27, 58.93319429, 1.26),  # 59Co
    Element("Ni", 28, 57.93534241, 1.24),  # 58Ni
    Element("Cu", 29, 62.92959772, 1.32),  # 63Cu
    Element("Zn", 30, 63.92914201, 1.22),  # 64Zn
    Element("Ga", 31, 68.9255735, 1.22),  # 69Ga
    Element("Ge", 32, 73.921177761, 1.20),  # 74Ge
    Element("As", 33, 74.92159457, 1.19),  # 75As
    Element("Se", 34, 79.9165218, 1.20),  # 80Se
    Element("Br", 35, 78.9183376, 1.20),  # 79Br
    Element("Kr", 36, 83.9114977282, 1.16),  # 84Kr
)

_BY_SYMBOL = {e.symbol: e for e in _TABLE}


def canonical_symbol(symbol: str) -> str:
    """Return ``symbol`` with the first letter upper case and the rest lower case."""
    return symbol.strip().capitalize()


def element(symbol: str) -> Element:
    """Look up an element by symbol (case-insensitive).

    Raises ``ValueError`` for unknown symbols and for elements beyond Kr.
    """
    try:
        return _BY_SYMBOL[canonical_symbol(symbol)]
    except KeyError:
        raise ValueError(
            f"unknown or unsupported element {symbol!r} (pyvtst supports H-Kr)"
        ) from None


def formula(symbols: Iterable[str]) -> str:
    """Return the Hill-order formula of ``symbols``, e.g. ``"CH4"`` or ``"H2O"``.

    Carbon comes first and hydrogen second when carbon is present; every other
    element (and hydrogen without carbon) follows alphabetically.
    """
    counts = Counter(canonical_symbol(s) for s in symbols)
    order = sorted(counts)
    if "C" in counts:
        order = (
            ["C"]
            + (["H"] if "H" in counts else [])
            + [s for s in order if s not in ("C", "H")]
        )
    return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order)
