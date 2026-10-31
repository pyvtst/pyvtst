"""Element table for H through Kr (Z = 1-36).

Each entry gives the element symbol, atomic number and the mass of its most
abundant isotope in amu. Masses are from the NIST "Atomic Weights and Isotopic
Compositions for All Elements" tables (J. S. Coursey et al., NIST Standard
Reference Database 144, https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses),
which take relative atomic masses from the Atomic Mass Evaluation. They are
tabulated data, not physical constants, so they are not taken from
``scipy.constants``.
"""

from typing import NamedTuple


class Element(NamedTuple):
    symbol: str
    atomic_number: int
    mass: float  # most abundant isotope, amu


_TABLE = (
    Element("H", 1, 1.00782503223),  # 1H
    Element("He", 2, 4.00260325413),  # 4He
    Element("Li", 3, 7.0160034366),  # 7Li
    Element("Be", 4, 9.012183065),  # 9Be
    Element("B", 5, 11.00930536),  # 11B
    Element("C", 6, 12.0),  # 12C
    Element("N", 7, 14.00307400443),  # 14N
    Element("O", 8, 15.99491461957),  # 16O
    Element("F", 9, 18.99840316273),  # 19F
    Element("Ne", 10, 19.9924401762),  # 20Ne
    Element("Na", 11, 22.989769282),  # 23Na
    Element("Mg", 12, 23.985041697),  # 24Mg
    Element("Al", 13, 26.98153853),  # 27Al
    Element("Si", 14, 27.97692653465),  # 28Si
    Element("P", 15, 30.97376199842),  # 31P
    Element("S", 16, 31.9720711744),  # 32S
    Element("Cl", 17, 34.968852682),  # 35Cl
    Element("Ar", 18, 39.9623831237),  # 40Ar
    Element("K", 19, 38.9637064864),  # 39K
    Element("Ca", 20, 39.962590863),  # 40Ca
    Element("Sc", 21, 44.95590828),  # 45Sc
    Element("Ti", 22, 47.94794198),  # 48Ti
    Element("V", 23, 50.94395704),  # 51V
    Element("Cr", 24, 51.94050623),  # 52Cr
    Element("Mn", 25, 54.93804391),  # 55Mn
    Element("Fe", 26, 55.93493633),  # 56Fe
    Element("Co", 27, 58.93319429),  # 59Co
    Element("Ni", 28, 57.93534241),  # 58Ni
    Element("Cu", 29, 62.92959772),  # 63Cu
    Element("Zn", 30, 63.92914201),  # 64Zn
    Element("Ga", 31, 68.9255735),  # 69Ga
    Element("Ge", 32, 73.921177761),  # 74Ge
    Element("As", 33, 74.92159457),  # 75As
    Element("Se", 34, 79.9165218),  # 80Se
    Element("Br", 35, 78.9183376),  # 79Br
    Element("Kr", 36, 83.9114977282),  # 84Kr
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
