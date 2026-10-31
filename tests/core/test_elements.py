import pytest

from pyvtst.core.elements import element


def test_carbon_12():
    assert element("C").mass == 12.0
    assert element("C").atomic_number == 6


def test_hydrogen_1():
    assert abs(element("H").mass - 1.00782503) < 1e-6


def test_case_insensitive():
    assert element("cl").symbol == "Cl"


def test_table_covers_h_to_kr():
    assert element("Kr").atomic_number == 36


@pytest.mark.parametrize("symbol", ["Xx", "Rb", ""])
def test_unknown_symbol(symbol):
    with pytest.raises(ValueError, match="element"):
        element(symbol)
