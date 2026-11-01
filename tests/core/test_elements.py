import pytest

from pyvtst.core.elements import _TABLE, element


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


@pytest.mark.parametrize(("symbol", "radius"), [("H", 0.31), ("C", 0.76), ("O", 0.66)])
def test_covalent_radius(symbol, radius):
    assert element(symbol).covalent_radius == radius


def test_every_element_has_a_positive_covalent_radius():
    assert len(_TABLE) == 36
    assert all(e.covalent_radius > 0 for e in _TABLE)
