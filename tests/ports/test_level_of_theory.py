import dataclasses
import itertools
import subprocess
import sys

import numpy as np
import pytest

from pyvtst.ports import LevelOfTheory


def test_case_and_order_do_not_matter():
    a = LevelOfTheory("B3LYP", "def2-SVP", (("grid", 3), ("conv_tol", 1e-10)))
    b = LevelOfTheory("b3lyp", "DEF2-svp", (("conv_tol", 1e-10), ("grid", 3)))
    assert a == b
    assert hash(a) == hash(b)
    assert a.key() == b.key()


def test_names_are_stripped():
    assert LevelOfTheory(" HF ", " STO-3G ") == LevelOfTheory("hf", "sto-3g")


def test_different_options_are_different_levels():
    a = LevelOfTheory("b3lyp", "def2-svp", (("grid", 3),))
    b = LevelOfTheory("b3lyp", "def2-svp", (("grid", 4),))
    assert a != b
    assert a.key() != b.key()


def test_not_equal_to_other_types():
    level = LevelOfTheory("hf", "sto-3g")
    assert level != "hf/sto-3g"
    assert level.__eq__("hf/sto-3g") is NotImplemented


@pytest.mark.parametrize("item", ["grid", ("grid",), ("grid", 3, 4), 3])
def test_option_must_be_a_pair(item):
    with pytest.raises(ValueError, match=r"\(name, value\) pairs"):
        LevelOfTheory("hf", "sto-3g", (item,))


def test_duplicate_option_rejected():
    with pytest.raises(ValueError, match="grid"):
        LevelOfTheory("b3lyp", "def2-svp", (("grid", 3), ("grid", 4)))


@pytest.mark.parametrize("value", [[1], None, 1j, float("nan"), float("inf")])
def test_invalid_option_value_rejected(value):
    with pytest.raises(ValueError, match="tol"):
        LevelOfTheory("hf", "sto-3g", (("tol", value),))


@pytest.mark.parametrize("name", ["", 3])
def test_invalid_option_name_rejected(name):
    with pytest.raises(ValueError, match="name"):
        LevelOfTheory("hf", "sto-3g", ((name, 1),))


@pytest.mark.parametrize(("method", "basis"), [("", "sto-3g"), ("hf", "  ")])
def test_empty_method_or_basis_rejected(method, basis):
    with pytest.raises(ValueError, match="non-empty string"):
        LevelOfTheory(method, basis)


def test_key_format():
    level = LevelOfTheory("B3LYP", "def2-SVP", (("grid_level", 3), ("conv_tol", 1e-10)))
    assert level.key() == "b3lyp/def2-svp?conv_tol=f:1e-10&grid_level=i:3"
    assert LevelOfTheory("HF", "STO-3G").key() == "hf/sto-3g"


def test_value_types_are_distinguished():
    levels = [LevelOfTheory("hf", "sto-3g", (("x", v),)) for v in (1, 1.0, True)]
    assert len({lv.key() for lv in levels}) == 3
    for a, b in itertools.combinations(levels, 2):
        assert a != b


def test_bool_and_str_rendering():
    level = LevelOfTheory("hf", "sto-3g", (("a", False), ("b", "x")))
    assert level.key() == "hf/sto-3g?a=b:false&b=s:x"


def test_numpy_scalars_render_as_python_numbers():
    a = LevelOfTheory("hf", "sto-3g", (("x", np.float64(0.1)),))
    b = LevelOfTheory("hf", "sto-3g", (("x", 0.1),))
    assert a.key() == b.key()
    assert type(a.options[0][1]) is float


def test_key_syntax_in_values_cannot_collide():
    a = LevelOfTheory("hf", "sto-3g", (("a", "1&b=s:2"),))
    b = LevelOfTheory("hf", "sto-3g", (("a", "1"), ("b", "2")))
    assert a.key() != b.key()
    assert LevelOfTheory("a/b", "c").key() != LevelOfTheory("a", "b/c").key()


def test_readable_basis_names_are_not_escaped():
    assert LevelOfTheory("hf", "6-311+G(d,p)").key() == "hf/6-311+g(d,p)"
    assert LevelOfTheory("hf", "6-31G**").key() == "hf/6-31g**"


def test_immutable():
    level = LevelOfTheory("hf", "sto-3g")
    with pytest.raises(dataclasses.FrozenInstanceError):
        level.method = "mp2"  # type: ignore[misc]


def test_key_is_stable_across_processes():
    script = (
        "from pyvtst.ports import LevelOfTheory\n"
        "print(LevelOfTheory('B3LYP', 'def2-SVP', "
        "(('grid', 3), ('conv_tol', 1e-10), ('x', True), ('y', 'z'))).key())"
    )
    keys = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    expected = LevelOfTheory(
        "B3LYP", "def2-SVP", (("grid", 3), ("conv_tol", 1e-10), ("x", True), ("y", "z"))
    ).key()
    assert keys == {expected}
