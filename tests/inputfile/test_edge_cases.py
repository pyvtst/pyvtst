"""Less common input mistakes and report sections."""

import pytest

from pyvtst.inputfile import DEFAULT_REGISTRY, InputError, load, run
from tests.inputfile.conftest import H3_INPUT, PYSCF_INPUT

TS_XYZ = '"""\nH -0.85 0.0 0.0\nH  0.0  0.0 0.0\nH  1.0  0.0 0.0\n"""'


def load_errors(write, text):
    with pytest.raises(InputError) as info:
        load(write(text))
    return info.value.errors


def test_unreadable_input(tmp_path):
    (tmp_path / "dir.toml").mkdir()
    with pytest.raises(InputError, match="cannot read the file"):
        load(tmp_path / "dir.toml")


def test_date_value(write):
    errors = load_errors(write, "title = 1979-05-27\n" + PYSCF_INPUT)
    assert "title: expected a string, found a date" in errors[0]


@pytest.mark.parametrize(
    ("reactants", "fragment"),
    [
        (
            'reactants = "H2"',
            "reactants: expected tables [reactants.r1], [reactants.r2], found a string",
        ),
        ("reactants = [1]", "reactants: write numbered tables"),
        ("reactants = {}", "reactants: expected at least one molecule"),
        (
            "reactants = { r1 = 1 }",
            "reactants.r1: expected a table [reactants.r1], found an integer",
        ),
    ],
)
def test_reactants_shape(write, reactants, fragment):
    text = PYSCF_INPUT.replace("[reactants.", "[unused.")
    errors = load_errors(write, reactants + "\n" + text)
    assert any(fragment in e for e in errors), errors


def test_products_are_not_limited_to_two(write):
    products = "".join(
        f'[products.p{i}]\nmult = 2\nxyz = "H 0 0 0"\n\n' for i in (1, 2, 3)
    )
    spec = load(write(PYSCF_INPUT.replace("[[rates]]", products + "[[rates]]")))
    assert spec.products is not None
    assert [m.name for m in spec.products] == ["p1", "p2", "p3"]


def test_adapter_type_not_a_string(write):
    errors = load_errors(write, PYSCF_INPUT.replace('type = "pyscf"', "type = 1"))
    assert "engine.type: expected a string, found an integer" in errors[0]


def test_rate_job_not_a_table(write):
    text = "rates = [300]\n" + PYSCF_INPUT.replace("[[rates]]\nT = 300\n", "")
    errors = load_errors(write, text)
    assert "rates[0]: expected a table, found an integer" in errors[0]


def test_temperature_range_wrong_type(write):
    text = PYSCF_INPUT.replace(
        "T = 300", 'T = { start = "200", stop = 1000, step = 100 }'
    )
    errors = load_errors(write, text)
    assert "rates[0].T.start: expected a number, found a string" in errors[0]


def test_temperature_list_wrong_type(write):
    errors = load_errors(write, PYSCF_INPUT.replace("T = 300", "T = [300, true]"))
    assert "rates[0].T[1]: expected a number, found a boolean" in errors[0]


def test_blank_lines_in_inline_geometry(write):
    spec = load(write(PYSCF_INPUT.replace("H 0.0  0.0 0.0\n", "\nH 0.0  0.0 0.0\n\n")))
    assert spec.reactants[0].n_atoms == 2


def test_empty_inline_geometry(write):
    errors = load_errors(write, PYSCF_INPUT.replace('xyz = "H 0 0 0"', 'xyz = " "'))
    assert "reactants.r2.xyz: no atoms" in errors[0]


def test_file_with_masses_and_no_comment(write):
    write("3\n\nH -0.85 0 0\nH 0 0 0\nH 1.0 0 0\n", "h3.xyz")
    text = PYSCF_INPUT.replace(
        "[ts]\nmult = 2\nsymmetry_number = 2\nxyz = " + TS_XYZ,
        '[ts]\nmult = 2\nfile = "h3.xyz"\nmasses = [1.0, 2.0, 1.0]',
    )
    spec = load(write(text))
    assert spec.ts.name == "ts"
    assert list(spec.ts.masses) == [1.0, 2.0, 1.0]


@pytest.mark.parametrize(
    ("extra", "xyz", "fragment"),
    [
        (
            "masses = [1.0]",
            "3\n\nH 0 0 0\nH 1 0 0\nH 2 0 0\n",
            "ts.masses: the molecule",
        ),
        ("masses = [1.0, -1.0, 1.0]", "3\n\nH 0 0 0\nH 1 0 0\nH 2 0 0\n", "ts.masses:"),
        ("", "3\n\nH 0 0 0\nH 1 x 0\nH 2 0 0\n", "ts.file:"),
    ],
)
def test_file_errors(write, extra, xyz, fragment):
    write(xyz, "h3.xyz")
    text = PYSCF_INPUT.replace(
        "[ts]\nmult = 2\nsymmetry_number = 2\nxyz = " + TS_XYZ,
        f'[ts]\nmult = 2\nfile = "h3.xyz"\n{extra}',
    )
    errors = load_errors(write, text)
    assert any(fragment in e for e in errors), errors


def test_internals_with_an_invalid_ts(write):
    text = PYSCF_INPUT.replace("[ts]\nmult = 2", "[ts]\nmult = 1").replace(
        "[[rates]]", '[mep]\ninternals = [["bond", 0, 1]]\n\n[[rates]]'
    )
    errors = load_errors(write, text)
    assert len(errors) == 1
    assert "ts.mult" in errors[0]


def test_geometric_entry_builds_the_optimizer(monkeypatch):
    built: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr("pyvtst.optimizers.GeometricOptimizer", Recorder)
    DEFAULT_REGISTRY.optimizers["geometric"].build({"convergence": "GAU"})
    assert built == {"convergence": "GAU"}


def test_report_with_path_and_products(write, registry, tmp_path):
    products = (
        '[products.p1]\nname = "H2 product"\nunits = "bohr"\n'
        'xyz = """\nH 0.0 0.0 0.0\nH 1.8 0.0 0.0\n"""\n\n'
        '[products.p2]\nname = "H product"\nmult = 2\nxyz = "H 0 0 0"\n\n'
    )
    text = H3_INPUT.replace("[[rates]]", products + "[[rates]]").replace(
        'tunneling = "eckart"', 'tunneling = "zct"'
    )
    output = tmp_path / "h3.out"
    with pytest.warns(UserWarning, match="the path ends"):
        run(write(text), output=output, registry=registry)
    report = output.read_text()
    assert "reaction path: compute_mep() defaults" in report
    assert "REACTION PATH" in report
    assert "s from -0.5000 to 0.5000 bohr, 101 points, method lqa" in report
    assert "product 'H2 product'" in report
    assert "product 'H product'" in report
