"""Loading and validating input files, without running them."""

import math
from pathlib import Path

import numpy as np
import pytest

from pyvtst.core.units import ANGSTROM_TO_BOHR
from pyvtst.inputfile import Field, InputError, load
from pyvtst.inputfile.schema import _convert
from pyvtst.io import read_xyz
from tests.inputfile.conftest import PYSCF_INPUT

TS_INLINE = '''[ts]
mult = 2
symmetry_number = 2
xyz = """
H -0.85 0.0 0.0
H  0.0  0.0 0.0
H  1.0  0.0 0.0
"""
'''


def load_errors(write, text):
    path = write(text)
    with pytest.raises(InputError) as info:
        load(path)
    return info.value.errors


def replace_ts(table):
    """PYSCF_INPUT with its [ts] table replaced by ``table``."""
    assert TS_INLINE in PYSCF_INPUT
    return PYSCF_INPUT.replace(TS_INLINE, table)


def with_rates(rates):
    """PYSCF_INPUT with its [[rates]] job replaced by ``rates``."""
    return PYSCF_INPUT.replace("[[rates]]\nT = 300\n", rates)


def only_error(errors, fragment):
    assert len(errors) == 1, errors
    assert fragment in errors[0], errors[0]
    return errors[0]


# Structure


def test_minimal_valid_input(write):
    spec = load(write(PYSCF_INPUT))
    assert len(spec.reactants) == 2
    assert spec.ts.n_atoms == 3
    assert spec.products is None
    assert spec.optimizer is None
    assert spec.store is None
    assert spec.freq_scale == 1.0
    assert spec.mep is None
    assert spec.title is None
    assert len(spec.rates) == 1
    job = spec.rates[0]
    assert (job.T, job.theory, job.tunneling) == ((300.0,), "tst", None)
    assert spec.engine.type == "pyscf"
    assert spec.engine.kwargs == {
        "method": "B3LYP",
        "basis": "def2-SVP",
        "reference": "auto",
        "hessian": "auto",
        "fd_step": 5e-3,
        "conv_tol": 1e-10,
        "grid_level": 3,
        "max_cycle": 50,
        "max_memory_mb": None,
        "verbose": False,
    }
    assert not spec.needs_path


def test_invalid_toml_syntax(write):
    lines = PYSCF_INPUT.splitlines()
    lines[6] = 'title = "unterminated'
    error = only_error(load_errors(write, "\n".join(lines)), "h3.toml")
    assert "invalid TOML" in error
    assert "line 7" in error


def test_missing_input_file(tmp_path):
    with pytest.raises(InputError) as info:
        load(tmp_path / "nothere.toml")
    only_error(info.value.errors, "nothere.toml: no such file")


def test_missing_rate_jobs(write):
    errors = load_errors(write, PYSCF_INPUT.replace("[[rates]]\nT = 300\n", ""))
    only_error(errors, "rates: missing required key")


R2_TABLE = '[reactants.r2]\nmult = 2\nxyz = "H 0 0 0"\n'


def test_three_reactants(write):
    third = '[reactants.r3]\nmult = 2\nxyz = "H 0 0 0"\n\n[ts]'
    errors = load_errors(write, PYSCF_INPUT.replace("[ts]", third, 1))
    only_error(
        errors, "reactants.r3: a reaction has one or two reactants (r1 and r2), got r3"
    )


def test_array_of_tables_is_rejected(write):
    text = PYSCF_INPUT.replace("[reactants.r1]", "[[reactants]]").replace(
        "[reactants.r2]", "[[reactants]]"
    )
    errors = load_errors(write, text)
    only_error(errors, "write numbered tables [reactants.r1], [reactants.r2], not")


def test_keys_directly_under_reactants(write):
    text = PYSCF_INPUT.replace(
        "[reactants.r1]\n", "[reactants]\nmult = 1\n\n[reactants.r1]\n"
    )
    error = only_error(load_errors(write, text), "reactants.mult: unknown key")
    assert "[reactants.r1], [reactants.r2]" in error


def test_numbering_gap(write):
    text = PYSCF_INPUT.replace("[reactants.r2]", "[reactants.r3]")
    errors = load_errors(write, text)
    assert any("reactants.r3: a reaction has one or two reactants" in e for e in errors)
    text = PYSCF_INPUT.replace("[reactants.r1]", "[reactants.r0]")
    errors = load_errors(write, text)
    assert any("reactants.r0: unknown key" in e for e in errors)
    assert any("reactants.r1: missing" in e for e in errors)


def test_single_reactant_must_be_r1(write):
    text = PYSCF_INPUT.replace("[reactants.r1]", "[reactants.r2]", 1).replace(
        R2_TABLE, ""
    )
    errors = load_errors(write, text)
    only_error(
        errors, "reactants.r1: missing: number the reactants from r1 without gaps"
    )


def test_reactants_ordered_by_number(write):
    r1_start = PYSCF_INPUT.index("[reactants.r1]")
    r2_start = PYSCF_INPUT.index("[reactants.r2]")
    ts_start = PYSCF_INPUT.index("[ts]")
    r1 = PYSCF_INPUT[r1_start:r2_start]
    r2 = PYSCF_INPUT[r2_start:ts_start]
    text = PYSCF_INPUT[:r1_start] + r2 + r1 + PYSCF_INPUT[ts_start:]
    spec = load(write(text))
    assert [m.name for m in spec.reactants] == ["r1", "r2"]
    assert [m.n_atoms for m in spec.reactants] == [2, 1]


def test_freq_scale_must_be_positive(write):
    errors = load_errors(write, "freq_scale = 0\n" + PYSCF_INPUT)
    only_error(errors, "freq_scale: must be finite and > 0")


def test_title_and_freq_scale(write):
    spec = load(write('title = "H3"\nfreq_scale = 0.97\n' + PYSCF_INPUT))
    assert spec.title == "H3"
    assert spec.freq_scale == 0.97


# Validation


def test_misspelled_key(write):
    text = PYSCF_INPUT.replace(
        "[reactants.r1]\nsymmetry_number", "[reactants.r1]\nmultt = 1\nsymmetry_number"
    )
    error = only_error(load_errors(write, text), "reactants.r1.multt: unknown key")
    assert "did you mean 'mult'?" in error


def test_wrong_type(write):
    text = PYSCF_INPUT.replace("[reactants.r2]\nmult = 2", '[reactants.r2]\nmult = "2"')
    error = only_error(load_errors(write, text), "reactants.r2.mult")
    assert "expected an integer, found a string" in error


def test_several_errors_at_once(write):
    text = replace_ts(TS_INLINE.replace("mult = 2", "mult = 2\ncolour = 1"))
    text = text.replace("T = 300", "T = -5")
    errors = load_errors(write, text)
    assert len(errors) == 2
    assert any("ts.colour: unknown key" in e for e in errors)
    assert any("rates[0].T: temperatures must be finite and > 0 K" in e for e in errors)


def test_unbalanced_reaction(write):
    text = replace_ts('[ts]\nxyz = """\nH 0 0 0\nH 1 0 0\nH 2 0 0\nH 3 0 0\n"""\n')
    error = only_error(load_errors(write, text), "ts: element composition differs")
    assert "H3" in error
    assert "H4" in error


def test_inconsistent_multiplicity(write):
    text = PYSCF_INPUT.replace("[reactants.r2]\nmult = 2", "[reactants.r2]\nmult = 1")
    only_error(
        load_errors(write, text), "reactants.r2.mult: multiplicity 1 is inconsistent"
    )


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("str", "abc", "abc"),
        ("int", 3, 3),
        ("int", 3.0, 3),
        ("float", 2, 2.0),
        ("float", 2.5, 2.5),
        ("bool", True, True),
        ("floats", [1, 2.5], (1.0, 2.5)),
        ("list", [1, "a"], [1, "a"]),
        ("table", {"a": 1}, {"a": 1}),
        ("any", "x", "x"),
    ],
)
def test_field_kind_accepts(kind, value, expected):
    assert _convert(value, Field("k", kind)) == (expected, None)


@pytest.mark.parametrize(
    ("kind", "value", "fragment"),
    [
        ("str", 1, "expected a string, found an integer"),
        ("int", True, "expected an integer, found a boolean"),
        ("int", 2.5, "expected an integer, found a float"),
        ("float", False, "expected a number, found a boolean"),
        ("float", "1", "expected a number, found a string"),
        ("float", math.inf, "must be finite"),
        ("bool", 1, "expected a boolean, found an integer"),
        ("floats", [1, "a"], "element 1 is not a finite number"),
        ("floats", 1.0, "expected an array of numbers, found a float"),
        ("list", {}, "expected an array, found a table"),
        ("table", [], "expected a table, found an array"),
    ],
)
def test_field_kind_rejects(kind, value, fragment):
    converted, problem = _convert(value, Field("k", kind))
    assert converted is None
    assert problem is not None
    assert fragment in problem


def test_choices_match_case_insensitively():
    f = Field("k", "str", choices=("GAU_TIGHT", "tric"))
    assert _convert("gau_tight", f) == ("GAU_TIGHT", None)
    problem = _convert("x", f)[1]
    assert problem is not None
    assert "expected one of 'GAU_TIGHT', 'tric', got 'x'" in problem


# Molecules


def test_inline_geometry_in_angstrom_by_default(write):
    spec = load(write(PYSCF_INPUT))
    np.testing.assert_allclose(
        spec.reactants[0].coords[1], [0.74 * ANGSTROM_TO_BOHR, 0, 0], rtol=0, atol=1e-12
    )


def test_inline_geometry_in_bohr(write):
    text = PYSCF_INPUT.replace(
        "symmetry_number = 2\nxyz", 'symmetry_number = 2\nunits = "bohr"\nxyz', 1
    )
    text = text.replace("H 0.74 0.0 0.0", "H 1.4 0.0 0.0")
    spec = load(write(text))
    np.testing.assert_allclose(spec.reactants[0].coords[1], [1.4, 0, 0], atol=1e-12)


def test_geometry_from_xyz_file(write, tmp_path):
    xyz = write("3\nH3 guess\nH -0.85 0 0\nH 0 0 0\nH 1.0 0 0\n", "h3.xyz")
    spec = load(write(replace_ts('[ts]\nmult = 2\nfile = "h3.xyz"\n')))
    assert spec.ts == read_xyz(xyz, mult=2)
    assert spec.ts.name == "H3 guess"


def test_both_geometry_sources(write):
    write("3\n\nH 0 0 0\nH 1 0 0\nH 2 0 0\n", "h3.xyz")
    text = replace_ts(TS_INLINE.replace("mult = 2", 'mult = 2\nfile = "h3.xyz"'))
    only_error(
        load_errors(write, text), "ts: exactly one of 'xyz' and 'file' is required"
    )


def test_neither_geometry_source(write):
    text = replace_ts("[ts]\nmult = 2\n")
    only_error(
        load_errors(write, text), "ts: exactly one of 'xyz' and 'file' is required"
    )


def test_units_with_a_file(write):
    write("2\n\nH 0 0 0\nH 0.74 0 0\n", "h2.xyz")
    text = PYSCF_INPUT.replace(
        "[reactants.r1]\nsymmetry_number = 2\n"
        'xyz = """\nH 0.0  0.0 0.0\nH 0.74 0.0 0.0\n"""',
        '[reactants.r1]\nfile = "h2.xyz"\nunits = "bohr"',
    )
    only_error(
        load_errors(write, text), "reactants.r1.units: XYZ files are always in angstrom"
    )


def test_malformed_inline_line(write):
    text = replace_ts(TS_INLINE.replace("H  1.0  0.0 0.0", "H 0.0 abc 0.0"))
    error = only_error(
        load_errors(write, text), "ts.xyz: line 3: non-numeric coordinate"
    )
    assert "'H 0.0 abc 0.0'" in error


def test_inline_line_with_extra_columns(write):
    text = replace_ts(TS_INLINE.replace("H  1.0  0.0 0.0", "H 1.0 0.0 0.0 7"))
    only_error(load_errors(write, text), "ts.xyz: line 3: expected 'symbol x y z'")


def test_unknown_element(write):
    text = replace_ts(TS_INLINE.replace("H  1.0  0.0 0.0", "Xx 1.0 0.0 0.0"))
    only_error(load_errors(write, text), "ts.xyz:")


def test_masses_for_an_isotopologue(write):
    text = PYSCF_INPUT.replace(
        "[reactants.r1]\nsymmetry_number = 2",
        "[reactants.r1]\nmasses = [2.014101778, 1.00782503207]",
    )
    spec = load(write(text))
    np.testing.assert_array_equal(
        spec.reactants[0].masses, [2.014101778, 1.00782503207]
    )


def test_wrong_mass_count(write):
    text = replace_ts(TS_INLINE.replace("mult = 2", "mult = 2\nmasses = [1.0, 1.0]"))
    only_error(
        load_errors(write, text), "ts.masses: the molecule has 3 atoms but 2 masses"
    )


def test_default_names(write):
    spec = load(write(PYSCF_INPUT))
    assert [m.name for m in spec.reactants] == ["r1", "r2"]
    assert spec.ts.name == "ts"


def test_missing_geometry_file(write, tmp_path):
    error = only_error(
        load_errors(write, replace_ts('[ts]\nmult = 2\nfile = "missing.xyz"\n')),
        "ts.file: no such file",
    )
    assert str(tmp_path / "missing.xyz") in error


def test_products(write):
    products = (
        '[products.p1]\nsymmetry_number = 2\nxyz = """\nH 0 0 0\nH 0.74 0 0\n"""\n\n'
    )
    products += '[products.p2]\nmult = 2\nxyz = "H 0 0 0"\n\n[[rates]]'
    spec = load(write(PYSCF_INPUT.replace("[[rates]]", products)))
    assert spec.products is not None
    assert len(spec.products) == 2


# Paths


def test_run_from_another_directory(write, tmp_path, monkeypatch):
    write('store = "h3.h5"\n' + PYSCF_INPUT, "calc/h3.toml")
    monkeypatch.chdir(tmp_path)
    spec = load("calc/h3.toml")
    assert spec.store == tmp_path.resolve() / "calc" / "h3.h5"


def test_home_directory_in_paths(write):
    spec = load(write('store = "~/h3.h5"\n' + PYSCF_INPUT))
    assert spec.store == Path.home() / "h3.h5"


# Reaction path


MEP = "[mep]\n{}\n\n[[rates]]"


def with_mep(body):
    return PYSCF_INPUT.replace("[[rates]]", MEP.format(body))


def test_no_mep_section(write):
    assert load(write(PYSCF_INPUT)).mep is None


def test_mep_defaults_filled_in(write):
    spec = load(write(with_mep("s_min = -1.0")))
    assert spec.mep is not None
    assert spec.mep.kwargs == {
        "s_min": -1.0,
        "s_max": 0.5,
        "ds": 0.01,
        "method": "lqa",
        "hessian_every": 1,
        "coordinates": "redundant",
        "internals": None,
    }
    assert spec.needs_path


def test_internal_coordinates(write):
    spec = load(
        write(
            with_mep(
                'internals = [["bond", 0, 1], ["bond", 1, 2], ["linear_bend", 0, 1, 2]]'
            )
        )
    )
    assert spec.mep is not None
    assert spec.mep.internals == (
        ("bond", 0, 1),
        ("bond", 1, 2),
        ("linear_bend", 0, 1, 2),
    )


def test_malformed_internal(write):
    errors = load_errors(write, with_mep('internals = [["bond", 0]]'))
    only_error(
        errors, "mep.internals[0]: ('bond', 0): a bond takes 2 atom indices, got 1"
    )


def test_mep_step_out_of_range(write):
    only_error(load_errors(write, with_mep("ds = 5")), "mep.ds: must be in (0, 0.1]")


def test_internals_with_cartesian_coordinates(write):
    body = 'coordinates = "cartesian"\ninternals = [["bond", 0, 1]]'
    only_error(load_errors(write, with_mep(body)), "mep.internals: only used with")


# Rate jobs


def test_temperature_range(write):
    spec = load(
        write(with_rates("[[rates]]\nT = { start = 200, stop = 1000, step = 200 }\n"))
    )
    assert spec.rates[0].T == (200.0, 400.0, 600.0, 800.0, 1000.0)


def test_temperature_range_stops_before_stop(write):
    spec = load(
        write(with_rates("[[rates]]\nT = { start = 200, stop = 1000, step = 300 }\n"))
    )
    assert spec.rates[0].T == (200.0, 500.0, 800.0)


def test_temperature_range_checks(write):
    errors = load_errors(
        write, with_rates("[[rates]]\nT = { start = 0, stop = -1, step = 0 }\n")
    )
    assert len(errors) == 3


def test_non_positive_temperature(write):
    errors = load_errors(write, with_rates("[[rates]]\nT = [300, -5]\n"))
    only_error(errors, "rates[0].T[1]: temperatures must be finite and > 0 K")


@pytest.mark.parametrize("value", ["true", '"300"', "[]"])
def test_temperature_type(write, value):
    load_errors(write, with_rates(f"[[rates]]\nT = {value}\n"))


def test_unknown_theory(write):
    errors = load_errors(write, with_rates('[[rates]]\nT = 300\ntheory = "us"\n'))
    error = only_error(errors, "rates[0].theory: expected one of")
    assert "'muvt'" in error


def test_explicit_no_tunneling_and_case(write):
    rates = '[[rates]]\nT = 300\ntheory = "CVT"\ntunneling = "none"\n'
    spec = load(write(with_rates(rates)))
    assert (spec.rates[0].theory, spec.rates[0].tunneling) == ("cvt", None)
    assert spec.needs_path


# Python API


def test_load_errors_as_an_exception(write):
    text = PYSCF_INPUT.replace("T = 300", "T = 0\ncolour = 1")
    with pytest.raises(ValueError, match="rates") as info:
        load(write(text))
    assert isinstance(info.value, InputError)
    assert len(info.value.errors) == 2
    assert str(info.value) == "\n".join(info.value.errors)


# Documentation examples


DOCS = Path(__file__).resolve().parents[2]


def toml_blocks(path):
    text = path.read_text(encoding="utf-8")
    return [block.split("```")[0] for block in text.split("```toml\n")[1:]]


@pytest.mark.parametrize("doc", ["docs/user/inputfile.md", "README.md"])
def test_documented_examples_load(write, doc):
    (example, *_) = toml_blocks(DOCS / doc)
    spec = load(write(example))
    assert spec.engine.type == "pyscf"
    assert len(spec.reactants) == 2
