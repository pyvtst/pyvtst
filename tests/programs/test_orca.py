"""ORCA keywords and blocks, result files and job runs."""

import re
import sys
import tempfile
from typing import Any

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.programs import orca
from pyvtst.programs.orca import (
    RunSettings,
    engrad_results,
    hess_result,
    normalize_blocks,
    normalize_keywords,
    parse_engrad,
    parse_hess,
    parse_xyz,
    run_job,
    xyz_coordinates,
)

# --- keywords ----------------------------------------------------------------


def test_keywords_fold_case_and_collapse_whitespace():
    assert (
        normalize_keywords("  D3BJ \t DefGrid3\n TightSCF ") == "d3bj defgrid3 tightscf"
    )


@pytest.mark.parametrize("keywords", ["", "   ", "D3BJ", "RIJCOSX def2/J", "SlowConv"])
def test_accepted_keywords(keywords):
    assert normalize_keywords(keywords) == " ".join(keywords.lower().split())


@pytest.mark.parametrize(
    "keyword",
    [
        "SP",
        "EnGrad",
        "NumGrad",
        "Opt",
        "TightOpt",
        "VeryTightOpt",
        "LooseOpt",
        "COpt",
        "OptTS",
        "L-OptTS",
        "ScanTS",
        "Freq",
        "AnFreq",
        "NumFreq",
        "IRC",
        "MD",
        "NEB",
        "NEB-TS",
        "GOAT",
        "GOAT-REACT",
        "Bohrs",
        "Angs",
        "PAL4",
        "pal16",
        "RHF",
        "UHF",
        "ROHF",
        "RKS",
        "UKS",
        "ROKS",
        "MORead",
        "UseSym",
        "tightopt",
    ],
)
def test_reserved_keywords_are_rejected(keyword):
    with pytest.raises(ValueError, match=keyword.replace("-", r"\-")):
        normalize_keywords(f"D3BJ {keyword}")


def test_keywords_comment_character_is_rejected():
    with pytest.raises(ValueError, match="#"):
        normalize_keywords("TightSCF # tight")


def test_keywords_must_be_a_string():
    with pytest.raises(ValueError, match="keywords"):
        normalize_keywords(["D3BJ"])  # type: ignore[arg-type]


# --- blocks ------------------------------------------------------------------


def test_blocks_keep_line_breaks_and_case():
    assert normalize_blocks("%scf\n  MaxIter   300\nend\n") == "%scf\nMaxIter 300\nend"
    assert normalize_blocks('%basis\n\n  GTOName "My-Basis.bas"\nend') == (
        '%basis\nGTOName "My-Basis.bas"\nend'
    )


def test_block_layout_does_not_change_the_text():
    assert normalize_blocks("%scf\n  MaxIter   300\nend\n") == normalize_blocks(
        "%scf\nMaxIter 300\nend"
    )


def test_no_blocks():
    assert normalize_blocks("") == ""
    assert normalize_blocks("  \n \n") == ""


@pytest.mark.parametrize(
    ("blocks", "needle"),
    [
        ("%pal nprocs 4 end", "%pal"),
        ("%scf MaxIter 3 end %PAL nprocs 4 end", "%PAL"),
        ("%maxcore 1000", "%maxcore"),
        ("%geom MaxIter 3 end", "%geom"),
        ("%freq TransInvar false end", "%freq"),
        ("%coords Units Angs end", "%coords"),
        ("%irc MaxIter 5 end", "%irc"),
        ("%neb NImages 5 end", "%neb"),
        ("%md Timestep 1 end", "%md"),
        ("%method\nRunTyp Opt\nend", "RunTyp"),
        ("%scf\nend\n! Opt", "'!'"),
        ("%scf\nend\n* xyz 0 1", "'*'"),
        ("%scf\nend\n$new_job", r"\$new_job"),
        ("MaxIter 3", "must start with '%'"),
        ("%scf\nMaxIter 3 # more\nend", "#"),
    ],
)
def test_conflicting_blocks_are_rejected(blocks, needle):
    with pytest.raises(ValueError, match=needle):
        normalize_blocks(blocks)


def test_pal_block_error_says_to_use_nprocs():
    with pytest.raises(ValueError, match="nprocs"):
        normalize_blocks("%pal nprocs 4 end")


def test_other_blocks_are_accepted():
    text = "%scf\nMaxIter 300\nend\n%method\nGrid 4\nend"
    assert normalize_blocks(text) == text


def test_blocks_must_be_a_string():
    with pytest.raises(ValueError, match="blocks"):
        normalize_blocks(3)  # type: ignore[arg-type]


# --- run settings ------------------------------------------------------------


@pytest.mark.parametrize(
    ("nprocs", "maxcore", "timeout", "needle"),
    [
        (0, 1000, None, "nprocs"),
        (True, 1000, None, "nprocs"),
        (1, 0, None, "maxcore"),
        (1, "2GB", None, "maxcore"),
        (1, 1000, 0, "timeout"),
        (1, 1000, "soon", "timeout"),
    ],
)
def test_bad_run_settings(nprocs, maxcore, timeout, needle):
    with pytest.raises(ValueError, match=needle):
        orca.check_run_settings(nprocs, maxcore, timeout)


def test_good_run_settings():
    orca.check_run_settings(4, 3000, 60.0)
    orca.check_run_settings(1, 1, None)


# --- result files ------------------------------------------------------------

H2O = Molecule(
    ["O", "H", "H"],
    [[0.1, -0.05, 0.08], [1.65, 0.35, -0.2], [-0.6, 1.55, 0.4]],
)


def engrad_text(mol=H2O, energy=-75.5, gradient=None, coords=None):
    gradient = np.arange(9.0).reshape(3, 3) * 0.01 if gradient is None else gradient
    coords = mol.coords if coords is None else coords
    numbers = {"O": 8, "H": 1, "C": 6, "N": 7}
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f" {mol.n_atoms}",
        "#",
        "# The current total energy in Eh",
        "#",
    ]
    lines.append(f"    {energy:.12f}")
    lines += ["#", "# The current gradient in Eh/bohr", "#"]
    lines += [f"   {g:.12f}" for g in np.asarray(gradient).ravel()]
    lines += ["#", "# The atomic numbers and current coordinates in Bohr", "#"]
    lines += [
        f"{numbers[s]:4d} {x:12.7f} {y:12.7f} {z:12.7f} "
        for s, (x, y, z) in zip(mol.symbols, coords, strict=True)
    ]
    return "\n".join(lines) + "\n\n"


def hess_text(hessian, symbols, coords, extra=""):
    n = hessian.shape[0]
    lines = [
        "",
        "$orca_hessian_file",
        "",
        "$act_energy",
        "      0.000000",
        "",
        "$hessian",
        str(n),
    ]
    for start in range(0, n, 5):
        cols = list(range(start, min(start + 5, n)))
        lines.append("".join(f"{c:>19d}" for c in cols) + "        ")
        for row in range(n):
            lines.append(
                f"{row:5d}" + "".join(f"{hessian[row, c]:19.10E}" for c in cols)
            )
    lines += [
        "",
        "$vibrational_frequencies",
        str(n),
        *[f"{i} 0.0" for i in range(n)],
        "",
    ]
    lines += [
        "#",
        "# The atoms: label  mass x y z (in bohrs)",
        "#",
        "$atoms",
        str(len(symbols)),
    ]
    lines += [
        f" {s:2s} {1.0:12.5f} {x:18.12f} {y:18.12f} {z:18.12f}"
        for s, (x, y, z) in zip(symbols, coords, strict=True)
    ]
    return "\n".join(lines) + "\n" + extra + "\n$end\n"


def random_hessian(n=9, seed=3):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a + a.T


def test_engrad_round_trip():
    gradient = np.random.default_rng(0).normal(size=(3, 3))
    energy, grad = engrad_results(
        parse_engrad(engrad_text(energy=-75.123456789012, gradient=gradient)), H2O
    )
    assert energy == -75.123456789012
    np.testing.assert_allclose(grad, gradient, atol=1e-12)


def test_engrad_gradient_is_a_new_array():
    engrad = parse_engrad(engrad_text())
    _, grad = engrad_results(engrad, H2O)
    grad[:] = 0.0
    assert np.any(engrad.gradient != 0.0)


def test_engrad_rotated_geometry_is_rejected():
    rotated = H2O.coords[:, [1, 0, 2]] * [1, -1, 1]
    with pytest.raises(ValueError, match="reoriented"):
        engrad_results(parse_engrad(engrad_text(coords=rotated)), H2O)


def test_engrad_other_atoms_are_rejected():
    other = Molecule(["N", "H", "H"], H2O.coords, mult=2)
    with pytest.raises(ValueError, match="atomic numbers"):
        engrad_results(parse_engrad(engrad_text(other)), H2O)


def test_engrad_rounding_of_input_is_within_tolerance():
    mol = Molecule(["O", "H", "H"], H2O.coords + 4e-8)
    engrad_results(parse_engrad(engrad_text(mol, coords=H2O.coords)), mol)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: "",
        lambda t: t.replace(" 3\n", " three\n", 1),
        lambda t: "\n".join(t.splitlines()[:-2]),
        lambda t: t.replace("-75.5", "abc"),
        lambda t: t.replace(" 3\n", " 4\n", 1),
        lambda t: t.replace("   8 ", "   8 9.0 ", 1),
        lambda t: t.replace("   8 ", "   O ", 1),
    ],
)
def test_bad_engrad_is_rejected(mutate):
    with pytest.raises(ValueError, match=r"job\.engrad"):
        parse_engrad(mutate(engrad_text()))


def test_hess_round_trip_with_centre_of_mass_shift():
    hessian = random_hessian()
    shifted = H2O.coords + np.array([0.1476, 0.0619, -0.0822])
    parsed = parse_hess(hess_text(hessian, H2O.symbols, shifted))
    np.testing.assert_allclose(hess_result(parsed, H2O), hessian, atol=1e-9)


def test_hess_with_two_column_blocks():
    n = 12
    mol = Molecule(["C", "N", "H", "H"], np.arange(12.0).reshape(4, 3) * 0.7, mult=2)
    hessian = random_hessian(n, seed=9)
    parsed = parse_hess(hess_text(hessian, mol.symbols, mol.coords))
    np.testing.assert_allclose(hess_result(parsed, mol), hessian, atol=1e-9)


def test_hess_is_symmetrized_and_a_new_array():
    hessian = random_hessian()
    hessian[0, 1] += 1e-3
    parsed = parse_hess(hess_text(hessian, H2O.symbols, H2O.coords))
    result = hess_result(parsed, H2O)
    np.testing.assert_allclose(result, result.T, atol=0)
    result[:] = 0.0
    assert np.any(parsed.hessian != 0.0)


def test_hess_rotated_geometry_is_rejected():
    rotated = H2O.coords[:, [1, 0, 2]] * [1, -1, 1]
    parsed = parse_hess(hess_text(random_hessian(), H2O.symbols, rotated))
    with pytest.raises(ValueError, match="translation"):
        hess_result(parsed, H2O)


def test_hess_other_atoms_are_rejected():
    parsed = parse_hess(hess_text(random_hessian(), ["N", "H", "H"], H2O.coords))
    with pytest.raises(ValueError, match="atomic numbers"):
        hess_result(parsed, H2O)


def test_hess_wrong_size_is_rejected():
    parsed = parse_hess(hess_text(random_hessian(6), H2O.symbols, H2O.coords))
    with pytest.raises(ValueError, match="9x9"):
        hess_result(parsed, H2O)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: "",
        lambda t: t.replace("$hessian", "$other"),
        lambda t: t.replace("$atoms", "$other"),
        lambda t: t.replace("\n9\n", "\nnine\n", 1),
        lambda t: "\n".join(t.splitlines()[:30]),
        lambda t: re.sub(r"(?m)^    3", "    7", t, count=1),
        lambda t: "\n".join(
            line for line in t.splitlines() if not line.startswith("    8 ")
        ),
        lambda t: t.rsplit("$atoms", 1)[0] + "$atoms\n3\n O 1.0 0.0 0.0\n",
        lambda t: t.rsplit("$atoms", 1)[0] + "$atoms\n2\n O 1.0 0 0 0\n H 1.0 0 0\n",
        lambda t: re.sub(r"(?m)^( +0 +1 +2 +3 +4) *$", " 0 1 2 3 99", t, count=1),
        lambda t: re.sub(r"(?m)^( +0 +1 +2 +3 +4) *$", " 0 one", t, count=1),
        lambda t: t.split("$vibrational")[0].rsplit("\n", 6)[0],
    ],
)
def test_bad_hess_is_rejected(mutate):
    text = hess_text(random_hessian(), H2O.symbols, H2O.coords)
    with pytest.raises(ValueError, match=r"job\.hess"):
        parse_hess(mutate(text))


def test_xyz_is_converted_to_bohr():
    text = (
        "2\nCoordinates from ORCA-job job E -1.0\n"
        " H 0.0 0.0 0.529177210903\n H 0.0 0.0 -0.529177210903\n"
    )
    symbols, coords = parse_xyz(text)
    assert symbols == ("H", "H")
    np.testing.assert_allclose(coords[:, 2], [1.0, -1.0], rtol=1e-8)


def test_xyz_atoms_are_checked():
    text = "3\ntitle\n O 0 0 0\n H 1 0 0\n H 0 1 0\n"
    assert xyz_coordinates(text, H2O).shape == (3, 3)
    with pytest.raises(ValueError, match="atomic numbers"):
        xyz_coordinates(text.replace("O 0", "N 0"), H2O)


@pytest.mark.parametrize(
    "text", ["", "x\ny\n", "3\ntitle\n O 0 0 0\n", "1\nt\n O 0 0\n"]
)
def test_bad_xyz_is_rejected(text):
    with pytest.raises(ValueError, match=r"job\.xyz"):
        parse_xyz(text)


# --- failure detection -------------------------------------------------------

BANNER = "   * O   R   C   A *\n   Program Version 6.0.1  -  RELEASE\n"
GOOD = BANNER + "work work\n                 ****ORCA TERMINATED NORMALLY****\n"


@pytest.mark.parametrize(
    ("status", "log", "err", "needle"),
    [
        (0, GOOD, "", None),
        (0, GOOD, "x" * 5000, None),
        (None, "", "No such file", "could not be started"),
        (0, "hello world\n", "", "may not be ORCA"),
        (
            0,
            BANNER + "ORCA finished by error termination in LEANSCF\n",
            "",
            "error termination",
        ),
        (3, GOOD, "", "status 3"),
        (4, BANNER + "INPUT ERROR\n", "", "status 4"),
        (0, BANNER + "cut short\n", "", "normal termination"),
        (
            25,
            BANNER + "ERROR: MP2 analytic Hessian calculations are not implemented"
            " - please use NumFreq\n",
            "",
            'hessian="numerical"',
        ),
        (
            0,
            BANNER + "please use NumFreq\nORCA finished by error termination in X\n",
            "",
            'hessian="numerical"',
        ),
    ],
)
def test_failure_detection(status, log, err, needle):
    failure = orca._failure(status, log, err)
    if needle is None:
        assert failure is None
    else:
        assert failure is not None
        assert needle in failure


# --- running jobs ------------------------------------------------------------


def make_orca(tmp_path, body):
    path = tmp_path / "orca"
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o755)
    return str(path)


SUCCESS = f"""
import sys
sys.stderr.write("mpi noise\\n" * 100)
print({BANNER!r})
open("job.engrad", "w").write("engrad")
open("job.gbw", "w").write("big")
open("job.tmp", "w").write("big")
print("****ORCA TERMINATED NORMALLY****")
"""


def test_successful_run_reads_files_and_keeps_stderr_apart(tmp_path):
    settings = RunSettings(make_orca(tmp_path, SUCCESS))
    result = run_job(settings, "! deck\n", "force-H2")
    assert result.failure is None
    assert result.files == {"job.engrad": "engrad"}
    assert "Program Version" in result.log
    assert "mpi noise" not in result.log
    assert "mpi noise" in result.err


def test_failure_message_quotes_the_log_not_stderr(tmp_path):
    body = SUCCESS.replace(
        "****ORCA TERMINATED NORMALLY****", "ORCA finished by error termination in X"
    )
    result = run_job(RunSettings(make_orca(tmp_path, body)), "d", "x")
    assert result.failure is not None
    text = result.describe(result.failure)
    assert "error termination in X" in text
    assert "mpi noise" not in text


def test_describe_falls_back_to_stderr_without_a_log(tmp_path):
    result = run_job(
        RunSettings(make_orca(tmp_path, "import sys; sys.stderr.write('boom\\n')")),
        "d",
        "x",
    )
    assert "boom" in result.describe("failed")
    assert result.failure is not None


def test_describe_keeps_only_the_last_lines():
    result = orca.JobResult("\n".join(f"line {i}" for i in range(100)), "", "f", {})
    text = result.describe("failed")
    assert "line 99" in text
    assert "line 70" not in text


def test_describe_without_any_output():
    assert orca.JobResult("", "", "f", {}).describe("failed") == "failed"


def test_deck_is_written_as_job_inp_and_directories_are_kept(tmp_path):
    work = tmp_path / "work"
    settings = RunSettings(make_orca(tmp_path, SUCCESS), workdir=work)
    run_job(settings, "! deck\n", "force-H2")
    run_job(settings, "! deck2\n", "freq-H2")
    kept = sorted(p.name for p in work.iterdir())
    assert kept == ["0001-force-H2", "0002-freq-H2"]
    names = sorted(p.name for p in (work / "0001-force-H2").iterdir())
    assert names == ["job.engrad", "job.err", "job.inp", "job.out"]
    assert (work / "0002-freq-H2" / "job.inp").read_text() == "! deck2\n"


def test_temporary_directory_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    run_job(RunSettings(make_orca(tmp_path, SUCCESS)), "d", "x")
    assert [p for p in tmp_path.iterdir() if p.is_dir()] == []


def test_timeout_stops_the_run(tmp_path):
    settings = RunSettings(
        make_orca(tmp_path, "import time; time.sleep(30)"), timeout=0.3
    )
    result = run_job(settings, "d", "x")
    assert result.failure is not None
    assert "did not finish" in result.failure


def test_verbose_says_where_the_run_is(tmp_path, capsys):
    run_job(RunSettings(make_orca(tmp_path, SUCCESS), verbose=True), "d", "force-H2")
    err = capsys.readouterr().err
    assert "running ORCA (force-H2)" in err
    assert "job.out" in err
    assert "mpi noise" not in err


def test_quiet_by_default(tmp_path, capsys):
    run_job(RunSettings(make_orca(tmp_path, SUCCESS)), "d", "x")
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_cycles_are_read_from_the_log():
    log = "GEOMETRY OPTIMIZATION CYCLE   1\nx\nGEOMETRY OPTIMIZATION CYCLE  12\n"
    assert orca.cycles(log) == [1, 12]
    assert orca.cycles("nothing") == []


# --- decks -------------------------------------------------------------------

H2 = Molecule(["H", "H"], [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
OH = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]], mult=2)
ATOMS_H2 = (
    "* xyz 0 1\n"
    "  H 0.000000000000 0.000000000000 -0.700000000000\n"
    "  H 0.000000000000 0.000000000000 0.700000000000\n"
    "*\n"
)


def engine(mol=H2, **changes):
    options: dict[str, Any] = {
        "method": "hf",
        "basis": "3-21g",
        "reference": "auto",
        "keywords": "",
        "blocks": "",
        "hessian": False,
        "numerical": False,
        "nprocs": 1,
        "maxcore": 2000,
    }
    options.update(changes)
    return orca.engine_deck(mol, **options)


def optimizer(mol=H2, **changes):
    options: dict[str, Any] = {
        "method": "hf",
        "basis": "3-21g",
        "reference": "auto",
        "keywords": "",
        "blocks": "",
        "saddle": False,
        "convergence": "tight",
        "max_cycles": 100,
        "numerical": False,
        "nprocs": 1,
        "maxcore": 2000,
    }
    options.update(changes)
    return orca.opt_deck(mol, **options)


def test_energy_gradient_deck_is_serial_by_default():
    assert engine() == (
        "# pyvtst engrad H2\n! hf 3-21g EnGrad Bohrs\n%maxcore 2000\n\n" + ATOMS_H2
    )


def test_analytic_hessian_deck():
    assert engine(hessian=True, nprocs=4, maxcore=3000) == (
        "# pyvtst freq H2\n"
        "! hf 3-21g EnGrad Freq Bohrs\n"
        "%pal nprocs 4 end\n"
        "%maxcore 3000\n"
        "%freq\n  TransInvar true\nend\n"
        "\n" + ATOMS_H2
    )


def test_numerical_hessian_deck():
    assert "! hf 3-21g EnGrad NumFreq Bohrs\n" in engine(hessian=True, numerical=True)


def test_numerical_flag_alone_does_not_ask_for_a_hessian():
    assert "Freq" not in engine(numerical=True)


def test_keywords_and_blocks_follow_the_method():
    deck = engine(
        method="b3lyp",
        basis="def2-svp",
        keywords="d3bj rijcosx def2/j",
        blocks="%scf\nMaxIter 300\nend",
        hessian=True,
    )
    assert (
        deck.splitlines()[1] == "! b3lyp def2-svp d3bj rijcosx def2/j EnGrad Freq Bohrs"
    )
    assert "%freq\n  TransInvar true\nend\n%scf\nMaxIter 300\nend\n\n* xyz" in deck


def test_composite_method_has_no_basis():
    assert engine(method="r2scan-3c", basis=orca.NO_BASIS).splitlines()[1] == (
        "! r2scan-3c EnGrad Bohrs"
    )


@pytest.mark.parametrize(
    ("reference", "mol", "line"),
    [
        ("auto", H2, "! hf 3-21g EnGrad Bohrs"),
        ("auto", OH, "! hf 3-21g EnGrad Bohrs"),
        ("unrestricted", H2, "! UHF hf 3-21g EnGrad Bohrs"),
        ("unrestricted", OH, "! UHF hf 3-21g EnGrad Bohrs"),
        ("restricted", H2, "! RHF hf 3-21g EnGrad Bohrs"),
    ],
)
def test_reference_keyword(reference, mol, line):
    assert engine(mol, reference=reference).splitlines()[1] == line


def test_restricted_open_shell_is_rejected():
    with pytest.raises(ValueError, match="multiplicity 1"):
        engine(OH, reference="restricted")


def test_charge_and_multiplicity_are_written():
    assert "* xyz 0 2\n" in engine(OH)
    cation = Molecule(["H", "H"], H2.coords, charge=1, mult=2)
    assert "* xyz 1 2\n" in engine(cation)


@pytest.mark.parametrize(
    ("convergence", "keyword"),
    [("normal", "Opt"), ("tight", "TightOpt"), ("verytight", "VeryTightOpt")],
)
def test_minimum_deck(convergence, keyword):
    assert optimizer(convergence=convergence, max_cycles=50, nprocs=2) == (
        "# pyvtst opt H2\n"
        f"! hf 3-21g {keyword} Bohrs\n"
        "%pal nprocs 2 end\n"
        "%maxcore 2000\n"
        "%geom\n  MaxIter 50\nend\n"
        "\n" + ATOMS_H2
    )


@pytest.mark.parametrize(
    ("convergence", "job"),
    [
        ("normal", "OptTS"),
        ("tight", "OptTS TightOpt"),
        ("verytight", "OptTS VeryTightOpt"),
    ],
)
def test_saddle_deck(convergence, job):
    assert optimizer(saddle=True, convergence=convergence) == (
        "# pyvtst ts H2\n"
        f"! hf 3-21g {job} Bohrs\n"
        "%maxcore 2000\n"
        "%geom\n  MaxIter 100\n  Calc_Hess true\nend\n"
        "\n" + ATOMS_H2
    )


def test_saddle_deck_with_numerical_hessian():
    assert "%geom\n  MaxIter 100\n  Calc_Hess true\n  NumHess true\nend\n" in (
        optimizer(saddle=True, numerical=True)
    )


def test_minimum_deck_ignores_the_hessian_mode():
    assert "Hess" not in optimizer(numerical=True)


def test_optimizer_deck_keeps_user_settings():
    deck = optimizer(
        mol=OH,
        reference="unrestricted",
        keywords="d3bj",
        blocks="%scf\nMaxIter 300\nend",
    )
    assert deck.splitlines()[1] == "! UHF hf 3-21g d3bj TightOpt Bohrs"
    assert "%geom\n  MaxIter 100\nend\n%scf\nMaxIter 300\nend\n\n* xyz 0 2" in deck
