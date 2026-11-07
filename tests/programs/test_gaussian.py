"""Gaussian keywords, input decks, formatted checkpoints and job runs."""

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.programs import gaussian
from pyvtst.programs.gaussian import (
    FORCE_JOB,
    FREQ_JOB,
    Fchk,
    RunSettings,
    build_deck,
    fchk_coordinates,
    fchk_results,
    job_name,
    normalize_keywords,
    opt_job,
    parse_fchk,
    run_job,
)

# --- keywords ----------------------------------------------------------------


def test_keywords_fold_case_and_collapse_whitespace():
    assert (
        normalize_keywords("  EmpiricalDispersion=GD3BJ \t Int=UltraFine\n")
        == "empiricaldispersion=gd3bj int=ultrafine"
    )
    assert normalize_keywords("") == ""


@pytest.mark.parametrize(
    ("keywords", "name"),
    [
        ("Opt Freq", "Opt"),
        ("scf=tight freq=noraman", "freq"),
        ("Force", "Force"),
        ("SP", "SP"),
        ("IRC=(CalcFC,MaxPoints=10)", "IRC"),
        ("Scan", "Scan"),
        ("Geom=Check", "Geom"),
        ("Guess(Read)", "Guess"),
        ("units=ang", "units"),
        ("NoSymm", "NoSymm"),
        ("Symmetry=None", "Symmetry"),
        ("Int=UltraFine,Opt", "Opt"),
    ],
)
def test_reserved_keywords_rejected(keywords, name):
    with pytest.raises(ValueError, match=f"keywords: '{name}'"):
        normalize_keywords(keywords)


@pytest.mark.parametrize(
    "keywords",
    ["SCF=(Tight, XQC)", "Pop=Full", "Optics", "SCRF=(Solvent=Water)", "Int(Grid=SG1)"],
)
def test_other_keywords_accepted(keywords):
    normalize_keywords(keywords)


def test_keywords_must_be_a_string():
    with pytest.raises(ValueError, match="keywords: expected a string"):
        normalize_keywords(3)  # type: ignore[arg-type]


# --- decks -------------------------------------------------------------------

H3 = Molecule(
    ["H", "H", "H"], [[-1.6, 0.0, 0.0], [0.0, 0.0, 0.0], [1.9, 0.08, 0.0]], mult=2
)
H2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

H3_ATOMS = (
    "0 2\n"
    "H -1.600000000000 0.000000000000 0.000000000000\n"
    "H 0.000000000000 0.000000000000 0.000000000000\n"
    "H 1.900000000000 0.080000000000 0.000000000000\n"
)


def deck(mol, job, reference="auto", keywords="scf=tight", title="force"):
    return build_deck(
        mol,
        method="hf",
        basis="3-21g",
        reference=reference,
        keywords=keywords,
        job=job,
        title=title,
        nprocs=4,
        memory="2GB",
    )


@pytest.mark.parametrize(
    ("job", "route_job"),
    [
        (FORCE_JOB, "Force"),
        (FREQ_JOB, "Freq=NoRaman"),
        (
            opt_job(saddle=False, max_cycles=50, convergence="default"),
            "Opt=(MaxCycles=50)",
        ),
        (
            opt_job(saddle=False, max_cycles=100, convergence="tight"),
            "Opt=(MaxCycles=100,Tight)",
        ),
        (
            opt_job(saddle=True, max_cycles=100, convergence="verytight"),
            "Opt=(TS,CalcFC,NoEigenTest,MaxCycles=100,VeryTight)",
        ),
    ],
)
@pytest.mark.parametrize(("reference", "prefix"), [("auto", ""), ("unrestricted", "U")])
def test_deck_text(job, route_job, reference, prefix):
    assert deck(H3, job, reference) == (
        "%NProcShared=4\n"
        "%Mem=2GB\n"
        "%Chk=job.chk\n"
        f"#P {prefix}hf/3-21g {route_job} NoSymm Units=AU scf=tight\n"
        "\n"
        "pyvtst force H3\n"
        "\n" + H3_ATOMS + "\n"
    )


def test_restricted_deck_and_no_keywords():
    text = deck(H2, FORCE_JOB, "restricted", keywords="", title="freq")
    assert "#P Rhf/3-21g Force NoSymm Units=AU\n\npyvtst freq H2\n" in text


def test_restricted_open_shell_rejected():
    with pytest.raises(ValueError, match=r"reference: .* multiplicity 1, got 2"):
        deck(H3, FORCE_JOB, "restricted")


# --- formatted checkpoints ---------------------------------------------------


def header(name, kind, count):
    return f"{name:<40}   {kind}   N={count:12d}\n"


def reals(name, values):
    rows = [
        "".join(f"{v:16.8E}" for v in values[i : i + 5]) + "\n"
        for i in range(0, len(values), 5)
    ]
    return header(name, "R", len(values)) + "".join(rows)


def fchk_text(mol, *, coords=None, numbers=None, hessian=True, energy=-1.5):
    n = 3 * mol.n_atoms
    coords = mol.coords if coords is None else coords
    numbers = [1] * mol.n_atoms if numbers is None else numbers
    lower = [0.1 * (i + 1) for i in range(n * (n + 1) // 2)]
    text = (
        "title line\n"
        "Freq      RHF                                                 3-21G\n"
        f"{'Number of atoms':<40}   I     {mol.n_atoms:12d}\n"
        + header("Route", "C", 6)
        + "#P hf/3-21g Freq=NoRaman NoSymm Units=AU                    \n"
        + "pyvtst      \n"
        + header("Atomic numbers", "I", len(numbers))
        + "".join(f"{z:12d}" for z in numbers)
        + "\n"
        + reals("Current cartesian coordinates", np.ravel(coords).tolist())
        + f"{'Total Energy':<40}   R     {energy:22.15E}\n"
        + f"{'Unrestricted':<40}   L     {'F':>12}\n"
        + reals("Cartesian Gradient", [0.01 * i for i in range(n)])
    )
    if hessian:
        text += reals("Cartesian Force Constants", lower)
    return text + header("Flags", "L", 2) + "  T F\n"


def test_fchk_entries():
    fchk = parse_fchk(fchk_text(H3))
    assert fchk.scalars["Number of atoms"] == 3
    assert fchk.scalar("Total Energy") == -1.5
    assert fchk.array("Atomic numbers").tolist() == [1, 1, 1]
    assert "Route" not in fchk.arrays
    assert "Unrestricted" not in fchk.scalars


def test_fchk_results_expand_the_lower_triangle():
    energy, gradient, hessian = fchk_results(
        parse_fchk(fchk_text(H2)), H2, hessian=True
    )
    assert energy == -1.5
    np.testing.assert_allclose(gradient, np.arange(6).reshape(2, 3) * 0.01)
    assert hessian is not None
    assert hessian.shape == (6, 6)
    np.testing.assert_array_equal(hessian, hessian.T)
    # Row-major lower triangle: (0,0), (1,0), (1,1), (2,0), ...
    assert hessian[0, 0] == pytest.approx(0.1)
    assert hessian[1, 0] == pytest.approx(0.2)
    assert hessian[0, 1] == pytest.approx(0.2)
    assert hessian[1, 1] == pytest.approx(0.3)
    assert hessian[5, 5] == pytest.approx(2.1)


def test_fchk_results_without_hessian():
    _, _, hessian = fchk_results(
        parse_fchk(fchk_text(H2, hessian=False)), H2, hessian=False
    )
    assert hessian is None


def test_missing_hessian_reported():
    with pytest.raises(ValueError, match="'Cartesian Force Constants'"):
        fchk_results(parse_fchk(fchk_text(H2, hessian=False)), H2, hessian=True)


def test_reoriented_geometry_rejected():
    swapped = H3.coords[:, [1, 0, 2]]
    with pytest.raises(ValueError, match="reoriented"):
        fchk_results(parse_fchk(fchk_text(H3, coords=swapped)), H3, hessian=False)


def test_geometry_within_tolerance_accepted():
    shifted = H3.coords + 4e-7
    fchk_results(parse_fchk(fchk_text(H3, coords=shifted)), H3, hessian=False)


def test_wrong_atoms_rejected():
    with pytest.raises(ValueError, match="atomic numbers"):
        fchk_coordinates(parse_fchk(fchk_text(H2, numbers=[1, 2])), H2)


def test_wrong_array_size_rejected():
    fchk = Fchk(
        {},
        {
            "Atomic numbers": np.array([1, 1]),
            "Current cartesian coordinates": np.zeros(5),
        },
    )
    with pytest.raises(ValueError, match="expected 6 values, got 5"):
        fchk_coordinates(fchk, H2)


def test_missing_scalar_reported():
    with pytest.raises(ValueError, match="'Total Energy'"):
        Fchk({}, {}).scalar("Total Energy")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("t\nt\nnot a header line\n", "not an entry header"),
        ("t\nt\n" + header("Atomic numbers", "I", 3) + "1 2\n", "file ends"),
        ("t\nt\n" + header("Atomic numbers", "I", 2) + "1 2 3\n", "got 3"),
        ("t\nt\n" + header("Atomic numbers", "I", 1) + "x\n", "integer"),
        ("t\nt\n" + header("Energies", "R", 1) + "abc\n", "number"),
        ("t\nt\n" + f"{'Atomic numbers':<40}   I   N=           x\n", "integer"),
    ],
)
def test_malformed_fchk(text, match):
    with pytest.raises(ValueError, match=match):
        parse_fchk(text)


def test_fortran_exponents_and_blank_lines():
    text = "t\nt\n\n" + f"{'Total Energy':<40}   R     -1.5D+00\n"
    assert parse_fchk(text).scalar("Total Energy") == -1.5


# --- running jobs ------------------------------------------------------------


@pytest.fixture
def settings(fake_gaussian):
    return RunSettings(str(fake_gaussian.g16), str(fake_gaussian.formchk))


def test_job_success(settings):
    result = run_job(settings, deck(H3, FREQ_JOB, keywords=""), "freq")
    assert result.failure is None
    assert result.fchk is not None
    assert result.fchk_failure is None
    assert "Normal termination" in result.log


def test_job_error_termination(settings, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "error")
    result = run_job(settings, deck(H3, FORCE_JOB, keywords=""), "force")
    assert result.failure == "Gaussian reported an error termination"
    assert result.fchk is None
    assert "Convergence failure" in result.describe("failed")


def test_job_formchk_after_failure(settings, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "steps_exceeded")
    job = opt_job(saddle=False, max_cycles=2, convergence="tight")
    result = run_job(
        settings, deck(H2, job, keywords=""), "opt", formchk_on_failure=True
    )
    assert result.failure is not None
    assert result.fchk is not None


def test_job_timeout(settings, monkeypatch):
    monkeypatch.setenv("FAKE_G16_MODE", "hang")
    result = run_job(
        RunSettings(settings.g16, settings.formchk, timeout=1.0),
        deck(H3, FORCE_JOB, keywords=""),
        "force",
        formchk_on_failure=True,
    )
    assert result.failure == "Gaussian did not finish within 1 s and was stopped"
    assert result.fchk is None


def script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    ("body", "failure"),
    [
        ("exit 3\n", "Gaussian exited with status 3"),
        ("exit 0\n", "Gaussian wrote no job.log"),
        ("echo running > job.log\n", "does not report a normal termination"),
    ],
)
def test_job_failures(tmp_path, body, failure):
    settings = RunSettings(script(tmp_path, "g16", body), "/nonexistent/formchk")
    result = run_job(settings, deck(H2, FORCE_JOB, keywords=""), "force")
    assert result.failure is not None
    assert failure in result.failure


def test_job_that_cannot_start(tmp_path):
    settings = RunSettings(str(tmp_path / "missing"), str(tmp_path / "missing"))
    result = run_job(settings, deck(H2, FORCE_JOB, keywords=""), "force")
    assert result.failure == "Gaussian could not be started"
    assert "missing" in result.describe("x")


def test_failure_without_output_has_no_tail(tmp_path):
    settings = RunSettings(script(tmp_path, "g16", "exit 1\n"), "/x")
    result = run_job(settings, deck(H2, FORCE_JOB, keywords=""), "force")
    assert result.describe("failed") == "failed"


NORMAL_G16 = "echo ' Normal termination' > job.log\n"


@pytest.mark.parametrize(
    ("g16", "formchk", "failure"),
    [
        (NORMAL_G16, "exit 0\n", "Gaussian wrote no job.chk"),
        (NORMAL_G16 + "touch job.chk\n", "echo boom; exit 2\n", "formchk failed: boom"),
        (NORMAL_G16 + "touch job.chk\n", "exit 2\n", "formchk failed"),
        (
            NORMAL_G16 + "touch job.chk\n",
            "echo 'garbage' > job.fchk\necho x >> job.fchk\necho y >> job.fchk\n",
            "cannot read job.fchk",
        ),
    ],
)
def test_formchk_failures(tmp_path, g16, formchk, failure):
    settings = RunSettings(
        script(tmp_path, "g16", g16), script(tmp_path, "formchk", formchk)
    )
    result = run_job(settings, deck(H2, FORCE_JOB, keywords=""), "force")
    assert result.failure is None
    assert result.fchk is None
    assert result.fchk_failure is not None
    assert result.fchk_failure.startswith(failure)


def test_workdir_keeps_files_but_not_the_checkpoint(settings, tmp_path, capsys):
    workdir = tmp_path / "runs"
    kept = RunSettings(settings.g16, settings.formchk, workdir=workdir, verbose=True)
    run_job(kept, deck(H3, FORCE_JOB, keywords=""), "force-H3")
    (job,) = workdir.iterdir()
    assert job.name == "0001-force-H3"
    assert sorted(p.name for p in job.iterdir()) == ["job.fchk", "job.gjf", "job.log"]
    assert str(job) in capsys.readouterr().err


def test_job_directories_are_numbered_in_run_order(settings, tmp_path):
    workdir = tmp_path / "runs"
    workdir.mkdir()
    (workdir / "0007-old-run").mkdir()
    (workdir / "notes.txt").write_text("")
    kept = RunSettings(settings.g16, settings.formchk, workdir=workdir)
    for name in ("opt-H2", "freq-H2"):
        run_job(kept, deck(H2, FORCE_JOB, keywords=""), name)
    assert sorted(p.name for p in workdir.iterdir()) == [
        "0007-old-run",
        "0008-opt-H2",
        "0009-freq-H2",
        "notes.txt",
    ]


def test_taken_directory_moves_to_the_next_number(tmp_path, monkeypatch):
    # Another run took 0001-freq-H2 after this one looked for the last number.
    (tmp_path / "0001-freq-H2").mkdir()
    monkeypatch.setattr(gaussian, "_last_number", lambda root: 0)
    assert gaussian._numbered_directory(tmp_path, "freq-H2").name == "0002-freq-H2"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("HCN-HNC TS", "ts-HCN-HNC_TS"), ("a/b:c", "ts-a_b_c"), ("", "ts-H2")],
)
def test_job_name(name, expected):
    assert job_name("ts", H2.replace(name=name)) == expected
