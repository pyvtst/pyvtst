"""The reaction-path geometries written next to the report."""

import numpy as np
import pytest

from pyvtst.core.units import BOHR_TO_ANGSTROM
from pyvtst.inputfile import load
from pyvtst.inputfile.build import execute
from pyvtst.inputfile.cli import main
from pyvtst.inputfile.report import mep_xyz_path, render, write_report
from tests.inputfile.conftest import H3_INPUT

PATH_INPUT = H3_INPUT.replace(
    'tunneling = "eckart"', 'tunneling = "eckart"\n\n[mep]\ns_min = -0.2\ns_max = 0.2'
)


def frames(text):
    """``(comment, symbols, coords)`` per frame of multi-frame XYZ text."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        rows = [line.split() for line in lines[i + 2 : i + 2 + n]]
        coords = np.array([[float(v) for v in r[1:]] for r in rows])
        out.append((lines[i + 1], [r[0] for r in rows], coords))
        i += n + 2
    return out


@pytest.fixture
def run_log(write, registry):
    return execute(load(write(PATH_INPUT), registry), registry)


def test_name_follows_the_report(tmp_path):
    assert mep_xyz_path(tmp_path / "h3.out") == tmp_path / "h3_mep.xyz"


def test_one_frame_per_point_in_s_order(run_log, tmp_path):
    write_report(run_log, tmp_path / "h3.out")
    path = run_log.path
    parsed = frames((tmp_path / "h3_mep.xyz").read_text())
    assert len(parsed) == len(path.s)
    s_values = [float(c.split("s_bohr=")[1].split()[0]) for c, _, _ in parsed]
    assert s_values == sorted(s_values)
    np.testing.assert_allclose(s_values, np.sort(path.s), atol=1e-6)
    for comment, symbols, coords in parsed:
        i = int(comment.split()[0].removeprefix("index="))
        assert symbols == list(path.molecules[i].symbols)
        np.testing.assert_allclose(
            coords, path.molecules[i].coords * BOHR_TO_ANGSTROM, atol=1e-8
        )
    ts_comment = parsed[int(np.argmin(np.abs(np.sort(path.s))))][0]
    assert "s_bohr=0.000000" in ts_comment


def test_report_names_the_file(run_log, tmp_path):
    write_report(run_log, tmp_path / "h3.out")
    report = (tmp_path / "h3.out").read_text()
    assert "geometries (angstrom): h3_mep.xyz" in report


def test_ase_reads_the_trajectory(run_log, tmp_path):
    ase_io = pytest.importorskip("ase.io")
    write_report(run_log, tmp_path / "h3.out")
    images = ase_io.read(tmp_path / "h3_mep.xyz", index=":")
    assert len(images) == len(run_log.path.s)
    assert images[0].info["s_bohr"] == pytest.approx(float(np.min(run_log.path.s)))
    assert "v_mep_kcal_mol" in images[0].info


def test_no_file_without_a_path(write, registry, tmp_path):
    assert main([str(write(H3_INPUT))], registry=registry) == 0
    assert not (tmp_path / "h3_mep.xyz").exists()


def test_command_writes_the_trajectory(write, registry, tmp_path):
    path = write(PATH_INPUT, "calc/h3.toml")
    assert (
        main([str(path), "-o", str(tmp_path / "out" / "run.out")], registry=registry)
        == 0
    )
    assert (tmp_path / "out" / "run_mep.xyz").is_file()
    assert not (tmp_path / "calc" / "h3_mep.xyz").exists()


def test_report_without_a_trajectory_file(run_log):
    report = render(run_log)
    assert "REACTION PATH" in report
    assert "geometries (angstrom)" not in report
