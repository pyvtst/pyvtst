"""Shared process code: executable links, separate output files, job directories."""

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pyvtst.programs import process


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# --- executable links --------------------------------------------------------


def test_symbolic_link_is_followed_when_asked(tmp_path):
    (tmp_path / "real").mkdir()
    real = _script(tmp_path / "real" / "prog", "")
    link = tmp_path / "prog-link"
    link.symlink_to(real)
    assert process.find_executable(
        "x", "PYVTST_X", "arg", link, resolve_links=True
    ) == str(real.resolve())
    assert process.find_executable("x", "PYVTST_X", "arg", link) == str(link.absolute())


def test_wrapper_script_is_not_resolved(tmp_path):
    wrapper = _script(tmp_path / "wrapper", "")
    assert process.find_executable(
        "x", "PYVTST_X", "arg", wrapper, resolve_links=True
    ) == str(wrapper.resolve())


def test_environment_variable_and_path_links_are_followed(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    real = _script(tmp_path / "bin" / "real-prog", "")
    link = tmp_path / "prog"
    link.symlink_to(real)
    monkeypatch.setenv("PYVTST_X", str(link))
    assert process.find_executable(
        "x", "PYVTST_X", "arg", None, resolve_links=True
    ) == str(real.resolve())
    monkeypatch.delenv("PYVTST_X")
    monkeypatch.setenv("PATH", str(tmp_path))
    link2 = tmp_path / "xprog"
    link2.symlink_to(real)
    assert process.find_executable(
        "xprog", "PYVTST_X", "arg", None, resolve_links=True
    ) == str(real.resolve())


# --- separate output files ---------------------------------------------------


def test_stdout_and_stderr_go_to_separate_files(tmp_path):
    prog = _script(
        tmp_path / "prog",
        "import sys\nprint('to stdout')\nprint('to stderr', file=sys.stderr)",
    )
    out, err = tmp_path / "job.out", tmp_path / "job.err"
    status, captured = process.execute(
        [str(prog)],
        tmp_path,
        timeout=None,
        verbose=False,
        stdout_path=out,
        stderr_path=err,
    )
    assert (status, captured) == (0, "")
    assert out.read_text() == "to stdout\n"
    assert err.read_text() == "to stderr\n"


def test_output_file_without_error_file_drops_stderr(tmp_path):
    prog = _script(
        tmp_path / "prog",
        "import sys\nprint('to stdout')\nprint('noise', file=sys.stderr)",
    )
    out = tmp_path / "job.out"
    process.execute([str(prog)], tmp_path, timeout=None, verbose=False, stdout_path=out)
    assert out.read_text() == "to stdout\n"


def test_captured_output_merges_the_streams(tmp_path):
    prog = _script(
        tmp_path / "prog",
        "import sys\nprint('a', flush=True)\nprint('b', file=sys.stderr)",
    )
    status, output = process.execute([str(prog)], tmp_path, timeout=None, verbose=False)
    assert status == 0
    assert sorted(output.split()) == ["a", "b"]


def test_program_that_cannot_start_reports_no_status(tmp_path):
    status, output = process.execute(
        [str(tmp_path / "missing")], tmp_path, timeout=None, verbose=False
    )
    assert status is None
    assert output


def test_timeout_kills_the_process_group(tmp_path):
    marker = tmp_path / "child-alive"
    child = _script(
        tmp_path / "child",
        "import pathlib, time\ntime.sleep(1.5)\n"
        f"pathlib.Path({str(marker)!r}).write_text('x')",
    )
    parent = _script(
        tmp_path / "parent",
        f"import subprocess, time\nsubprocess.Popen([{str(child)!r}])\ntime.sleep(60)",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        process.execute([str(parent)], tmp_path, timeout=0.5, verbose=False)
    time.sleep(2.0)
    assert not marker.exists()


# --- job directories ---------------------------------------------------------


def test_taken_directory_moves_to_the_next_number(tmp_path, monkeypatch):
    # Another run took 0001-freq-H2 after this one looked for the last number.
    (tmp_path / "0001-freq-H2").mkdir()
    monkeypatch.setattr(process, "last_number", lambda root: 0)
    assert process.numbered_directory(tmp_path, "freq-H2").name == "0002-freq-H2"


def test_temporary_job_directory_is_removed(tmp_path):
    with process.job_directory(None, "x", prefix="pyvtst-test-") as path:
        assert path.name.startswith("pyvtst-test-")
        assert path.is_dir()
    assert not path.exists()


def test_kept_job_directories_are_numbered(tmp_path):
    (tmp_path / "0007-old").mkdir()
    with process.job_directory(tmp_path, "a", prefix="p-") as first:
        pass
    with process.job_directory(tmp_path, "b", prefix="p-") as second:
        pass
    assert (first.name, second.name) == ("0008-a", "0009-b")
    assert os.listdir(tmp_path)
