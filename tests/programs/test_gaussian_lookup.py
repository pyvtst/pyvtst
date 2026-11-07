"""Gaussian and formchk lookup: argument, then environment variable, then PATH."""

import pytest

from pyvtst.programs.gaussian import find_executable, find_programs


def executable(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No PYVTST_* variables and a PATH holding only an empty directory."""
    monkeypatch.delenv("PYVTST_G16", raising=False)
    monkeypatch.delenv("PYVTST_FORMCHK", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return monkeypatch


def on_path(monkeypatch, directory):
    monkeypatch.setenv("PATH", str(directory))


def g16(value=None):
    return find_executable("g16", "PYVTST_G16", "executable", value)


def formchk(value=None):
    return find_executable("formchk", "PYVTST_FORMCHK", "formchk", value)


def test_environment_variable_takes_precedence_over_path(clean_env, tmp_path):
    executable(tmp_path / "path", "g16")
    chosen = executable(tmp_path / "chosen", "g16")
    on_path(clean_env, tmp_path / "path")
    clean_env.setenv("PYVTST_G16", str(chosen))
    assert g16() == str(chosen)


def test_argument_takes_precedence_over_environment_variable(clean_env, tmp_path):
    clean_env.setenv("PYVTST_FORMCHK", str(executable(tmp_path / "env", "formchk")))
    chosen = executable(tmp_path / "arg", "formchk")
    assert formchk(chosen) == str(chosen)


def test_path_is_the_fallback(clean_env, tmp_path):
    found = executable(tmp_path / "path", "g16")
    on_path(clean_env, tmp_path / "path")
    assert g16() == str(found)


def test_empty_environment_variable_counts_as_unset(clean_env, tmp_path):
    found = executable(tmp_path / "path", "g16")
    on_path(clean_env, tmp_path / "path")
    clean_env.setenv("PYVTST_G16", "")
    assert g16() == str(found)


def test_broken_environment_variable_is_an_error(clean_env, tmp_path):
    executable(tmp_path / "path", "g16")
    on_path(clean_env, tmp_path / "path")
    missing = tmp_path / "nowhere" / "g16"
    clean_env.setenv("PYVTST_G16", str(missing))
    with pytest.raises(FileNotFoundError, match="PYVTST_G16") as info:
        g16()
    assert str(missing) in str(info.value)


def test_broken_argument_is_an_error(clean_env, tmp_path):
    clean_env.setenv("PYVTST_G16", str(executable(tmp_path / "env", "g16")))
    not_executable = tmp_path / "g16"
    not_executable.write_text("")
    with pytest.raises(FileNotFoundError, match="executable") as info:
        g16(not_executable)
    assert str(not_executable) in str(info.value)


def test_nothing_found(clean_env):
    with pytest.raises(FileNotFoundError) as info:
        formchk()
    message = str(info.value)
    assert "formchk" in message
    assert "PYVTST_FORMCHK" in message
    assert "PATH" in message


def test_find_programs_returns_both(clean_env, tmp_path):
    exe = executable(tmp_path / "bin", "g16")
    chk = executable(tmp_path / "bin", "formchk")
    on_path(clean_env, tmp_path / "bin")
    assert find_programs(None, None) == (str(exe), str(chk))
