"""ORCA lookup: argument, then environment variable, then PATH; links resolved."""

import pytest

from pyvtst.programs.orca import find_orca


def executable(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No PYVTST_ORCA variable and a PATH holding only an empty directory."""
    monkeypatch.delenv("PYVTST_ORCA", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return monkeypatch


def test_environment_variable_takes_precedence_over_path(clean_env, tmp_path):
    on_path = executable(tmp_path / "path", "orca")
    from_env = executable(tmp_path / "env", "my-orca")
    clean_env.setenv("PATH", str(on_path.parent))
    clean_env.setenv("PYVTST_ORCA", str(from_env))
    assert find_orca(None) == str(from_env)


def test_argument_takes_precedence_over_environment_variable(clean_env, tmp_path):
    from_env = executable(tmp_path / "env", "orca-env")
    from_arg = executable(tmp_path / "arg", "orca-arg")
    clean_env.setenv("PYVTST_ORCA", str(from_env))
    assert find_orca(from_arg) == str(from_arg)


def test_path_is_the_fallback(clean_env, tmp_path):
    on_path = executable(tmp_path / "path", "orca")
    clean_env.setenv("PATH", str(on_path.parent))
    assert find_orca(None) == str(on_path)


def test_empty_environment_variable_counts_as_unset(clean_env, tmp_path):
    on_path = executable(tmp_path / "path", "orca")
    clean_env.setenv("PATH", str(on_path.parent))
    clean_env.setenv("PYVTST_ORCA", "")
    assert find_orca(None) == str(on_path)


def test_broken_environment_variable_is_an_error(clean_env, tmp_path):
    on_path = executable(tmp_path / "path", "orca")
    clean_env.setenv("PATH", str(on_path.parent))
    missing = tmp_path / "missing" / "orca"
    clean_env.setenv("PYVTST_ORCA", str(missing))
    with pytest.raises(FileNotFoundError) as info:
        find_orca(None)
    assert "PYVTST_ORCA" in str(info.value)
    assert str(missing) in str(info.value)


def test_broken_argument_is_an_error_without_fallback(clean_env, tmp_path):
    on_path = executable(tmp_path / "path", "orca")
    clean_env.setenv("PATH", str(on_path.parent))
    with pytest.raises(FileNotFoundError, match="executable"):
        find_orca(tmp_path / "missing")


def test_nothing_found(clean_env):
    with pytest.raises(FileNotFoundError) as info:
        find_orca(None)
    message = str(info.value)
    assert "executable" in message
    assert "PYVTST_ORCA" in message
    assert "PATH" in message


def test_symbolic_link_is_resolved(clean_env, tmp_path):
    real = executable(tmp_path / "opt" / "orca_6_0_1", "orca")
    link = tmp_path / "bin" / "orca6"
    link.parent.mkdir()
    link.symlink_to(real)
    assert find_orca(link) == str(real.resolve())
    clean_env.setenv("PYVTST_ORCA", str(link))
    assert find_orca(None) == str(real.resolve())
