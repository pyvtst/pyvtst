import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from pyvtst import __version__
from pyvtst.ports import StoreFormatError, StoreLockedError
from pyvtst.ports.store import Value
from pyvtst.stores import HDF5Store
from tests.stores.contract import StoreContract, assert_records_equal

h5py = pytest.importorskip("h5py")


class TestHDF5Contract(StoreContract):
    @pytest.fixture
    def store(self, tmp_path):
        return HDF5Store(tmp_path / "store.h5")


@pytest.fixture
def path(tmp_path):
    return tmp_path / "cache.h5"


def run_python(script, *args, **kwargs):
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )


def spawn_python(script, *args):
    return subprocess.Popen(
        [sys.executable, "-c", script, *map(str, args)],
        stdout=subprocess.PIPE,
        text=True,
    )


# Holds the file open for writing (an exclusive HDF5 lock) until killed.
HOLD_SCRIPT = """
import sys, time, h5py
f = h5py.File(sys.argv[1], "a")
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


# Persistence and modes


PUT_SCRIPT = """
import sys
import numpy as np
from pyvtst.stores import HDF5Store
HDF5Store(sys.argv[1]).put("a/b", {"h": np.arange(6.0).reshape(2, 3), "e": -1.5})
"""


def test_persistence_across_processes(path):
    run_python(PUT_SCRIPT, path)
    got = HDF5Store(path).get("a/b")
    assert_records_equal(got, {"h": np.arange(6.0).reshape(2, 3), "e": -1.5})


def test_missing_file_is_created(path):
    HDF5Store(path)
    assert path.exists()
    with h5py.File(path, "r") as f:
        assert f.attrs["pyvtst_store"] == "pyvtst-result-store"
        assert f.attrs["format_version"] == 1
        assert f.attrs["created_by"] == f"pyvtst {__version__}"


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        HDF5Store(tmp_path / "nope" / "cache.h5")


def test_read_only_missing_file(path):
    with pytest.raises(FileNotFoundError):
        HDF5Store(path, mode="r")
    assert not path.exists()


def test_read_only_mode(path):
    HDF5Store(path, backup=False).put("k", {"x": 1.0})
    before = path.read_bytes()
    ro = HDF5Store(path, mode="r")
    assert ro.get("k") == {"x": 1.0}
    assert list(ro.keys()) == ["k"]
    with pytest.raises(PermissionError):
        ro.put("k2", {"x": 2.0})
    assert path.read_bytes() == before
    assert not (path.parent / "cache.h5.bak").exists()


def test_read_only_empty_hdf5_file(path):
    h5py.File(path, "w").close()
    ro = HDF5Store(path, mode="r")
    assert ro.get("k") is None
    assert list(ro.keys()) == []


def test_empty_hdf5_file_is_initialized(path):
    h5py.File(path, "w").close()
    store = HDF5Store(path)
    store.put("k", {"x": 1})
    assert store.get("k") == {"x": 1}


def test_layout(path):
    HDF5Store(path).put("a/b", {"h": np.zeros(2), "e": 1.0})
    with h5py.File(path, "r") as f:
        data = f["records/a/b/.data"]
        assert data["h"].shape == (2,)
        assert data.attrs["e"] == 1.0
        assert len(f.get(".staging", {})) == 0
        assert len(f.get(".trash", {})) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"mode": "w"}, "mode"), ({"lock_timeout": -1.0}, "lock_timeout")],
)
def test_invalid_arguments(path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        HDF5Store(path, **kwargs)


def test_repr(path):
    assert repr(HDF5Store(path)) == f"HDF5Store({str(path)!r}, mode='a')"


# File format versioning


def test_foreign_hdf5_file(path):
    with h5py.File(path, "w") as f:
        f["data"] = np.arange(3)
    before = path.read_bytes()
    for mode in ("a", "r"):
        with pytest.raises(StoreFormatError, match="not a pyvtst result store"):
            HDF5Store(path, mode=mode)
    assert path.read_bytes() == before


def test_foreign_file_with_only_attributes(path):
    with h5py.File(path, "w") as f:
        f.attrs["author"] = "someone else"
    with pytest.raises(StoreFormatError):
        HDF5Store(path)


def test_newer_format(path):
    HDF5Store(path)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = np.int64(2)
    with pytest.raises(StoreFormatError, match="upgrade pyvtst"):
        HDF5Store(path)


def test_newer_format_written_after_open(path):
    store = HDF5Store(path)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = np.int64(2)
    with pytest.raises(StoreFormatError, match="upgrade pyvtst"):
        store.get("k")


def test_wrong_marker(path):
    with h5py.File(path, "w") as f:
        f.attrs["pyvtst_store"] = "something-else"
        f.attrs["format_version"] = 1
    with pytest.raises(StoreFormatError, match="unrecognized"):
        HDF5Store(path)


def test_corrupt_file(path):
    path.write_bytes(b"this is not an HDF5 file" * 10)
    with pytest.raises(StoreFormatError) as excinfo:
        HDF5Store(path)
    assert str(path) in str(excinfo.value)
    assert ".bak" not in str(excinfo.value)


def test_corrupt_file_names_the_backup(path):
    path.write_bytes(b"garbage" * 10)
    (path.parent / "cache.h5.bak").write_bytes(b"")
    with pytest.raises(StoreFormatError, match=r"cache\.h5\.bak"):
        HDF5Store(path)


def test_file_corrupted_after_open(path):
    store = HDF5Store(path)
    path.write_bytes(b"garbage" * 10)
    with pytest.raises(StoreFormatError, match="cannot be read as an HDF5 file"):
        store.get("k")


# Crash safety

SEED_SCRIPT = """
import sys
from pyvtst.stores import HDF5Store
store = HDF5Store(sys.argv[1], backup=False)
for i in range(int(sys.argv[2])):
    store.put(f"r/{i}", {"i": i})
"""


def test_backup_before_first_write(path):
    run_python(SEED_SCRIPT, path, 2)
    backup = path.parent / "cache.h5.bak"
    assert not backup.exists()
    store = HDF5Store(path)
    store.put("r/2", {"i": 2})
    assert list(HDF5Store(backup, mode="r").keys()) == ["r/0", "r/1"]
    store.put("r/3", {"i": 3})
    # Only the first write of a session is backed up.
    assert list(HDF5Store(backup, mode="r").keys()) == ["r/0", "r/1"]
    assert list(store.keys()) == ["r/0", "r/1", "r/2", "r/3"]


def test_backup_once_per_process(path):
    run_python(SEED_SCRIPT, path, 1)
    HDF5Store(path).put("r/1", {"i": 1})
    HDF5Store(path).put("r/2", {"i": 2})
    assert list(HDF5Store(path.parent / "cache.h5.bak").keys()) == ["r/0"]


def test_backup_disabled(path):
    store = HDF5Store(path, backup=False)
    store.put("k", {"x": 1})
    assert not (path.parent / "cache.h5.bak").exists()


@pytest.fixture
def fail_second_dataset(monkeypatch):
    """Return a function that makes the second dataset created from then on raise."""
    real = h5py.Group.create_dataset

    def arm():
        calls = []

        def create_dataset(self, name, *args, **kwargs):
            calls.append(name)
            if len(calls) == 2:
                raise RuntimeError("simulated failure mid-write")
            return real(self, name, *args, **kwargs)

        monkeypatch.setattr(h5py.Group, "create_dataset", create_dataset)

    return arm


@pytest.mark.parametrize("key", ["k", "new"], ids=["existing", "new"])
def test_exception_during_a_write(path, fail_second_dataset, monkeypatch, key):
    store = HDF5Store(path)
    old: dict[str, Value] = {"a": np.zeros(2), "b": 1.0}
    other = {"z": np.ones(3)}
    store.put("k", old)
    store.put("other", other)

    fail_second_dataset()
    with pytest.raises(RuntimeError, match="simulated"):
        store.put(key, {"x": np.ones(2), "y": np.ones(2)})
    monkeypatch.undo()

    assert_records_equal(store.get("k"), old)
    assert_records_equal(store.get("other"), other)
    assert list(store.keys()) == ["k", "other"]
    assert ("new" in store) is False

    store.put(key, {"x": np.ones(2)})
    assert_records_equal(store.get(key), {"x": np.ones(2)})
    assert_records_equal(store.get("other"), other)
    with h5py.File(path, "r") as f:
        assert len(f[".staging"]) == 0
        assert len(f.get(".trash", {})) == 0


KILL_SCRIPT = """
import sys, time
from pyvtst.stores import HDF5Store
store = HDF5Store(sys.argv[1])
for i in range(10):
    store.put(f"r/{i:06d}", {"i": i})
    print(f"done {i}", flush=True)
time.sleep(60)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
def test_killed_between_puts(path):
    proc = spawn_python(KILL_SCRIPT, path)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip() == "done 9":
                break
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    store = HDF5Store(path)
    assert list(store.keys()) == [f"r/{i:06d}" for i in range(10)]
    for i in range(10):
        assert store.get(f"r/{i:06d}") == {"i": i}


# Multiple processes

WRITER_SCRIPT = """
import os, sys, time
from pyvtst.stores import HDF5Store
path, name, go = sys.argv[1:4]
store = HDF5Store(path)
while not os.path.exists(go):
    time.sleep(0.005)
for i in range(50):
    store.put(f"{name}/{i:06d}", {"i": i, "who": name})
"""


def test_two_concurrent_writers(path):
    HDF5Store(path)
    go = path.parent / "go"
    procs = [spawn_python(WRITER_SCRIPT, path, name, go) for name in ("a", "b")]
    time.sleep(0.5)
    go.touch()
    for proc in procs:
        assert proc.wait(timeout=120) == 0
    store = HDF5Store(path)
    keys = list(store.keys())
    assert len(keys) == 100
    assert store.get("b/000049") == {"i": 49, "who": "b"}


def test_lock_timeout(path):
    store = HDF5Store(path, lock_timeout=0.5)
    holder = spawn_python(HOLD_SCRIPT, path, 60)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        start = time.monotonic()
        with pytest.raises(StoreLockedError) as excinfo:
            store.get("k")
        elapsed = time.monotonic() - start
    finally:
        holder.kill()
        holder.wait(timeout=10)
    assert excinfo.value.path == str(path)
    assert str(path) in str(excinfo.value)
    assert 0.5 <= elapsed < 5.0


def test_lock_released_within_timeout(path):
    store = HDF5Store(path, lock_timeout=30.0)
    holder = spawn_python(HOLD_SCRIPT, path, 0.5)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        store.put("k", {"x": 1})
    finally:
        holder.wait(timeout=10)
    assert store.get("k") == {"x": 1}
