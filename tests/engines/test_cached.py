import dataclasses
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.keys import KEY_COORD_RESOLUTION
from pyvtst.engines import CachedEngine
from pyvtst.ports import Engine, EngineResult, LevelOfTheory
from pyvtst.stores import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
K = 0.5  # hartree/bohr²


class CountingEngine(Engine):
    """Harmonic surface E = K/2 |x|² + charge/10 that counts its runs.

    Runs without a Hessian count as ``"force"``, runs with one as ``"freq"``.
    With ``always_hessian``, every run returns a Hessian.
    """

    def __init__(self, basis="def2-svp", *, always_hessian=False):
        self._level = LevelOfTheory("fake", basis)
        self.calls: Counter[str] = Counter()
        self.always_hessian = always_hessian

    @property
    def level(self):
        return self._level

    def compute(self, mol, *, hessian):
        self.calls["freq" if hessian else "force"] += 1
        energy = float(0.5 * K * np.sum(mol.coords**2) + 0.1 * mol.charge)
        full = K * np.eye(3 * mol.n_atoms)
        give = hessian or self.always_hessian
        return EngineResult(energy, K * mol.coords.copy(), full if give else None)


class OtherEngine(CountingEngine):
    pass


class BrokenEngine(CountingEngine):
    """Puts ``bad`` in place of one field of each result.

    ``bad`` maps a field name to its value. ``result`` replaces the whole
    result, and is raised if it is an exception.
    """

    def __init__(self, bad=None, result=None):
        super().__init__()
        self.bad = bad or {}
        self.result = result

    def compute(self, mol, *, hessian):
        good = super().compute(mol, hessian=hessian)
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is not None:
            return self.result
        return dataclasses.replace(good, **self.bad)


def water(**changes):
    coords = [[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8857], [0.0, -1.4309, -0.8857]]
    return Molecule(("O", "H", "H"), coords).replace(**changes)


def methane(**changes):
    coords = [[0.0, 0.0, 0.0], [1.2, 1.2, 1.2], [-1.2, -1.2, 1.2], [-1.2, 1.2, -1.2]]
    coords.append([1.2, -1.2, -1.2])
    return Molecule(("C", "H", "H", "H", "H"), coords).replace(**changes)


@pytest.fixture
def engine():
    return CountingEngine()


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def cached(engine, store):
    return CachedEngine(engine, store)


def call_all(eng, mol):
    return eng.energy(mol), eng.gradient(mol), eng.hessian(mol)


def test_is_an_engine(cached):
    assert isinstance(cached, Engine)


def test_transparent_results(cached):
    mol = water()
    reference = CountingEngine()
    energy, gradient, hessian = call_all(cached, mol)
    assert type(energy) is float
    assert energy == reference.energy(mol)
    np.testing.assert_array_equal(gradient, reference.gradient(mol))
    np.testing.assert_array_equal(hessian, reference.hessian(mol))
    assert gradient.dtype == np.float64
    assert gradient.shape == (3, 3)
    assert hessian.shape == (9, 9)


def test_hits_return_equal_results(cached):
    mol = water()
    first = call_all(cached, mol)
    second = call_all(cached, mol)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


def test_level_passes_through(cached, engine):
    assert cached.level == engine.level


def test_second_call_is_a_hit(cached, engine):
    mol = water()
    first = cached.hessian(mol)
    second = cached.hessian(mol)
    assert engine.calls["freq"] == 1
    np.testing.assert_array_equal(first, second)


def test_each_kind_is_cached_once(cached, engine):
    for _ in range(3):
        call_all(cached, water())
    assert engine.calls == {"force": 1, "freq": 1}


def test_one_run_serves_energy_and_gradient(cached, engine):
    cached.energy(water())
    cached.gradient(water())
    assert engine.calls == {"force": 1}


def test_hessian_run_serves_energy_and_gradient(cached, engine):
    mol = water()
    cached.hessian(mol)
    energy, gradient = cached.energy(mol), cached.gradient(mol)
    assert engine.calls == {"freq": 1}
    reference = CountingEngine().compute(mol, hessian=False)
    assert energy == reference.energy
    np.testing.assert_array_equal(gradient, reference.gradient)


def test_unrequested_hessian_is_stored(store):
    engine = CountingEngine(always_hessian=True)
    cached = CachedEngine(engine, store)
    cached.energy(water())
    cached.hessian(water())
    assert engine.calls == {"force": 1}


def test_compute_is_served_from_the_store(cached, engine):
    mol = water()
    first = cached.compute(mol, hessian=True)
    second = cached.compute(mol, hessian=True)
    without = cached.compute(mol, hessian=False)
    assert engine.calls == {"freq": 1}
    assert second.energy == first.energy == without.energy
    np.testing.assert_array_equal(second.gradient, first.gradient)
    np.testing.assert_array_equal(second.hessian, first.hessian)
    assert without.hessian is None


def test_compute_needs_every_record(cached, engine):
    mol = water()
    cached.compute(mol, hessian=False)
    result = cached.compute(mol, hessian=True)
    assert engine.calls == {"force": 1, "freq": 1}
    assert result.hessian.shape == (9, 9)
    store = cached._store
    store.put(key_of(store, "gradient"), {})
    cached.compute(mol, hessian=False)
    assert engine.calls == {"force": 2, "freq": 1}


def test_wraps_another_cached_engine(engine, store):
    outer = CachedEngine(CachedEngine(engine, MemoryStore()), store)
    outer.hessian(water())
    outer.hessian(water())
    outer.energy(water())
    assert engine.calls == {"freq": 1}
    assert outer.level == engine.level


def test_returned_arrays_can_be_modified(cached):
    gradient = cached.gradient(water())
    gradient[:] = 99.0
    result = cached.compute(water(), hessian=True)
    result.gradient[:] = 99.0
    result.hessian[:] = 99.0
    reference = CountingEngine()
    np.testing.assert_array_equal(cached.gradient(water()), reference.gradient(water()))
    np.testing.assert_array_equal(cached.hessian(water()), reference.hessian(water()))


def test_different_basis_is_a_miss(store):
    svp, tzvp = CountingEngine("def2-svp"), CountingEngine("def2-tzvp")
    CachedEngine(svp, store).hessian(water())
    CachedEngine(tzvp, store).hessian(water())
    assert tzvp.calls["freq"] == 1


@pytest.mark.parametrize(
    "changes",
    [{"charge": 2}, {"mult": 3}],
    ids=["charge", "mult"],
)
def test_charge_and_mult_cause_a_miss(cached, engine, changes):
    cached.energy(water())
    cached.energy(water(**changes))
    assert engine.calls["force"] == 2


def test_coordinate_change_beyond_resolution_is_a_miss(cached, engine):
    mol = water()
    coords = mol.coords.copy()
    coords[1, 1] += 2 * KEY_COORD_RESOLUTION
    cached.gradient(mol)
    cached.gradient(mol.replace(coords=coords))
    assert engine.calls["force"] == 2


def test_isotopologue_reuses_the_hessian(cached, engine):
    ch4 = methane()
    masses = ch4.masses.copy()
    masses[1] = 2.01410177812
    ch3d = ch4.replace(masses=masses, name="CH3D")
    h = cached.hessian(ch4)
    np.testing.assert_array_equal(cached.hessian(ch3d), h)
    assert engine.calls["freq"] == 1


def test_different_engine_is_a_miss(store):
    a, b = CountingEngine(), OtherEngine()
    assert a.level == b.level
    CachedEngine(a, store).energy(water())
    CachedEngine(b, store).energy(water())
    assert b.calls["force"] == 1
    CachedEngine(OtherEngine(), store).energy(water())
    assert len(list(store.keys())) == 4  # energy and gradient per engine


def test_engine_id_defaults_to_class_name(cached):
    assert cached.engine_id == f"{__name__}.CountingEngine"


def test_engine_id_can_be_pinned(store):
    a, b = CountingEngine(), OtherEngine()
    CachedEngine(a, store, engine_id="mine").energy(water())
    CachedEngine(b, store, engine_id="mine").energy(water())
    assert b.calls["force"] == 0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gradient", np.zeros((3, 2)), "shape"),
        ("gradient", np.full((3, 3), np.nan), "non-finite"),
        ("gradient", [[0.0] * 3] * 3, "numpy array"),
        ("gradient", np.zeros((3, 3), dtype=complex), "real"),
        ("hessian", np.zeros((3, 3)), "shape"),
        ("hessian", np.full((9, 9), np.inf), "non-finite"),
        ("hessian", None, "no Hessian"),
        ("energy", float("nan"), "finite"),
        ("energy", np.zeros(1), "float"),
        ("energy", True, "float"),
        ("energy", None, "float"),
    ],
)
@pytest.mark.parametrize("method", ["energy", "gradient", "hessian"])
def test_invalid_results_are_not_cached(store, method, field, value, match):
    if field == "hessian" and method != "hessian":
        pytest.skip("only a Hessian request checks the Hessian")
    engine = BrokenEngine(bad={field: value})
    cached = CachedEngine(engine, store)
    for _ in range(2):
        with pytest.raises(ValueError, match=match):
            getattr(cached, method)(water())
    assert sum(engine.calls.values()) == 2
    assert list(store.keys()) == []


def test_result_must_be_an_engine_result(store):
    engine = BrokenEngine(result=(0.0, np.zeros((3, 3))))
    with pytest.raises(ValueError, match="EngineResult"):
        CachedEngine(engine, store).energy(water())
    assert list(store.keys()) == []


def test_numeric_results_are_converted(store):
    engine = BrokenEngine(
        bad={"energy": np.float32(-1.5), "gradient": np.zeros((3, 3), dtype=np.int64)}
    )
    cached = CachedEngine(engine, store)
    assert type(cached.energy(water())) is float
    assert cached.gradient(water()).dtype == np.float64


def test_engine_failure_is_not_cached(store):
    error = RuntimeError("SCF not converged")
    engine = BrokenEngine(result=error)
    cached = CachedEngine(engine, store)
    with pytest.raises(RuntimeError) as excinfo:
        cached.energy(water())
    assert excinfo.value is error
    assert list(store.keys()) == []


@pytest.mark.parametrize("kind", ["energy", "gradient", "hessian"])
def test_records_are_self_describing(cached, engine, store, kind):
    mol = water()
    cached.hessian(mol)
    assert len(list(store.keys())) == 3
    record = store.get(key_of(store, kind))
    assert set(record) == {kind, "level", "engine", "symbols", "coords"}
    assert record["level"] == engine.level.key()
    assert record["engine"] == cached.engine_id
    assert record["symbols"] == "O H H"
    np.testing.assert_array_equal(record["coords"], mol.coords)


def key_of(store, kind):
    """The key of the one record holding a result of this kind."""
    (key,) = (k for k in store.keys(prefix="") if kind in store.get(k))
    return key


def tamper(store, kind, **fields):
    key = key_of(store, kind)
    record = store.get(key)
    record.update(fields)
    store.put(key, record)
    return key


@pytest.mark.parametrize(
    "fields",
    [
        {"level": "b3lyp/def2-tzvp"},
        {"engine": "someone.Else"},
        {"symbols": "O H He"},
        {"symbols": "Xx H H"},
        {"symbols": 3},
        {"coords": np.zeros((3, 3))},
        {"coords": np.zeros(2)},
        {"coords": "nope"},
        {"hessian": np.zeros((2, 2))},
        {"hessian": "nope"},
    ],
    ids=lambda f: "-".join(f"{k}={v!r:.12}" for k, v in f.items()),
)
def test_mismatched_record_is_recomputed_and_overwritten(cached, engine, store, fields):
    mol = water()
    expected = cached.hessian(mol)
    key = tamper(store, "hessian", **fields)
    np.testing.assert_array_equal(cached.hessian(mol), expected)
    assert engine.calls["freq"] == 2
    record = store.get(key)
    assert record["level"] == engine.level.key()
    np.testing.assert_array_equal(record["hessian"], expected)
    cached.hessian(mol)
    assert engine.calls["freq"] == 2


def test_record_missing_fields_is_recomputed(cached, engine, store):
    cached.energy(water())
    store.put(key_of(store, "energy"), {})
    cached.energy(water())
    assert engine.calls["force"] == 2


RUN_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from pyvtst.engines import CachedEngine
from pyvtst.stores import HDF5Store
from tests.engines.test_cached import CountingEngine, methane, water

engine = CountingEngine()
cached = CachedEngine(engine, HDF5Store(sys.argv[2]))
results = []
for mol in (water(), water(charge=2), methane()):
    results.append([
        cached.energy(mol),
        cached.gradient(mol).tolist(),
        cached.hessian(mol).tolist(),
    ])
print(json.dumps({"calls": dict(engine.calls), "results": results}))
"""


def test_zero_engine_calls_in_a_new_process(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "cache.h5"

    def run():
        out = subprocess.run(
            [sys.executable, "-c", RUN_SCRIPT, str(ROOT), str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout)

    first, second = run(), run()
    assert first["calls"] == {"force": 3, "freq": 3}
    assert second["calls"] == {}
    assert second["results"] == first["results"]
