import warnings
from collections import Counter

import numpy as np
import pytest

from pyvtst import Molecule, RateResult, Reaction
from pyvtst.core.rates import zpe_energy
from pyvtst.core.stationary import StationaryPointError
from pyvtst.core.tunneling import eckart
from pyvtst.engines import CachedEngine
from pyvtst.optimizers import GeometricOptimizer
from pyvtst.ports import ConvergenceError, OptimizationResult
from pyvtst.stores import MemoryStore
from tests.fakes.leps import R0, LEPSEngine


class CountingEngine:
    """The LEPS engine, counting calls per method."""

    def __init__(self):
        self.inner = LEPSEngine()
        self.calls: Counter[str] = Counter()

    @property
    def level(self):
        return self.inner.level

    def energy(self, mol):
        self.calls["energy"] += 1
        return self.inner.energy(mol)

    def gradient(self, mol):
        self.calls["gradient"] += 1
        return self.inner.gradient(mol)

    def hessian(self, mol):
        self.calls["hessian"] += 1
        return self.inner.hessian(mol)


class CountingOptimizer:
    """Wraps another optimizer and records the molecules it was asked to optimize."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[str, Molecule]] = []

    def key(self):
        return self.inner.key()

    def minimize(self, mol, engine):
        self.calls.append(("minimize", mol))
        return self.inner.minimize(mol, engine)

    def saddle(self, mol, engine):
        self.calls.append(("saddle", mol))
        return self.inner.saddle(mol, engine)


class FixedOptimizer:
    """Returns fixed coordinates (or the input, for None) without optimizing."""

    def __init__(self, saddle=None, minimum=None, error=None):
        self.coords = {"saddle": saddle, "minimum": minimum}
        self.error = error

    def key(self):
        return "fixed"

    def minimize(self, mol, engine):
        return self._result(mol, engine, "minimum")

    def saddle(self, mol, engine):
        return self._result(mol, engine, "saddle")

    def _result(self, mol, engine, kind):
        if self.error is not None:
            raise self.error
        coords = self.coords[kind]
        out = mol if coords is None else mol.replace(coords=coords)
        return OptimizationResult(out, engine.energy(out), 0)


@pytest.fixture
def geometric_optimizer():
    pytest.importorskip("geometric")
    return CountingOptimizer(GeometricOptimizer())


@pytest.fixture
def engine():
    return CountingEngine()


@pytest.fixture
def rxn(engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer):
    return Reaction(
        [h2_stretched, h_atom], h3_ts_guess, engine, optimizer=geometric_optimizer
    )


def ch4():
    c = 1.19
    return Molecule(
        ["C", "H", "H", "H", "H"],
        [[0.0, 0.0, 0.0], [c, c, c], [-c, -c, c], [-c, c, -c], [c, -c, -c]],
    )


# Construction


def test_valid_reaction_makes_no_engine_calls(
    engine, h2_stretched, h_atom, h3_ts_guess
):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    assert sum(engine.calls.values()) == 0
    assert isinstance(rxn.engine, CachedEngine)
    assert isinstance(rxn.store, MemoryStore)


def test_single_reactant_molecule_is_accepted(engine, h2_stretched):
    rxn = Reaction(h2_stretched, h2_stretched, engine, optimizer=FixedOptimizer())
    assert rxn.reactants == (h2_stretched,)


def test_element_mismatch(engine, h_atom, h3_ts_guess):
    with pytest.raises(ValueError, match=r"reactants CH5, transition state H3"):
        Reaction([ch4(), h_atom], h3_ts_guess, engine)


def test_product_element_mismatch(engine, h2_stretched, h_atom, h3_ts_guess):
    with pytest.raises(ValueError, match="products H2"):
        Reaction([h2_stretched, h_atom], h3_ts_guess, engine, products=[h2_stretched])


def test_charge_mismatch(engine, h2_stretched, h_atom, h3_ts_guess):
    anion = h3_ts_guess.replace(charge=-1, mult=1)
    with pytest.raises(
        ValueError, match=r"charge differs: reactants 0, transition state -1"
    ):
        Reaction([h2_stretched, h_atom], anion, engine)


@pytest.mark.parametrize("n", [0, 3])
def test_reactant_count(engine, h_atom, h3_ts_guess, n):
    with pytest.raises(ValueError, match="one or two"):
        Reaction([h_atom] * n, h3_ts_guess, engine)


def test_empty_products_are_rejected(engine, h2_stretched, h_atom, h3_ts_guess):
    with pytest.raises(ValueError, match="products"):
        Reaction([h2_stretched, h_atom], h3_ts_guess, engine, products=[])


def test_spin_mismatch_warns(engine, h2_stretched, h_atom, h3_ts_guess):
    quartet = h3_ts_guess.replace(mult=4)
    with pytest.warns(
        UserWarning, match="multiplicities 2, but the transition state has 4"
    ):
        Reaction([h2_stretched, h_atom], quartet, engine)


def test_matching_spin_does_not_warn(engine, h2_stretched, h_atom, h3_ts_guess):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Reaction(
            [h2_stretched, h_atom], h3_ts_guess, engine, products=[h_atom, h2_stretched]
        )


def test_already_cached_engine_is_not_wrapped_again(
    engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    store = MemoryStore()
    cached = CachedEngine(engine, store)
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        cached,
        optimizer=geometric_optimizer,
        store=store,
    )
    assert rxn.engine is cached
    rxn.optimize()

    engine_keys = [
        k
        for k in list(store.keys())
        if k.split("/")[0] in ("energy", "gradient", "hessian")
    ]
    assert engine_keys
    records = [store.get(k) for k in engine_keys]
    ids = {r["engine"] for r in records if r is not None}
    assert ids == {"tests.test_reaction.CountingEngine"}
    assert list(store.keys("opt/"))


# Optimization


def test_h_plus_h2_on_the_analytic_surface(rxn):
    points = rxn.optimize()
    ts = points.ts
    assert ts.is_linear
    assert ts.n_imaginary == 1
    assert ts.kind == "saddle"
    h2 = points.reactants[0]
    assert h2.frequencies.shape == (1,)
    assert h2.frequencies[0] > 0
    assert np.linalg.norm(
        h2.molecule.coords[0] - h2.molecule.coords[1]
    ) == pytest.approx(R0, abs=1e-3)
    assert points.products is None


def test_single_atoms_are_not_optimized(rxn, geometric_optimizer, h_atom):
    points = rxn.optimize()
    assert all(mol.n_atoms > 1 for _, mol in geometric_optimizer.calls)
    atom = points.reactants[1]
    assert atom.molecule == h_atom
    assert atom.energy == 0.0
    assert atom.frequencies.shape == (0,)


def test_guess_converges_to_a_minimum(engine, h2_stretched, h_atom, h3_ts_guess):
    separated = [[0.0, 0.0, 0.0], [R0, 0.0, 0.0], [R0 + 9.0, 0.0, 0.0]]
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        optimizer=FixedOptimizer(saddle=separated),
    )
    with (
        warnings.catch_warnings(action="ignore"),  # near-zero modes of the far H
        pytest.raises(
            StationaryPointError,
            match="transition state 'H3' has 0 imaginary frequencies, expected 1",
        ),
    ):
        rxn.optimize()


def test_saddle_as_reactant_is_rejected(engine, h3_ts_guess, rxn):
    ts = rxn.optimize().ts
    saddle_opt = FixedOptimizer(minimum=ts.molecule.coords)
    bad = Reaction(h3_ts_guess, h3_ts_guess, engine, optimizer=saddle_opt)
    with pytest.raises(StationaryPointError, match=r"reactant 1 'H3' has 1 imaginary"):
        bad.optimize()


def test_convergence_error_names_the_species(engine, h2_stretched, h_atom, h3_ts_guess):
    last = h3_ts_guess.replace(coords=h3_ts_guess.coords * 1.01)
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        optimizer=FixedOptimizer(error=ConvergenceError("no luck", last)),
    )
    with pytest.raises(ConvergenceError, match="reactant 1 'H2': no luck") as info:
        rxn.optimize()
    assert info.value.last_molecule is last


def test_second_call_is_free(rxn, engine, geometric_optimizer):
    first = rxn.optimize()
    calls, n_opt = sum(engine.calls.values()), len(geometric_optimizer.calls)
    second = rxn.optimize()
    assert sum(engine.calls.values()) == calls
    assert len(geometric_optimizer.calls) == n_opt
    assert second == first


def test_new_reaction_on_same_store_reuses_optimizations(
    engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    store = MemoryStore()
    first = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        optimizer=geometric_optimizer,
        store=store,
    ).optimize()
    fresh = CountingEngine()
    opt = CountingOptimizer(geometric_optimizer.inner)
    second = Reaction(
        [h2_stretched, h_atom], h3_ts_guess, fresh, optimizer=opt, store=store
    ).optimize()
    assert sum(fresh.calls.values()) == 0
    assert opt.calls == []
    assert second == first


def test_damaged_optimization_record_is_recomputed(
    engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    store = MemoryStore()
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        optimizer=geometric_optimizer,
        store=store,
    )
    first = rxn.optimize()
    for key in store.keys("opt/"):
        store.put(key, {"coords": np.zeros((7, 3)), "energy": 0.0})
    opt = CountingOptimizer(geometric_optimizer.inner)
    again = Reaction(
        [h2_stretched, h_atom], h3_ts_guess, engine, optimizer=opt, store=store
    ).optimize()
    assert len(opt.calls) == 2
    np.testing.assert_allclose(again.ts.frequencies, first.ts.frequencies)


# Rates


def test_auto_run_prerequisites(rxn):
    result = rxn.rates([300, 1000])
    assert isinstance(result, RateResult)
    assert result.T.shape == (2,)
    assert result.units == "cm^3 molecule^-1 s^-1"
    assert np.all(np.isfinite(result.k) & (result.k > 0))
    np.testing.assert_array_equal(result.kappa, 1.0)
    assert result.theory == "tst"
    assert result.tunneling is None
    assert 0 < result.barrier_kcal < 10


def test_scalar_temperature(rxn):
    result = rxn.rates(500)
    for a in (result.T, result.k, result.kappa, result.k_tst):
        assert a.shape == (1,)


def test_rates_over_a_wide_range_and_second_call_is_free(rxn, engine):
    t = np.linspace(200.0, 2000.0, 10)
    first = rxn.rates(t, tunneling="eckart")
    calls = sum(engine.calls.values())
    second = rxn.rates(t, tunneling="eckart")
    assert sum(engine.calls.values()) == calls
    np.testing.assert_array_equal(second.k, first.k)
    assert np.all(np.isfinite(first.k) & (first.k > 0))


def test_cvt_not_yet_available(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(NotImplementedError, match=r"cvt.*not implemented yet"):
        rxn.rates(300, theory="cvt")
    assert sum(engine.calls.values()) == 0


def test_zct_not_yet_available(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(NotImplementedError, match=r"zct.*not implemented yet"):
        rxn.rates(300, tunneling="zct")


def test_unknown_theory(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(ValueError, match="'tst'"):
        rxn.rates(300, theory="vtst")


def test_unknown_tunneling_method(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(ValueError, match="None, 'wigner' or 'eckart'"):
        rxn.rates(300, tunneling="sct")


def test_invalid_temperature(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(ValueError, match="-5"):
        rxn.rates([300, -5])
    assert sum(engine.calls.values()) == 0


def test_eckart_is_symmetric_without_products(rxn):
    t = np.array([300.0, 600.0])
    result = rxn.rates(t, tunneling="eckart")
    points = rxn.optimize()
    nu = points.ts.imaginary_frequency
    assert nu is not None
    v1 = result.barrier_kcal
    assert v1 > 0
    barrier = zpe_energy(points.ts) - sum(zpe_energy(r) for r in points.reactants)
    np.testing.assert_array_equal(result.kappa, eckart(barrier, barrier, nu, t))


def test_eckart_uses_products(
    engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    h2_product = h2_stretched.replace(coords=h2_stretched.coords * 0.9)
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        products=[h_atom, h2_product],
        optimizer=geometric_optimizer,
    )
    t = np.array([300.0, 600.0])
    result = rxn.rates(t, tunneling="eckart")
    points = rxn.optimize()
    assert points.products is not None
    ts_level = zpe_energy(points.ts)
    v1 = ts_level - sum(zpe_energy(r) for r in points.reactants)
    v2 = ts_level - sum(zpe_energy(p) for p in points.products)
    nu = points.ts.imaginary_frequency
    assert nu is not None
    np.testing.assert_array_equal(result.kappa, eckart(v1, v2, nu, t))


# RateResult


def test_arrays_are_consistent(rxn):
    result = rxn.rates([300.0, 500.0, 1000.0], tunneling="wigner")
    np.testing.assert_array_equal(result.k, result.k_tst * result.kappa)
    assert np.all(result.kappa > 1)
    assert result.tunneling == "wigner"
    with pytest.raises(ValueError, match="read-only"):
        result.k[0] = 1.0


def test_table_output(rxn):
    result = rxn.rates([300.0, 500.0, 1000.0])
    lines = result.table().splitlines()
    assert len(lines) == 4
    assert "cm^3 molecule^-1 s^-1" in lines[0]
    assert lines[1].split()[0] == "300.00"


def test_result_is_immutable(rxn):
    result = rxn.rates(300.0)
    with pytest.raises(AttributeError):
        result.theory = "cvt"


# Persistent store


@pytest.fixture
def hdf5_path(tmp_path):
    pytest.importorskip("h5py")
    return tmp_path / "rxn.h5"


def test_path_creates_a_persistent_store(
    hdf5_path, engine, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    rxn = Reaction(
        [h2_stretched, h_atom],
        h3_ts_guess,
        engine,
        optimizer=geometric_optimizer,
        store=hdf5_path,
    )
    rxn.rates([300.0, 1000.0])
    assert hdf5_path.exists()


def test_persistent_store_makes_reruns_free(
    hdf5_path, h2_stretched, h_atom, h3_ts_guess, geometric_optimizer
):
    def run():
        engine = CountingEngine()
        opt = CountingOptimizer(geometric_optimizer.inner)
        rxn = Reaction(
            [h2_stretched, h_atom],
            h3_ts_guess,
            engine,
            optimizer=opt,
            store=str(hdf5_path),
        )
        return rxn.rates([300.0, 1000.0], tunneling="eckart"), engine, opt

    first, engine1, _ = run()
    assert sum(engine1.calls.values()) > 0
    second, engine2, opt2 = run()
    assert sum(engine2.calls.values()) == 0
    assert opt2.calls == []
    np.testing.assert_array_equal(second.k, first.k)
