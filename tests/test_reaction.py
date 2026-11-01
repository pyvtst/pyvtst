import warnings
from collections import Counter

import numpy as np
import pytest

from pyvtst import Molecule, RateResult, Reaction, ReactionPath
from pyvtst.core.mep import generalized_zpe
from pyvtst.core.rates import tst_rate, zpe_energy
from pyvtst.core.stationary import StationaryPointError
from pyvtst.core.tunneling import eckart
from pyvtst.core.units import HARTREE_TO_KCAL_PER_MOL
from pyvtst.core.vib import curvilinear_generalized_modes, generalized_normal_modes
from pyvtst.engines import CachedEngine
from pyvtst.optimizers import GeometricOptimizer
from pyvtst.ports import ConvergenceError, OptimizationResult
from pyvtst.stores import MemoryStore
from tests.fakes.leps import R0, LEPSEngine, collinear_h3, saddle_distance


class CountingEngine:
    """The LEPS engine, counting calls per method.

    With ``fail_after``, gradient calls beyond that many raise, like a crash.
    """

    def __init__(self, fail_after=None):
        self.inner = LEPSEngine()
        self.calls: Counter[str] = Counter()
        self.gradient_coords: list[np.ndarray] = []
        self.fail_after = fail_after

    @property
    def level(self):
        return self.inner.level

    def energy(self, mol):
        self.calls["energy"] += 1
        return self.inner.energy(mol)

    def gradient(self, mol):
        if self.fail_after is not None and self.calls["gradient"] >= self.fail_after:
            raise RuntimeError("engine crashed")
        self.calls["gradient"] += 1
        self.gradient_coords.append(np.array(mol.coords))
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


def test_zct_not_yet_available(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(NotImplementedError, match=r"zct.*not implemented yet"):
        rxn.rates(300, tunneling="zct")


def test_unknown_theory(engine, h2_stretched, h_atom, h3_ts_guess):
    rxn = Reaction([h2_stretched, h_atom], h3_ts_guess, engine)
    with pytest.raises(ValueError, match="'tst' or 'cvt'"):
        rxn.rates(300, theory="icvt")


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


# Minimum-energy path

D_MASS = 2.014101778  # amu


def exact_optimizer():
    """Returns the exact LEPS saddle point and H2 minimum without optimizing."""
    r = saddle_distance()
    return CountingOptimizer(
        FixedOptimizer(
            saddle=[[-r, 0.0, 0.0], [0.0, 0.0, 0.0], [r, 0.0, 0.0]],
            minimum=[[0.0, 0.0, 0.0], [R0, 0.0, 0.0]],
        )
    )


def mep_reaction(engine, store=None, masses=(None, None, None)):
    """H2 + H on LEPS with exact stationary points; masses per role."""
    h2_masses, h_masses, ts_masses = masses
    h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], masses=h2_masses)
    h = Molecule(["H"], [[0.0, 0.0, 0.0]], mult=2, masses=h_masses)
    ts = collinear_h3(1.6, 1.9, masses=ts_masses)
    return Reaction([h2, h], ts, engine, optimizer=exact_optimizer(), store=store)


@pytest.fixture(scope="module")
def lqa_path():
    return mep_reaction(LEPSEngine()).compute_mep()


def mass_scaled(path):
    return np.array(
        [np.repeat(np.sqrt(m.masses), 3) * m.coords.ravel() for m in path.molecules]
    )


def test_auto_run_prerequisites_for_the_path(rxn, geometric_optimizer):
    path = rxn.compute_mep()
    assert geometric_optimizer.calls
    assert isinstance(path, ReactionPath)
    assert path.s[0] == pytest.approx(-0.5)
    assert path.s[-1] == pytest.approx(0.5)


def test_requested_limits():
    path = mep_reaction(LEPSEngine()).compute_mep(s_min=-0.4, s_max=0.4)
    assert path.s[0] == pytest.approx(-0.4, abs=1e-12)
    assert path.s[-1] == pytest.approx(0.4, abs=1e-12)
    assert (path.stop_reactant, path.stop_product) == ("s_limit", "s_limit")
    assert len(path) == 81


def test_first_points_straddle_the_saddle(lqa_path):
    i = lqa_path.ts_index
    assert lqa_path.energies[i - 1] < lqa_path.energies[i]
    assert lqa_path.energies[i + 1] < lqa_path.energies[i]


def test_energy_decreases_away_from_the_saddle(lqa_path):
    i = lqa_path.ts_index
    assert np.all(np.diff(lqa_path.energies[: i + 1]) > 0)
    assert np.all(np.diff(lqa_path.energies[i:]) < 0)


def test_points_are_evenly_spaced_in_arc_length(lqa_path):
    np.testing.assert_allclose(
        lqa_path.s / 0.01, np.round(lqa_path.s / 0.01), rtol=0, atol=1e-10
    )
    steps = np.linalg.norm(np.diff(mass_scaled(lqa_path), axis=0), axis=1)
    np.testing.assert_allclose(steps, 0.01, rtol=0, atol=1e-4)


def test_mode_count_along_a_collinear_path(lqa_path):
    assert all(f is not None and f.shape == (3,) for f in lqa_path.frequencies)
    assert np.all(np.isfinite(lqa_path.zpe))


def test_saddle_point_entry(lqa_path):
    rxn = mep_reaction(LEPSEngine())
    points = rxn.optimize()
    barrier = tst_rate(points.reactants, points.ts, [300.0]).barrier_zpe
    assert lqa_path.s[lqa_path.ts_index] == 0.0
    assert lqa_path.v_ag[lqa_path.ts_index] == pytest.approx(barrier, abs=1e-10)
    assert lqa_path.v_mep[lqa_path.ts_index] == pytest.approx(
        points.ts.energy - sum(r.energy for r in points.reactants), abs=1e-14
    )


def test_path_metadata(lqa_path):
    assert lqa_path.method == "lqa"
    assert lqa_path.ds == 0.01
    assert lqa_path.oriented
    assert lqa_path.gradients.shape == (101, 3, 3)
    assert lqa_path.molecules[lqa_path.ts_index].coords[2, 0] == pytest.approx(
        saddle_distance()
    )


# Argument validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s_max": -0.2},
        {"s_min": 0.1},
        {"s_min": float("nan")},
        {"s_max": "0.5"},
        {"ds": 0.0},
        {"ds": 0.2},
        {"method": "euler", "hessian_every": 0},
        {"method": "euler", "hessian_every": 2.5},
        {"method": "euler", "hessian_every": True},
    ],
)
def test_invalid_arguments_fail_fast(kwargs):
    engine = CountingEngine()
    rxn = mep_reaction(engine)
    with pytest.raises(ValueError, match=list(kwargs)[-1]):
        rxn.compute_mep(**kwargs)
    assert sum(engine.calls.values()) == 0
    assert rxn.optimizer.calls == []


def test_unknown_method():
    engine = CountingEngine()
    rxn = mep_reaction(engine)
    with pytest.raises(ValueError, match="'lqa' or 'euler'"):
        rxn.compute_mep(method="rk4")
    assert sum(engine.calls.values()) == 0
    assert rxn.optimizer.calls == []


def test_lqa_with_sparse_hessians():
    engine = CountingEngine()
    with pytest.raises(ValueError, match="LQA needs a Hessian at every point"):
        mep_reaction(engine).compute_mep(hessian_every=5)
    assert sum(engine.calls.values()) == 0


def test_euler_defaults():
    path = mep_reaction(LEPSEngine()).compute_mep(-0.03, 0.03, method="euler")
    assert path.ds == 0.001
    np.testing.assert_array_equal(
        np.flatnonzero(path.has_frequencies), [0, 10, 20, 30, 40, 50, 60]
    )


# Variational TST


def test_cvt_runs_the_path_when_needed():
    engine = CountingEngine()
    rxn = mep_reaction(engine)
    result = rxn.rates([300.0, 1000.0], theory="cvt")
    assert engine.calls["hessian"] > 3
    assert result.theory == "cvt"
    assert result.cvt is not None
    assert result.cvt.s[0] == pytest.approx(-0.5)
    assert result.cvt.s[-1] == pytest.approx(0.5)
    assert np.all(result.cvt.k <= result.k_tst)


def test_cvt_reuses_the_stored_path():
    store = MemoryStore()
    mep_reaction(CountingEngine(), store).compute_mep()
    engine = CountingEngine()
    mep_reaction(engine, store).rates(300.0, theory="cvt")
    assert sum(engine.calls.values()) == 0


def test_cvt_on_a_given_path():
    rxn = mep_reaction(LEPSEngine())
    p = rxn.compute_mep(s_min=-1.0, s_max=1.0, coordinates="cartesian")
    result = rxn.rates(300.0, theory="cvt", path=p)
    assert result.cvt is not None
    np.testing.assert_array_equal(result.cvt.s, p.s[p.has_frequencies])
    assert result.cvt.s[0] < -0.5
    assert result.cvt.s[-1] > 0.5


@pytest.mark.filterwarnings("ignore:the transition state's atoms do not match")
def test_path_from_another_reaction():
    other = mep_reaction(LEPSEngine(), masses=(None, None, [D_MASS, 1.008, 1.008]))
    p = other.compute_mep(s_min=-0.1, s_max=0.1)
    rxn = mep_reaction(LEPSEngine())
    with pytest.raises(ValueError, match="does not belong to this reaction"):
        rxn.rates(300.0, theory="cvt", path=p)


def test_path_without_cvt(lqa_path):
    engine = CountingEngine()
    rxn = mep_reaction(engine)
    with pytest.raises(ValueError, match='only with theory="cvt"'):
        rxn.rates(300.0, path=lqa_path)
    assert sum(engine.calls.values()) == 0
    assert not rxn.optimizer.calls


def test_path_must_be_a_reaction_path():
    rxn = mep_reaction(CountingEngine())
    with pytest.raises(TypeError, match="ReactionPath"):
        rxn.rates(300.0, theory="cvt", path=[0.0, 0.1])


def test_cvt_arrays_are_consistent():
    rxn = mep_reaction(LEPSEngine())
    t = [300.0, 600.0, 1000.0]
    result = rxn.rates(t, theory="cvt", tunneling="eckart")
    tst = rxn.rates(t, tunneling="eckart")
    assert result.cvt is not None
    assert tst.cvt is None
    np.testing.assert_array_equal(result.k, result.cvt.k * result.kappa)
    np.testing.assert_array_equal(result.k_tst, tst.k_tst)
    np.testing.assert_array_equal(result.kappa, tst.kappa)
    assert result.units == tst.units == result.cvt.units


def test_cvt_table_output():
    result = mep_reaction(LEPSEngine()).rates([300.0, 500.0, 1000.0], theory="cvt")
    lines = result.table().splitlines()
    assert len(lines) == 4
    assert "s*" in lines[0]
    assert "k_CVT" in lines[0]
    assert "cm^3 molecule^-1 s^-1" in lines[0]
    assert len(lines[1].split()) == 6


# Caching and resume


def test_repeated_call_is_free():
    store = MemoryStore()
    first = mep_reaction(CountingEngine(), store).compute_mep()
    engine = CountingEngine()
    second = mep_reaction(engine, store).compute_mep()
    assert sum(engine.calls.values()) == 0
    assert second == first


@pytest.fixture(params=["memory", "hdf5"])
def store_factory(request, tmp_path):
    if request.param == "memory":
        store = MemoryStore()
        return lambda: store
    pytest.importorskip("h5py")
    return lambda: str(tmp_path / "mep.h5")


def test_resume_after_interruption(store_factory):
    crashing = CountingEngine(fail_after=15)
    with pytest.raises(RuntimeError, match="crashed"):
        mep_reaction(crashing, store_factory()).compute_mep()
    stored = crashing.gradient_coords

    engine = CountingEngine()
    resumed = mep_reaction(engine, store_factory()).compute_mep()
    for coords in engine.gradient_coords:
        assert not any(np.allclose(coords, c, rtol=0, atol=1e-9) for c in stored)
    # Every point but the 15 stored ones, plus the transition state.
    assert engine.calls["gradient"] == len(resumed) - 1 - 15 + 1
    assert resumed == mep_reaction(LEPSEngine()).compute_mep()


def test_extending_the_range():
    store = MemoryStore()
    mep_reaction(CountingEngine(), store).compute_mep(s_min=-0.2, s_max=0.2)
    engine = CountingEngine()
    wide = mep_reaction(engine, store).compute_mep(s_min=-0.4, s_max=0.4)
    new = np.sum(np.abs(wide.s) > 0.2 + 1e-9)
    assert engine.calls["gradient"] == new == 40
    assert engine.calls["hessian"] == new
    assert wide == mep_reaction(LEPSEngine()).compute_mep(s_min=-0.4, s_max=0.4)


def test_shrinking_the_range_is_free():
    store = MemoryStore()
    full = mep_reaction(CountingEngine(), store).compute_mep()
    engine = CountingEngine()
    short = mep_reaction(engine, store).compute_mep(s_min=-0.2, s_max=0.1)
    assert sum(engine.calls.values()) == 0
    keep = (full.s >= -0.2 - 1e-9) & (full.s <= 0.1 + 1e-9)
    np.testing.assert_array_equal(short.energies, full.energies[keep])


def test_different_step_size_is_a_different_path():
    store = MemoryStore()
    mep_reaction(CountingEngine(), store).compute_mep(ds=0.01)
    engine = CountingEngine()
    coarse = mep_reaction(engine, store).compute_mep(ds=0.02)
    assert len(coarse) == 51
    assert engine.calls["gradient"] == 50
    prefixes = {k.rsplit("/", 2)[0] for k in store.keys("mep/")}
    assert len(prefixes) == 2


def test_another_engine_does_not_reuse_optimizations_or_path_points():
    # Same level of theory, different engine class.
    store = MemoryStore()
    path = mep_reaction(LEPSEngine(), store).compute_mep(-0.1, 0.1)
    engine = CountingEngine()
    rxn = mep_reaction(engine, store)
    other = rxn.compute_mep(-0.1, 0.1)
    assert [kind for kind, _ in rxn.optimizer.calls] == ["minimize", "saddle"]
    # Every point but the transition state is traced again.
    assert engine.calls["gradient"] == len(other)
    assert len({k.rsplit("/", 2)[0] for k in store.keys("mep/")}) == 2
    assert other == path


def test_pinned_engine_id_shares_the_cache():
    store = MemoryStore()
    mep_reaction(LEPSEngine(), store).compute_mep(-0.1, 0.1)
    engine = CountingEngine()
    pinned = CachedEngine(engine, store, engine_id="tests.fakes.leps.LEPSEngine")
    rxn = mep_reaction(pinned, store)
    rxn.compute_mep(-0.1, 0.1)
    assert sum(engine.calls.values()) == 0
    assert rxn.optimizer.calls == []


class RecordingStore(MemoryStore):
    """A memory store that records the keys written."""

    def __init__(self):
        super().__init__()
        self.written: list[str] = []

    def put(self, key, record):
        self.written.append(key)
        super().put(key, record)


def test_damaged_point_record_is_recomputed_with_its_successors():
    store = RecordingStore()
    first = mep_reaction(CountingEngine(), store).compute_mep()
    minus = sorted(k for k in store.keys("mep/") if "/minus/" in k)
    assert len(minus) == 50
    store.put(minus[9], {"coords": np.zeros((2, 3)), "energy": 0.0})
    store.written.clear()

    again = mep_reaction(CountingEngine(), store).compute_mep()
    assert [k for k in store.written if k.startswith("mep/")] == minus[9:]
    repaired = store.get(minus[9])
    assert repaired is not None
    assert np.shape(repaired["coords"]) == (3, 3)
    assert again == first


# Methods, masses and orientation


def test_methods_agree():
    rxn = mep_reaction(LEPSEngine())
    lqa = rxn.compute_mep(-0.3, 0.3, ds=0.01)
    euler = rxn.compute_mep(-0.3, 0.3, ds=0.0005, method="euler")
    for s in (-0.3, 0.3):
        a = lqa.v_mep[np.argmin(np.abs(lqa.s - s))]
        b = euler.v_mep[np.argmin(np.abs(euler.s - s))]
        assert abs(a - b) * HARTREE_TO_KCAL_PER_MOL < 0.05


def test_euler_hessian_spacing():
    engine = CountingEngine()
    path = mep_reaction(engine).compute_mep(-0.1, 0.1, method="euler", hessian_every=10)
    with_freqs = np.round(path.s[path.has_frequencies] / path.ds).astype(int)
    np.testing.assert_array_equal(with_freqs, np.arange(-100, 101, 10))
    assert all(
        f is None
        for f, h in zip(path.frequencies, path.has_frequencies, strict=True)
        if not h
    )
    assert np.all(np.isnan(path.zpe[~path.has_frequencies]))
    # The two stationary-point Hessians plus one per tenth point.
    assert engine.calls["hessian"] == 2 + 20


def test_isotope_substitution_changes_the_path():
    d = [D_MASS] * 3
    h_path = mep_reaction(LEPSEngine()).compute_mep(-0.1, 0.1)
    d_path = mep_reaction(LEPSEngine(), masses=(d[:2], d[:1], d)).compute_mep(-0.1, 0.1)
    assert d_path.energies[-1] > h_path.energies[-1]


def test_asymmetric_isotopic_reaction():
    masses = ([D_MASS, D_MASS], None, [D_MASS, D_MASS, 1.00782503223])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = mep_reaction(LEPSEngine(), masses=masses).compute_mep(-0.2, 0.2)
    assert path.oriented

    def dd(mol):
        return abs(np.linalg.norm(mol.coords[0] - mol.coords[1]) - R0)

    assert dd(path.molecules[0]) < dd(path.molecules[-1])


def test_atom_order_mismatch():
    masses = ([D_MASS, D_MASS], None, [1.00782503223, D_MASS, D_MASS])
    rxn = mep_reaction(LEPSEngine(), masses=masses)
    with pytest.warns(UserWarning, match="sign of s is arbitrary"):
        path = rxn.compute_mep(-0.1, 0.1)
    assert not path.oriented
    assert len(path) == 21


def test_side_stopping_early_warns():
    rxn = mep_reaction(LEPSEngine())
    with pytest.warns(UserWarning, match=r"product side stopped at s = \+"):
        path = rxn.compute_mep(-0.2, 30.0, ds=0.1, method="euler", hessian_every=1000)
    assert path.stop_product in ("minimum", "energy_rise")
    assert path.stop_reactant == "s_limit"
    assert path.s[-1] < 30.0


def test_large_imaginary_generalized_frequency_warns(monkeypatch):
    def with_imaginary(mol, hessian, gradient=None, direction=None):
        freqs, modes = generalized_normal_modes(mol, hessian, gradient, direction)
        if gradient is not None:
            freqs = np.array([-200.0, *freqs[1:]])
        return freqs, modes

    monkeypatch.setattr("pyvtst.reaction.generalized_normal_modes", with_imaginary)
    with pytest.warns(UserWarning, match=r"s = -0\.0100 \(\[-200\.0\] cm\^-1\)"):
        path = mep_reaction(LEPSEngine()).compute_mep(
            -0.01, 0.01, coordinates="cartesian"
        )
    assert path.frequencies[0][0] == -200.0
    assert path.zpe[0] == pytest.approx(generalized_zpe(path.frequencies[0][1:]))


# Coordinates of the generalized frequencies

H3_INTERNALS = (("bond", 0, 1), ("bond", 1, 2), ("linear_bend", 0, 1, 2))


def test_default_is_curvilinear(lqa_path):
    assert lqa_path.coordinates == "redundant"
    assert lqa_path.internals == H3_INTERNALS
    assert all(f is not None and f.shape == (3,) for f in lqa_path.frequencies)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coordinates": "polar"}, "coordinates"),
        ({"coordinates": "cartesian", "internals": [("bond", 0, 1)]}, "internals"),
        ({"internals": [("bond", 0, 7)]}, r"\('bond', 0, 7\)"),
        ({"internals": [("wag", 0, 1, 2)]}, "wag"),
    ],
)
def test_invalid_coordinate_arguments_fail_fast(kwargs, match):
    engine = CountingEngine()
    rxn = mep_reaction(engine)
    with pytest.raises(ValueError, match=match):
        rxn.compute_mep(**kwargs)
    assert sum(engine.calls.values()) == 0
    assert rxn.optimizer.calls == []


def test_insufficient_internal_coordinates_at_the_transition_state():
    store = MemoryStore()
    rxn = mep_reaction(LEPSEngine(), store)
    with pytest.raises(ValueError, match="2 of 4 internal directions at the trans"):
        rxn.compute_mep(internals=[("bond", 0, 1), ("bond", 1, 2)])
    assert list(store.keys("mep/")) == []


def test_coverage_error_names_the_point(monkeypatch):
    def bonds_only(mol, hessian, internals, *args, **kwargs):
        return curvilinear_generalized_modes(mol, hessian, internals[:2], **kwargs)

    monkeypatch.setattr("pyvtst.reaction.curvilinear_generalized_modes", bonds_only)
    with pytest.raises(ValueError, match=r"2 of 4 .* at s = -0\.0100 bohr"):
        mep_reaction(LEPSEngine()).compute_mep(-0.01, 0.01)


def test_automatic_set_is_recorded():
    store = MemoryStore()
    path = mep_reaction(LEPSEngine(), store).compute_mep(-0.1, 0.1)
    again = mep_reaction(LEPSEngine(), store).compute_mep(
        -0.1, 0.1, internals=path.internals
    )
    assert again == path


def test_switching_coordinates_is_free():
    store = MemoryStore()
    redundant = mep_reaction(CountingEngine(), store).compute_mep(-0.1, 0.1)
    engine = CountingEngine()
    cartesian = mep_reaction(engine, store).compute_mep(
        -0.1, 0.1, coordinates="cartesian"
    )
    assert sum(engine.calls.values()) == 0
    assert cartesian.coordinates == "cartesian"
    assert cartesian.internals == ()
    np.testing.assert_array_equal(cartesian.s, redundant.s)
    np.testing.assert_array_equal(cartesian.energies, redundant.energies)
    np.testing.assert_array_equal(cartesian.gradients, redundant.gradients)
    assert cartesian != redundant


def inject_imaginary(analysis):
    """Wrap a frequency analysis so that every point gets -200 cm^-1."""

    def wrapped(*args, **kwargs):
        freqs, modes = analysis(*args, **kwargs)
        return np.array([-200.0, *freqs[1:]]), modes

    return wrapped


def test_cartesian_warning_suggests_internal_coordinates(monkeypatch):
    monkeypatch.setattr(
        "pyvtst.reaction.generalized_normal_modes",
        inject_imaginary(generalized_normal_modes),
    )
    with pytest.warns(UserWarning, match=r"s = -0\.4000 .*coordinates=\"redundant\""):
        mep_reaction(LEPSEngine()).compute_mep(
            -0.4, 0.4, ds=0.1, coordinates="cartesian"
        )


def test_redundant_warning_has_no_hint(monkeypatch):
    monkeypatch.setattr(
        "pyvtst.reaction.curvilinear_generalized_modes",
        inject_imaginary(curvilinear_generalized_modes),
    )
    with pytest.warns(UserWarning, match=r"s = -0\.4000") as record:
        mep_reaction(LEPSEngine()).compute_mep(-0.4, 0.4, ds=0.1)
    assert all("coordinates=" not in str(w.message) for w in record)
