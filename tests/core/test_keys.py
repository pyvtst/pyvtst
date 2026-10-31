import os
import subprocess
import sys

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.keys import KEY_COORD_RESOLUTION, cache_key, molecule_key
from pyvtst.core.level import LevelOfTheory
from pyvtst.ports.store import validate_key

WATER_COORDS = [[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8857], [0.0, -1.4309, -0.8857]]
LEVEL = LevelOfTheory("b3lyp", "def2-svp", (("grid_level", 3),))

# Reference keys for water() at LEVEL. They must never change unless the key
# scheme's version tags are bumped on purpose.
REF_MOLECULE_KEY = "6df05037e9e968fcbfedd63edaff3555"
REF_MOLECULE_KEY_NO_MASSES = "c9348e838f370e8413ed587d5783ae17"
REF_HESSIAN_KEY = (
    "hessian/f5949fb7b4c707de3e3d41551b2ba3a9/6df05037e9e968fcbfedd63edaff3555"
)
REF_ENGINE_HESSIAN_KEY = (
    "hessian/f5949fb7b4c707de3e3d41551b2ba3a9/c9348e838f370e8413ed587d5783ae17"
    "/b9a539a7db0418a1bba1cf52cd8ef3ce"
)
ENGINE_ID = "pyvtst.engines.pyscf.PySCFEngine"


def water(**changes):
    return Molecule(("O", "H", "H"), WATER_COORDS).replace(**changes)


def shifted(mol, delta, atom=1, axis=1):
    coords = mol.coords.copy()
    coords[atom, axis] += delta
    return mol.replace(coords=coords)


def test_reference_keys():
    mol = water()
    assert molecule_key(mol) == REF_MOLECULE_KEY
    assert molecule_key(mol, include_masses=False) == REF_MOLECULE_KEY_NO_MASSES
    assert cache_key("hessian", mol, LEVEL) == REF_HESSIAN_KEY
    assert (
        cache_key("hessian", mol, LEVEL, include_masses=False, engine=ENGINE_ID)
        == REF_ENGINE_HESSIAN_KEY
    )


KEY_SCRIPT = f"""
from pyvtst import Molecule
from pyvtst.core.keys import cache_key
from pyvtst.core.level import LevelOfTheory
mol = Molecule(("O", "H", "H"), {WATER_COORDS!r})
level = LevelOfTheory("b3lyp", "def2-svp", (("grid_level", 3),))
print(cache_key("hessian", mol, level, include_masses=False, engine={ENGINE_ID!r}))
"""


def test_keys_stable_across_processes():
    keys = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", KEY_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        keys.append(result.stdout.strip())
    assert keys == [REF_ENGINE_HESSIAN_KEY, REF_ENGINE_HESSIAN_KEY]


def test_key_is_a_32_char_hex_digest():
    key = molecule_key(water())
    assert len(key) == 32
    int(key, 16)


def test_name_and_symmetry_number_are_ignored():
    assert molecule_key(water()) == molecule_key(water(name="water", symmetry_number=2))


@pytest.mark.parametrize(
    "changes",
    [
        {"charge": 2},
        {"mult": 3},
        {"symbols": ("H", "O", "H")},
    ],
    ids=["charge", "mult", "symbol-order"],
)
def test_charge_mult_and_symbols_matter(changes):
    assert molecule_key(water()) != molecule_key(water(**changes))


def test_masses_matter_by_default():
    coords = [[0.0, 0.0, 0.0], [1.2, 1.2, 1.2], [-1.2, -1.2, 1.2], [-1.2, 1.2, -1.2]]
    coords.append([1.2, -1.2, -1.2])
    ch4 = Molecule(("C", "H", "H", "H", "H"), coords)
    masses = ch4.masses.copy()
    masses[1] = 2.01410177812
    ch3d = ch4.replace(masses=masses)
    assert molecule_key(ch4) != molecule_key(ch3d)
    assert molecule_key(ch4, include_masses=False) == molecule_key(
        ch3d, include_masses=False
    )


def test_float_noise_below_resolution():
    # 0.1234563 bohr sits 0.2 steps from the nearest cell boundary.
    base = water()
    coords = base.coords.copy()
    coords[0, 0] = 0.1234563
    a = base.replace(coords=coords)
    b = shifted(a, 1e-12, atom=0, axis=0)
    assert not np.array_equal(a.coords, b.coords)
    assert molecule_key(a) == molecule_key(b)


def test_real_displacement():
    assert molecule_key(water()) != molecule_key(
        shifted(water(), 2 * KEY_COORD_RESOLUTION)
    )


def test_signed_zero():
    pos = water()
    coords = pos.coords.copy()
    coords[0, 0] = -0.0
    neg = pos.replace(coords=coords)
    assert np.signbit(neg.coords[0, 0])
    assert molecule_key(pos) == molecule_key(neg)


def test_cartesian_frame_matters():
    mol = water()
    moved = mol.replace(coords=mol.coords + 0.1)
    assert molecule_key(mol) != molecule_key(moved)


def test_resolution_constant():
    assert KEY_COORD_RESOLUTION == 1e-6


def test_level_changes_the_key():
    svp = LevelOfTheory("b3lyp", "def2-svp")
    tzvp = LevelOfTheory("b3lyp", "def2-tzvp")
    assert cache_key("e", water(), svp) != cache_key("e", water(), tzvp)


def test_option_changes_the_key():
    grid3 = LevelOfTheory("b3lyp", "def2-svp", (("grid_level", 3),))
    grid4 = LevelOfTheory("b3lyp", "def2-svp", (("grid_level", 4),))
    assert cache_key("e", water(), grid3) != cache_key("e", water(), grid4)


def test_kind_changes_the_key():
    assert cache_key("energy", water(), LEVEL) != cache_key("gradient", water(), LEVEL)


def test_parameter_order_is_irrelevant():
    assert cache_key("x", water(), LEVEL, a=1, b="y") == cache_key(
        "x", water(), LEVEL, b="y", a=1
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ({"p": 1}, {"p": 1.0}),
        ({"p": 1}, {"p": True}),
        ({"p": 0}, {"p": None}),
        ({"p": "1"}, {"p": 1}),
        ({"p": 1}, {"q": 1}),
        ({"p": 1}, {"p": 2}),
        ({"p": 1}, {}),
    ],
)
def test_parameter_names_and_values_change_the_key(a, b):
    assert cache_key("x", water(), LEVEL, **a) != cache_key("x", water(), LEVEL, **b)


def test_numpy_scalars_are_converted():
    assert cache_key("x", water(), LEVEL, a=np.int64(3), b=np.float64(0.5)) == (
        cache_key("x", water(), LEVEL, a=3, b=0.5)
    )


def test_include_masses_is_not_a_parameter():
    key = cache_key("x", water(), LEVEL, include_masses=False)
    assert key.count("/") == 2
    assert key.endswith(REF_MOLECULE_KEY_NO_MASSES)


@pytest.mark.parametrize("value", [[1, 2], (1,), {"a": 1}, object()])
def test_unsupported_parameter_type(value):
    with pytest.raises(TypeError, match="'steps'"):
        cache_key("x", water(), LEVEL, steps=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_parameter(value):
    with pytest.raises(ValueError, match="'t'"):
        cache_key("x", water(), LEVEL, t=value)


@pytest.mark.parametrize("kind", ["", "a b", "a//b", "/a", "a/", ".a", "a/.b", "é"])
def test_invalid_kind(kind):
    with pytest.raises(ValueError, match="kind"):
        cache_key(kind, water(), LEVEL)


def test_overlong_kind():
    with pytest.raises(ValueError, match="kind"):
        cache_key("k" * 450, water(), LEVEL)


def test_shared_prefix():
    a = cache_key("mep/forward", water(), LEVEL, step=1)
    b = cache_key("mep/forward", shifted(water(), 0.5), LEVEL, step=1)
    prefix = a.rsplit("/", 2)[0] + "/"
    assert prefix.count("/") == 3
    assert prefix.startswith("mep/forward/")
    assert b.startswith(prefix)
    assert a != b


def test_any_object_with_key_is_a_level():
    class Custom:
        def key(self):
            return LEVEL.key()

    assert cache_key("x", water(), Custom()) == cache_key("x", water(), LEVEL)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("energy", {}),
        ("mep/forward", {"step": 3}),
        ("a.b-c_d/e", {"engine": "x.y", "t": 1.5, "flag": True, "none": None}),
    ],
)
@pytest.mark.parametrize("include_masses", [True, False])
def test_produced_keys_are_valid_store_keys(kind, params, include_masses):
    key = cache_key(kind, water(), LEVEL, include_masses=include_masses, **params)
    assert validate_key(key) == key
