import re

import numpy as np
import pytest

from pyvtst import Molecule
from pyvtst.core.internals import (
    KINDS,
    Internal,
    auto_internals,
    b_derivatives,
    check_span,
    validate_internals,
    values,
    wilson_b,
)
from pyvtst.core.units import ANGSTROM_TO_BOHR
from tests.core.hessians import random_rotation, water
from tests.fakes.leps import collinear_h3

STEP = 1e-5

# Four atoms with no symmetry: the 0-1-2 angle is about 98°, so every kind is
# well defined.
SKEW = Molecule(
    ["C", "O", "N", "H"],
    [[0.1, 0.2, -0.3], [1.9, 0.1, 0.4], [2.5, 1.8, -0.2], [4.1, 2.0, 0.9]],
)
ALL_KINDS: list[Internal] = [
    ("bond", 0, 1),
    ("angle", 0, 1, 2),
    ("linear_bend", 0, 1, 2),
    ("torsion", 0, 1, 2, 3),
]


def displaced(mol, p, h):
    step = np.zeros(3 * mol.n_atoms)
    step[p] = h
    return mol.replace(coords=mol.coords + step.reshape(-1, 3))


# Validation


def test_kinds():
    assert KINDS == ("bond", "angle", "linear_bend", "torsion")


def test_unknown_kind():
    with pytest.raises(ValueError, match=r"'wag'.*'bond', 'angle', 'linear_bend'"):
        validate_internals([("wag", 0, 1, 2)], 3)


@pytest.mark.parametrize("entry", [("bond", 0, 3), ("angle", 1, 1, 2)])
def test_bad_indices(entry):
    with pytest.raises(ValueError, match=re_entry(entry)):
        validate_internals([("bond", 0, 1), entry], 3)


@pytest.mark.parametrize(
    "entry", [("bond", 0), ("torsion", 0, 1, 2), ("bond", 0, 1.0), ("bond", True, 1)]
)
def test_malformed_entries(entry):
    with pytest.raises(ValueError, match=re_entry(entry)):
        validate_internals([entry], 4)


@pytest.mark.parametrize("entry", ["bond", 3, ()])
def test_entry_must_be_a_tuple(entry):
    with pytest.raises(ValueError, match="kind"):
        validate_internals([entry], 3)


def test_internals_must_be_a_sequence():
    with pytest.raises(ValueError, match="internals"):
        validate_internals("bond", 3)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("bond", 0, 1), ("bond", 1, 0)),
        (("angle", 0, 1, 2), ("angle", 2, 1, 0)),
        (("linear_bend", 0, 1, 2), ("linear_bend", 2, 1, 0)),
        (("torsion", 0, 1, 2, 3), ("torsion", 3, 2, 1, 0)),
        (("bond", 0, 1), ("bond", 0, 1)),
    ],
)
def test_duplicate_coordinate(first, second):
    with pytest.raises(ValueError, match=re_entry(second)):
        validate_internals([first, second], 4)


def test_angle_and_linear_bend_on_the_same_atoms_are_different():
    validate_internals([("angle", 0, 1, 2), ("linear_bend", 0, 1, 2)], 3)


def test_redundant_set():
    internals = [("bond", 0, 1), ("bond", 0, 2), ("bond", 1, 2), ("angle", 1, 0, 2)]
    assert validate_internals(internals, 3) == tuple(internals)


def test_lists_and_numpy_integers_are_normalized():
    assert validate_internals([["bond", np.int64(0), 1]], 2) == (("bond", 0, 1),)


def re_entry(entry):
    return re.escape(repr(tuple(entry)))


# Values and first derivatives


def test_values():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    mol = Molecule(["C", "C", "C", "C"], x)
    q = values(mol, [("bond", 0, 1), ("angle", 0, 1, 2), ("torsion", 0, 1, 2, 3)])
    np.testing.assert_allclose(q, [1.0, np.pi / 2, np.pi / 2], rtol=0, atol=1e-14)


def test_first_derivatives():
    b = wilson_b(SKEW, ALL_KINDS)
    assert b.shape == (5, 12)
    fd = np.column_stack(
        [
            (
                values(displaced(SKEW, p, STEP), ALL_KINDS, reference=SKEW)
                - values(displaced(SKEW, p, -STEP), ALL_KINDS, reference=SKEW)
            )
            / (2 * STEP)
            for p in range(12)
        ]
    )
    np.testing.assert_allclose(b, fd, rtol=0, atol=1e-7)


def test_translation_invariance():
    b = wilson_b(SKEW, ALL_KINDS)
    np.testing.assert_allclose(b.reshape(5, 4, 3).sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("mol", [SKEW, collinear_h3(1.6, 1.9)], ids=["skew", "H3"])
def test_rigid_motion(mol):
    internals: list[Internal] = (
        ALL_KINDS if mol.n_atoms == 4 else [("linear_bend", 0, 1, 2)]
    )
    moved = mol.replace(coords=mol.coords @ random_rotation(7).T + [1.3, -0.4, 2.2])
    np.testing.assert_allclose(
        values(moved, internals), values(mol, internals), rtol=0, atol=1e-12
    )


def test_linear_bend_is_zero_when_collinear():
    assert np.all(values(collinear_h3(1.6, 1.9), [("linear_bend", 0, 1, 2)]) == 0.0)


def test_linear_bend_components_follow_the_bend():
    # e_1 lies in the bending plane, so all of the bend is in the first
    # component and none in the second.
    mol = Molecule(["O", "C", "O"], SKEW.coords[:3])
    q = values(mol, [("linear_bend", 0, 1, 2)])
    assert q[0] > 0.5
    assert q[1] == pytest.approx(0.0, abs=1e-14)


def test_reference_must_match():
    with pytest.raises(ValueError, match="reference"):
        values(SKEW, ALL_KINDS, reference=water())


# Second derivatives


def test_second_derivatives():
    c = b_derivatives(SKEW, ALL_KINDS)
    assert c.shape == (5, 12, 12)
    fd = np.stack(
        [
            (
                wilson_b(displaced(SKEW, p, STEP), ALL_KINDS, reference=SKEW)
                - wilson_b(displaced(SKEW, p, -STEP), ALL_KINDS, reference=SKEW)
            )
            / (2 * STEP)
            for p in range(12)
        ],
        axis=1,
    )
    np.testing.assert_allclose(c, fd, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(c, c.transpose(0, 2, 1))


def test_bond_second_derivative():
    d = SKEW.coords[0] - SKEW.coords[1]
    r = np.linalg.norm(d)
    block = (np.eye(3) - np.outer(d, d) / r**2) / r
    expected = np.zeros((12, 12))
    expected[:3, :3] = expected[3:6, 3:6] = block
    expected[:3, 3:6] = expected[3:6, :3] = -block
    c = b_derivatives(SKEW, [("bond", 0, 1)])
    np.testing.assert_allclose(c[0], expected, rtol=0, atol=1e-8)


# Automatic coordinates


def test_bent_molecule():
    assert auto_internals(water()) == (
        ("bond", 0, 1),
        ("bond", 0, 2),
        ("angle", 1, 0, 2),
    )


def test_collinear_transition_state():
    assert auto_internals(collinear_h3(1.73, 1.73)) == (
        ("bond", 0, 1),
        ("bond", 1, 2),
        ("linear_bend", 0, 1, 2),
    )


def abstraction_ts():
    """Planar O, H_a, H_b, H_c with O-H_b-H_c at 165°."""
    hb = 2.30 * np.array([1.0, 0.0, 0.0])
    a_hoh = np.deg2rad(97.0)
    ha = 1.80 * np.array([np.cos(a_hoh), np.sin(a_hoh), 0.0])
    # H_c continues past H_b, 15° off the O-H_b line, away from H_a.
    a_c = np.deg2rad(15.0)
    hc = hb + 1.75 * np.array([np.cos(a_c), -np.sin(a_c), 0.0])
    coords = np.array([np.zeros(3), ha, hb, hc])
    return Molecule(["O", "H", "H", "H"], coords, mult=2)


def test_abstraction_transition_state():
    mol = abstraction_ts()
    assert values(mol, [("angle", 0, 2, 3)])[0] == pytest.approx(np.deg2rad(165.0))
    assert auto_internals(mol) == (
        ("bond", 0, 1),
        ("bond", 0, 2),
        ("bond", 2, 3),
        ("angle", 0, 2, 3),
        ("angle", 1, 0, 2),
        ("torsion", 1, 0, 2, 3),
    )


def two_h2():
    """Two H2 molecules whose closest atoms (1 and 2) are 6 bohr apart."""
    r = 0.7414 * ANGSTROM_TO_BOHR
    return Molecule(
        ["H"] * 4,
        [[-r, 0.5, 0.0], [0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0 + r, 0.3, 0.6]],
    )


def test_separated_fragments():
    internals = auto_internals(two_h2())
    bonds = [c for c in internals if c[0] == "bond"]
    assert bonds == [("bond", 0, 1), ("bond", 1, 2), ("bond", 2, 3)]


def test_ties_go_to_the_lowest_pair():
    # Atom 1 is 2.5 bohr from both 0 and 2, too far to be bonded.
    mol = Molecule(["He"] * 3, [[-2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0]])
    assert [c for c in auto_internals(mol) if c[0] == "bond"] == [
        ("bond", 0, 1),
        ("bond", 1, 2),
    ]


def test_single_atom_has_no_coordinates():
    assert auto_internals(Molecule(["He"], [[0.0, 0.0, 0.0]])) == ()


def test_order_is_bonds_angles_linear_bends_torsions():
    # Abstraction TS with the O-H_b-H_c line straightened: the angle becomes a
    # linear bend and the torsion through it disappears.
    mol = abstraction_ts()
    x = mol.coords.copy()
    x[3] = x[2] + 1.75 * np.array([1.0, 0.0, 0.0])
    kinds = [c[0] for c in auto_internals(mol.replace(coords=x))]
    assert kinds == ["bond", "bond", "bond", "angle", "linear_bend"]


# Coverage


def test_missing_direction():
    with pytest.raises(ValueError, match=r"2 of 3 internal directions") as info:
        check_span(water(), [("bond", 0, 1), ("bond", 0, 2)])
    assert "1 (H)" in str(info.value)


def test_where_is_reported():
    with pytest.raises(ValueError, match=r"directions at s = 0\.1"):
        check_span(water(), [("bond", 0, 1)], where=" at s = 0.1")


def test_empty_set_spans_nothing():
    with pytest.raises(ValueError, match="0 of 3"):
        check_span(water(), [])


@pytest.mark.parametrize(
    "mol",
    [water(), collinear_h3(1.73, 1.73), abstraction_ts(), two_h2()],
    ids=["water", "H3", "abstraction", "two H2"],
)
def test_automatic_sets_span(mol):
    check_span(mol, auto_internals(mol))


def test_linear_bend_spans_bent_geometries():
    # A collinear reference with a bent geometry: the bonds and both bend
    # components over-span the three internal directions.
    ref = collinear_h3(1.73, 1.73)
    bent = ref.replace(
        coords=ref.coords + np.array([[0.0, 0.2, 0.0], [0.0] * 3, [0.0, 0.1, 0.0]])
    )
    check_span(bent, auto_internals(ref), reference=ref)
