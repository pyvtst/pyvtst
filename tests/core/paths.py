"""Hand-made reaction paths around the H + H2 test species, without an engine."""

import numpy as np

from pyvtst.core.mep import ReactionPath, generalized_zpe
from pyvtst.core.rates import zpe_energy
from tests.core.test_rates import h2, h3_ts, h_atom


def reactants():
    return [h2(), h_atom()]


def make_path(
    s,
    energies,
    frequencies,
    molecules=None,
    oriented=True,
    freq_scale=1.0,
    curvature=None,
):
    """A hand-made path; ``frequencies`` holds one array (or None) per point.

    ``curvature``, when given, holds one array of components (or None) per
    point, like ``frequencies``.
    """
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    mols = (h3_ts().molecule,) * n if molecules is None else tuple(molecules)
    freqs = tuple(None if f is None else np.asarray(f, float) for f in frequencies)
    zpe = [np.nan if f is None else generalized_zpe(f) for f in freqs]
    if curvature is not None:
        curvature = tuple(
            None if c is None else np.asarray(c, float) for c in curvature
        )
    reacts = reactants()
    return ReactionPath(
        s=s,
        molecules=mols,
        energies=np.asarray(energies, dtype=np.float64),
        gradients=np.zeros((n, mols[0].n_atoms, 3)),
        frequencies=freqs,
        zpe=np.array(zpe),
        reference_energy=sum(r.energy for r in reacts),
        reference_zpe_energy=sum(zpe_energy(r) for r in reacts),
        ts_index=int(np.argmin(np.abs(s))),
        method="lqa",
        ds=0.05,
        stop_reactant="s_limit",
        stop_product="s_limit",
        oriented=oriented,
        coordinates="redundant",
        freq_scale=freq_scale,
        curvature=curvature,
    )
