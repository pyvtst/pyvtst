"""OH + H2 -> H2O + H CVT/ZCT and CVT/SCT at UHF/6-31G(d) with PySCF (slow)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("geometric")

from tests.test_curvilinear_pyscf import rxn  # noqa: F401

pytestmark = pytest.mark.slow

T = [200.0, 300.0, 500.0, 1000.0, 2000.0]


def test_cvt_zct_and_sct_are_finite_and_sct_tunnels_more(rxn):  # noqa: F811
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        zct = rxn.rates(T, theory="cvt", tunneling="zct")
        sct = rxn.rates(T, theory="cvt", tunneling="sct")
    for result in (zct, sct):
        assert result.path_tunneling is not None
        assert np.all(np.isfinite(result.kappa) & (result.kappa > 0))
        assert np.all(np.isfinite(result.k) & (result.k > 0))
    assert np.all(sct.kappa >= zct.kappa)
    # The default ±0.5 bohr path ends far above the reactants' zero-point level.
    assert any("tunneling: the path ends" in str(w.message) for w in record)
