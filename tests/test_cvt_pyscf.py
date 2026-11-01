"""OH + H2 -> H2O + H canonical variational TST at UHF/6-31G(d) with PySCF (slow)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("geometric")

from tests.test_curvilinear_pyscf import rxn  # noqa: F401

pytestmark = pytest.mark.slow


def test_cvt_is_finite_and_below_tst(rxn):  # noqa: F811
    t = [200.0, 300.0, 500.0, 1000.0, 2000.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rxn.rates(t, theory="cvt")
    cvt = result.cvt
    assert cvt is not None
    assert np.all(np.isfinite(cvt.k) & (cvt.k > 0))
    assert np.all(cvt.k <= result.k_tst)
    assert np.all((cvt.s_star >= cvt.s[0]) & (cvt.s_star <= cvt.s[-1]))
