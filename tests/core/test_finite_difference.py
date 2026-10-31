import numpy as np
import pytest

from pyvtst.core.finite_difference import fd_hessian


@pytest.fixture
def quadratic():
    """Gradient of E = ½ xᵀAx for a random symmetric 9x9 A, and A itself."""
    rng = np.random.default_rng(0)
    m = rng.normal(size=(9, 9))
    a = m + m.T
    return (lambda x: (a @ x.ravel()).reshape(-1, 3)), a


def test_exact_for_quadratic_surface(quadratic):
    grad, a = quadratic
    x0 = np.arange(9.0).reshape(3, 3)
    np.testing.assert_allclose(fd_hessian(grad, x0), a, rtol=0, atol=1e-8)


def test_six_n_gradient_calls(quadratic):
    grad, _ = quadratic
    calls = []

    def counting(x):
        calls.append(x.copy())
        return grad(x)

    fd_hessian(counting, np.zeros((3, 3)))
    assert len(calls) == 18
    assert all(x.shape == (3, 3) for x in calls)


def test_result_is_symmetric():
    def grad(x):
        x = x.ravel()
        return np.array([x[0] * x[1] ** 2, x[0] ** 2 + x[2], np.sin(x[1])])

    h = fd_hessian(grad, np.array([[0.3, -0.2, 0.5]]))
    np.testing.assert_array_equal(h, h.T)


def test_step_must_be_positive(quadratic):
    grad, _ = quadratic
    with pytest.raises(ValueError, match="step"):
        fd_hessian(grad, np.zeros((3, 3)), step=0.0)
