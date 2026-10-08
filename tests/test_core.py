import numpy as np
import pytest
from promptwork.core import GaussianPosterior, ReferenceGaussianPosterior


@pytest.mark.parametrize('dim', [1, 7, 32])
@pytest.mark.parametrize('counts', [(0, 1, 4, 12), (8, 3, 0, 2)])
def test_reference_monotonicity_telescoping(dim, counts):
    rng = np.random.default_rng(43)
    fast = GaussianPosterior(dim, 1.7)
    slow = ReferenceGaussianPosterior(dim, 1.7)
    gains = []
    for count in counts:
        z = rng.normal(size=(count, dim))
        alpha = rng.uniform(0, 4, count)
        alpha[::3] = 0
        gain = fast.update(z, alpha)
        assert gain >= 0
        assert gain == pytest.approx(slow.update(z, alpha), abs=1e-8)
        np.testing.assert_allclose(fast.Sigma, slow.Sigma, atol=1e-8)
        gains.append(gain)
    assert sum(gains) == pytest.approx(0.5 * (dim * np.log(1.7)
                                    - np.linalg.slogdet(fast.Sigma)[1]), abs=1e-8)


def test_repetition_and_zero():
    p = GaussianPosterior(3)
    before = p.Sigma.copy()
    assert p.update(np.ones((2, 3)), np.zeros(2)) == 0.0
    np.testing.assert_array_equal(before, p.Sigma)
    gains = [p.update(np.array([[2., 1., 0.]]), np.array([4.])) for _ in range(8)]
    assert all(a > b > 0 for a, b in zip(gains, gains[1:]))


def test_raw_embeddings():
    assert GaussianPosterior(1).update([[2]], [4]) == pytest.approx(0.5 * np.log(17))


@pytest.mark.parametrize('z,alpha', [([[1, 2]], [-1]), ([[np.nan, 2]], [1]), ([[1]], [1])])
def test_invalid_update(z, alpha):
    with pytest.raises(ValueError):
        GaussianPosterior(2).update(z, alpha)
