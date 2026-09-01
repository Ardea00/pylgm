import numpy as np
import pytest

from pylgm.exceptions import DataContractError, ModelValidationError
from pylgm.likelihoods import CompiledGaussian, CompiledMixture, CompiledPoisson


def _parts(n=6):
    mask_a = np.zeros(n, dtype=bool)
    mask_a[:3] = True
    mask_b = ~mask_a
    return ((mask_a, CompiledGaussian(1.5)), (mask_b, CompiledPoisson()))


def test_mixture_dispatches_each_method_per_row():
    n = 6
    parts = _parts(n)
    mixture = CompiledMixture(parts, n)
    eta = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert mixture.log_likelihood(eta, y) == pytest.approx(
        parts[0][1].log_likelihood(eta[parts[0][0]], y[parts[0][0]])
        + parts[1][1].log_likelihood(eta[parts[1][0]], y[parts[1][0]])
    )
    for method in ("gradient", "working_weights", "third_derivative", "pointwise_log_density"):
        got = getattr(mixture, method)(eta, y)
        for mask, likelihood in parts:
            assert got[mask] == pytest.approx(getattr(likelihood, method)(eta[mask], y[mask]))


def test_mixture_rejects_overlapping_masks():
    n = 4
    overlap = np.array([True, True, True, False])
    other = np.array([False, True, True, True])
    with pytest.raises(ModelValidationError, match="disjoint"):
        CompiledMixture(((overlap, CompiledGaussian(1.0)), (other, CompiledPoisson())), n)


def test_mixture_rejects_uncovered_rows():
    n = 4
    partial = np.array([True, True, False, False])
    with pytest.raises(ModelValidationError, match="cover"):
        CompiledMixture(((partial, CompiledGaussian(1.0)),), n)


def test_mixture_validate_response_checks_each_part_on_its_own_rows():
    # A negative value on the Gaussian rows is fine; on the Poisson rows it is not.
    mixture = CompiledMixture(_parts(6), 6)
    mixture.validate_response(np.array([-1.0, 0.5, 2.0, 1.0, 2.0, 3.0]))
    with pytest.raises(DataContractError):
        mixture.validate_response(np.array([-1.0, 0.5, 2.0, -1.0, 2.0, 3.0]))


def test_restrict_reindexes_masks_into_observed_space():
    mixture = CompiledMixture(_parts(6), 6)
    observed = np.array([True, False, True, True, False, True])
    restricted = mixture.restrict(observed)
    assert restricted.n_rows == 4
    assert [mask.tolist() for mask, _ in restricted.parts] == [
        [True, True, False, False],
        [False, False, True, True],
    ]


def test_restrict_defaults_to_self_for_ordinary_likelihoods():
    likelihood = CompiledPoisson()
    assert likelihood.restrict(np.array([True, False])) is likelihood
