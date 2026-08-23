import numpy as np
import pytest
from scipy.integrate import quad

from pylgm.priors import PCBYM2Phi


def _gamma():
    # positive eigenvalues of pinv(R*) for a 10-node chain, via the shipped scaler
    from pylgm.effects.besag import _scaled_structure
    from pylgm.effects.graph import normalize_graph

    graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 9] for i in range(10)}
    nodes, w = normalize_graph(graph)
    p = np.linalg.pinv(_scaled_structure(w, nodes, True))
    values = np.linalg.eigvalsh(p)
    return values[values > 1e-10]


def test_density_integrates_to_one():
    prior = PCBYM2Phi(upper=0.5, alpha=2 / 3).bind(_gamma())
    mass, _ = quad(lambda p: np.exp(prior.logpdf(p)), 0.0, 1.0, limit=400)
    assert mass == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(("upper", "alpha"), [(0.5, 2 / 3), (0.5, 0.9), (0.8, 0.95)])
def test_calibration_matches_requested_probability(upper, alpha):
    prior = PCBYM2Phi(upper=upper, alpha=alpha).bind(_gamma())
    below, _ = quad(lambda p: np.exp(prior.logpdf(p)), 0.0, upper, limit=400)
    assert below == pytest.approx(alpha, abs=1e-6)


def test_distance_is_zero_at_base_and_increasing():
    prior = PCBYM2Phi().bind(_gamma())
    assert prior.distance(0.0) == pytest.approx(0.0)
    grid = np.linspace(0.0, 1.0, 50)
    distances = [prior.distance(p) for p in grid]
    assert np.all(np.diff(distances) > 0)
    assert np.isfinite(prior.distance(1.0))


def test_distance_derivative_at_zero_matches_finite_difference():
    prior = PCBYM2Phi().bind(_gamma())
    analytic = prior.distance_derivative(0.0)
    numeric = (prior.distance(1e-6) - prior.distance(0.0)) / 1e-6
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_unattainable_alpha_names_the_range():
    gamma = _gamma()
    prior = PCBYM2Phi(upper=0.5, alpha=0.05)  # below d(U)/d(1)
    with pytest.raises(ValueError, match="attainable"):
        prior.bind(gamma)


def test_unbound_logpdf_raises():
    with pytest.raises(ValueError, match="bound"):
        PCBYM2Phi().logpdf(0.5)


@pytest.mark.parametrize("kwargs", [{"upper": 0.0}, {"upper": 1.0}, {"alpha": 0.0}, {"alpha": 1.0}])
def test_invalid_calibration_rejected(kwargs):
    with pytest.raises(ValueError):
        PCBYM2Phi(**kwargs)
