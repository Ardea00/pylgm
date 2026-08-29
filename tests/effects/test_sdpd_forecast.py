import numpy as np

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sdpd_forecast import sdpd_forecast


def _w(n):
    graph = {str(i): [str((i + 1) % n)] for i in range(n)}
    _, w = normalize_directed_graph(graph)
    return row_standardize(w)


def test_one_step_mean_matches_closed_form():
    n, rho, gamma, eta, tau = 5, 0.4, 0.3, 0.1, 2.0
    w = _w(n)
    x_last = np.arange(n, dtype=float)
    v_last = np.ones(n)
    a = np.eye(n) - rho * w.toarray()
    b = gamma * np.eye(n) + eta * w.toarray()
    expected_mean = np.linalg.solve(a, b @ x_last)
    (mean, var), = sdpd_forecast(x_last, v_last, rho, gamma, eta, tau, [w])
    assert np.allclose(mean, expected_mean)
    assert np.all(var > 0)


def test_multi_step_iterates():
    n = 4
    w = _w(n)
    steps = sdpd_forecast(np.ones(n), np.ones(n), 0.3, 0.5, 0.0, 1.0, [w, w, w])
    assert len(steps) == 3
    for mean, var in steps:
        assert mean.shape == (n,)
        assert np.all(np.isfinite(mean))
        assert np.all(var > 0)
