"""Forward forecasting for the dynamic spatial panel (SDPD).

Given the fitted last-period latent ``x̂_T`` (mean + marginal variance) and the
next periods' directed networks, propagate the SDPD recursion forward:

    x̂_{t+1} = A_{t+1}⁻¹ B_{t+1} x̂_t
    Var(x_{t+1}) = A_{t+1}⁻¹ [B_{t+1} diag(Var x_t) B_{t+1}ᵀ + τ⁻¹ I] A_{t+1}⁻ᵀ

with ``A = I - ρW``, ``B = γI + ηW``. Variance is carried diagonal-only
(marginal), consistent with the gaussian latent strategy above the sparse guard.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from pylgm.effects.directed_graph import (
    normalize_directed_graph,
    row_standardize,
    sorted_time_keys,
)
from pylgm.effects.sar import _panel_networks
from pylgm.parameters import Hyperparameter


def sdpd_forecast(
    x_last: np.ndarray,
    v_last: np.ndarray,
    rho: float,
    gamma: float,
    eta: float,
    tau: float,
    future_ws: list[csr_matrix],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward SDPD recursion; returns ``[(mean, marginal_var), ...]`` per step."""
    x = np.asarray(x_last, dtype=float)
    v = np.asarray(v_last, dtype=float)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for w in future_ws:
        n = w.shape[0]
        ident = identity(n, format="csc")
        a = (ident - rho * w).tocsc()
        b = (gamma * ident + eta * w).tocsc()
        lu = splu(a)
        x_next = lu.solve(b @ x)
        # ponytail: dense A^-1 (via sparse LU solve against I) — O(n^3), fine for
        # a forecast horizon; a diagonal-only fast path is a later optimization.
        a_inv = lu.solve(np.eye(n))
        inner = (b @ diags(v) @ b.T).toarray() + (1.0 / tau) * np.eye(n)
        cov_next = a_inv @ inner @ a_inv.T
        var_next = np.clip(np.diag(cov_next), 0.0, None)
        out.append((x_next, var_next))
        x, v = x_next, var_next
    return out


def _align_to_units(graph: Mapping, units: tuple[str, ...]) -> csr_matrix:
    """Row-standardized ``W`` for ``graph`` on the fitted ``units`` ordering."""
    nodes, w = normalize_directed_graph(graph)
    unknown = sorted(set(nodes) - set(units))
    if unknown:
        raise ValueError(
            f"forecast network references unit(s) not in the fitted panel: {unknown!r}"
        )
    position = {u: i for i, u in enumerate(units)}
    coo = w.tocoo()
    rows = [position[nodes[i]] for i in coo.row]
    cols = [position[nodes[j]] for j in coo.col]
    big = csr_matrix((coo.data, (rows, cols)), shape=(len(units), len(units)))
    return row_standardize(big)


def forecast_dynamic_spatial_panel(result, effect, future_graphs: Mapping) -> pd.DataFrame:
    """Forecast future periods for a fitted ``DynamicSpatialPanel`` effect.

    ``result`` is the fit output, ``effect`` the ``DynamicSpatialPanel`` spec,
    ``future_graphs`` a ``{time: directed_graph}`` mapping for the periods to
    forecast. Returns a frame with columns ``unit, time, mean, variance``.
    """

    def _value(param):
        if isinstance(param, Hyperparameter):
            return float(result.hyperparameters[param.name])
        return float(param)

    rho, gamma, eta, tau = (
        _value(effect.rho), _value(effect.gamma), _value(effect.eta), _value(effect.precision)
    )
    fitted_graphs = {t: dict(g) for t, g in dict(effect.graphs).items()}
    units, times, _ = _panel_networks(fitted_graphs)
    n, t_count = len(units), len(times)
    marginals = result.latent_marginals(effect.name)
    x_grid = np.asarray(marginals.mean, dtype=float).reshape(t_count, n)
    v_grid = np.asarray(marginals.variance, dtype=float).reshape(t_count, n)
    future_times = sorted_time_keys(future_graphs)
    future_ws = [
        _align_to_units(dict(future_graphs[_key(future_graphs, t)]), units) for t in future_times
    ]
    steps = sdpd_forecast(x_grid[-1], v_grid[-1], rho, gamma, eta, tau, future_ws)
    records = []
    for t, (mean, var) in zip(future_times, steps):
        for unit, mean_i, var_i in zip(units, mean, var):
            records.append({"unit": unit, "time": t, "mean": mean_i, "variance": var_i})
    return pd.DataFrame.from_records(records)


def _key(graphs: Mapping, text: str):
    for key in graphs:
        if str(key) == text:
            return key
    raise KeyError(text)
