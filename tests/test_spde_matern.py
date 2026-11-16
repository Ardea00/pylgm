"""Matérn / SPDE spatial effect: FEM assembly, covariance oracle, design, fits, EB gradient."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import splu
from scipy.special import kv

from pylgm import LGM, Fixed, Gaussian, Hyperparameter, Matern, Poisson
from pylgm.effects import spde
from pylgm.effects.spde import (
    barycentric_design, fem_matrices, matern_parts, matern_precision, mesh_arrays,
)
from pylgm.exceptions import CompilationError
from pylgm.inference import gaussian, laplace
from pylgm.optimization import empirical_bayes


def _grid(lo, hi, h):
    """Regular grid on [lo, hi]^2, each square split along the same diagonal."""
    xs = np.arange(lo, hi + h / 2, h)
    n = len(xs)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel()])
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    return points, np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])


def _matern_covariance(points, range_, sigma):
    """Analytic Matérn nu = 1: sigma^2 (kappa r) K_1(kappa r), kappa = sqrt(8) / range."""
    kappa = np.sqrt(8.0) / range_
    r = np.linalg.norm(points[:, None] - points[None], axis=2)
    with np.errstate(invalid="ignore"):
        cov = sigma**2 * kappa * r * kv(1, kappa * r)
    cov[r == 0] = sigma**2
    return cov


# --- 1. FEM assembly -------------------------------------------------------

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
UNIT_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def test_fem_matrices_match_hand_computation_on_the_unit_square():
    mass, stiffness = fem_matrices(*mesh_arrays(UNIT_SQUARE, UNIT_TRIANGLES))
    # lumped mass: |T|/3 per triangle per vertex; vertices 0, 2 sit in both triangles
    np.testing.assert_allclose(mass, [1 / 3, 1 / 6, 1 / 3, 1 / 6])
    expected = np.array([
        [1.0, -0.5, 0.0, -0.5],
        [-0.5, 1.0, -0.5, 0.0],
        [0.0, -0.5, 1.0, -0.5],
        [-0.5, 0.0, -0.5, 1.0],
    ])
    np.testing.assert_allclose(stiffness.toarray(), expected, atol=1e-15)


def test_stiffness_has_constant_null_vector_and_q_is_spd():
    rng = np.random.default_rng(0)
    points = np.vstack([[[0, 0], [1, 0], [0, 1], [1, 1]], rng.uniform(0, 1, (40, 2))])
    vertices, triangles = mesh_arrays(points)
    c, g, gcg = matern_parts(vertices, triangles)
    np.testing.assert_allclose(g @ np.ones(len(vertices)), 0.0, atol=1e-12)
    np.testing.assert_allclose(c.sum(), 1.0)  # lumped mass integrates 1 over the unit square
    q = matern_precision((c, g, gcg), 0.3, 2.0).toarray()
    assert np.array_equal(q, q.T)
    assert np.linalg.eigvalsh(q).min() > 0.0


def test_invalid_meshes_raise():
    with pytest.raises(ValueError, match="belong to no triangle"):
        mesh_arrays(np.vstack([UNIT_SQUARE, [[5.0, 5.0]]]), UNIT_TRIANGLES)
    with pytest.raises(ValueError, match="degenerate"):
        mesh_arrays(np.array([[0, 0], [1, 0], [2, 0.0]]), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        mesh_arrays(np.zeros((4, 3)))


# --- 2. Covariance oracle ---------------------------------------------------

def _covariance_errors(h, range_=1.0, sigma=1.5, half_width=4.0):
    """Max |Q^{-1}[c, j] - Matérn(r_cj)| over 0 < r <= 2 range, and the variance
    ratio at c, for the centre node c of [-4, 4]^2 (4 ranges from every edge)."""
    points, triangles = _grid(-half_width, half_width, h)
    q = matern_precision(matern_parts(*mesh_arrays(points, triangles)), range_, sigma)
    centre = int(np.argmin((points**2).sum(axis=1)))
    unit = np.zeros(len(points))
    unit[centre] = 1.0
    column = splu(q.tocsc()).solve(unit)  # one column of Q^{-1}, never densified
    r = np.linalg.norm(points - points[centre], axis=1)
    near = (r > 0) & (r <= 2 * range_)
    kappa = np.sqrt(8.0) / range_
    exact = sigma**2 * kappa * r[near] * kv(1, kappa * r[near])
    return np.max(np.abs(column[near] - exact)) / sigma**2, column[centre] / sigma**2


def test_interior_covariance_matches_analytic_matern_and_converges():
    # Measured: h = range/5 -> 2.7% max error (variance ratio 1.094);
    #           h = range/10 -> 1.7% (variance ratio 1.039).
    coarse_error, coarse_variance = _covariance_errors(0.2)
    fine_error, fine_variance = _covariance_errors(0.1)
    assert fine_error < coarse_error
    assert abs(fine_variance - 1.0) < abs(coarse_variance - 1.0)
    assert coarse_error < 0.035 and fine_error < 0.02
    assert abs(fine_variance - 1.0) < 0.05


# --- 3. Design --------------------------------------------------------------

def _random_mesh(seed=1):
    rng = np.random.default_rng(seed)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return mesh_arrays(np.vstack([corners, rng.uniform(0, 1, (60, 2))]))


def test_barycentric_design_is_a_partition_of_unity_and_reproduces_linears():
    vertices, triangles = _random_mesh()
    rng = np.random.default_rng(2)
    locations = np.vstack([rng.uniform(0, 1, (200, 2)), vertices[:5], [[0.0, 0.5], [1.0, 1.0]]])
    design = barycentric_design(vertices, triangles, locations)
    assert design.shape == (len(locations), len(vertices))
    assert np.diff(design.indptr).max() <= 3
    assert design.min() >= 0.0
    np.testing.assert_allclose(design.sum(axis=1).A1, 1.0)
    linear = 0.7 - 1.3 * vertices[:, 0] + 2.1 * vertices[:, 1]
    np.testing.assert_allclose(
        design @ linear, 0.7 - 1.3 * locations[:, 0] + 2.1 * locations[:, 1], atol=1e-12
    )


def test_full_scan_fallback_gives_the_same_design(monkeypatch):
    vertices, triangles = _random_mesh()
    locations = np.random.default_rng(3).uniform(0, 1, (50, 2))
    expected = barycentric_design(vertices, triangles, locations).toarray()
    monkeypatch.setattr(spde, "_CANDIDATES", 1)  # most points miss the first centroid
    np.testing.assert_allclose(barycentric_design(vertices, triangles, locations).toarray(), expected)


def test_point_outside_the_mesh_raises():
    vertices, triangles = _random_mesh()
    with pytest.raises(ValueError, match="outside the Matern mesh"):
        barycentric_design(vertices, triangles, np.array([[0.5, 0.5], [1.2, 0.5]]))
    frame = pd.DataFrame({"x": [0.5, 1.2], "y": [0.5, 0.5], "t": [0, 1], "z": [0.0, 1.0]})
    model = LGM("z", Gaussian(1.0), Fixed("1") + Matern("s", ("x", "y"), vertices, range=0.3),
                time="t")
    with pytest.raises(CompilationError, match="outside the Matern mesh"):
        model.fit(frame)


# --- 4. End to end ----------------------------------------------------------

TRUE_RANGE, TRUE_SIGMA = 1.5, 1.0
# Data on [0, 6]^2 (4 ranges wide); mesh extended one range beyond it.
MESH, _ = _grid(-1.5, 7.5, 0.3)  # 961 nodes: dense Gaussian and dense Laplace engines


def _simulated(seed=2, n=300):
    """A field drawn from the *analytic* Matérn covariance (not from Q)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 6, (n, 2))
    cov = _matern_covariance(points, TRUE_RANGE, TRUE_SIGMA)
    field = np.linalg.cholesky(cov + 1e-9 * np.eye(n)) @ rng.normal(size=n)
    return pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "t": np.arange(n)}), field, rng


def _matern():
    return Matern(
        "s", ("x", "y"), MESH,
        range=Hyperparameter("s.range", initial=3.0),
        sigma=Hyperparameter("s.sigma", initial=0.5),
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_empirical_bayes_recovers_range_and_sigma_and_predicts(family):
    frame, field, rng = _simulated()
    if family == "gaussian":
        frame["z"] = 1.0 + field + 0.3 * rng.normal(size=len(field))
        likelihood, engine = Gaussian(Hyperparameter("noise", initial=0.5)), "exact_gaussian"
    else:
        frame["z"] = rng.poisson(np.exp(1.0 + field)).astype(float)
        likelihood, engine = Poisson(), "laplace"
    result = LGM("z", likelihood, Fixed("1") + _matern(), time="t").fit(frame, engine=engine)

    # Loose: a factor of 2 on the range, 1.5 on sigma. Measured (seed 2):
    # Gaussian range 1.54, sigma 1.04 (noise 0.34, true 0.3); Poisson range 1.75, sigma 1.07.
    estimates = result.hyperparameters
    assert TRUE_RANGE / 2 < estimates["s.range"] < 2 * TRUE_RANGE
    assert TRUE_SIGMA / 1.5 < estimates["s.sigma"] < 1.5 * TRUE_SIGMA

    np.testing.assert_allclose(
        result.predict(frame).predictive_mean, result.predictive_mean, rtol=1e-12, atol=1e-12
    )
    new = pd.DataFrame({"x": [1.0, 3.3, 7.0], "y": [2.0, 5.9, -1.0], "t": [0, 1, 2]})
    predicted = result.predict(new)
    assert np.isfinite(predicted.predictive_mean).all()
    assert np.isfinite(predicted.predictive_variance).all()
    # A new point at a fit row's location predicts exactly as that row did.
    np.testing.assert_allclose(
        result.predict(frame.iloc[[7]]).predictive_mean, result.predictive_mean[7], rtol=1e-12
    )


def test_large_mesh_takes_the_sparse_path():
    points, triangles = _grid(0.0, 6.4, 0.1)  # 65^2 = 4225 nodes > dense latent limit
    rng = np.random.default_rng(4)
    locations = rng.uniform(0.5, 5.9, (200, 2))
    frame = pd.DataFrame({"x": locations[:, 0], "y": locations[:, 1], "t": np.arange(200),
                          "z": rng.normal(size=200)})
    effect = Matern("s", ("x", "y"), points, range=1.0, sigma=1.0, triangles=triangles)
    result = LGM("z", Gaussian(0.5), Fixed("1") + effect, time="t").fit(frame)
    assert result._sparse_posterior is not None
    assert result.diagnostics["sparse_dimension"] == len(points)


def test_poisson_on_a_large_mesh_takes_the_sparse_laplace_path(monkeypatch):
    points, triangles = _grid(0.0, 6.4, 0.2)  # 33^2 = 1089 nodes >= sparse Laplace switch
    rng = np.random.default_rng(6)
    locations = rng.uniform(0.5, 5.9, (150, 2))
    frame = pd.DataFrame({"x": locations[:, 0], "y": locations[:, 1], "t": np.arange(150),
                          "z": rng.poisson(2.0, 150).astype(float)})
    effect = Matern("s", ("x", "y"), points, range=1.5, sigma=0.5, triangles=triangles)
    dense = {"calls": 0}
    original = laplace._fit_laplace_dense

    def counting(*args, **kwargs):
        dense["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(laplace, "_fit_laplace_dense", counting)
    result = LGM("z", Poisson(), Fixed("1") + effect, time="t").fit(frame, engine="laplace")
    assert dense["calls"] == 0
    np.testing.assert_allclose(
        result.predict(frame).predictive_mean, result.predictive_mean, rtol=1e-10, atol=1e-10
    )


def test_integrate_and_predict_outside_the_mesh():
    vertices, _ = _random_mesh()
    rng = np.random.default_rng(7)
    locations = rng.uniform(0, 1, (60, 2))
    frame = pd.DataFrame({"x": locations[:, 0], "y": locations[:, 1], "t": np.arange(60),
                          "z": np.sin(3 * locations[:, 0]) + 0.3 * rng.normal(size=60)})
    effect = Matern("s", ("x", "y"), vertices, range=Hyperparameter("s.range", initial=0.5),
                    sigma=1.0)
    result = LGM("z", Gaussian(0.3), Fixed("1") + effect, time="t").fit(
        frame, hyperparameters="integrate"
    )
    assert np.isfinite(result.predict(frame).predictive_mean).all()
    outside = pd.DataFrame({"x": [1.5], "y": [0.5], "t": [0]})
    with pytest.raises(ValueError, match="outside the Matern mesh"):
        result.predict(outside)


def test_spark_adapter_collects_the_coordinate_columns():
    from pylgm.data.spark import _required_columns

    vertices, _ = _random_mesh()
    model = LGM("z", Gaussian(1.0), Fixed("1") + Matern("s", ("lon", "lat"), vertices, range=0.3),
                time="t")
    assert {"lon", "lat"} <= _required_columns(model)


# --- 6. EB gradient ---------------------------------------------------------

@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_eb_gradient_is_analytic_and_matches_central_differences(family, monkeypatch):
    points, _ = _grid(-1.0, 5.0, 0.5)  # 169 nodes
    frame, field, rng = _simulated(seed=5, n=120)
    frame[["x", "y"]] = frame[["x", "y"]] * (4.0 / 6.0)
    if family == "gaussian":
        frame["z"] = field + 0.4 * rng.normal(size=len(field))
        likelihood, engine, module, name = (
            Gaussian(Hyperparameter("noise", initial=0.6)), "exact_gaussian", gaussian, "_fit_dense",
        )
    else:
        frame["z"] = rng.poisson(np.exp(0.5 + field)).astype(float)
        likelihood, engine, module, name = Poisson(), "laplace", laplace, "_fit_laplace_dense"
    effect = Matern("s", ("x", "y"), points, range=Hyperparameter("s.range", initial=1.2),
                    sigma=Hyperparameter("s.sigma", initial=0.8))

    fits = {"calls": 0, "analytic": 0}
    original = getattr(module, name)

    def counting(*args, **kwargs):
        fits["calls"] += 1
        result = original(*args, **kwargs)
        fits["analytic"] += "lml_gradient" in result.diagnostics
        return result

    monkeypatch.setattr(module, name, counting)
    captured = {}

    def fake_minimize(objective, start, *, jac, **_):
        point = start + 0.2
        objective(point)  # L-BFGS-B evaluates f before g at the same point
        before = fits["calls"]
        captured["g"] = jac(point)
        captured["extra_fits"] = fits["calls"] - before
        h = 1e-4
        captured["fd"] = np.array([
            (objective(point + h * e) - objective(point - h * e)) / (2 * h)
            for e in np.eye(point.size)
        ])
        return OptimizeResult(x=start, fun=0.0, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)
    LGM("z", likelihood, Fixed("1") + effect, time="t").fit(frame, engine=engine)
    assert fits["analytic"] > 0
    assert captured["extra_fits"] == 0  # every gradient component came from the analytic path
    np.testing.assert_allclose(captured["g"], captured["fd"], rtol=1e-4, atol=1e-7)
