"""Matérn fields on a triangulated planar domain via the SPDE / P1 finite-element link.

The field ``x`` solves ``(kappa^2 - Delta) (tau x) = W`` on the mesh domain
``Omega`` (``W`` Gaussian white noise), with homogeneous Neumann boundary
conditions ``d x / d n = 0`` on ``dOmega``. Discretised with continuous
piecewise-linear (P1) elements -- the field is ``sum_k w_k phi_k(s)`` over the
mesh's hat functions -- and the **lumped** (diagonal) mass matrix
``C_kk = int phi_k``, the weights' precision is

    Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^{-1} G),   G_kl = int grad phi_k . grad phi_l.

This is the alpha = 2 case of Lindgren, Rue & Lindström (2011) (citation
unverified here; the formula itself is checked numerically against the
analytic Matérn covariance in tests/test_spde_matern.py). In d = 2, alpha = 2
the Matérn smoothness is nu = alpha - d/2 = 1, the range is
``rho = sqrt(8 nu) / kappa`` and the marginal variance is
``sigma^2 = 1 / (4 pi kappa^2 tau^2)``.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.spatial import Delaunay, cKDTree

from pylgm.ir.model import LatentBlock

_INSIDE_TOLERANCE = 1e-10  # barycentric slack for a point on an edge/vertex
_CANDIDATES = 16           # nearest triangle centroids tried before a full scan


def mesh_arrays(vertices, triangles=None) -> tuple[np.ndarray, np.ndarray]:
    """Validated ``(vertices (m, 2) float, triangles (T, 3) int)``; Delaunay when ``triangles`` is None."""
    points = np.array(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("Matern mesh vertices must be an (m, 2) array with m >= 3")
    if not np.isfinite(points).all():
        raise ValueError("Matern mesh vertices must be finite")
    if triangles is None:
        triangles = Delaunay(points).simplices
    simplices = np.array(triangles)
    if simplices.ndim != 2 or simplices.shape[1] != 3 or not simplices.size:
        raise ValueError("Matern mesh triangles must be a non-empty (T, 3) integer array")
    if not np.issubdtype(simplices.dtype, np.integer):
        raise ValueError("Matern mesh triangles must hold integer vertex indices")
    if simplices.min() < 0 or simplices.max() >= len(points):
        raise ValueError("Matern mesh triangles reference a vertex index out of range")
    simplices = simplices.astype(np.int64)
    if np.any(np.abs(_signed_areas(points, simplices)) <= 1e-14 * np.ptp(points, axis=0).prod()):
        raise ValueError("Matern mesh has a degenerate (zero-area) triangle")
    unused = np.setdiff1d(np.arange(len(points)), simplices)
    if unused.size:
        raise ValueError(
            f"Matern mesh vertices {unused[:10].tolist()} belong to no triangle "
            "(duplicate points?); their mass would be zero and C^{-1} undefined"
        )
    points.setflags(write=False)
    simplices.setflags(write=False)
    return points, simplices


def _signed_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def fem_matrices(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
    """Lumped P1 mass diagonal ``c`` (``C = diag(c)``) and stiffness ``G`` (CSR)."""
    area = np.abs(_signed_areas(points, triangles))
    m = len(points)
    mass = np.bincount(triangles.ravel(), weights=np.repeat(area / 3.0, 3), minlength=m)
    # Edge opposite vertex k of each triangle: e_k = p_{k+2} - p_{k+1}. The P1
    # stiffness is G_kl = e_k . e_l / (4 |T|), whatever the orientation.
    p = [points[triangles[:, k]] for k in range(3)]
    edges = [p[(k + 2) % 3] - p[(k + 1) % 3] for k in range(3)]
    rows, cols, vals = [], [], []
    for k in range(3):
        for j in range(3):
            rows.append(triangles[:, k])
            cols.append(triangles[:, j])
            vals.append(np.einsum("ij,ij->i", edges[k], edges[j]) / (4.0 * area))
    stiffness = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m)
    ).tocsr()
    return mass, stiffness


def matern_parts(points: np.ndarray, triangles: np.ndarray) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """``(C, G, G C^{-1} G)``: Q is ``tau^2 (kappa^4 C + 2 kappa^2 G + G C^{-1} G)``."""
    mass, stiffness = fem_matrices(points, triangles)
    gcg = stiffness @ diags(1.0 / mass) @ stiffness
    # Exactly symmetric parts keep every linear combination exactly symmetric;
    # assembly and the product round differently on (k, l) and (l, k).
    return (
        diags(mass, format="csr"),
        (0.5 * (stiffness + stiffness.T)).tocsr(),
        (0.5 * (gcg + gcg.T)).tocsr(),
    )


def matern_precision(parts, range_: float, sigma: float) -> csr_matrix:
    """``Q`` at a range ``rho`` and marginal standard deviation ``sigma`` (nu = 1, d = 2)."""
    c, g, gcg = parts
    kappa2 = 8.0 / range_**2
    tau2 = 1.0 / (4.0 * np.pi * kappa2 * sigma**2)
    return csr_matrix(tau2 * (kappa2**2 * c + 2.0 * kappa2 * g + gcg))


def barycentric_design(points: np.ndarray, triangles: np.ndarray, locations: np.ndarray) -> csr_matrix:
    """``(n, m)`` P1 interpolation: each row holds a location's barycentric
    weights on the three vertices of its triangle. A location in no triangle
    raises ``ValueError``."""
    locations = np.asarray(locations, dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 2 or not np.isfinite(locations).all():
        raise ValueError("Matern locations must be finite (n, 2) coordinates")
    n = len(locations)
    a = points[triangles[:, 0]]
    # Per-triangle inverse of [b - a, c - a], so lambda_{1,2} = inv @ (s - a).
    basis = np.stack(
        [points[triangles[:, 1]] - a, points[triangles[:, 2]] - a], axis=2
    )  # (T, 2, 2), columns are the two edges
    inverse = np.linalg.inv(basis)

    def weights(point_ids, tri_ids):
        lam = np.einsum("tij,tj->ti", inverse[tri_ids], locations[point_ids] - a[tri_ids])
        return np.column_stack([1.0 - lam.sum(axis=1), lam])

    found = np.full(n, -1)
    bary = np.zeros((n, 3))
    centroids = points[triangles].mean(axis=1)
    k = min(_CANDIDATES, len(triangles))
    _, near = cKDTree(centroids).query(locations, k=k)
    near = near.reshape(n, k)
    for column in range(k):
        todo = np.flatnonzero(found < 0)
        if not todo.size:
            break
        w = weights(todo, near[todo, column])
        hit = (w >= -_INSIDE_TOLERANCE).all(axis=1)
        found[todo[hit]] = near[todo[hit], column]
        bary[todo[hit]] = w[hit]
    # ponytail: full O(points x triangles) scan for the rare location whose
    # triangle is not among its nearest centroids (long thin triangles).
    for i in np.flatnonzero(found < 0):
        w = weights(np.full(len(triangles), i), np.arange(len(triangles)))
        inside = np.flatnonzero((w >= -_INSIDE_TOLERANCE).all(axis=1))
        if inside.size:
            found[i] = inside[0]
            bary[i] = w[inside[0]]
    outside = np.flatnonzero(found < 0)
    if outside.size:
        raise ValueError(
            f"{outside.size} location(s) lie outside the Matern mesh, e.g. row "
            f"{int(outside[0])} at {locations[outside[0]].tolist()}; extend the mesh to cover them"
        )
    bary = np.clip(bary, 0.0, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return coo_matrix(
        (bary.ravel(), (np.repeat(np.arange(n), 3), triangles[found].ravel())),
        shape=(n, len(points)),
    ).tocsr()


def build_matern(frame, name: str, coordinates, points, triangles, parts, range_: float,
                 sigma: float) -> LatentBlock:
    missing = [c for c in coordinates if c not in frame.columns]
    if missing:
        raise KeyError(missing[0])
    locations = frame[list(coordinates)].to_numpy(dtype=float)
    design = barycentric_design(points, triangles, locations)
    labels = tuple(str(k) for k in range(len(points)))
    return LatentBlock(
        name, labels, design, matern_precision(parts, range_, sigma), np.empty((0, len(points)))
    )
