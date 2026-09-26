"""Gaussian observations and exact constraints on a predictor grid."""

from dataclasses import dataclass, field

from collections.abc import Callable

import numpy as np
from scipy.linalg import orth, qr
from scipy.sparse import csr_matrix, diags, issparse, vstack

from pylgm.exceptions import InferenceError, ModelValidationError, UnsupportedEngineError
from pylgm.ir.model import CompiledLGM, LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledMixture
from pylgm.parameters import Hyperparameter


_MAX_DENSE_CONSTRAINT_WORKSPACE_BYTES = 64 * 1024 * 1024


def _matrix(value: object, name: str) -> csr_matrix:
    if issparse(value):
        if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value.data):
            raise TypeError(f"{name} must be a real numeric 2D matrix")
        result = csr_matrix(value, dtype=float, copy=True)
    else:
        array = np.asarray(value)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
            raise TypeError(f"{name} must be a real numeric 2D matrix")
        result = csr_matrix(array, dtype=float)
    if not np.isfinite(result.data).all():
        raise ValueError(f"{name} must be finite")
    result.sort_indices()
    return result


def _vector(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
        raise TypeError(f"{name} must be a real numeric one-dimensional array")
    result = np.array(result, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class LinearObservation:
    """Independent Gaussian observations ``values = operator @ eta + error``.

    The operator has one column per predictor-grid row. ``sigma`` is one positive
    standard deviation, a vector aligned with ``values``, or a ``Hyperparameter``:
    one scalar standard deviation for the whole block, estimated by empirical
    Bayes or integrated by INLA like any effect hyperparameter. With
    ``scale="log"`` the operator acts on ``exp(eta)`` (predictor on the log
    scale, aggregates on levels), fitted by Gauss-Newton relinearization.
    """

    values: np.ndarray = field(repr=False)
    operator: csr_matrix = field(repr=False)
    sigma: np.ndarray | Hyperparameter = field(repr=False)
    scale: str = field(default="identity")

    def __init__(self, values: object, operator: object, sigma: object, scale: str = "identity") -> None:
        values = _vector(values, "LinearObservation values")
        operator = _matrix(operator, "LinearObservation operator")
        if operator.shape[0] != values.size:
            raise ValueError("LinearObservation operator rows must match values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "operator", operator)
        if scale not in ("identity", "log"):
            raise ValueError("LinearObservation scale must be 'identity' or 'log'")
        object.__setattr__(self, "scale", scale)
        if isinstance(sigma, Hyperparameter):
            if sigma.transform != "log":
                raise ValueError(
                    "LinearObservation sigma is a standard deviation; its Hyperparameter "
                    f"must use transform='log', not {sigma.transform!r}"
                )
            object.__setattr__(self, "sigma", sigma)
            return
        raw_sigma = np.asarray(sigma)
        if raw_sigma.ndim == 0:
            if not np.issubdtype(raw_sigma.dtype, np.number) or not np.isrealobj(raw_sigma):
                raise TypeError("LinearObservation sigma must be numeric")
            try:
                sigma = np.full(values.size, float(raw_sigma))
            except (TypeError, ValueError) as error:
                raise TypeError("LinearObservation sigma must be numeric") from error
        else:
            sigma = _vector(sigma, "LinearObservation sigma")
        if sigma.shape != values.shape:
            raise ValueError("LinearObservation sigma must be scalar or match values")
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("LinearObservation sigma must be finite and positive")
        sigma = np.array(sigma, dtype=float, copy=True)
        sigma.setflags(write=False)
        object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True, init=False)
class LinearConstraint:
    """An exact equality ``operator @ eta = rhs`` on the predictor grid.

    With ``scale="log"`` the operator acts on ``exp(eta)`` (predictor on the
    log scale, aggregates on levels), fitted by Gauss-Newton relinearization.
    """

    operator: csr_matrix = field(repr=False)
    rhs: np.ndarray = field(repr=False)
    scale: str = field(default="identity")

    def __init__(self, operator: object, rhs: object, scale: str = "identity") -> None:
        operator = _matrix(operator, "LinearConstraint operator")
        rhs = _vector(rhs, "LinearConstraint rhs")
        if operator.shape[0] != rhs.size:
            raise ValueError("LinearConstraint operator rows must match rhs")
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "rhs", rhs)
        if scale not in ("identity", "log"):
            raise ValueError("LinearConstraint scale must be 'identity' or 'log'")
        object.__setattr__(self, "scale", scale)


def _aligned(operator: csr_matrix, rows: int, name: str) -> csr_matrix:
    if operator.shape[1] != rows:
        raise ModelValidationError(
            f"{name} has {operator.shape[1]} columns; expected one per grid row ({rows})"
        )
    return operator


def _constraint_rows(model: CompiledLGM, constraints: tuple[LinearConstraint, ...]):
    """Translate ``C eta=e`` to independent, compatible ``C Z x=e-C o`` rows."""
    if not constraints:
        return model.extra_constraints, model.extra_constraint_rhs

    rows, rhs = [], []
    for constraint in constraints:
        operator = _aligned(
            constraint.operator,
            model.prediction_design.shape[0],
            "LinearConstraint operator",
        )
        rows.append(operator @ model.prediction_design)
        rhs.append(constraint.rhs - np.asarray(operator @ model.prediction_offset).reshape(-1))
    proposed = vstack(rows, format="csr")
    proposed_rhs = np.concatenate(rhs)

    # Only (rows x latent) arrays are formed: the rank reduction projects out
    # the structural row space instead of building its latent x latent null space.
    dense_bytes = (
        (model.constraints.shape[0] + proposed.shape[0])
        * proposed.shape[1]
        * np.dtype(float).itemsize
    )
    if dense_bytes > _MAX_DENSE_CONSTRAINT_WORKSPACE_BYTES:
        raise ModelValidationError(
            "LinearConstraint rank reduction requires up to "
            f"{dense_bytes / 1024**2:.1f} MiB of dense workspace for a "
            f"{model.constraints.shape[0] + proposed.shape[0]}x{proposed.shape[1]} "
            "constraint matrix; reduce or split the constraints"
        )
    proposed = proposed.toarray()

    combined = np.vstack([model.constraints, proposed])
    combined_rhs = np.concatenate([model.constraint_rhs, proposed_rhs])
    solution, *_ = np.linalg.lstsq(combined, combined_rhs, rcond=None)
    scale = max(1.0, float(np.linalg.norm(combined_rhs, ord=np.inf)))
    if np.max(np.abs(combined @ solution - combined_rhs), initial=0.0) > 1e-9 * scale:
        raise ModelValidationError("linear constraints are mutually inconsistent")

    # Components of the proposed rows outside the structural row space. With
    # an orthonormal basis N of null(C0), proposed @ N and proposed @ N @ N.T
    # differ by an orthogonal map, so rank and pivoted QR are unchanged.
    reduced = proposed
    if model.constraints.shape[0]:
        basis = orth(model.constraints.T)
        reduced = proposed - (proposed @ basis) @ basis.T
    # The projection cancels the structural component, so the singular values
    # of ``reduced`` are judged against the scale of the original rows.
    tolerance = max(reduced.shape) * np.finfo(float).eps * np.linalg.norm(proposed, 2)
    rank = np.linalg.matrix_rank(reduced, tol=tolerance) if proposed.size else 0
    if rank:
        keep = np.sort(qr(reduced.T, mode="economic", pivoting=True)[2][:rank])
        proposed, proposed_rhs = proposed[keep], proposed_rhs[keep]
    else:
        proposed = np.empty((0, combined.shape[1]))
        proposed_rhs = np.empty(0)
    return (
        np.vstack([model.extra_constraints, proposed]),
        np.concatenate([model.extra_constraint_rhs, proposed_rhs]),
    )


def project_gaussian_model(
    model: CompiledLGM,
    observations: tuple[LinearObservation, ...],
    constraints: tuple[LinearConstraint, ...],
) -> CompiledLGM:
    """Fit in aggregate-observation space while predicting on the original grid."""
    if not isinstance(model.likelihood, CompiledGaussian):
        raise UnsupportedEngineError("LinearObservation requires a Gaussian likelihood")
    if any(item.scale != "identity" for item in (*observations, *constraints)):
        raise ModelValidationError(
            "scale='log' observations and constraints must be linearized before projection"
        )

    designs, values, offsets, sigmas = [], [], [], []
    observed = model.observed
    if observed.any():
        designs.append(model.design[observed])
        values.append(model.y[observed])
        offsets.append(model.offset[observed])
        sigmas.append(np.full(np.count_nonzero(observed), model.likelihood.sigma))
    for observation in observations:
        if isinstance(observation.sigma, Hyperparameter):
            raise ModelValidationError(
                f"LinearObservation sigma {observation.sigma.name!r} is unresolved; "
                "it is estimated through LGM.fit"
            )
        operator = _aligned(
            observation.operator,
            model.prediction_design.shape[0],
            "LinearObservation operator",
        )
        designs.append(operator @ model.prediction_design)
        values.append(observation.values)
        offsets.append(np.asarray(operator @ model.prediction_offset).reshape(-1))
        sigmas.append(observation.sigma)

    width = model.design.shape[1]
    if designs:
        design = vstack(designs, format="csr")
        y, offset, sigma = map(np.concatenate, (values, offsets, sigmas))
        inverse_sigma = 1.0 / sigma
        design = design.multiply(inverse_sigma[:, None]).tocsr()
        y, offset = y * inverse_sigma, offset * inverse_sigma
        normalization = model.log_likelihood_normalization - float(np.log(sigma).sum())
    else:
        design = csr_matrix((0, width))
        y = offset = np.empty(0)
        normalization = model.log_likelihood_normalization

    blocks, start = [], 0
    for block in model.blocks:
        stop = start + block.design.shape[1]
        blocks.append(
            LatentBlock(
                block.name, block.labels, design[:, start:stop],
                block.precision, block.constraints,
            )
        )
        start = stop

    extra, extra_rhs = _constraint_rows(model, constraints)
    intrinsic_count = model.constraints.shape[0] - model.extra_constraints.shape[0]
    all_constraints = np.vstack([model.constraints[:intrinsic_count], extra])
    return CompiledLGM(
        y=y,
        observed=np.ones(y.size, dtype=bool),
        offset=offset,
        design=design,
        precision=model.precision,
        constraints=all_constraints,
        labels=model.labels,
        likelihood=CompiledGaussian(1.0),
        blocks=tuple(blocks),
        extra_constraints=extra,
        extra_constraint_rhs=extra_rhs,
        prediction_design=model.prediction_design,
        prediction_offset=model.prediction_offset,
        prediction_observation_variance=model.likelihood.variance,
        log_likelihood_normalization=normalization,
        data_constraint_count=extra.shape[0] - model.extra_constraints.shape[0],
        row_log_scale=np.log(sigma) if designs else None,
    )


def project_joint_model(model, observations, constraints):
    """Append standardized ``LinearObservation`` pseudo-rows to a stacked joint model.

    Unlike :func:`project_gaussian_model`, the sub-model rows keep their own
    likelihoods; only the pseudo-rows are divided by ``sigma``, under a unit
    Gaussian part appended to the mixture after every grid row.
    """
    if any(item.scale != "identity" for item in (*observations, *constraints)):
        raise ModelValidationError(
            "scale='log' observations and constraints must be linearized before projection"
        )
    rows = model.design.shape[0]
    grid_rows = model.prediction_design.shape[0]
    designs, values, offsets, sigmas = [], [], [], []
    for observation in observations:
        if isinstance(observation.sigma, Hyperparameter):
            raise ModelValidationError(
                f"LinearObservation sigma {observation.sigma.name!r} is unresolved; "
                "it is estimated through Joint.fit"
            )
        operator = _aligned(observation.operator, grid_rows, "LinearObservation operator")
        designs.append(operator @ model.prediction_design)
        values.append(observation.values)
        offsets.append(np.asarray(operator @ model.prediction_offset).reshape(-1))
        sigmas.append(observation.sigma)
    if designs:
        added_design = vstack(designs, format="csr")
        added_y, added_offset, sigma = map(np.concatenate, (values, offsets, sigmas))
        inverse_sigma = 1.0 / sigma
        added_design = added_design.multiply(inverse_sigma[:, None]).tocsr()
        added_y, added_offset = added_y * inverse_sigma, added_offset * inverse_sigma
        normalization = model.log_likelihood_normalization - float(np.log(sigma).sum())
    else:
        added_design = csr_matrix((0, model.design.shape[1]))
        added_y = added_offset = np.empty(0)
        normalization = model.log_likelihood_normalization
    added = added_y.size
    design = vstack([model.design, added_design], format="csr")

    blocks, start = [], 0
    for block in model.blocks:
        stop = start + block.design.shape[1]
        blocks.append(
            LatentBlock(
                block.name, block.labels, design[:, start:stop],
                block.precision, block.constraints,
            )
        )
        start = stop

    likelihood = model.likelihood
    if added:
        parts = (
            model.likelihood.parts
            if isinstance(model.likelihood, CompiledMixture)
            else ((np.ones(rows, dtype=bool), model.likelihood),)
        )
        padding = np.zeros(added, dtype=bool)
        parts = tuple((np.concatenate([mask, padding]), lk) for mask, lk in parts)
        pseudo = np.concatenate([np.zeros(rows, dtype=bool), np.ones(added, dtype=bool)])
        likelihood = CompiledMixture((*parts, (pseudo, CompiledGaussian(1.0))), rows + added)
    extra, extra_rhs = _constraint_rows(model, constraints)
    intrinsic_count = model.constraints.shape[0] - model.extra_constraints.shape[0]
    return CompiledLGM(
        y=np.concatenate([model.y, added_y]),
        observed=np.concatenate([model.observed, np.ones(added, dtype=bool)]),
        offset=np.concatenate([model.offset, added_offset]),
        design=design, precision=model.precision,
        constraints=np.vstack([model.constraints[:intrinsic_count], extra]),
        labels=model.labels, likelihood=likelihood, blocks=tuple(blocks),
        extra_constraints=extra, extra_constraint_rhs=extra_rhs,
        prediction_design=model.prediction_design, prediction_offset=model.prediction_offset,
        log_likelihood_normalization=normalization,
        data_constraint_count=extra.shape[0] - model.extra_constraints.shape[0],
        row_log_scale=(
            np.concatenate([np.zeros(rows), np.log(sigma)]) if designs else model.row_log_scale
        ),
    )


def observation_hyperparameters(observations) -> tuple[Hyperparameter, ...]:
    return tuple(item.sigma for item in observations if isinstance(item.sigma, Hyperparameter))


@dataclass(frozen=True)
class _ConstantFamily:
    """A model with no hyperparameters of its own, as a family of one member."""

    model: CompiledLGM
    parameter_names: tuple = ()
    parameter_bounds = {}
    parameter_priors = {}

    def materialize(self, values):
        return self.model


@dataclass(frozen=True)
class _ProjectedGaussianFamily:
    base: object
    observations: tuple[LinearObservation, ...]
    constraints: tuple[LinearConstraint, ...]

    @property
    def hyperparameters(self) -> tuple[Hyperparameter, ...]:
        """The ``LinearObservation`` sigmas this family estimates beyond ``base``."""
        return observation_hyperparameters(self.observations)

    @property
    def parameter_names(self):
        return (*self.base.parameter_names, *(hp.name for hp in self.hyperparameters))

    @property
    def parameter_bounds(self):
        from pylgm.compiler import _log_bounds

        bounds = dict(self.base.parameter_bounds)
        bounds.update({hp.name: _log_bounds(hp) for hp in self.hyperparameters})
        return bounds

    @property
    def parameter_priors(self):
        priors = dict(self.base.parameter_priors)
        priors.update({hp.name: hp.prior for hp in self.hyperparameters if hp.prior is not None})
        return priors

    def materialize(self, values):
        base = self.base.materialize({name: values[name] for name in self.base.parameter_names})
        observations = tuple(
            replace_sigma(item, values[item.sigma.name])
            if isinstance(item.sigma, Hyperparameter) else item
            for item in self.observations
        )
        return project_gaussian_model(base, observations, self.constraints)


@dataclass(frozen=True)
class _ProjectedJointFamily(_ProjectedGaussianFamily):
    """``_ProjectedGaussianFamily`` for a stacked joint: sub-model rows keep their likelihoods."""

    def materialize(self, values):
        base = self.base.materialize({name: values[name] for name in self.base.parameter_names})
        observations = tuple(
            replace_sigma(item, values[item.sigma.name])
            if isinstance(item.sigma, Hyperparameter) else item
            for item in self.observations
        )
        return project_joint_model(base, observations, self.constraints)


@dataclass(frozen=True)
class _RelinearizedFamily(_ProjectedGaussianFamily):
    """A projected family whose ``scale='log'`` items are relinearized at every ``theta``.

    Each ``materialize`` runs the fixed point of :func:`relinearize`, warm-started
    from the previous solution, so empirical Bayes and INLA see an ordinary
    linear projected model.
    """

    project: Callable = field(default=project_gaussian_model, compare=False)
    inner_fit: Callable | None = field(default=None, compare=False)
    _start: dict = field(default_factory=dict, compare=False, repr=False)

    def materialize(self, values):
        base = self.base.materialize({name: values[name] for name in self.base.parameter_names})
        observations = tuple(
            replace_sigma(item, values[item.sigma.name])
            if isinstance(item.sigma, Hyperparameter) else item
            for item in self.observations
        )
        model, eta = relinearize(
            base, observations, self.constraints,
            project=self.project, inner_fit=self.inner_fit, start=self._start.get("eta"),
        )
        self._start["eta"] = eta
        return model


def replace_sigma(observation: LinearObservation, sigma: float) -> LinearObservation:
    return LinearObservation(observation.values, observation.operator, sigma, scale=observation.scale)


def project_gaussian_family(family, observations, constraints, *, base_model=None, family_type=None):
    """Wrap ``family`` (or, when it is ``None``, the fixed ``base_model``)."""
    if family_type is None:
        family_type = _ProjectedGaussianFamily
    if family is None:
        family = _ConstantFamily(base_model)
    names = list(family.parameter_names) + [
        hp.name for hp in observation_hyperparameters(observations)
    ]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ModelValidationError(f"hyperparameter names must be unique: {duplicated}")
    return family_type(family, observations, constraints)


def reorder_linear_inputs(observations, constraints, source_positions):
    """Move caller-ordered operator columns into canonical panel order."""
    rows = len(source_positions)
    for item in (*observations, *constraints):
        _aligned(item.operator, rows, f"{type(item).__name__} operator")
    observations = tuple(
        LinearObservation(
            item.values, item.operator[:, source_positions], item.sigma, scale=item.scale
        )
        for item in observations
    )
    constraints = tuple(
        LinearConstraint(item.operator[:, source_positions], item.rhs, scale=item.scale)
        for item in constraints
    )
    return observations, constraints


_RELINEARIZATION_TOLERANCE = 1e-9
_RELINEARIZATION_MAX_ITERATIONS = 100


def linearize(observations, constraints, eta):
    """Replace every ``scale='log'`` item by its tangent at ``eta``; identity items pass through."""
    level = np.exp(eta)
    shift = level * (1.0 - eta)

    def tangent(operator):
        return (operator @ diags(level)).tocsr(), np.asarray(operator @ shift).reshape(-1)

    linear_observations = []
    for item in observations:
        if item.scale == "log":
            operator, offset = tangent(item.operator)
            item = LinearObservation(item.values - offset, operator, item.sigma)
        linear_observations.append(item)
    linear_constraints = []
    for item in constraints:
        if item.scale == "log":
            operator, offset = tangent(item.operator)
            item = LinearConstraint(operator, item.rhs - offset)
        linear_constraints.append(item)
    return tuple(linear_observations), tuple(linear_constraints)


def _log_columns(observations, constraints) -> np.ndarray:
    """Grid rows referenced by a nonzero entry of some ``scale='log'`` operator (sorted ints)."""
    columns = set()
    for item in (*observations, *constraints):
        if item.scale == "log":
            operator = item.operator.copy()
            operator.eliminate_zeros()
            columns.update(np.unique(operator.indices).tolist())
    return np.array(sorted(columns), dtype=int)


def _initial_log_predictor(model, observations, constraints) -> np.ndarray:
    """A starting point: the prior offset, with each log-aggregated row set to its flat share."""
    eta = np.array(model.prediction_offset, dtype=float)
    proposals: dict[int, list[float]] = {}
    for item in (*observations, *constraints):
        if item.scale != "log":
            continue
        targets = item.values if isinstance(item, LinearObservation) else item.rhs
        operator = item.operator.tocsr()
        for row in range(operator.shape[0]):
            start, stop = operator.indptr[row], operator.indptr[row + 1]
            cols = operator.indices[start:stop]
            weights = operator.data[start:stop]
            if cols.size == 0:
                continue
            target = targets[row]
            if not np.all(weights > 0) or not target > 0:
                continue
            proposal = float(np.log(target / weights.sum()))
            for column in cols:
                proposals.setdefault(int(column), []).append(proposal)
    for column, values in proposals.items():
        eta[column] = float(np.mean(values))
    return eta


def relinearize(base, observations, constraints, *, project, inner_fit, start=None):
    """Fixed-point Gauss-Newton relinearization of the ``scale='log'`` items.

    Returns ``(model, eta)``: ``base`` projected with the items linearized at ``eta``,
    where the fitted grid predictor reproduces ``eta`` on every log-referenced row.
    """
    columns = _log_columns(observations, constraints)
    eta = (
        _initial_log_predictor(base, observations, constraints)
        if start is None else np.array(start, dtype=float)
    )
    previous, damping, change = np.inf, 1.0, np.inf
    for _ in range(_RELINEARIZATION_MAX_ITERATIONS):
        model = project(base, *linearize(observations, constraints, eta))
        fit = inner_fit(model)
        target = model.prediction_offset + np.asarray(model.prediction_design @ fit.mean).reshape(-1)
        if not columns.size:
            return model, target
        change = float(np.max(np.abs(target[columns] - eta[columns])))
        if change <= _RELINEARIZATION_TOLERANCE * max(1.0, float(np.max(np.abs(eta[columns])))):
            return model, eta
        # Damping only tames oscillation; it does not move the fixed point.
        if change > previous:
            damping = max(damping / 2.0, 1.0 / 16.0)
        previous = change
        eta = eta + damping * (target - eta)
    raise InferenceError(
        f"scale='log' relinearization did not converge in {_RELINEARIZATION_MAX_ITERATIONS} "
        f"iterations (last change {change:.3e})"
    )


__all__ = ["LinearConstraint", "LinearObservation"]
