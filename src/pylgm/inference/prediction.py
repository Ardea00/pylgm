"""Out-of-sample prediction: score new rows against a fitted latent posterior.

``predict_from`` rebuilds the fixed/structured design for rows that were not
passed to ``fit`` and reuses the already-fitted posterior mean/covariance. It
cannot create a new latent column, so an index level absent from the fitted
domain is an error rather than a silently dropped or zero-filled contribution.

``predictive_variance`` is the linear-predictor (eta) posterior variance for
every likelihood, computed the same way ``pylgm.inference.gaussian`` and
``pylgm.inference.laplace`` compute it. ``fitted_mean`` is not re-derived
either: it calls the same ``likelihood.response_prediction`` method those
modules use, so a fit-row round trip reproduces the fitted result exactly --
with one documented exception.

**Integrated fits with a non-linear link.** ``integrate_inla`` builds
``fitted_mean`` by mixing the *transformed* per-hyperparameter values,
``sum_k w_k g(mu_k, s_k)``, whereas prediction transforms the *integrated*
moments, ``g(sum_k w_k mu_k, Var_mixture)``. For a non-linear ``g`` (the
Poisson log link's lognormal mean, say) Jensen's inequality makes these differ
slightly, so ``predict().fitted_mean`` is a moment-matched approximation of
``INLAResult.fitted_mean`` rather than an exact reproduction; the gap scales
with the hyperparameter uncertainty. Reproducing it exactly would require
retaining every grid point's latent covariance, which is not affordable for a
wide latent field. ``predictive_mean`` and ``predictive_variance`` are exact,
as is ``fitted_mean`` for the identity link and for all plug-in and
empirical-Bayes fits.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from formulaic import ModelSpec

from pylgm.exceptions import NumericalError
from pylgm.inference.result import _readonly_array


@dataclass(frozen=True)
class PredictionContext:
    """Everything needed to rebuild the fitted design for new rows.

    ``entries`` is an ordered tuple of block descriptors, in fitted block
    order: ``("fixed", ModelSpec)``,
    ``("structured", (block_name, index_column, labels_tuple))``,
    ``("midas", (block_name, columns_tuple))``,
    ``("spacetime", (block_name, space, time, area_labels, time_labels))``,
    ``("dynamic_spatial_panel", (block_name, unit, time, unit_labels, time_labels))``,
    ``("grouped_structured", (block_name, group, index, group_labels, level_labels))``,
    or ``("shared", (block_name, index, labels, scale_spec, fitted_scale))`` (joint
    models only -- a shared field's predict-time design is ``scale * incidence``).
    """

    entries: tuple[tuple[str, object], ...]
    likelihood: object
    offset: str | None
    trials: str | None = None
    width: int = 0
    column_slices: tuple[tuple[int, int], ...] = ()
    """Per-entry ``(start, stop)`` column spans in the fitted latent.

    Empty means the entries are contiguous and gapless from column 0, which is
    every single-response model. A joint model's per-outcome context sets this
    because its entries occupy scattered spans of the stacked latent.
    """


@dataclass(frozen=True)
class JointPredictionContext:
    """Per-outcome prediction contexts for a joint model."""

    contexts: Mapping[str, PredictionContext]

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(self.contexts)


def _fixed_block(spec: ModelSpec, new_data: pd.DataFrame) -> np.ndarray:
    # na_action="raise" rather than formulaic's default "drop". Dropping is
    # doubly unsafe here: a NaN covariate would silently shorten the design
    # (and a single surviving row would then broadcast over every requested
    # row), and an unseen categorical level becomes NaN, which contrast coding
    # would otherwise render as an all-zero dummy -- indistinguishable from the
    # reference level. Both must fail loudly, as unseen structured levels do.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matrix = spec.get_model_matrix(new_data, na_action="raise")
        # An unseen categorical level is cast to NaN inside contrast coding,
        # after na_action is applied, so it only surfaces as a warning; left
        # alone it becomes an all-zero dummy row that scores as the reference
        # level. Escalate it to the same hard error the structured blocks give.
        mismatch = next(
            (
                item
                for item in caught
                if type(item.message).__name__ == "DataMismatchWarning"
            ),
            None,
        )
        if mismatch is not None:
            raise ValueError(str(mismatch.message))
        for item in caught:  # keep every other warning visible
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )
    except Exception as error:
        raise ValueError(
            "predict() could not rebuild the fixed-effect design from new_data "
            f"({error}). A null covariate, or a categorical level that was not "
            "in the fitted data, will do this: predict reuses the fitted design, "
            "so it cannot score a level the model never saw."
        ) from error
    return np.asarray(matrix, dtype=float)


def _structured_block(entry: tuple[str, str, tuple[str, ...]], new_data: pd.DataFrame) -> np.ndarray:
    name, index_column, labels = entry
    if index_column not in new_data.columns:
        raise ValueError(
            f"predict() new_data is missing column {index_column!r} required by the "
            f"{name!r} block"
        )
    positions = {label: column for column, label in enumerate(labels)}
    keys = new_data[index_column].map(str)
    known = keys.isin(positions)
    if not known.all():
        missing = sorted(set(keys[~known].tolist()))
        raise ValueError(
            f"predict() cannot score rows whose {name!r} level was not in the fitted "
            f"model: {missing!r}. predict reuses the fitted latent posterior, so it "
            "cannot create a new latent component. To forecast new levels, include "
            "those rows at fit time with a NaN response instead."
        )
    design = np.zeros((len(new_data), len(labels)))
    design[np.arange(len(new_data)), keys.map(positions).to_numpy()] = 1.0
    return design


def _midas_block(entry: tuple[str, tuple[str, ...]], new_data: pd.DataFrame) -> np.ndarray:
    name, columns = entry
    missing = [column for column in columns if column not in new_data.columns]
    if missing:
        raise ValueError(
            f"predict() new_data is missing lag columns {missing!r} required by the "
            f"{name!r} block"
        )
    design = new_data[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(design).all():
        raise ValueError(
            f"predict() new_data has non-finite values in the {name!r} lag columns"
        )
    return design


def _midas_parametric_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    from pylgm.effects.midas import midas_weights

    name, columns, kernel, theta = entry
    missing = [column for column in columns if column not in new_data.columns]
    if missing:
        raise ValueError(
            f"predict() new_data is missing lag columns {missing!r} required by the "
            f"{name!r} block"
        )
    values = new_data[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"predict() new_data has non-finite values in the {name!r} lag columns")
    weights = midas_weights(kernel, len(columns), theta)
    return (values @ weights).reshape(-1, 1)


def _paired_cell_block(
    name: str,
    outer_column: str,
    inner_column: str,
    outer_labels: tuple[str, ...],
    inner_labels: tuple[str, ...],
    new_data: pd.DataFrame,
    *,
    block_label: str,
    pair_label: str,
    hint: str,
) -> np.ndarray:
    """One-hot design for a block whose latent cell is an (outer, inner) pair.

    Shared by the space-time interaction, the dynamic spatial panel, and the
    grouped AR1: all three index a latent cell by two columns and lay the cells
    out ``outer_position * len(inner_labels) + inner_position``. Callers pass
    their own labels in that order, so the caller owns the layout convention;
    only the lookup, validation, and one-hot construction are shared.
    """
    for column in (outer_column, inner_column):
        if column not in new_data.columns:
            raise ValueError(
                f"predict() new_data is missing column {column!r} required by the "
                f"{name!r} {block_label} block"
            )
    outer_pos = {label: i for i, label in enumerate(outer_labels)}
    inner_pos = {label: j for j, label in enumerate(inner_labels)}
    outer = new_data[outer_column].map(str)
    inner = new_data[inner_column].map(str)
    unseen = sorted(set(outer[~outer.isin(outer_pos)]) | set(inner[~inner.isin(inner_pos)]))
    if unseen:
        raise ValueError(
            f"predict() cannot score rows whose {name!r} {pair_label} was not in the "
            f"fitted model: {unseen!r}. predict reuses the fitted latent posterior, so it "
            f"cannot create a new latent cell. {hint}"
        )
    cells = outer.map(outer_pos).to_numpy() * len(inner_labels) + inner.map(inner_pos).to_numpy()
    design = np.zeros((len(new_data), len(outer_labels) * len(inner_labels)))
    design[np.arange(len(new_data)), cells] = 1.0
    return design


def _spacetime_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    name, space, time, area_labels, time_labels = entry
    return _paired_cell_block(
        name, space, time, area_labels, time_labels, new_data,
        block_label="space-time",
        pair_label="space/time level",
        hint="To forecast new cells, include those rows at fit time with a NaN "
             "response instead.",
    )


def _dynamic_spatial_panel_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    # Time-major, unlike the other two: the SDPD operator stacks whole periods.
    name, unit, time, unit_labels, time_labels = entry
    return _paired_cell_block(
        name, time, unit, time_labels, unit_labels, new_data,
        block_label="dynamic-spatial-panel",
        pair_label="unit/time",
        hint="To forecast future periods, use the SDPD forecast() helper instead.",
    )


def _grouped_structured_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    name, group, index_column, group_labels, level_labels = entry
    return _paired_cell_block(
        name, group, index_column, group_labels, level_labels, new_data,
        block_label="grouped",
        pair_label="group/level",
        hint="To forecast new levels, include those rows at fit time with a NaN "
             "response instead.",
    )


def _shared_design_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    """Rebuild a shared field's design for one outcome: scale_k * incidence.

    Named to distinguish it from ``compiler._shared_block``, which builds the
    fit-time LatentBlock; this one rebuilds the dense predict-time design.
    """
    name, index, labels, scale_spec, fitted = entry
    if index not in new_data.columns:
        raise ValueError(f"predict() new_data is missing the index column {index!r}")
    position_of = {label: i for i, label in enumerate(labels)}
    block = np.zeros((len(new_data), len(labels)))
    scale = fitted if isinstance(scale_spec, str) else float(scale_spec)
    for row, value in enumerate(new_data[index].astype(str)):
        if value not in position_of:
            raise ValueError(
                f"predict() new_data has an unseen level {value!r} in {index!r} "
                f"for shared effect {name!r}"
            )
        block[row, position_of[value]] = scale
    return block


def _weighted_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    """Scale a nested entry's rebuilt design by a weight column from new_data."""
    inner_entry, by_column = entry
    if by_column not in new_data.columns:
        raise ValueError(
            f"predict() new_data is missing the weight column {by_column!r}"
        )
    weights = pd.to_numeric(new_data[by_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError(
            f"predict() weight column {by_column!r} must be numeric and finite"
        )
    return weights[:, None] * _design_block_for(inner_entry, new_data)


def _design_block_for(entry: tuple[str, object], new_data: pd.DataFrame) -> np.ndarray:
    """Rebuild the dense predict-time design block for one ``(kind, payload)`` entry."""
    kind, payload = entry
    if kind == "fixed":
        return _fixed_block(payload, new_data)
    elif kind == "structured":
        return _structured_block(payload, new_data)
    elif kind == "midas":
        return _midas_block(payload, new_data)
    elif kind == "midas_parametric":
        return _midas_parametric_block(payload, new_data)
    elif kind == "spacetime":
        return _spacetime_block(payload, new_data)
    elif kind == "dynamic_spatial_panel":
        return _dynamic_spatial_panel_block(payload, new_data)
    elif kind == "grouped_structured":
        return _grouped_structured_block(payload, new_data)
    elif kind == "shared":
        return _shared_design_block(payload, new_data)
    elif kind == "weighted":
        return _weighted_block(payload, new_data)
    else:
        raise ValueError(f"predict() context has an unknown block kind {kind!r}")


def _design_for(context: PredictionContext, new_data: pd.DataFrame) -> np.ndarray:
    if not isinstance(new_data, pd.DataFrame) or new_data.empty:
        raise ValueError("predict() new_data must be a non-empty pandas DataFrame")
    blocks = [_design_block_for(entry, new_data) for entry in context.entries]
    if context.column_slices:
        if len(context.column_slices) != len(blocks):
            raise ValueError(
                "predict() context column_slices must align one-to-one with entries"
            )
        design = np.zeros((len(new_data), context.width))
        for block, (start, stop) in zip(blocks, context.column_slices):
            if stop - start != block.shape[1]:
                raise ValueError(
                    f"predict() rebuilt a block of width {block.shape[1]} for a "
                    f"column span of width {stop - start}"
                )
            design[:, start:stop] = block
    else:
        design = np.hstack(blocks) if blocks else np.empty((len(new_data), 0))
    # Defence in depth against a block silently losing rows: a length-1 design
    # would broadcast against the offset and fabricate a prediction for every
    # requested row.
    if design.shape[0] != len(new_data):
        raise ValueError(
            f"predict() rebuilt a design with {design.shape[0]} rows for "
            f"{len(new_data)} input rows; new_data may contain nulls in a column "
            "the model reads"
        )
    if design.shape[1] != context.width:
        raise ValueError(
            f"predict() rebuilt design width {design.shape[1]} does not match the "
            f"fitted latent width {context.width}"
        )
    return design


def _offset_for(context: PredictionContext, new_data: pd.DataFrame) -> np.ndarray:
    if context.offset is None:
        return np.zeros(len(new_data))
    if context.offset not in new_data.columns:
        raise ValueError(
            f"predict() new_data is missing the offset column {context.offset!r}"
        )
    return np.asarray(new_data[context.offset], dtype=float)


def _prediction_likelihood(context: PredictionContext, new_data: pd.DataFrame) -> object:
    """Rebind a Binomial likelihood to new_data's own trials so it predicts n*p.

    Non-binomial likelihoods have no trials column and return unchanged.
    """
    if context.trials is None:
        return context.likelihood
    if context.trials not in new_data.columns:
        raise ValueError(
            f"predict() new_data is missing the trials column {context.trials!r}"
        )
    return context.likelihood.for_observations(
        {"trials": np.asarray(new_data[context.trials], dtype=float)}
    )


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise NumericalError(f"predict() produced non-finite {name}")


def predict_from(
    context: PredictionContext,
    mean: np.ndarray,
    covariance: np.ndarray,
    new_data: pd.DataFrame,
) -> "Prediction":
    """Score ``new_data`` against the fitted posterior described by ``context``."""
    design = _design_for(context, new_data)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    offset = _offset_for(context, new_data)

    eta = np.asarray(design @ mean + offset, dtype=float)
    predictive_variance = np.einsum("ij,jk,ik->i", design, covariance, design)

    likelihood = _prediction_likelihood(context, new_data)
    fitted_mean = np.asarray(likelihood.response_prediction(eta, predictive_variance), dtype=float)

    _require_finite("predictive mean", eta)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)

    return Prediction(
        predictive_mean=eta,
        predictive_variance=predictive_variance,
        fitted_mean=fitted_mean,
        keys=new_data.index,
    )


def predict_from_sparse(
    context: PredictionContext,
    mean: np.ndarray,
    sparse_posterior,
    new_data: pd.DataFrame,
) -> "Prediction":
    """Like ``predict_from``, but predictive variance comes from the sparse
    posterior (``diag(design @ Sigma_c @ design.T)``) instead of a dense
    covariance -- for results past the dense-materialisation guard.
    """
    design = _design_for(context, new_data)
    mean = np.asarray(mean, dtype=float)
    offset = _offset_for(context, new_data)

    eta = np.asarray(design @ mean + offset, dtype=float)
    predictive_variance = np.asarray(sparse_posterior.predictive_variances(design), dtype=float)

    likelihood = _prediction_likelihood(context, new_data)
    fitted_mean = np.asarray(likelihood.response_prediction(eta, predictive_variance), dtype=float)

    _require_finite("predictive mean", eta)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)

    return Prediction(
        predictive_mean=eta,
        predictive_variance=predictive_variance,
        fitted_mean=fitted_mean,
        keys=new_data.index,
    )


@dataclass(frozen=True, init=False)
class Prediction:
    """Out-of-sample scores for rows not passed to ``fit``."""

    _predictive_mean: np.ndarray = field(repr=False)
    _predictive_variance: np.ndarray = field(repr=False)
    _fitted_mean: np.ndarray = field(repr=False)
    keys: pd.Index

    def __init__(
        self,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
        fitted_mean: np.ndarray,
        keys: pd.Index,
    ) -> None:
        object.__setattr__(self, "_predictive_mean", _readonly_array(predictive_mean))
        object.__setattr__(self, "_predictive_variance", _readonly_array(predictive_variance))
        object.__setattr__(self, "_fitted_mean", _readonly_array(fitted_mean))
        object.__setattr__(self, "keys", pd.Index(keys))

    @property
    def predictive_mean(self) -> np.ndarray:
        return _readonly_array(self._predictive_mean)

    @property
    def predictive_variance(self) -> np.ndarray:
        return _readonly_array(self._predictive_variance)

    @property
    def fitted_mean(self) -> np.ndarray:
        return _readonly_array(self._fitted_mean)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "predictive_mean": self._predictive_mean,
                "predictive_sd": np.sqrt(self._predictive_variance),
                "fitted_mean": self._fitted_mean,
            },
            index=self.keys,
        )
