"""Out-of-sample prediction: score new rows against a fitted latent posterior.

``predict_from`` rebuilds the fixed/structured design for rows that were not
passed to ``fit`` and reuses the already-fitted posterior mean/covariance. It
cannot create a new latent column, so an index level absent from the fitted
domain is an error rather than a silently dropped or zero-filled contribution.

Response-scale conventions (``predictive_variance`` / ``fitted_mean``) are not
re-derived here: they call the same likelihood methods
(``likelihood.variance``, ``likelihood.response_prediction``) that
``pylgm.inference.gaussian`` and ``pylgm.inference.laplace`` use, so a
fit-row round trip matches the fitted result exactly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from formulaic import ModelSpec

from pylgm.exceptions import NumericalError
from pylgm.inference.result import _readonly_array
from pylgm.likelihoods import CompiledGaussian


@dataclass(frozen=True)
class PredictionContext:
    """Everything needed to rebuild the fitted design for new rows.

    ``entries`` is an ordered tuple of block descriptors, in fitted block
    order: ``("fixed", ModelSpec)`` or
    ``("structured", (block_name, index_column, labels_tuple))``.
    """

    entries: tuple[tuple[str, object], ...]
    likelihood: object
    offset: str | None
    width: int


def _fixed_block(spec: ModelSpec, new_data: pd.DataFrame) -> np.ndarray:
    try:
        matrix = spec.get_model_matrix(new_data)
    except Exception as error:
        raise ValueError(
            f"predict() could not rebuild the fixed-effect design from new_data: {error}"
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


def _design_for(context: PredictionContext, new_data: pd.DataFrame) -> np.ndarray:
    if not isinstance(new_data, pd.DataFrame) or new_data.empty:
        raise ValueError("predict() new_data must be a non-empty pandas DataFrame")
    blocks = []
    for kind, payload in context.entries:
        if kind == "fixed":
            blocks.append(_fixed_block(payload, new_data))
        elif kind == "structured":
            blocks.append(_structured_block(payload, new_data))
        else:
            raise ValueError(f"predict() context has an unknown block kind {kind!r}")
    design = np.hstack(blocks) if blocks else np.empty((len(new_data), 0))
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
    eta_variance = np.einsum("ij,jk,ik->i", design, covariance, design)

    likelihood = context.likelihood
    if isinstance(likelihood, CompiledGaussian):
        predictive_variance = eta_variance + likelihood.variance
    else:
        predictive_variance = eta_variance
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
