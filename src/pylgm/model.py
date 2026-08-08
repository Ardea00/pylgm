"""Declarative latent Gaussian model API."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import Fixed, IID, Predictor, RW1, RW2
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference import GaussianResult, fit_gaussian


_ROW_KEY = "__pylgm_row__"


def _align_predictions_with_source_rows(
    result: GaussianResult, source_positions: np.ndarray
) -> GaussianResult:
    caller_order = np.argsort(source_positions)
    return GaussianResult(
        labels=result.labels,
        mean=result.mean,
        covariance=result.covariance,
        log_marginal_likelihood=result.log_marginal_likelihood,
        predictive_mean=result.predictive_mean[caller_order],
        predictive_variance=result.predictive_variance[caller_order],
        block_slices=result.block_slices,
        diagnostics=result.diagnostics,
    )


@dataclass(frozen=True)
class LGM:
    """A declarative latent Gaussian model."""

    response: str
    likelihood: object
    predictor: Predictor | Fixed | IID | RW1 | RW2
    panel: tuple[str, ...] = ()
    time: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, str) or not self.response:
            raise ValueError("response must be a non-empty string")
        if not isinstance(self.predictor, Predictor):
            if not isinstance(self.predictor, (Fixed, IID, RW1, RW2)):
                raise TypeError("predictor must contain declarative effects")
            object.__setattr__(self, "predictor", Predictor((self.predictor,)))
        try:
            panel = tuple(self.panel)
        except TypeError as error:
            raise TypeError("panel must be an iterable of column names") from error
        if any(not isinstance(column, str) or not column for column in panel):
            raise ValueError("panel columns must be non-empty strings")
        if self.time is not None and (not isinstance(self.time, str) or not self.time):
            raise ValueError("time must be a non-empty string or None")
        object.__setattr__(self, "panel", panel)

    def fit(self, frame: object, engine: str = "exact_gaussian"):
        """Compile and fit this model with an explicitly selected engine."""
        if not isinstance(frame, pd.DataFrame):
            if callable(getattr(frame, "toPandas", None)) or type(frame).__module__.startswith(
                "pyspark"
            ):
                raise DataContractError(
                    "PySpark input is unsupported in pyLGM 0.3; "
                    "support is planned for a later release"
                )
            raise DataContractError("frame must be a Pandas DataFrame")

        prepared = frame.copy(deep=True)
        time = self.time
        if time is None:
            if _ROW_KEY in prepared.columns:
                raise DataContractError(f"reserved row key column already exists: {_ROW_KEY!r}")
            prepared[_ROW_KEY] = np.arange(len(prepared), dtype=np.int64)
            time = _ROW_KEY

        data = DataConfig(time=time, response=self.response, panel=self.panel)
        panel = CanonicalPanel.from_frame(prepared, data)

        from pylgm.compiler import compile_lgm

        compiled = compile_lgm(self, panel)
        engines = {"exact_gaussian": fit_gaussian}
        try:
            fit = engines[engine]
        except KeyError as error:
            raise UnsupportedEngineError(f"unknown inference engine: {engine}") from error
        result = fit(compiled)
        return _align_predictions_with_source_rows(result, panel.source_positions)


__all__ = ["LGM"]
