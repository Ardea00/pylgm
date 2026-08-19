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
        prediction_keys=result.prediction_keys,
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

    def fit(
        self,
        frame: object,
        engine: str = "exact_gaussian",
        *,
        max_driver_rows: int | None = 100_000,
    ):
        """Compile and fit this model with an explicitly selected engine.

        Pandas input keeps caller-row prediction order. A Spark DataFrame is
        collected through the optional adapter and its predictions are returned
        in canonical ``(*panel, time)`` order with immutable ``prediction_keys``.
        ``max_driver_rows`` bounds the Spark driver collection only.
        """
        if isinstance(frame, pd.DataFrame):
            return self._fit_pandas(frame, engine)

        from pylgm.data.spark import is_spark_dataframe

        if is_spark_dataframe(frame):
            return self._fit_spark(frame, engine, max_driver_rows=max_driver_rows)
        raise DataContractError("frame must be a Pandas DataFrame")

    def _engine(self, engine: str):
        engines = {"exact_gaussian": fit_gaussian}
        try:
            return engines[engine]
        except KeyError as error:
            raise UnsupportedEngineError(f"unknown inference engine: {engine}") from error

    def _fit_pandas(self, frame: pd.DataFrame, engine: str) -> GaussianResult:
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
        result = self._engine(engine)(compiled)
        return _align_predictions_with_source_rows(result, panel.source_positions)

    def _fit_spark(
        self, frame: object, engine: str, *, max_driver_rows: int | None
    ) -> GaussianResult:
        from pylgm.data.spark import canonicalize_spark_frame

        canonical = canonicalize_spark_frame(frame, self, max_driver_rows=max_driver_rows)

        from pylgm.compiler import compile_lgm

        compiled = compile_lgm(self, canonical.panel)
        result = self._engine(engine)(compiled)
        return GaussianResult(
            labels=result.labels,
            mean=result.mean,
            covariance=result.covariance,
            log_marginal_likelihood=result.log_marginal_likelihood,
            predictive_mean=result.predictive_mean,
            predictive_variance=result.predictive_variance,
            block_slices=result.block_slices,
            diagnostics=result.diagnostics,
            prediction_keys=canonical.prediction_keys,
        )


__all__ = ["LGM"]
