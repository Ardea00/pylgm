"""Declarative latent Gaussian model API."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import Fixed, IID, Predictor, RW1, RW2
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference import GaussianResult, LaplaceResult, fit_gaussian, fit_laplace
from pylgm.likelihoods import Gaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes


_ROW_KEY = "__pylgm_row__"


def _rebuild_result(
    result: GaussianResult | LaplaceResult,
    *,
    caller_order: np.ndarray | None = None,
    prediction_keys: pd.DataFrame | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> GaussianResult | LaplaceResult:
    """Rebuild a result of the same type, optionally reordering rows or overriding metadata.

    Any override left as ``None`` carries the corresponding value forward from ``result``
    unchanged, so callers only pass what they mean to change.
    """
    predictive_mean = result.predictive_mean
    predictive_variance = result.predictive_variance
    if caller_order is not None:
        predictive_mean = predictive_mean[caller_order]
        predictive_variance = predictive_variance[caller_order]
    common = dict(
        labels=result.labels,
        mean=result.mean,
        covariance=result.covariance,
        log_marginal_likelihood=result.log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        block_slices=result.block_slices,
        diagnostics=diagnostics if diagnostics is not None else result.diagnostics,
        prediction_keys=prediction_keys if prediction_keys is not None else result.prediction_keys,
        hyperparameters=hyperparameters if hyperparameters is not None else result.hyperparameters,
    )
    if isinstance(result, LaplaceResult):
        fitted_mean = (
            result.fitted_mean[caller_order] if caller_order is not None else result.fitted_mean
        )
        return LaplaceResult(fitted_mean=fitted_mean, link_name=result.link_name, **common)
    return GaussianResult(**common)


def _align_predictions_with_source_rows(
    result: GaussianResult | LaplaceResult, source_positions: np.ndarray
) -> GaussianResult | LaplaceResult:
    return _rebuild_result(result, caller_order=np.argsort(source_positions))


def _attach_estimates(
    result: GaussianResult | LaplaceResult,
    hyperparameters: Mapping[str, float],
    diagnostics: Mapping[str, object],
) -> GaussianResult | LaplaceResult:
    return _rebuild_result(result, hyperparameters=hyperparameters, diagnostics=diagnostics)


@dataclass(frozen=True)
class LGM:
    """A declarative latent Gaussian model."""

    response: str
    likelihood: object
    predictor: Predictor | Fixed | IID | RW1 | RW2
    panel: tuple[str, ...] = ()
    time: str | None = None
    offset: str | None = None

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
        if self.offset is not None and (not isinstance(self.offset, str) or not self.offset):
            raise ValueError("offset must be a non-empty string or None")
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
        engines = {"exact_gaussian": fit_gaussian, "laplace": fit_laplace}
        try:
            fit = engines[engine]
        except KeyError as error:
            raise UnsupportedEngineError(f"unknown inference engine: {engine}") from error
        is_gaussian = isinstance(self.likelihood, Gaussian)
        if engine == "exact_gaussian" and not is_gaussian:
            raise UnsupportedEngineError(
                "exact_gaussian requires a Gaussian likelihood; "
                "use engine='laplace' for a non-Gaussian likelihood"
            )
        if engine == "laplace" and is_gaussian:
            raise UnsupportedEngineError(
                "engine='laplace' is for non-Gaussian likelihoods; "
                "use engine='exact_gaussian' for a Gaussian likelihood"
            )
        return fit

    def _run_empirical_bayes(self, family, engine: str) -> GaussianResult | LaplaceResult:
        from pylgm.compiler import _model_hyperparameters

        fit = self._engine(engine)
        hyperparameters = list(_model_hyperparameters(self))
        bounds = {}
        initial = {}
        for _, hyperparameter in hyperparameters:
            bounds[hyperparameter.name] = OptimizationBounds(
                hyperparameter.initial, hyperparameter.lower, hyperparameter.upper
            )
            initial[hyperparameter.name] = hyperparameter.initial
        priored = [hp for _, hp in hyperparameters if hp.prior is not None]
        penalty = None
        if priored:
            def penalty(values, priored=priored):
                return sum(hp.prior.logpdf(values[hp.name]) for hp in priored)
        eb = optimize_empirical_bayes(family, bounds, initial=initial, fit=fit, penalty=penalty)
        diagnostics = dict(eb.fit.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        diagnostics["hyperparameter_penalized"] = penalty is not None
        return _attach_estimates(eb.fit, dict(eb.parameters), diagnostics)

    def _fit_pandas(self, frame: pd.DataFrame, engine: str) -> GaussianResult | LaplaceResult:
        prepared = frame.copy(deep=True)
        time = self.time
        if time is None:
            if _ROW_KEY in prepared.columns:
                raise DataContractError(f"reserved row key column already exists: {_ROW_KEY!r}")
            prepared[_ROW_KEY] = np.arange(len(prepared), dtype=np.int64)
            time = _ROW_KEY

        data = DataConfig(time=time, response=self.response, panel=self.panel)
        panel = CanonicalPanel.from_frame(prepared, data)

        from pylgm.compiler import compile_family, compile_lgm

        family = compile_family(self, panel)
        if family is None:
            compiled = compile_lgm(self, panel)
            result = self._engine(engine)(compiled)
        else:
            result = self._run_empirical_bayes(family, engine)
        return _align_predictions_with_source_rows(result, panel.source_positions)

    def _fit_spark(
        self, frame: object, engine: str, *, max_driver_rows: int | None
    ) -> GaussianResult | LaplaceResult:
        from pylgm.data.spark import canonicalize_spark_frame

        canonical = canonicalize_spark_frame(frame, self, max_driver_rows=max_driver_rows)

        from pylgm.compiler import compile_family, compile_lgm

        family = compile_family(self, canonical.panel)
        if family is None:
            compiled = compile_lgm(self, canonical.panel)
            result = self._engine(engine)(compiled)
        else:
            result = self._run_empirical_bayes(family, engine)
        return _rebuild_result(result, prediction_keys=canonical.prediction_keys)


__all__ = ["LGM"]
