"""Declarative latent Gaussian model API."""

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import Predictor
from pylgm.effects.spec import EffectSpec, _ComposableEffect
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference import GaussianResult, INLAResult, LaplaceResult, fit_gaussian, fit_laplace
from pylgm.likelihoods import Gaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization.inla import integrate_inla


_ROW_KEY = "__pylgm_row__"


def _normalize_constraints(constraints: object) -> tuple[tuple[dict[str, float], float], ...]:
    """Validate and freeze the model-level ``A x = e`` constraints.

    Each entry is either a bare mapping ``{label: coefficient}`` (right-hand side
    ``e = 0``) or a ``(mapping, rhs)`` pair carrying a nonzero right-hand side.
    Labels are resolved against the compiled latent labels later (in the
    compiler), where the full ordering is known; here we only check shape and
    coefficient sanity so a malformed constraint fails at model construction.
    """
    try:
        rows = tuple(constraints)
    except TypeError as error:
        raise TypeError("constraints must be an iterable of label->coefficient mappings") from error
    normalized: list[tuple[dict[str, float], float]] = []
    for row in rows:
        if isinstance(row, Mapping):
            mapping, rhs = row, 0.0
        elif isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], Mapping):
            mapping = row[0]
            try:
                rhs = float(row[1])
            except (TypeError, ValueError) as error:
                raise ValueError("constraint right-hand side must be a real number") from error
            if not np.isfinite(rhs):
                raise ValueError("constraint right-hand side must be finite")
        else:
            raise TypeError(
                "each constraint must be a {label: coefficient} mapping "
                "or a (mapping, rhs) pair"
            )
        if not mapping:
            raise ValueError("each constraint must reference at least one latent label")
        clean: dict[str, float] = {}
        for label, coefficient in mapping.items():
            if not isinstance(label, str) or not label:
                raise ValueError("constraint labels must be non-empty strings")
            try:
                value = float(coefficient)
            except (TypeError, ValueError) as error:
                raise ValueError(f"constraint coefficient for {label!r} must be a real number") from error
            if not np.isfinite(value):
                raise ValueError(f"constraint coefficient for {label!r} must be finite")
            clean[label] = value
        if not any(value != 0.0 for value in clean.values()):
            raise ValueError("each constraint must have at least one nonzero coefficient")
        normalized.append((clean, rhs))
    return tuple(normalized)


def _context_with_fitted_likelihood(context, result, model):
    """Replace the context's plug-in likelihood with the fitted one.

    ``build_prediction_context`` reads the likelihood off a ``compile_lgm``
    result, which resolves a ``Hyperparameter`` sigma to its ``.initial``. On
    the optimise and integrate paths that is only the starting guess, so
    ``predict`` would add the wrong observation variance -- silently, and by
    orders of magnitude. Substitute what was actually estimated.
    """
    from dataclasses import replace

    from pylgm.likelihoods import CompiledGaussian, Gaussian, WeibullSurv
    from pylgm.parameters import Hyperparameter

    if isinstance(model.likelihood, Gaussian) and isinstance(model.likelihood.sigma, Hyperparameter):
        name = model.likelihood.sigma.name

        marginals = getattr(result, "hyperparameter_marginals", None)
        if callable(marginals):
            table = marginals() or {}
            if name in table:
                # The integrated fit-row variance mixes per-theta sigma^2, so the
                # matching plug-in is sqrt(E[sigma^2]) = sqrt(mean^2 + variance).
                entry = table[name]
                second = float(entry.mean[0]) ** 2 + float(entry.variance[0])
                return replace(context, likelihood=CompiledGaussian(float(np.sqrt(second))))

        fitted = result.hyperparameters
        if fitted and name in fitted:
            return replace(context, likelihood=CompiledGaussian(float(fitted[name])))
        return context

    if isinstance(model.likelihood, WeibullSurv) and isinstance(model.likelihood.shape, Hyperparameter):
        name = model.likelihood.shape.name
        marginals = getattr(result, "hyperparameter_marginals", None)
        if callable(marginals):
            table = marginals() or {}
            if name in table:
                fitted_shape = float(table[name].mean[0])
                return replace(context, likelihood=model.likelihood.materialize({name: fitted_shape}))
        fitted = getattr(result, "hyperparameters", None) or {}
        if name in fitted:
            return replace(context, likelihood=model.likelihood.materialize({name: float(fitted[name])}))
    return context


def _context_with_fitted_weights(context, result):
    """Substitute fitted shape estimates into parametric-MIDAS design entries.

    ``build_prediction_context`` runs on a ``compile_lgm`` result, which resolves
    an estimated shape Hyperparameter to its ``.initial`` (the starting guess). A
    parametric-MIDAS design is a function of that shape, so predict() would
    otherwise aggregate new rows with the initial weights, not the fitted ones.
    Mirror ``_context_with_fitted_likelihood`` and swap in the estimates. Under
    ``hyperparameters="integrate"`` the point estimate lives in the INLA
    marginal table (``result.hyperparameters`` is ``None`` there), so check that
    first, exactly as the likelihood fixup does for sigma.
    """
    from dataclasses import replace

    marginals = getattr(result, "hyperparameter_marginals", None)
    table = marginals() if callable(marginals) else None
    fitted = getattr(result, "hyperparameters", None) or {}

    def _resolve(name):
        if table and name in table:
            return float(table[name].mean[0])
        return float(fitted[name])

    new_entries = []
    changed = False
    for kind, payload in context.entries:
        if kind == "midas_parametric":
            name, columns, kernel, theta_spec = payload
            theta = tuple(s if not isinstance(s, str) else _resolve(s) for s in theta_spec)
            new_entries.append((kind, (name, columns, kernel, theta)))
            changed = True
        else:
            new_entries.append((kind, payload))
    if not changed:
        return context
    return replace(context, entries=tuple(new_entries))


def _rebuild_result(
    result: GaussianResult | LaplaceResult | INLAResult,
    *,
    caller_order: np.ndarray | None = None,
    prediction_keys: pd.DataFrame | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    diagnostics: Mapping[str, object] | None = None,
    prediction_context: object | None = None,
) -> GaussianResult | LaplaceResult | INLAResult:
    """Rebuild a result of the same type, optionally reordering rows or overriding metadata.

    Any override left as ``None`` carries the corresponding value forward from ``result``
    unchanged, so callers only pass what they mean to change.
    """
    predictive_mean = result.predictive_mean
    # ponytail: read result._covariance/_predictive_variance (private) here --
    # _rebuild_result reconstructs the object, so it needs the raw stored value,
    # not the guard-raising public property.
    predictive_variance = result._predictive_variance
    if caller_order is not None:
        predictive_mean = predictive_mean[caller_order]
        if predictive_variance is not None:
            predictive_variance = predictive_variance[caller_order]
    common = dict(
        labels=result.labels,
        mean=result.mean,
        covariance=result._covariance,
        log_marginal_likelihood=result.log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        block_slices=result.block_slices,
        diagnostics=diagnostics if diagnostics is not None else result.diagnostics,
        prediction_keys=prediction_keys if prediction_keys is not None else result.prediction_keys,
        hyperparameters=hyperparameters if hyperparameters is not None else result.hyperparameters,
        prediction_context=(
            prediction_context if prediction_context is not None else result.prediction_context
        ),
    )
    if isinstance(result, LaplaceResult):
        fitted_mean = (
            result.fitted_mean[caller_order] if caller_order is not None else result.fitted_mean
        )
        return LaplaceResult(fitted_mean=fitted_mean, link_name=result.link_name, **common)
    if isinstance(result, INLAResult):
        fitted_mean = (
            result.fitted_mean[caller_order]
            if caller_order is not None and result.fitted_mean is not None
            else result.fitted_mean
        )
        criteria = (
            result.criteria.reordered(caller_order)
            if caller_order is not None else result.criteria
        )
        return INLAResult(
            hyperparameter_marginals=result.hyperparameter_marginals(),
            criteria=criteria,
            fitted_mean=fitted_mean,
            link_name=result.link_name,
            latent_marginal_table=result.latent_marginal_table,
            observation_variance=result.observation_variance,
            # ponytail: latent_variances is a full-latent diagonal (not row-indexed
            # like predictive_mean/predictive_variance), so caller_order does not
            # permute it -- straight pass-through is correct.
            latent_variances=getattr(result, "_latent_variances", None),
            **common,
        )
    # ponytail: caller_order only permutes prediction rows (predictive_mean/
    # predictive_variance above), never the latent index space, so the sparse
    # posterior needs no reindexing -- straight pass-through is correct.
    return GaussianResult(
        observation_variance=result.observation_variance,
        sparse_posterior=getattr(result, "_sparse_posterior", None),
        **common,
    )


def _align_predictions_with_source_rows(
    result: GaussianResult | LaplaceResult | INLAResult, source_positions: np.ndarray
) -> GaussianResult | LaplaceResult | INLAResult:
    return _rebuild_result(result, caller_order=np.argsort(source_positions))


def _parameters_at_bound(
    values: dict[str, float], bounds: Mapping[str, object], rtol: float = 1e-3
) -> tuple[str, ...]:
    """Names whose estimate landed on the edge of its declared interval.

    A pinned estimate means the optimizer wanted to keep going and the bound,
    not the data, is setting the value -- so the fit is shaped by a default the
    caller probably never chose (``Hyperparameter`` derives its bounds from
    ``initial`` when they are not given).

    Closeness is measured on the transform's own scale, which is where the
    optimizer works: a precision of 9999.98 against an upper bound of 10000 is
    pinned for every practical purpose, but is nowhere near it in natural units.
    """
    pinned = []
    for name, bound in bounds.items():
        value = values.get(name)
        if value is None:
            continue
        to_internal = bound.transform.to_internal
        try:
            scaled = to_internal(value)
            low, high = to_internal(float(bound.lower)), to_internal(float(bound.upper))
        except (ValueError, FloatingPointError, OverflowError):
            continue
        span = high - low
        if not span > 0.0 or not np.isfinite(span):
            continue
        tolerance = span * rtol
        if scaled <= low + tolerance or scaled >= high - tolerance:
            pinned.append(name)
    return tuple(sorted(pinned))


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
    predictor: Predictor | EffectSpec
    panel: tuple[str, ...] = ()
    time: str | None = None
    offset: str | None = None
    constraints: tuple = ()
    """Extra linear constraints ``A x = e`` on the latent field.

    Each entry is either a mapping ``{qualified_label: coefficient}`` (right-hand
    side ``e = 0``) or a ``(mapping, rhs)`` pair for a nonzero right-hand side --
    the label-keyed equivalent of R-INLA's ``extraconstr``. Labels are qualified
    as ``"effect:level"`` (the same strings that appear in ``result.labels``); e.g.
    ``[{"region:oslo": 1.0, "region:bergen": -1.0}]`` forces the two region effects
    to coincide, and ``[({"region:oslo": 1.0}, 2.5)]`` pins one to ``2.5``. These
    compose with any intrinsic constraints an effect already carries (such as a
    Besag sum-to-zero). A nonzero ``e`` is imposed by conditioning the prior on
    ``A x = e`` (conditioning by kriging), matching R-INLA's semantics.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.response, str) or not self.response:
            raise ValueError("response must be a non-empty string")
        if not isinstance(self.predictor, Predictor):
            # Any single effect is a one-term predictor. Checked against the
            # shared base rather than a listed tuple, which silently went stale
            # as effects were added and rejected bare AR1/Besag/MIDAS/... .
            if not isinstance(self.predictor, _ComposableEffect):
                raise TypeError(
                    "predictor must be an effect or a sum of effects, got "
                    f"{type(self.predictor).__name__}"
                )
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
        object.__setattr__(self, "constraints", _normalize_constraints(self.constraints))

    def fit(
        self,
        frame: object,
        engine: str = "exact_gaussian",
        *,
        max_driver_rows: int | None = 100_000,
        hyperparameters: str = "optimize",
        latent_strategy: str = "gaussian",
    ):
        """Compile and fit this model with an explicitly selected engine.

        Pandas input keeps caller-row prediction order. A Spark DataFrame is
        collected through the optional adapter and its predictions are returned
        in canonical ``(*panel, time)`` order with immutable ``prediction_keys``.
        ``max_driver_rows`` bounds the Spark driver collection only.

        ``hyperparameters`` selects how any declared ``Hyperparameter`` is
        resolved: ``"optimize"`` (default) fits the empirical Bayes mode, and
        ``"integrate"`` runs INLA grid integration over the hyperparameter
        posterior instead (requires at least one declared ``Hyperparameter``).

        ``latent_strategy`` selects how latent marginals are summarized under
        ``hyperparameters="integrate"``: ``"gaussian"`` (default) keeps the
        Gaussian conditional summaries, ``"simplified_laplace"`` fits
        skew-normal marginals (Rue-Martino-Chopin 2009), and ``"laplace"``
        fits full-Laplace tabulated marginals (unconstrained models only);
        both non-``"gaussian"`` strategies require ``hyperparameters="integrate"``.
        """
        if hyperparameters not in ("optimize", "integrate"):
            raise ValueError(
                f"hyperparameters must be 'optimize' or 'integrate', got {hyperparameters!r}"
            )
        if latent_strategy not in ("gaussian", "simplified_laplace", "laplace"):
            raise ValueError(
                f"latent_strategy must be 'gaussian', 'simplified_laplace', or 'laplace', "
                f"got {latent_strategy!r}"
            )
        if latent_strategy != "gaussian" and hyperparameters != "integrate":
            raise ValueError(
                f"latent_strategy={latent_strategy!r} requires hyperparameters='integrate'"
            )
        if isinstance(frame, pd.DataFrame):
            return self._fit_pandas(
                frame, engine, hyperparameters=hyperparameters, latent_strategy=latent_strategy
            )

        from pylgm.data.spark import is_spark_dataframe

        if is_spark_dataframe(frame):
            return self._fit_spark(
                frame, engine, max_driver_rows=max_driver_rows,
                hyperparameters=hyperparameters, latent_strategy=latent_strategy,
            )
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

    def _family_optimization_inputs(self, family):
        from pylgm.compiler import _model_hyperparameters

        hyperparameters = list(_model_hyperparameters(self))
        if family.parameter_bounds:
            bounds = dict(family.parameter_bounds)
        else:
            bounds = {hp.name: OptimizationBounds(hp.initial, hp.lower, hp.upper) for _, hp in hyperparameters}
        initial = {hp.name: hp.initial for _, hp in hyperparameters}
        family_priors = dict(getattr(family, "parameter_priors", {}) or {})
        priored = [hp for _, hp in hyperparameters if hp.prior is not None]
        penalty = None
        if family_priors or priored:
            def penalty(values, family_priors=family_priors, priored=priored):
                total = 0.0
                for name, prior in family_priors.items():
                    total += float(prior.logpdf(values[name]))
                for hp in priored:
                    if hp.name not in family_priors:
                        total += float(hp.prior.logpdf(values[hp.name]))
                return total
        return bounds, initial, penalty

    def _run_empirical_bayes(self, family, engine: str) -> GaussianResult | LaplaceResult:
        fit = self._engine(engine)
        bounds, initial, penalty = self._family_optimization_inputs(family)
        eb = optimize_empirical_bayes(family, bounds, initial=initial, fit=fit, penalty=penalty)
        diagnostics = dict(eb.fit.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        diagnostics["hyperparameter_penalized"] = penalty is not None
        pinned = _parameters_at_bound(dict(eb.parameters), bounds)
        # Stored as a string: diagnostics values must be immutable scalars.
        diagnostics["hyperparameters_at_bound"] = ", ".join(pinned)
        if pinned:
            warnings.warn(
                f"empirical-Bayes estimate(s) {list(pinned)} landed on the edge of "
                "the declared interval, so the bound rather than the data is "
                "setting the value. Widen lower/upper on those Hyperparameters "
                "(the defaults are initial*1e-3 to initial*1e3) and refit.",
                UserWarning,
                stacklevel=3,
            )
        return _attach_estimates(eb.fit, dict(eb.parameters), diagnostics)

    def _run_inla(self, family, engine: str, latent_strategy: str = "gaussian") -> INLAResult:
        fit = self._engine(engine)
        bounds, initial, penalty = self._family_optimization_inputs(family)
        return integrate_inla(
            family, bounds, initial=initial, fit=fit, penalty=penalty,
            latent_strategy=latent_strategy,
        )

    def _fit_pandas(
        self, frame: pd.DataFrame, engine: str, *,
        hyperparameters: str = "optimize", latent_strategy: str = "gaussian",
    ) -> GaussianResult | LaplaceResult | INLAResult:
        prepared = frame.copy(deep=True)
        time = self.time
        if time is None:
            if _ROW_KEY in prepared.columns:
                raise DataContractError(f"reserved row key column already exists: {_ROW_KEY!r}")
            prepared[_ROW_KEY] = np.arange(len(prepared), dtype=np.int64)
            time = _ROW_KEY

        data = DataConfig(time=time, response=self.response, panel=self.panel)
        panel = CanonicalPanel.from_frame(prepared, data)

        from pylgm.compiler import build_prediction_context, compile_family, compile_lgm

        family = compile_family(self, panel)
        if hyperparameters == "integrate":
            if family is None:
                raise ValueError(
                    "hyperparameters='integrate' requires a declared Hyperparameter"
                )
            result = self._run_inla(family, engine, latent_strategy)
            compiled = compile_lgm(self, panel)
        elif family is None:
            compiled = compile_lgm(self, panel)
            result = self._engine(engine)(compiled)
        else:
            result = self._run_empirical_bayes(family, engine)
            compiled = compile_lgm(self, panel)
        context = build_prediction_context(self, panel, compiled, result)
        context = _context_with_fitted_likelihood(context, result, self)
        context = _context_with_fitted_weights(context, result)
        result = _rebuild_result(result, prediction_context=context)
        return _align_predictions_with_source_rows(result, panel.source_positions)

    def _fit_spark(
        self,
        frame: object,
        engine: str,
        *,
        max_driver_rows: int | None,
        hyperparameters: str = "optimize",
        latent_strategy: str = "gaussian",
    ) -> GaussianResult | LaplaceResult | INLAResult:
        from pylgm.data.spark import canonicalize_spark_frame

        canonical = canonicalize_spark_frame(frame, self, max_driver_rows=max_driver_rows)

        from pylgm.compiler import build_prediction_context, compile_family, compile_lgm

        family = compile_family(self, canonical.panel)
        if hyperparameters == "integrate":
            if family is None:
                raise ValueError(
                    "hyperparameters='integrate' requires a declared Hyperparameter"
                )
            result = self._run_inla(family, engine, latent_strategy)
            compiled = compile_lgm(self, canonical.panel)
        elif family is None:
            compiled = compile_lgm(self, canonical.panel)
            result = self._engine(engine)(compiled)
        else:
            result = self._run_empirical_bayes(family, engine)
            compiled = compile_lgm(self, canonical.panel)
        context = build_prediction_context(self, canonical.panel, compiled, result)
        context = _context_with_fitted_likelihood(context, result, self)
        context = _context_with_fitted_weights(context, result)
        return _rebuild_result(
            result, prediction_keys=canonical.prediction_keys, prediction_context=context
        )


__all__ = ["LGM"]
