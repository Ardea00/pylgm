"""Declarative latent Gaussian model API."""

from collections.abc import Mapping
from dataclasses import dataclass
import warnings
from functools import partial

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import Predictor
from pylgm.effects.spec import EffectSpec, _ComposableEffect
from pylgm.exceptions import DataContractError, ModelValidationError, UnsupportedEngineError
from pylgm.inference import GaussianResult, INLAResult, LaplaceResult, fit_gaussian, fit_laplace
from pylgm.likelihoods import Gaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization.inla import integrate_inla, require_single_mean_shift
from pylgm.observations import LinearConstraint, LinearObservation
from pylgm.parallel import blas_limit, validate_blas_threads, validate_workers
from pylgm.parameters import Hyperparameter


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


def _hyperparameter_table(result) -> Mapping:
    """An integrated result's hyperparameter marginals; empty for any other result."""
    marginals = getattr(result, "hyperparameter_marginals", None)
    return (marginals() if callable(marginals) else None) or {}


def _point_estimates(result) -> dict[str, float]:
    """Hyperparameter point estimates: the optimum, or the integrated posterior mean."""
    estimates = {name: float(value) for name, value in (result.hyperparameters or {}).items()}
    estimates.update(
        {name: float(entry.mean[0]) for name, entry in _hyperparameter_table(result).items()}
    )
    return estimates


def _fitted_context(context, model, estimates: Mapping[str, float], table: Mapping | None = None):
    """Substitute estimated hyperparameters into a prediction context.

    ``build_prediction_context`` reads a ``compile_lgm`` result, which resolves
    every ``Hyperparameter`` to its ``.initial`` -- only the starting guess on
    the optimise and integrate paths. Swap in the estimates:

    - the likelihood: a Gaussian sigma sets ``predict``'s observation variance
      (from ``table``, the integrated plug-in is ``sqrt(E[sigma^2])``, matching
      the grid-mixed fit-row variance); any other likelihood's parameters (a
      NegativeBinomial size, a ZeroInflated pi, a Weibull shape) score the rows
      ``update()`` absorbs, with the fitted rows' trials rebound;
    - parametric-MIDAS shapes and ``Copy`` scales, whose designs are functions
      of the value.
    """
    from dataclasses import replace

    from pylgm.likelihoods import CompiledGaussian

    if not estimates:
        return context
    table = table or {}
    likelihood = model.likelihood
    if isinstance(likelihood, Gaussian):
        name = getattr(likelihood.sigma, "name", None)
        if name in table:
            sigma = float(np.sqrt(table[name].mean[0] ** 2 + table[name].variance[0]))
            context = replace(context, likelihood=CompiledGaussian(sigma))
        elif name in estimates:
            context = replace(context, likelihood=CompiledGaussian(estimates[name]))
    else:
        materialized = likelihood.materialize(dict(estimates))
        trials = getattr(context.likelihood, "trials", None)
        if trials is not None:
            materialized = materialized.for_observations({"trials": trials})
        context = replace(context, likelihood=materialized)

    def resolve(spec, value):
        return float(estimates[spec]) if isinstance(spec, str) else value

    entries = []
    for entry in context.entries:
        kind, payload = entry
        if kind not in ("midas_parametric", "copied"):
            entries.append(entry)
            continue
        if kind == "midas_parametric":
            name, columns, kernel, theta_spec = payload
            payload = (name, columns, kernel, tuple(resolve(t, t) for t in theta_spec))
        elif kind == "copied":
            base_entry, copies = payload
            payload = (base_entry, tuple(
                (index, labels, spec, resolve(spec, value))
                for index, labels, spec, value in copies
            ))
        entries.append((kind, payload))
    if all(new is old for new, old in zip(entries, context.entries, strict=True)):
        return context      # nothing theta-dependent: keep the entries object shared
    return replace(context, entries=tuple(entries))


def _rebuild_result(
    result: GaussianResult | LaplaceResult | INLAResult,
    *,
    caller_order: np.ndarray | None = None,
    prediction_keys: pd.DataFrame | None = None,
    hyperparameters: Mapping[str, float] | None = None,
    diagnostics: Mapping[str, object] | None = None,
    prediction_context: object | None = None,
    reorder_criteria: bool = True,
    grid: object | None = None,
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
        sampler = result._sampler
        if sampler is not None and caller_order is not None:
            sampler = sampler.reordered(caller_order)
        return LaplaceResult(
            fitted_mean=fitted_mean, link_name=result.link_name, sampler=sampler,
            sparse_posterior=getattr(result, "_sparse_posterior", None), **common
        )
    if isinstance(result, INLAResult):
        fitted_mean = (
            result.fitted_mean[caller_order]
            if caller_order is not None and result.fitted_mean is not None
            else result.fitted_mean
        )
        criteria = result._criteria
        if criteria is not None and caller_order is not None and reorder_criteria:
            criteria = criteria.reordered(caller_order)
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
            grid=grid if grid is not None else result._grid,
            **common,
        )
    # ponytail: caller_order only permutes prediction rows (predictive_mean/
    # predictive_variance above), never the latent index space, so the sparse
    # posterior needs no reindexing -- straight pass-through is correct.
    return GaussianResult(
        observation_variance=result.observation_variance,
        sparse_posterior=getattr(result, "_sparse_posterior", None),
        sampler=(
            result._sampler.reordered(caller_order)
            if result._sampler is not None and caller_order is not None else result._sampler
        ),
        **common,
    )


def _finished(result, context_at, *, caller_order=None, prediction_keys=None,
              reorder_criteria=True):
    """Attach the fitted prediction context and put rows in caller order.

    ``context_at(estimates, table)`` builds the prediction context at the given
    hyperparameter point estimates (``table``: an integrated fit's marginals).
    An INLA grid is bound too: each point's conditional gets the context of its
    own theta, so ``update`` builds the new rows' design and likelihood there,
    and the integrated context is rebuilt from the updated marginals.
    """
    from dataclasses import replace

    grid = getattr(result, "_grid", None)
    if grid is not None:
        def conditional_at(conditional, theta):
            return _rebuild_result(
                conditional(), caller_order=caller_order, hyperparameters=theta,
                prediction_context=context_at(theta, {}),
            )

        grid = replace(
            grid,
            conditionals=tuple(
                partial(conditional_at, conditional, theta)
                for conditional, theta in zip(grid.conditionals, grid.thetas, strict=True)
            ),
            context=lambda marginals: context_at(
                {name: float(entry.mean[0]) for name, entry in marginals.items()}, marginals
            ),
        )
    return _rebuild_result(
        result, caller_order=caller_order, prediction_keys=prediction_keys,
        prediction_context=context_at(_point_estimates(result), _hyperparameter_table(result)),
        reorder_criteria=reorder_criteria, grid=grid,
    )


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


def _warm_fit(fit, previous):
    """``fit``, starting a Laplace mode at ``previous``'s latent mean, matched by label.

    Only the dict is kept, never ``previous``: holding a result would chain
    every earlier window's result into the next one's memory.
    """
    from functools import wraps
    import inspect

    if previous is None or "initial_mode" not in inspect.signature(fit).parameters:
        return fit
    by_label = dict(zip(previous.labels, np.asarray(previous.mean, dtype=float).tolist()))

    @wraps(fit)
    def warm(model, *args, initial_mode=None, **kwargs):
        if initial_mode is None:   # a search's own warm start takes precedence
            initial_mode = np.array([by_label.get(label, 0.0) for label in model.labels])
        return fit(model, *args, initial_mode=initial_mode, **kwargs)

    return warm


def _warm_estimates(previous) -> dict[str, float]:
    """Hyperparameter starting point from ``previous``: its optimum, or its INLA mode.

    An estimate pinned at a bound is left out: the objective is flat out there
    (a precision running to infinity), so a search started on the bound stalls
    on the plateau instead of coming back to an interior optimum.
    """
    if previous is None:
        return {}
    diagnostics = previous.diagnostics
    estimates = {name: float(value) for name, value in (previous.hyperparameters or {}).items()}
    prefix = "inla_mode_"
    estimates.update({
        key[len(prefix):]: float(value)
        for key, value in diagnostics.items() if key.startswith(prefix)
    })
    pinned = ",".join(str(diagnostics.get(key, "")) for key in
                      ("hyperparameters_at_bound", "inla_active_bounds"))
    pinned = {name.strip() for name in pinned.split(",")}
    return {name: value for name, value in estimates.items() if name not in pinned}


def _optimization_inputs(family, declared, warm_start=None):
    """``(bounds, initial, penalty)`` for a family's hyperparameter search.

    A family-bound prior (e.g. a graph-bound PC prior) wins over the one the
    Hyperparameter declares; ``warm_start`` moves the initial values.
    """
    declared = list(declared) + list(getattr(family, "hyperparameters", ()))
    bounds = (
        dict(family.parameter_bounds) if family.parameter_bounds
        else {hp.name: OptimizationBounds(hp.initial, hp.lower, hp.upper) for hp in declared}
    )
    initial = {hp.name: hp.initial for hp in declared if hp.name in bounds}
    for name, value in _warm_estimates(warm_start).items():
        if name in bounds:
            initial[name] = min(max(value, bounds[name].lower), bounds[name].upper)
    family_priors = dict(getattr(family, "parameter_priors", {}) or {})
    priored = [hp for hp in declared if hp.prior is not None and hp.name not in family_priors]
    penalty = None
    if family_priors or priored:
        def penalty(values):
            total = 0.0
            for name, prior in family_priors.items():
                total += float(prior.logpdf(values[name]))
            for hp in priored:
                total += float(hp.prior.logpdf(values[hp.name]))
            return total

    return bounds, initial, penalty


def _fit_family(family, direct, fit, declared, *, hyperparameters, latent_strategy,
                warm_start, num_workers, blas_threads, stacklevel=5):
    """Fit a compiled family: directly, by empirical Bayes, or by INLA integration.

    ``direct()`` is the model fitted when nothing is estimated; ``declared`` are
    the model's Hyperparameter declarations (bounds, initial values, priors).
    ``stacklevel`` points a bound warning at the caller's ``fit`` line.
    """
    if warm_start is not None and not isinstance(
        warm_start, (GaussianResult, LaplaceResult, INLAResult)
    ):
        raise TypeError("warm_start must be a result returned by fit()")
    fit = _warm_fit(fit, warm_start)
    if family is None or not family.parameter_names:
        if hyperparameters == "integrate":
            raise ValueError("hyperparameters='integrate' requires a declared Hyperparameter")
        with blas_limit(1, blas_threads):
            return fit(direct())

    bounds, initial, penalty = _optimization_inputs(family, declared, warm_start)
    if hyperparameters == "integrate":
        return integrate_inla(
            family, bounds, initial=initial, fit=fit, penalty=penalty,
            latent_strategy=latent_strategy, num_workers=num_workers, blas_threads=blas_threads,
        )
    eb = optimize_empirical_bayes(
        family, bounds, initial=initial, fit=fit, penalty=penalty,
        num_workers=num_workers, blas_threads=blas_threads,
    )
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
            stacklevel=stacklevel,
        )
    return _rebuild_result(eb.fit, hyperparameters=dict(eb.parameters), diagnostics=diagnostics)


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
        mean_correction: bool = False,
        observations: object = (),
        constraints: object = (),
        num_workers: int = 1,
        blas_threads: int | None = None,
        warm_start: object = None,
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

        ``observations`` may contain ``LinearObservation`` blocks whose operators
        aggregate the predictor-grid rows, and ``constraints`` may contain exact
        ``LinearConstraint`` equalities on that same grid, for any likelihood: a
        non-Gaussian model keeps its own row likelihood and takes the observations
        as Gaussian pseudo-rows on the Laplace engine. Operator columns follow the
        caller's frame order. When either is supplied, the response column may be
        absent entirely.

        ``num_workers`` runs a hyperparameter search's independent conditional
        fits (a finite-difference gradient's points, an INLA grid) concurrently
        on a thread pool, with BLAS limited to ``blas_threads`` (default 1) per
        worker; results are bit-identical to ``num_workers=1, blas_threads=1``.
        ``blas_threads`` alone caps BLAS threads for the whole fit. See
        docs/empirical-bayes.md.

        ``warm_start`` takes an earlier result of this model -- the previous
        window of a rolling or expanding experiment -- and starts the
        hyperparameter search at its estimates (the INLA mode for an integrated
        result) and a Laplace mode at its latent mean, matched by label, with
        new levels at zero. The rows may differ freely. It changes where the
        search starts, not what it converges to: results match a cold fit to
        the optimizer's tolerance.
        """
        try:
            observations = tuple(observations)
            constraints = tuple(constraints)
        except TypeError as error:
            raise TypeError("observations and constraints must be iterable") from error
        num_workers = validate_workers(num_workers, "num_workers")
        blas_threads = validate_blas_threads(blas_threads)
        if any(not isinstance(item, LinearObservation) for item in observations):
            raise TypeError("observations must contain only LinearObservation instances")
        if any(not isinstance(item, LinearConstraint) for item in constraints):
            raise TypeError("constraints must contain only LinearConstraint instances")
        if any(item.scale == "below_threshold" for item in (*observations, *constraints)):
            raise ModelValidationError(
                "scale='below_threshold' needs a Joint with a CensoredHurdle"
            )
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
        require_single_mean_shift(latent_strategy, mean_correction)
        options = dict(
            hyperparameters=hyperparameters, latent_strategy=latent_strategy,
            mean_correction=mean_correction, warm_start=warm_start,
            num_workers=num_workers, blas_threads=blas_threads,
        )
        if isinstance(frame, pd.DataFrame):
            return self._fit_pandas(
                frame, engine, observations=observations, constraints=constraints, **options,
            )

        from pylgm.data.spark import is_spark_dataframe

        if is_spark_dataframe(frame):
            if observations or constraints:
                raise UnsupportedEngineError(
                    "linear observations and predictor constraints currently require pandas input"
                )
            return self._fit_spark(frame, engine, max_driver_rows=max_driver_rows, **options)
        raise DataContractError("frame must be a Pandas DataFrame")

    def _engine(self, engine: str, mean_correction: bool = False):
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
        if mean_correction and engine == "laplace":
            return partial(fit, mean_correction=True)
        # An exact Gaussian posterior has no mode/mean gap to correct: its third
        # derivative is identically zero, so the shift would be too. Ignoring the
        # flag there is the honest reading, not a silent omission.
        return fit

    def _declared_hyperparameters(self) -> list:
        from pylgm.compiler import _model_hyperparameters

        return [hp for _, hp in _model_hyperparameters(self)]

    def _fit_family(self, family, direct, engine, *, mean_correction, **options):
        return _fit_family(
            family, direct, self._engine(engine, mean_correction),
            self._declared_hyperparameters(), **options,
        )

    def _fit_pandas(
        self, frame: pd.DataFrame, engine: str, *,
        observations: tuple[LinearObservation, ...] = (),
        constraints: tuple[LinearConstraint, ...] = (),
        **options,
    ) -> GaussianResult | LaplaceResult | INLAResult:
        prepared = frame.copy(deep=True)
        if (observations or constraints) and self.response not in prepared.columns:
            prepared[self.response] = np.nan
        time = self.time
        if time is None:
            if _ROW_KEY in prepared.columns:
                raise DataContractError(f"reserved row key column already exists: {_ROW_KEY!r}")
            prepared[_ROW_KEY] = np.arange(len(prepared), dtype=np.int64)
            time = _ROW_KEY

        data = DataConfig(time=time, response=self.response, panel=self.panel)
        panel = CanonicalPanel.from_frame(
            prepared, data, require_observed=not (observations or constraints)
        )

        from pylgm.compiler import build_prediction_context, compile_family, compile_lgm
        from pylgm.observations import (
            _ProjectedGaussianFamily,
            _ProjectedMixtureFamily,
            _RelinearizedFamily,
            observation_hyperparameters,
            project_gaussian_family,
            project_gaussian_model,
            project_mixture_model,
            reorder_linear_inputs,
        )

        # A Gaussian model folds the observations into one standardized Gaussian
        # likelihood; any other keeps its rows and gains Gaussian pseudo-rows.
        if isinstance(self.likelihood, Gaussian):
            project, projected_family = project_gaussian_model, _ProjectedGaussianFamily
            inner_fit = partial(fit_gaussian, predictive_variances=False)
        else:
            project, projected_family = project_mixture_model, _ProjectedMixtureFamily
            inner_fit = partial(fit_laplace, predictive_variances=False)

        observations, constraints = reorder_linear_inputs(
            observations, constraints, panel.source_positions
        )
        if (
            (observations or constraints)
            and not panel.observed.any()
            and isinstance(getattr(self.likelihood, "sigma", None), Hyperparameter)
        ):
            # Without row responses the row likelihood is a placeholder: its sigma
            # moves neither the predictions nor the LML, so it would be a flat
            # direction for the optimizer rather than an estimate.
            raise ModelValidationError(
                f"the Gaussian sigma {self.likelihood.sigma.name!r} cannot be estimated: "
                "the model has no row responses, only linear observations or constraints; "
                "give it a fixed value, or estimate a LinearObservation sigma instead"
            )

        compiled = compile_lgm(self, panel)
        if any(item.scale != "identity" for item in (*observations, *constraints)):
            family = project_gaussian_family(
                compile_family(self, panel), observations, constraints,
                base_model=compiled,
                family_type=partial(
                    _RelinearizedFamily, project=project, inner_fit=inner_fit,
                    curvature=project is project_mixture_model,
                ),
            )

            def direct():
                return family.materialize({})
        else:
            family = compile_family(self, panel)
            if observation_hyperparameters(observations):
                family = project_gaussian_family(
                    family, observations, constraints,
                    base_model=compiled if family is None else None,
                    family_type=projected_family,
                )
            elif family is not None and (observations or constraints):
                family = project_gaussian_family(
                    family, observations, constraints, family_type=projected_family,
                )

            def direct():
                if observations or constraints:
                    return project(compiled, observations, constraints)
                return compiled
        result = self._fit_family(family, direct, engine, **options)
        return _finished(
            result,
            partial(_fitted_context, build_prediction_context(self, panel, compiled, result), self),
            caller_order=np.argsort(panel.source_positions),
            reorder_criteria=not (observations or constraints),
        )

    def _fit_spark(
        self, frame: object, engine: str, *, max_driver_rows: int | None, **options,
    ) -> GaussianResult | LaplaceResult | INLAResult:
        from pylgm.data.spark import canonicalize_spark_frame

        canonical = canonicalize_spark_frame(frame, self, max_driver_rows=max_driver_rows)

        from pylgm.compiler import build_prediction_context, compile_family, compile_lgm

        compiled = compile_lgm(self, canonical.panel)
        result = self._fit_family(
            compile_family(self, canonical.panel), lambda: compiled, engine, **options
        )
        return _finished(
            result,
            partial(
                _fitted_context, build_prediction_context(self, canonical.panel, compiled, result),
                self,
            ),
            prediction_keys=canonical.prediction_keys,
        )


__all__ = ["LGM"]
