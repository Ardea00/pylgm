# pyLGM Penalized MAP-II Design

**Status:** Approved 2026-08-20

## Purpose

Turn the currently declaration-only `Hyperparameter.prior` into a regularizer.
Instead of pure type-II maximum likelihood (empirical Bayes), estimate the
hyperparameters by maximum a posteriori (MAP-II): maximize the marginal
likelihood **plus** the log prior. This is the natural next step after the
declarative empirical-Bayes slice, using the prior metadata that slice
deliberately left unused.

It is a small extension of the existing `optimize_empirical_bayes` optimizer and
the declarative `LGM.fit` empirical-Bayes dispatch. It adds no new inference
engine, no hyperparameter *integration* (that is the later INLA slice), and no
new public API surface beyond behavior on an already-public field.

## Objective and Scale

For hyperparameters `θ` with declared priors `π_i`, the penalized objective is

\[ \mathcal{L}(\theta) \;=\; \log p(y \mid \theta) \;+\; \sum_i \log \pi_i(\theta_i), \]

maximized over `θ` (equivalently, the optimizer minimizes `-\mathcal{L}`).

Each prior's `logpdf` is defined on its **native parameter scale**: `GaussianPrior`
on the value, `PCPrecision` on the precision (the standard PC-precision density
`π(τ) = (θ/2) τ^{-3/2} e^{-θ/√τ}`, `θ = -ln(α)/upper_sd`). The penalty is therefore
evaluated at the natural hyperparameter value, and this is MAP on the native
scale. **No Jacobian term is added**: MAP finds the mode of the fixed function
`\mathcal{L}(θ)`; the optimizer's internal log-space search is only a change of
search coordinate and does not move the argmax, so no change-of-variable
correction applies. (A deliberately different MAP on a transformed scale, which
*would* need a Jacobian, is out of scope.)

Only hyperparameters that declare a `prior` contribute a penalty term; those
without a prior remain pure type-II ML. A model may mix both — the penalty sums
over the prior'd parameters only.

## Trigger

Automatic: a declared `Hyperparameter` with a `prior` is estimated by MAP; one
with no prior stays pure ML. This matches the target-state API (declaring a prior
means it is used) and the empirical-Bayes slice's recorded intent. No new
`LGM.fit` argument is introduced.

## Scope

### Included
- An optional `penalty: Callable[[Mapping[str, float]], float] | None = None` on
  `optimize_empirical_bayes`; the objective subtracts `penalty(θ)` from
  `-log_marginal_likelihood`. Default `None` leaves behavior identical to today.
- `LGM.fit` (both engines) builds the penalty from the declared hyperparameters'
  priors and passes it, so declaring a prior yields a MAP estimate on
  `result.hyperparameters`.
- A diagnostics flag recording that estimation was penalized (MAP vs ML).
- Docs, a runnable example, and the roadmap update marking MAP-II shipped.

### Excluded (later slices)
- MAP on a log/other scale with explicit Jacobians.
- Hyperparameter posterior *marginals* / integration (INLA).
- New priors beyond the existing `GaussianPrior`/`PCPrecision`.
- YAML expression of priors (YAML stays numeric plug-in).
- Any change to the legacy schema-v2 `Experiment` path.

## Public Contract

```python
from pylgm import LGM, Gaussian, Fixed, IID, Hyperparameter, PCPrecision

model = LGM(
    response="y",
    likelihood=Gaussian(1.0),
    predictor=Fixed("1") + IID(
        "region", index="region",
        precision=Hyperparameter("region_prec", initial=1.0,
                                 prior=PCPrecision(upper_sd=1.0, alpha=0.01)),
    ),
    panel=("region",), time="t",
)
result = model.fit(frame)                     # region_prec estimated by MAP-II
result.hyperparameters["region_prec"]          # MAP estimate
result.diagnostics["hyperparameter_penalized"] # True (a prior was applied)
```

- A `Hyperparameter` with no `prior` behaves exactly as in the empirical-Bayes
  slice (pure type-II ML); `result` for a model with no `Hyperparameter` at all
  is unchanged with `hyperparameters is None`.
- Works identically for `engine="exact_gaussian"` and `engine="laplace"`, and
  preserves the Pandas caller-order and Spark canonical-key contracts.

## Architecture

### Optimizer (`optimization/empirical_bayes.py`)

Add `penalty: Callable[[Mapping[str, float]], float] | None = None`. In the
objective, after computing `raw_objective = -float(result.log_marginal_likelihood)`,
when `penalty is not None` subtract the penalty evaluated at the natural
parameter mapping: `raw_objective -= float(penalty(parameters))`. The existing
`if not np.isfinite(raw_objective): raise NumericalError` guard already handles a
`-inf` log prior (out-of-support value) by marking that evaluation invalid so the
optimizer routes away from it. The `OptimizationDiagnostics.objective` then
reports the penalized objective at the optimum; no other optimizer change.

### Declarative dispatch (`model.py`)

`_run_empirical_bayes` builds the penalty from the model's declared
hyperparameters that carry a prior:

```python
priored = [(hp) for _, hp in _model_hyperparameters(self) if hp.prior is not None]
penalty = None
if priored:
    def penalty(values):
        return sum(hp.prior.logpdf(values[hp.name]) for hp in priored)
```

Pass `penalty=penalty` to `optimize_empirical_bayes`. Record
`diagnostics["hyperparameter_penalized"] = penalty is not None` alongside the
existing `empirical_bayes_converged`/`empirical_bayes_evaluations` scalars. The
estimate attachment, caller-order realignment, and Spark key handling are
unchanged from the empirical-Bayes slice.

## Errors

- A hyperparameter value outside a prior's support (e.g. a non-positive value
  into `PCPrecision`) yields `-inf` log prior → `+inf` objective → the existing
  `NumericalError`/failure-routing path; it never returns a bad fit as success.
  In practice `Hyperparameter` bounds are strictly positive, so `PCPrecision`
  always sees a valid precision.
- All existing empirical-Bayes and engine/likelihood errors are unchanged.

## Testing and Validation

1. **Optimizer penalty seam.** `optimize_empirical_bayes` with a `penalty`
   callable produces a different, correctly-shifted optimum than without it on a
   small fixture; `penalty=None` reproduces the pure-ML result exactly.
2. **Penalty direction.** With a `PCPrecision` prior (which shrinks the effect
   standard deviation toward zero, i.e. pulls the precision estimate upward),
   the MAP precision estimate differs from the ML estimate in the prior-favored
   direction on a fixture where they diverge; a hand-checked penalized objective
   value matches.
3. **Automatic trigger, both engines.** A Gaussian (`exact_gaussian`) and a
   Poisson (`laplace`) model with a prior'd `Hyperparameter` return a converged
   result whose `hyperparameters` holds the MAP estimate and whose
   `diagnostics["hyperparameter_penalized"]` is `True`.
4. **ML invariance.** A prior-free `Hyperparameter` model produces the exact
   empirical-Bayes (pure ML) result, with `hyperparameter_penalized` `False`; a
   no-`Hyperparameter` model is unchanged with `hyperparameters is None`.
5. **Priors.** `GaussianPrior`/`PCPrecision` `logpdf` values used by the penalty
   are correct at hand-checked points (extend existing prior tests if needed).

## Acceptance Criteria

1. Declaring a `Hyperparameter` with a `prior` yields a MAP-II estimate on
   `result.hyperparameters` for both engines, flagged in diagnostics.
2. `penalty=None` and prior-free models reproduce the empirical-Bayes (pure ML)
   behavior exactly; no-`Hyperparameter` models are unchanged.
3. The penalty is the native-scale log prior with no Jacobian; out-of-support
   values are routed away, never returned as a successful fit.
4. No new runtime dependencies; full suite green; Pandas/Spark contracts and the
   legacy path preserved.
