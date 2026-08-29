# pyLGM Survival Likelihoods Design

**Status:** Approved 2026-08-29

## Purpose

This slice adds the first **survival (time-to-event)** likelihoods to pyLGM:
`WeibullSurv` and `ExponentialSurv`, fit on the existing Laplace engine. Unlike
the GLM families, these are not plain response-scale likelihoods: each row
carries an event/right-censoring indicator and, optionally, a left-truncation
entry time, alongside the follow-up time. That data contract is the substance of
the slice.

The target audience is econometric **duration analysis** — unemployment spells,
firm survival, time-to-default — where right-censoring is ubiquitous and
left-truncation (delayed entry / stock sampling) is a first-class concern.

It reuses the existing Laplace inference engine, `CompiledLGM` IR, canonical
panel, effect builders, constraint handling, hyperparameter optimisation
(EB / MAP-II / INLA), and the `phi`-style hyperparameter machinery. It adds no
new inference engine and no new latent structure.

## Scope

### Included

- `WeibullSurv` likelihood — proportional-hazards parameterisation, shape `α`
  fixed or estimable.
- `ExponentialSurv` likelihood — the `α ≡ 1` special case, no shape
  hyperparameter.
- **Right-censoring** via a per-row `event` indicator column (1 = event,
  0 = right-censored).
- **Left-truncation** (delayed entry) via an optional per-row `entry` time
  column; absent → entry at 0.
- Generalisation of the compiled-likelihood auxiliary-column hook
  (`for_observations`) from a single vector to a named mapping, so a likelihood
  may bind more than one per-row data vector. `Binomial` migrates onto it
  unchanged in behaviour.
- Shape-`α` estimation through the existing empirical-Bayes / MAP-II / INLA
  marginal, identical in wiring to `NegativeBinomial.phi`.
- `response_mean` (E[T]), `cdf` (for PIT/CPO/DIC/WAIC criteria),
  `response_prediction`, and `validate_response` for the new families.
- The YAML frontend: `weibullsurv` / `exponentialsurv` config types.
- A runnable econometric duration example with right-censoring and a frailty
  `IID` effect.

### Excluded (explicit later slices; design left open)

- **Interval-censoring / grouped-duration data.** It requires a two-time-bound
  response contract, and discrete-time duration is already expressible today via
  the existing `Bernoulli` likelihood (logit or, with a future cloglog link,
  complementary-log-log) on person-period ("episode-split") data. Deferred.
- **Time-varying covariates.** These are handled by episode-splitting at the
  data-preparation layer; once rows are split, the row-wise likelihood here
  already applies. A documentation note, not code.
- **Competing risks, cure/mixture models, frailty as an explicit likelihood
  term.** Unobserved heterogeneity is already available structurally: an `IID`
  effect over the individual/group index is a shared frailty; `RW1`/`RW2`/CAR
  give structured frailty. No new likelihood code is needed for it.
- Any new inference engine, latent effect, or link object beyond what the
  families need internally.

The design must not preclude the excluded items. See **Extensibility**.

## Background: proportional-hazards parameterisation

Following R-INLA's `weibullsurv` (variant 0), the linear predictor `η` enters as
the **log relative risk**:

- hazard `h(t) = α · t^(α−1) · exp(η)`
- cumulative hazard `H(t) = t^α · exp(η)`
- survival `S(t) = exp(−H(t)) = exp(−t^α · exp(η))`
- density `f(t) = h(t) · S(t)`

with Weibull shape `α > 0`. The exponential model is exactly `α ≡ 1`:
`h(t) = exp(η)`, `H(t) = t · exp(η)`.

`η` carries the usual offset and all latent effects; a coefficient `β_k` has the
proportional-hazards reading `exp(β_k)` = hazard ratio.

## The unified censored + truncated log-likelihood

For row `i` with follow-up time `tᵢ > 0`, event indicator `δᵢ ∈ {0, 1}`, and
left-truncation entry time `vᵢ ∈ [0, tᵢ)` (with `vᵢ = 0` when no truncation
column is supplied):

```
log Lᵢ = δᵢ · ( log α + (α − 1)·log tᵢ + ηᵢ )  −  (tᵢ^α − vᵢ^α) · exp(ηᵢ)
```

Derivation: the delayed-entry likelihood conditions the risk set on survival to
`vᵢ`, giving `δᵢ·log h(tᵢ) + log S(tᵢ) − log S(vᵢ)`. Substituting the forms above
and collecting the `η`-free constant `cᵢ = log α + (α−1)·log tᵢ` yields the line
above. An event (`δ=1`) contributes the log-hazard; the `−ΔH` term is the
row's exposure over its at-risk window. Right-censoring is `δ=0`
(hazard term drops, exposure remains); no truncation is `vᵢ = 0`.

### η-derivatives (what the Laplace engine consumes)

Let `Aᵢ = (tᵢ^α − vᵢ^α) · exp(ηᵢ)` — the at-risk cumulative hazard,
`Aᵢ > 0` whenever `tᵢ > vᵢ`. Then:

| Quantity | Value | Engine method |
|---|---|---|
| `∂ log Lᵢ / ∂ηᵢ` | `δᵢ − Aᵢ` | `gradient` |
| `−∂² log Lᵢ / ∂ηᵢ²` | `Aᵢ` (> 0, so posterior precision is PD) | `working_weights` |
| `∂³ log Lᵢ / ∂ηᵢ³` | `−Aᵢ` | `third_derivative` |

This is **structurally identical to a Poisson likelihood** with response `δ` and
mean `A` (the classic Poisson–survival equivalence). The existing damped-Newton
Laplace engine therefore fits it unchanged: `working_weights` is strictly
positive, `third_derivative` is smooth, and no new solver behaviour is required.

### Shape-α derivatives (for hyperparameter estimation)

`α` is a hyperparameter, optimised over the Laplace marginal exactly as
`NegativeBinomial.phi` is. The marginal depends on `α` through both `cᵢ`
(the `log α + (α−1)log tᵢ` term) and `Aᵢ` (through `tᵢ^α`, `vᵢ^α`), so the
existing marginal-likelihood optimiser can estimate it. No analytic `α`-gradient
is required by the current optimiser (it works over the marginal); the family
only needs `α` to enter `log_likelihood` / `pointwise_log_density` and the
`Aᵢ`-bearing terms, which it does. `ExponentialSurv` fixes `α = 1` and exposes no
hyperparameter.

## Public Contract

```python
from pylgm import LGM, WeibullSurv, ExponentialSurv, Fixed, IID
from pylgm.parameters import Hyperparameter

# Weibull PH, shape estimated, right-censoring only
model = LGM(
    response="duration",                       # follow-up time t (> 0)
    likelihood=WeibullSurv(
        event="failed",                        # per-row δ column (0/1)
        shape=Hyperparameter("alpha", initial=1.0),   # or a fixed float
    ),
    predictor=Fixed("1 + x") + IID("firm_frailty", index="firm", precision=2.0),
    panel=("firm",),
    time="t",
)
result = model.fit(df, engine="laplace")

# Exponential PH with left-truncation (delayed entry)
model = LGM(
    response="duration",
    likelihood=ExponentialSurv(event="failed", entry="entry_time"),
    predictor=Fixed("1 + x"),
    panel=("firm",),
    time="t",
)
```

### Likelihood spec classes

```python
@dataclass(frozen=True)
class WeibullSurv:
    event: str                                 # required; per-row 0/1 column
    shape: float | Hyperparameter = 1.0        # Weibull α > 0, fixed or estimable
    entry: str | None = None                   # optional left-truncation column

@dataclass(frozen=True)
class ExponentialSurv:
    event: str
    entry: str | None = None
```

Both validate `event` (and `entry`) as non-empty column names at construction and
`shape` as a positive real when fixed, mirroring the existing `Binomial` /
`NegativeBinomial` constructors.

### Compiled classes

`CompiledWeibullSurv(alpha, event, entry)` and `CompiledExponentialSurv` (the
latter delegating to the former with `alpha=1.0` or implemented directly).
Both implement the full compiled-likelihood contract:

- `log_likelihood`, `pointwise_log_density` — the unified line above.
- `gradient`, `working_weights`, `third_derivative` — `δ−A`, `A`, `−A`.
- `response_mean(eta) = exp(−η/α)·Γ(1 + 1/α)` — unconditional E[T].
- `response_prediction(eta_mean, eta_variance)` — point estimate via
  `response_mean(eta_mean)` (η-variance ignored, as for `Bernoulli`/`Beta`).
- `cdf(eta, t) = 1 − exp(−t^α·exp(η))` — marginal event CDF for PIT/CPO.
- `validate_response(y)` — `t` finite and `> 0`.

### Auxiliary-column binding (`for_observations` generalisation)

The compiled-likelihood hook changes from a single trials vector to a named
mapping so a family can bind more than one per-row vector:

```python
class _CompiledLikelihood:
    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "_CompiledLikelihood":
        return self            # default no-op (phi-less GLM families)
```

- `CompiledBinomial.for_observations` reads `aux["trials"]` (behaviour
  unchanged).
- `CompiledWeibullSurv` / `CompiledExponentialSurv` read `aux["event"]` and
  `aux["entry"]` (the latter defaulting to a zero vector when the column is
  absent).

The compiler's `_binomial_trials` generalises to `_likelihood_columns(model,
frame)`, returning the per-family aux mapping (`{"trials": …}` for `Binomial`;
`{"event": …, "entry": …}` for survival; `None` otherwise). The two existing
call sites (`compile_lgm` observed-row validation, and the EB/INLA `factory`
closure) pass this mapping through `for_observations`, sliced to observed rows
where they already slice `trials`.

`validate_response` on the response `t` stays as today. The **auxiliary columns**
(`event`, `entry`) are validated in `_likelihood_columns` at extraction, raising
`DataContractError` / `CompilationError` consistently with `_binomial_trials`:

- `event` ∈ {0, 1}, finite, present in the frame.
- `entry` (if named) finite, `≥ 0`, and `< t` row-wise; present in the frame.

## Architecture & data flow

No new modules. Changes are localised:

- `src/pylgm/likelihoods.py` — `WeibullSurv`, `ExponentialSurv`, and their
  compiled counterparts; the `for_observations(aux)` signature change and the
  `CompiledBinomial` migration.
- `src/pylgm/compiler.py` — `_binomial_trials` → `_likelihood_columns`; update
  the two `for_observations` call sites. No survival-specific compiled-model
  metadata is carried: `event`/`entry` are fit-time-only inputs, and survival's
  `response_prediction` (E[T] from `η`) needs no per-row aux at prediction rows,
  unlike `Binomial`'s `n·p`. The existing `trials` metadata field is left as is.
- `src/pylgm/config/schema.py`, `src/pylgm/config/model.py` — `weibullsurv` /
  `exponentialsurv` declarative types.
- `src/pylgm/__init__.py` — export `WeibullSurv`, `ExponentialSurv`.
- Prediction path (`inference/prediction.py`, `model.py`) — survival is a
  non-Gaussian Laplace family, so it flows through the existing Laplace
  prediction context; the shape `α` fixup mirrors the existing `phi` fixup
  (`_context_with_fitted_estimates`) so out-of-sample prediction uses the fitted
  `α`.

Flow is unchanged from the GLM families: `LGM.fit` → `compile_lgm` /
`compile_gaussian_family` → Laplace EB/INLA optimiser → `LaplaceResult`, with the
new per-row aux vectors bound onto the compiled likelihood alongside the response.

## Error handling

All contract violations raise the existing `DataContractError` (bad response /
event / entry values) or `CompilationError` (missing columns / structural
problems), matching `_binomial_trials` and the GLM `validate_response` methods.
Specific cases:

- response `t` not finite or `≤ 0` → `DataContractError`.
- `event` column missing → `CompilationError`; values outside {0,1} →
  `DataContractError`.
- `entry` column named but missing → `CompilationError`; `entry < 0`, non-finite,
  or `entry ≥ t` for any row → `DataContractError`.
- `shape` fixed but not a positive real → `ValueError` at construction.

## Testing

TDD, one behaviour per test, no new frameworks. Planned coverage:

- **Parameter recovery.** Simulate Weibull PH data (known `β`, `α`); fit and
  recover both within tolerance. Repeat for exponential (`α = 1`).
- **Poisson-equivalence sanity.** On a fixed `η`, assert `gradient`,
  `working_weights`, `third_derivative` equal the Poisson-trick construction
  (`δ − A`, `A`, `−A`).
- **Right-censoring.** A censored row (`δ = 0`) contributes only `−H(t)`; check
  `log_likelihood` against a hand-computed value.
- **Left-truncation.** A delayed-entry row subtracts `H(v)`; check the likelihood
  and that ignoring `entry` (vs `entry = 0`) is identical.
- **Exponential = Weibull(α = 1).** The two compiled likelihoods agree termwise.
- **Contract methods.** `cdf` monotone in `t` and → 1; `response_mean` equals
  `exp(−η/α)·Γ(1+1/α)`; `validate_response` and aux-column validators raise the
  right errors on bad input.
- **Shape estimation.** EB/MAP-II recovers a non-unit `α` from simulated data.
- **YAML round-trip.** `weibullsurv` / `exponentialsurv` configs compile to the
  same model as the Python API, including `event`/`entry`/`shape`.
- **Binomial regression guard.** Existing Binomial tests stay green after the
  `for_observations(aux)` migration.

## Extensibility

- **Interval-censoring** slots in as a variant that binds a second time bound
  (`t_lower`, `t_upper`) through the now-general aux mapping and adds an
  `S(t_lower) − S(t_upper)` term — no further plumbing change.
- **cloglog link + discrete-time duration** reuses the existing `Bernoulli`
  family once a `CLogLogLink` is added; orthogonal to this slice.
- **Additional parametric hazards** (log-logistic, log-normal, Gompertz) follow
  the same compiled-likelihood template; only the `H(t)`/`h(t)` forms differ.
- The `for_observations(aux)` mapping is the general seam for any future
  likelihood needing multiple per-row data vectors.
