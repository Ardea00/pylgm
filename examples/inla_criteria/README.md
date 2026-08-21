# Runnable INLA model-assessment example (exact Gaussian)

This directory fits the same twenty-four-region, six-timepoint Gaussian panel
as [`examples/inla`](../inla/README.md) — a genuine per-region offset gives
the region precision hyperparameter an interior posterior — and then reports
the **model-assessment criteria** that come attached to every integrated fit.
Run it from the repository root:

```bash
PYTHONPATH=src python examples/inla_criteria/run.py
```

The model is identical to `examples/inla`; the only difference is what gets
printed:

```python
result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

criteria = result.criteria
criteria.dic                        # deviance information criterion
criteria.waic                       # widely applicable information criterion
criteria.waic_effective_parameters  # WAIC's effective-parameter count
criteria.cpo_failures               # count of unreliable per-obs CPO estimates
criteria.log_cpo_sum                # sum of log CPO, a leave-one-out log score
criteria.pit                        # per-observation PIT values
```

`result.criteria` is always populated for an integrated fit (`hyperparameters
="integrate"`), for both the `exact_gaussian` and `laplace` engines — there is
no separate call needed to compute it.

## What each criterion means

- **DIC** and **WAIC** are model-*comparison* criteria: lower is better when
  comparing two fits of the same data. `waic_effective_parameters` (and its
  DIC counterpart, `dic_effective_parameters`) estimate how many effective
  parameters the fitted model uses, as a complexity penalty.
- **CPO** (conditional predictive ordinate) is the leave-one-out predictive
  density for each observation, computed via the harmonic-mean identity
  rather than by refitting the model once per held-out point. That identity
  can be numerically unreliable for individual observations, so
  `cpo_failures` counts how many of the per-observation CPO values hit the
  reliability guard — a nonzero count means some LOO estimates in
  `result.criteria.cpo` should be treated with caution. `log_cpo_sum` is the
  sum of the per-observation log CPO values, a leave-one-out log predictive
  score.
- **PIT** (probability integral transform) is the predictive CDF evaluated
  at each observed response. If the model is well calibrated, the PIT values
  should look like a sample from Uniform(0, 1) — clustering near 0 or 1
  indicates poorly calibrated predictive intervals. For discrete response
  likelihoods, `pit` is the non-randomized `P(Y <= y)`, which is
  conservative (not exactly uniform even for a correctly specified model);
  the randomized version is not yet implemented.

**Discrete-response caveat:** for Poisson/Bernoulli responses, the
harmonic-mean CPO/PIT estimator integrates a heavy/divergent-tailed `E[1/p]`
against the Gaussian eta-approximation, so both the CPO/PIT values and
`cpo_failures` are sensitive to the Gauss-Hermite quadrature node count;
`cpo_failures` is a lower bound on unreliable observations, not a guarantee
that the rest are trustworthy. Better tail handling (randomized PIT, robust
leave-one-out) is deferred.

See the ["Model-assessment criteria" section](../../README.md#model-assessment-criteria)
of the project README for the full contract.
