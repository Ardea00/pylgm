# Joint models cross-checked against MCMC

Validation of pyLGM's joint (multi-likelihood) machinery against NUTS ground truth.

## Why MCMC and not R-INLA

R-INLA was the obvious reference, but agreeing with it would only show that pyLGM
reproduces **another Laplace approximation of the same family** — it could not
detect an error both implementations share. NUTS is asymptotically exact, so it
tests the approximation itself. This is how Rue, Martino & Chopin (2009, §5)
validate INLA. It also removes the R toolchain from the loop entirely.

## The model

Knorr-Held & Best shared-component structure over 40 districts:

    log mu_oral_i   = alpha_1 + delta   * u_i
    log mu_larynx_i = alpha_2 + delta^-1 * u_i
    u_i     ~ N(0, sigma_u^2),  sigma_u = 0.7
    alpha_k ~ N(0, 1/1e-6)                     <- pyLGM's Fixed default prior precision

`delta = 1.6` is **fixed in both fits**, deliberately. pyLGM estimates it by
empirical Bayes — a point maximising the marginal likelihood, not a posterior —
so leaving it free would have the two sides answering different questions. What
this cross-check validates is the latent posterior; `delta` recovery is covered
separately by the convergence study noted below.

## What the comparison found

pyLGM's `gaussian` latent strategy reports the **joint posterior mode**. MCMC
reports **marginal posterior means**. In 42 dimensions these genuinely differ,
even though the marginals are near-symmetric (the `alpha_oral` marginal has skew
−0.087): the mode is not a typical point in high dimensions.

So the tests do *not* assert that pyLGM's means equal the MCMC means — that
would assert something false. They assert what is true:

| Assertion | Evidence |
|---|---|
| pyLGM's latent mean is the exact posterior mode | matches an independent scipy L-BFGS-B optimisation of the exact log-posterior to <1e-5; log-posterior there is **higher** than at the MCMC mean (−53.5129 vs −54.5715) |
| Posterior SDs agree with NUTS | ratio pyLGM/NUTS in [0.949, 1.018], mean 0.982 — the curvature is right |
| Joint models add no error of their own | a plain single-response `LGM` shows the same gap: max abs z 0.813 (control) vs 1.058 (joint), mean 0.034 vs 0.083 |

That last row is the control that matters. The mode-vs-mean gap is a property of
the Gaussian latent strategy, present in pyLGM's pre-existing single-response
path, not something joint models introduce. Correcting it is what the
`simplified_laplace` and `laplace` latent strategies are for.

## Running

The tests need no MCMC — they read `nuts_reference.json`:

    PYTHONPATH=src python -m pytest tests/integration/test_mcmc_crosscheck.py -q

To regenerate the reference you need PyMC, in a **separate** environment (it
pins PyTensor, which constrains numpy/scipy and can move numpy under pyLGM):

    python -m venv /tmp/venv-mcmc
    /tmp/venv-mcmc/bin/pip install pymc
    /tmp/venv-mcmc/bin/python examples/joint_mcmc_crosscheck/generate_reference.py

Reference quality: 4 chains x 8000 draws, min bulk ESS 23,765, max R-hat 1.00062,
0 divergences.

## Not covered here

- **`delta` estimation.** Fixed in both fits, as explained above. Recovery is
  checked separately: mean estimate 1.471 / 1.538 / 1.517 against a true 1.6 at
  40 / 150 / 600 districts, with SD falling 0.299 -> 0.106 -> 0.101 — consistent,
  no systematic bias.
- **`hyperparameters="integrate"`.** The INLA grid path on a `Joint` has no test
  coverage yet, so the `simplified_laplace` and `laplace` latent strategies are
  not exercised on joint models.
- **Real data.** This is simulated. A published-data benchmark (Knorr-Held &
  Best used oral cavity and oesophageal cancer over 544 German districts) remains
  future work; the oesophageal counts are not publicly available.
