# Theory

pyLGM implements the **latent Gaussian model** (LGM) class and fits it with
**Laplace approximations** and INLA-style integration, rather than MCMC. This
page sketches the model class, the inference, and the structured effects, with
pointers into the literature. It is background reading; the [guide](index.md)
pages document the API.

## The latent Gaussian model

An LGM is a three-stage Bayesian hierarchy.

**1. Observations.** Each response `y_i` is conditionally independent given a
linear predictor `η_i`, through a likelihood in the exponential family:

```
y_i | η_i, θ  ~  p(y_i | η_i, θ)
```

pyLGM ships Gaussian, Poisson, and Bernoulli likelihoods (see
[likelihoods](likelihoods.md)). A link connects the mean to `η_i` — identity for
Gaussian, log for Poisson, logit for Bernoulli.

**2. Latent Gaussian field.** The linear predictor is an additive sum of
effects,

```
η_i = β0 + Σ_k f_k(i)
```

where the fixed coefficients `β` and every random/structured effect `f_k`
together form one big latent vector `x`, given a **Gaussian** prior:

```
x | θ  ~  Normal(0, Q(θ)^-1)
```

`Q(θ)` is the *precision* (inverse covariance) matrix. Because most effects
couple only a few neighbouring elements, `Q` is **sparse** — a Gaussian Markov
random field (GMRF). This is the defining structure of the model class and the
reason inference is tractable (Rue & Held, 2005).

**3. Hyperparameters.** A typically low-dimensional `θ` — precisions,
correlations, mixing weights — controls `Q` and the likelihood, with priors of
its own (see [empirical Bayes and priors](empirical-bayes.md)).

The name of the class is exactly this: the latent field `x` is Gaussian; the
data need not be. INLA (Rue, Martino & Chopin, 2009) is the standard inference
method for it, and pyLGM is a pure-Python implementation of the same ideas.

## Inference: Laplace, not Monte Carlo

The joint posterior factorises as `p(x, θ | y) ∝ p(θ) p(x | θ) p(y | x, θ)`.
Two approximations make it deterministic and fast.

**Latent field given hyperparameters.** For fixed `θ`, `p(x | y, θ)` is
approximated by a Gaussian matched at its mode — a **Laplace approximation**
(Tierney & Kadane, 1986). The mode is found by Newton iteration on the sparse
system; the curvature there gives the covariance. For a **Gaussian likelihood
this is exact**, so pyLGM has a dedicated exact-Gaussian engine and a Laplace
engine for the non-Gaussian likelihoods.

**Hyperparameters.** The marginal likelihood `p(θ | y)` is itself
Laplace-approximated (the Gaussian normalising constant evaluated at the mode).
pyLGM offers two ways to use it:

- **Empirical Bayes** — optimise `p(θ | y)` (type-II maximum likelihood) or its
  penalised version with priors (MAP-II), then condition on the point estimate.
- **INLA integration** — place a grid over `θ`, weight each node by
  `p(θ | y)`, and mix the per-node latent marginals, propagating hyperparameter
  uncertainty (Rue, Martino & Chopin, 2009; Martins et al., 2013). See
  [INLA integration](inla.md).

**Latent marginals.** Given `θ`, the per-element marginals `p(x_j | y, θ)` come
at three fidelities: the plain **Gaussian** from the mode, the **simplified
Laplace** (a skewness correction), and the **full Laplace**. These are the
INLA accuracy levels.

## Gaussian Markov random fields

A GMRF is a multivariate Gaussian whose precision matrix `Q` is sparse: `Q_jk =
0` exactly when `x_j` and `x_k` are conditionally independent given the rest.
Sparsity is what makes the Newton solves and log-determinants cheap. Every
effect below is, in the end, a rule for filling in a sparse `Q`. The canonical
reference is Rue & Held (2005).

## Structured effects

### Temporal: random walks and AR1

`RW1` and `RW2` are **intrinsic** GMRFs — smoothness priors penalising first or
second differences of a sequence. Being intrinsic, they are rank-deficient
(improper) and carry sum-to-zero constraints. `AR1` is a stationary,
*proper* first-order autoregression. See [effects](effects.md).

### Spatial: the CAR family

Conditional autoregressions place a GMRF on the nodes of a neighbour graph.

**Besag / ICAR.** The intrinsic CAR uses the graph Laplacian,

```
Q = τ · (D − W)
```

with `W` the adjacency matrix and `D` the diagonal of node degrees. Each
connected component contributes one null eigenvalue (the constant vector), so
`Q` is improper — handled with one sum-to-zero constraint per component
(Besag, 1974; Besag, York & Mollié, 1991).

**Scaling matters.** The precision `τ` is only interpretable, and priors only
transfer between graphs, if the effect is **scaled** so that the geometric mean
of its marginal variances equals 1 (Sørbye & Rue, 2014). pyLGM applies this
scaling by default. Computing it correctly requires discarding exactly the one
null eigenvalue per connected component — a step that is subtle enough to have
been a real bug in this library, fixed and regression-tested (see the
[disease-mapping example](examples-disease-mapping.md)).

**Proper CAR.** `ProperCAR` adds a spatial-dependence parameter `ρ` that makes
`Q` full-rank and invertible — a proper prior, no constraint needed. `ρ` can be
fixed or estimated.

**BYM2.** The BYM2 reparameterisation (Riebler, Sørbye, Simpson & Rue, 2016)
combines a *scaled* ICAR component and an IID component under a single
interpretable precision `τ` and a mixing parameter `φ ∈ [0, 1]` — the fraction
of the marginal variance that is spatially structured. Both parameters are
identifiable and prior-friendly, which the original BYM parameterisation was
not. See [spatial effects](spatial-effects.md).

## Linear constraints

Beyond the intrinsic constraints an effect carries, `LGM(constraints=...)`
imposes arbitrary linear constraints `A x = e` on the latent field (R-INLA's
`extraconstr`; see [effects](effects.md#linear-constraints-extraconstr)). pyLGM
enforces them by **null-space reparametrisation**: with `B` an orthonormal basis
of `null(A)` and `x_p` any particular solution of `A x_p = e`, writing
`x = x_p + B z` satisfies `A x = e` for every `z`, and inference proceeds in the
reduced coordinates `z`. For the homogeneous case (`e = 0`) this is exact and
`x_p = 0`.

For a nonzero right-hand side the prior on `z` is not centred: conditioning the
field prior `x ~ N(0, Q⁻¹)` on `A x = e` induces a Gaussian on `z` with mean
`m = −(BᵀQB)⁻¹ Bᵀ Q x_p` — the **conditioning-by-kriging** solution R-INLA uses.
Because the same prior linear term corrects for the choice of `x_p`, the fit is
independent of *which* particular solution is picked, so pyLGM takes the cheap
least-squares one. Contradictory constraints are not rejected; the least-squares
`x_p` gives the closest satisfiable field.

## Priors: penalised complexity

pyLGM's default hyperparameter priors are **PC (penalised-complexity) priors**
(Simpson, Rue, Riebler, Martins & Sørbye, 2017). A PC prior penalises the
*distance* of a component from a simpler base model (e.g. "no effect") at a
constant rate, giving an exponential prior on that distance. For a precision it
becomes a prior on the standard deviation `σ` set by a single interpretable
statement,

```
P(σ > u) = α        # e.g. "P(sd > 1) = 0.01"
```

which is how `PCPrecision(upper_sd=u, alpha=α)` is specified. The BYM2 mixing
parameter has its own PC prior (`PCBYM2Phi`). See
[empirical Bayes and priors](empirical-bayes.md).

## Model assessment

For non-Gaussian fits pyLGM reports the standard INLA criteria: **DIC**
(Spiegelhalter, Best, Carlin & van der Linde, 2002), **WAIC** (Watanabe, 2010),
and leave-one-out **CPO/PIT** (Held, Schrödle & Rue, 2010). See
[INLA integration](inla.md).

## How the theory maps to the API

| Concept | API |
| --- | --- |
| Likelihood / link | [`Gaussian`, `Poisson`, `Bernoulli`](likelihoods.md) |
| Fixed effects, IID random effects | [`Fixed`, `IID`](effects.md) |
| Temporal GMRFs | [`RW1`, `RW2`, `AR1`](effects.md) |
| Spatial CAR GMRFs | [`Besag`, `ProperCAR`, `BYM2`](spatial-effects.md) |
| Linear constraints (`extraconstr`) | [`LGM(constraints=...)`](effects.md#linear-constraints-extraconstr) |
| PC priors on hyperparameters | [`PCPrecision`, `PCBYM2Phi`](empirical-bayes.md) |
| Empirical Bayes / MAP-II | [`hyperparameters="optimize"`](empirical-bayes.md) |
| INLA grid integration, DIC/WAIC/CPO | [`hyperparameters="integrate"`](inla.md) |

## References

1. Besag, J. (1974). Spatial interaction and the statistical analysis of
   lattice systems. *Journal of the Royal Statistical Society B*, 36(2),
   192–236.
2. Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with
   two applications in spatial statistics. *Annals of the Institute of
   Statistical Mathematics*, 43(1), 1–20.
3. Tierney, L., & Kadane, J. B. (1986). Accurate approximations for posterior
   moments and marginal densities. *Journal of the American Statistical
   Association*, 81(393), 82–86.
4. Spiegelhalter, D. J., Best, N. G., Carlin, B. P., & van der Linde, A.
   (2002). Bayesian measures of model complexity and fit. *Journal of the Royal
   Statistical Society B*, 64(4), 583–639.
5. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
   Applications*. Chapman & Hall/CRC.
6. Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for
   latent Gaussian models by using integrated nested Laplace approximations.
   *Journal of the Royal Statistical Society B*, 71(2), 319–392.
7. Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
   widely applicable information criterion in singular learning theory.
   *Journal of Machine Learning Research*, 11, 3571–3594.
8. Held, L., Schrödle, B., & Rue, H. (2010). Posterior and cross-validatory
   predictive checks: a comparison of MCMC and INLA. In *Statistical Modelling
   and Regression Structures*, 91–110. Physica-Verlag.
9. Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between
   Gaussian fields and Gaussian Markov random fields: the stochastic partial
   differential equation approach. *Journal of the Royal Statistical Society B*,
   73(4), 423–498.
10. Martins, T. G., Simpson, D., Lindgren, F., & Rue, H. (2013). Bayesian
    computing with INLA: New features. *Computational Statistics & Data
    Analysis*, 67, 68–83.
11. Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random
    field priors in spatial modelling. *Spatial Statistics*, 8, 39–51.
12. Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
    Bayesian spatial model for disease mapping that accounts for scaling.
    *Statistical Methods in Medical Research*, 25(4), 1145–1165.
13. Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017).
    Penalising model component complexity: a principled, practical approach to
    constructing priors. *Statistical Science*, 32(1), 1–28.
