# Theory

pyLGM implements the **latent Gaussian model** (LGM) class and fits it with
**Laplace approximations** and INLA-style integration, rather than MCMC. This
page sketches the model class, the inference, and the structured effects, with
pointers into the literature. It is background reading; the [guide](index.md)
pages document the API, and [how pyLGM compares](comparison.md) places the
method against regression, gradient boosting, and MCMC.

## The latent Gaussian model

An LGM is a three-stage Bayesian hierarchy.

**1. Observations.** Each response \(y_i\) is conditionally independent given a
linear predictor \(\eta_i\), through a likelihood in the exponential family:

\[
y_i \mid \eta_i, \theta \;\sim\; p(y_i \mid \eta_i, \theta)
\]

pyLGM ships Gaussian, Poisson, Bernoulli, Binomial, negative-binomial, Gamma
and Beta likelihoods, plus Weibull and exponential **survival** likelihoods with
censoring and truncation (see [likelihoods](likelihoods.md)). A link connects
the mean to \(\eta_i\) — identity for Gaussian, log for Poisson, logit for
Bernoulli and Beta.

**2. Latent Gaussian field.** The linear predictor is an additive sum of
effects,

\[
\eta_i \;=\; \beta_0 \;+\; \sum_k f_k(i)
\]

where the fixed coefficients \(\beta\) and every random/structured effect
\(f_k\) together form one latent vector \(x\), given a **Gaussian** prior:

\[
x \mid \theta \;\sim\; \mathcal{N}\!\left(0,\; Q(\theta)^{-1}\right)
\]

\(Q(\theta)\) is the *precision* (inverse covariance) matrix. Because most
effects couple only a few neighbouring elements, \(Q\) is **sparse** — a
Gaussian Markov random field (GMRF). This is the defining structure of the model
class and the reason inference is tractable (Rue & Held, 2005).

**3. Hyperparameters.** A typically low-dimensional \(\theta\) — precisions,
correlations, mixing weights — controls \(Q\) and the likelihood, with priors of
its own (see [empirical Bayes and priors](empirical-bayes.md)).

The name of the class is exactly this: the latent field \(x\) is Gaussian; the
data need not be. INLA (Rue, Martino & Chopin, 2009) is the standard inference
method for it, and pyLGM is a pure-Python implementation of the same ideas.

## Inference: Laplace, not Monte Carlo

The joint posterior factorises as

\[
p(x, \theta \mid y) \;\propto\; p(\theta)\, p(x \mid \theta)\, p(y \mid x, \theta).
\]

Two approximations make it deterministic and fast.

**Latent field given hyperparameters.** For fixed \(\theta\),
\(p(x \mid y, \theta)\) is approximated by a Gaussian matched at its mode — a
**Laplace approximation** (Tierney & Kadane, 1986). Writing the log-posterior as

\[
\log p(x \mid y,\theta) \;=\; \text{const} \;-\; \tfrac{1}{2}x^\top Q(\theta) x
\;+\; \sum_i \log p(y_i \mid \eta_i, \theta),
\]

the mode \(x^\star\) solves \(\nabla = 0\) by Newton iteration, and the
curvature there gives the approximating covariance:

\[
x \mid y, \theta \;\overset{\text{approx}}{\sim}\;
\mathcal{N}\!\left(x^\star,\; \left[Q(\theta) + \operatorname{diag}(c)\right]^{-1}\right),
\qquad
c_i = -\left.\frac{\partial^2 \log p(y_i \mid \eta_i)}{\partial \eta_i^2}\right|_{\eta^\star}.
\]

For a **Gaussian likelihood** \(c\) is constant and the approximation is
**exact**, so pyLGM has a dedicated exact-Gaussian engine and a Laplace engine
for the non-Gaussian likelihoods.

**Hyperparameters.** The marginal likelihood \(p(\theta \mid y)\) is itself
Laplace-approximated,

\[
\log p(\theta \mid y) \;\approx\; \log p(\theta) + \log p(y \mid x^\star, \theta)
+ \tfrac{1}{2}\log|Q(\theta)| - \tfrac{1}{2}\log|Q(\theta) + \operatorname{diag}(c)|
- \tfrac{1}{2}x^{\star\top} Q(\theta)\, x^\star ,
\]

and pyLGM offers two ways to use it:

- **Empirical Bayes** — optimise \(p(\theta \mid y)\) (type-II maximum
  likelihood) or its penalised version with priors (MAP-II), then condition on
  the point estimate \(\hat\theta\).
- **INLA integration** — place a grid \(\{\theta_m\}\) over \(\theta\), weight
  each node by \(p(\theta_m \mid y)\), and mix the per-node latent marginals,

  \[
  p(x_j \mid y) \;\approx\; \sum_m p(x_j \mid y, \theta_m)\, w_m ,
  \]

  propagating hyperparameter uncertainty (Rue, Martino & Chopin, 2009; Martins
  et al., 2013). See [INLA integration](inla.md).

**Latent marginals.** Given \(\theta\), the per-element marginals
\(p(x_j \mid y, \theta)\) come at three fidelities: the plain **Gaussian** from
the mode, the **simplified Laplace** (a skewness correction), and the **full
Laplace**. These are the INLA accuracy levels.

## Gaussian Markov random fields

A GMRF is a multivariate Gaussian whose precision matrix is sparse:

\[
Q_{jk} = 0 \quad\Longleftrightarrow\quad
x_j \perp\!\!\!\perp x_k \;\big|\; x_{-jk},
\]

that is, a zero in \(Q\) is exactly a statement of conditional independence.
Sparsity is what makes the Newton solves and log-determinants cheap. Every
effect below is, in the end, a rule for filling in a sparse \(Q\). The canonical
reference is Rue & Held (2005).

## Structured effects

### Temporal: random walks, AR1, seasonality

`RW1` and `RW2` are **intrinsic** GMRFs — smoothness priors penalising first or
second differences. With \(D\) the difference operator, \(Q = \tau D^\top D\).
Being intrinsic they are rank-deficient (improper) and carry sum-to-zero
constraints.

`AR1` is a stationary, *proper* first-order autoregression, with

\[
Q \;=\; \frac{\tau}{1-\rho^2}
\begin{pmatrix}
1 & -\rho & & \\
-\rho & 1+\rho^2 & \ddots & \\
& \ddots & \ddots & -\rho \\
& & -\rho & 1
\end{pmatrix},
\qquad
\operatorname{Cov}(x_s, x_t) = \tau^{-1}\rho^{|s-t|}.
\]

Setting `group=` gives one independent series per panel unit — the precision
becomes \(I_G \otimes Q\), block diagonal, sharing \(\rho\) and \(\tau\) but not
the realizations.

`Seasonal` penalises the sum of every \(m\) consecutive levels, so a pattern
repeating with period \(m\) is unpenalised and only *drift* away from it costs:

\[
Q \;=\; \tau\, S^\top S + \delta P_0,
\qquad
(S x)_i = \sum_{j=0}^{m-1} x_{i+j}.
\]

The null space of \(S\) is exactly the fixed seasonal patterns, so — unlike
`RW1`/`RW2`, whose null space is a nuisance absorbed by the intercept — it must
*not* be constrained away; the fixed ridge \(\delta P_0\) on its orthogonal
projector keeps it estimable and \(Q\) positive definite. See
[effects](effects.md#seasonal-effect).

### Mixed-frequency: MIDAS

`MIDAS` regresses a low-frequency response on many lags of a high-frequency
covariate, tying the lag coefficients together with a random-walk smoothness
prior over the lag index. `MIDASParametric` instead restricts the lag curve to a
low-dimensional kernel — exp-Almon or Beta — with the shape estimated:

\[
w_k(\vartheta) \;=\; \frac{\exp(\vartheta_1 k + \vartheta_2 k^2)}
{\sum_j \exp(\vartheta_1 j + \vartheta_2 j^2)} .
\]

### Spatial: the CAR family

Conditional autoregressions place a GMRF on the nodes of a neighbour graph.

**Besag / ICAR.** The intrinsic CAR uses the graph Laplacian,

\[
Q \;=\; \tau\,(D - W),
\]

with \(W\) the adjacency matrix and \(D\) the diagonal of node degrees. Each
connected component contributes one null eigenvalue (the constant vector), so
\(Q\) is improper — handled with one sum-to-zero constraint per component
(Besag, 1974; Besag, York & Mollié, 1991). Graphs may be **weighted**, so the
same family models ownership, interbank-exposure or supply-chain networks, not
only geographic adjacency.

**Scaling matters.** The precision \(\tau\) is only interpretable, and priors
only transfer between graphs, if the effect is **scaled** so that the geometric
mean of its marginal variances equals 1 (Sørbye & Rue, 2014). pyLGM applies this
scaling by default. Computing it correctly requires discarding exactly the one
null eigenvalue per connected component — a step subtle enough to have been a
real bug in this library, fixed and regression-tested (see the
[disease-mapping example](examples-disease-mapping.md)).

**Proper CAR.** `ProperCAR` adds a spatial-dependence parameter \(\rho\),
\(Q = \tau (D - \rho W)\), which is full-rank for \(\rho\) strictly inside the
interval set by the graph spectrum — a proper prior, no constraint needed.

**BYM2.** The BYM2 reparameterisation (Riebler, Sørbye, Simpson & Rue, 2016)
combines a *scaled* ICAR component \(u^\star\) and an IID component \(v\) under
a single interpretable precision and a mixing parameter \(\phi \in [0,1]\):

\[
x \;=\; \frac{1}{\sqrt{\tau}}\left(\sqrt{\phi}\, u^\star + \sqrt{1-\phi}\, v\right),
\]

so \(\phi\) reads as the fraction of marginal variance that is spatially
structured. Both parameters are identifiable and prior-friendly, which the
original BYM parameterisation was not.

### Directed and dynamic networks: SAR and SDPD

The CAR family requires a **symmetric** graph. Economic influence is usually
directed — who is exposed to whom, who supplies whom — and symmetrising it
discards the asymmetry. `SAR` instead builds the precision from a
spatial-autoregressive operator on a row-standardized, generally asymmetric
\(W\):

\[
Q \;=\; \tau\, M^\top M, \qquad M = I - \rho W ,
\]

which is symmetric and positive definite by construction for
\(\rho \in (-1,1)\), so \(\rho\) reads as network contagion strength: how much
of a unit's latent state is explained by its counterparties' rather than its own
shock.

`DynamicSpatialPanel` (SDPD) is the time-varying generalisation over a balanced
\(\text{unit} \times \text{time}\) grid. Stacking the periods, \(M\) becomes
block bidiagonal,

\[
M \;=\;
\begin{pmatrix}
A_1 & & & \\
-B_2 & A_2 & & \\
& \ddots & \ddots & \\
& & -B_T & A_T
\end{pmatrix},
\qquad
\begin{aligned}
A_t &= I - \rho W_t, \\
B_t &= \gamma I + \eta W_t,
\end{aligned}
\]

with \(\rho\) contemporaneous network dependence, \(\gamma\) own-unit
persistence, and \(\eta\) spatio-temporal diffusion. \(T=1\) reduces exactly to
`SAR`. See [spatial effects](spatial-effects.md).

### Space-time interaction

`SpaceTime` implements the Knorr-Held (2000) interaction types I–IV as a
Kronecker product of a spatial and a temporal structure,
\(Q = \tau\,(K_s \otimes K_t)\), with the null space of the product removed by
constraints so the interaction does not absorb the main effects.

## Linear constraints

Beyond the intrinsic constraints an effect carries, `LGM(constraints=...)`
imposes arbitrary linear constraints \(A x = e\) on the latent field (R-INLA's
`extraconstr`). pyLGM enforces them by **null-space reparametrisation**: with
\(B\) an orthonormal basis of \(\operatorname{null}(A)\) and \(x_p\) any
particular solution of \(A x_p = e\), writing \(x = x_p + Bz\) satisfies
\(Ax = e\) for every \(z\), and inference proceeds in the reduced coordinates
\(z\). For the homogeneous case (\(e = 0\)) this is exact and \(x_p = 0\).

For a nonzero right-hand side the prior on \(z\) is not centred: conditioning
\(x \sim \mathcal{N}(0, Q^{-1})\) on \(Ax = e\) induces a Gaussian on \(z\) with
mean

\[
m \;=\; -\left(B^\top Q B\right)^{-1} B^\top Q\, x_p ,
\]

the **conditioning-by-kriging** solution R-INLA uses. Because the same prior
linear term corrects for the choice of \(x_p\), the fit is independent of
*which* particular solution is picked, so pyLGM takes the cheap least-squares
one. Contradictory constraints are not rejected; the least-squares \(x_p\) gives
the closest satisfiable field.

## Priors: penalised complexity

pyLGM's default hyperparameter priors are **PC (penalised-complexity) priors**
(Simpson, Rue, Riebler, Martins & Sørbye, 2017). A PC prior penalises the
*distance* of a component from a simpler base model (e.g. "no effect") at a
constant rate, giving an exponential prior on that distance. For a precision it
becomes a prior on the standard deviation \(\sigma\) set by one interpretable
statement,

\[
\mathbb{P}(\sigma > u) = \alpha
\qquad\text{e.g.}\qquad
\mathbb{P}(\sigma > 1) = 0.01,
\]

which is how `PCPrecision(upper_sd=u, alpha=α)` is specified. The BYM2 mixing
parameter has its own PC prior (`PCBYM2Phi`).

## Scaling: why sparsity is the whole game

A dense Gaussian posterior over \(n\) latent elements costs \(O(n^3)\) to
factorise and \(O(n^2)\) to store — about 200 GB at \(n = 10^5\). A GMRF with
bandwidth/fill \(b\) costs roughly \(O(n b^2)\) instead. pyLGM keeps a dense
reference engine for small models and switches, past a guard, to a sparse
constrained-Gaussian solver that delivers the posterior mean, marginal
likelihood, and the full uncertainty surface — marginal, predictive and
linear-combination variances — via a **Takahashi selected inverse**, which
computes the entries of \(\Sigma = Q^{-1}\) on the sparsity pattern of the
Cholesky factor without ever forming \(\Sigma\). See
[internals](internals.md).

## Model assessment

For non-Gaussian fits pyLGM reports the standard INLA criteria: **DIC**
(Spiegelhalter, Best, Carlin & van der Linde, 2002), **WAIC** (Watanabe, 2010),
and leave-one-out **CPO/PIT** (Held, Schrödle & Rue, 2010). See
[INLA integration](inla.md).

## How the theory maps to the API

| Concept | API |
| --- | --- |
| Likelihood / link | [`Gaussian`, `Poisson`, `Bernoulli`, `Binomial`, `NegativeBinomial`, `Gamma`, `Beta`](likelihoods.md) |
| Survival with censoring/truncation | [`WeibullSurv`, `ExponentialSurv`](likelihoods.md#survival-likelihoods) |
| Fixed effects, IID random effects | [`Fixed`, `IID`](effects.md) |
| Temporal GMRFs | [`RW1`, `RW2`, `AR1` (optionally group-wise), `Seasonal`](effects.md) |
| Mixed-frequency lag curves | [`MIDAS`, `MIDASParametric`](effects.md#midas-smooth-lag-effect) |
| Spatial CAR GMRFs | [`Besag`, `ProperCAR`, `BYM2`](spatial-effects.md) |
| Directed / dynamic networks | [`SAR`, `DynamicSpatialPanel`](spatial-effects.md#directed-spatial-autoregressive-sar-effect) |
| Space-time interaction | [`SpaceTime`](effects.md#spacetime-effect-knorr-held-interaction) |
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
4. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
5. Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
   in disease risk. *Statistics in Medicine*, 19(17–18), 2555–2567.
6. Spiegelhalter, D. J., Best, N. G., Carlin, B. P., & van der Linde, A.
   (2002). Bayesian measures of model complexity and fit. *Journal of the Royal
   Statistical Society B*, 64(4), 583–639.
7. Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch: mixed
   data sampling regression models. *Working paper*.
8. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
   Applications*. Chapman & Hall/CRC.
9. Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for
   latent Gaussian models by using integrated nested Laplace approximations.
   *Journal of the Royal Statistical Society B*, 71(2), 319–392.
10. Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
    widely applicable information criterion in singular learning theory.
    *Journal of Machine Learning Research*, 11, 3571–3594.
11. Held, L., Schrödle, B., & Rue, H. (2010). Posterior and cross-validatory
    predictive checks: a comparison of MCMC and INLA. In *Statistical Modelling
    and Regression Structures*, 91–110. Physica-Verlag.
12. Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between
    Gaussian fields and Gaussian Markov random fields: the stochastic partial
    differential equation approach. *Journal of the Royal Statistical Society B*,
    73(4), 423–498.
13. Yu, J., de Jong, R., & Lee, L.-F. (2008). Quasi-maximum likelihood
    estimators for spatial dynamic panel data with fixed effects. *Journal of
    Econometrics*, 146(1), 118–134.
14. Martins, T. G., Simpson, D., Lindgren, F., & Rue, H. (2013). Bayesian
    computing with INLA: New features. *Computational Statistics & Data
    Analysis*, 67, 68–83.
15. Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random
    field priors in spatial modelling. *Spatial Statistics*, 8, 39–51.
16. Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
    Bayesian spatial model for disease mapping that accounts for scaling.
    *Statistical Methods in Medical Research*, 25(4), 1145–1165.
17. Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017).
    Penalising model component complexity: a principled, practical approach to
    constructing priors. *Statistical Science*, 32(1), 1–28.
