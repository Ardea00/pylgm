# Calibration checking

`pylgm.validation.calibrate` answers a question no goodness-of-fit statistic
can: **if the data really came from this model, does the reported posterior
cover the truth at the rate it advertises?**

It is a simulation-based calibration check (SBC, [Talts et al.
2018](https://arxiv.org/abs/1804.06788)). Nothing about it needs MCMC, and it
works on your own model, not just on the library's test suite.

## The short version

```python
from pylgm import Fixed, Hyperparameter, IID, LGM, Poisson
from pylgm.priors import PCPrecision
from pylgm.validation import calibrate

model = LGM(
    response="y",
    likelihood=Poisson(),
    predictor=Fixed("1", prior_precision=4.0)
    + IID("u", index="g", precision=9.0),
)
print(calibrate(model, frame))
```

```
calibration over 128 replicates (seed 0, Bonferroni threshold p >= 0.001)
  component         p(location)   p(dispersion)   mean PIT
  fixed:Intercept        0.7343          0.7207     0.5118
  u:a1                   0.4412         0.03567     0.4731
  u:a3                   0.1968          0.8213     0.4354
  u:a5                   0.2347          0.1878     0.4460
  u:a7                   0.4797          0.4188     0.4766
  verdict: PASS
```

`frame` supplies the *design* only — its index columns, sizes and covariates are
used as-is and its response column is overwritten with simulated data, so
whatever values are in it do not matter. The engine is chosen from the
likelihood, so there is nothing else to pass.

`report.ok` is the boolean, `report.failures` the offending components.

## How it works

For each of `replicates` rounds:

1. draw a latent field from the model's own prior, $x \sim N(0, Q^{-1})$;
2. simulate a response from the likelihood at $\eta = Ax$;
3. refit;
4. record where the truth fell in each reported marginal — the probability
   integral transform $u = F(x_\text{true})$.

Under exact inference $u \sim \mathrm{Uniform}(0,1)$ exactly, so the check is a
uniformity test. Talts et al. phrase this as a rank among $L$ posterior draws;
$u$ is the $L \to \infty$ limit, and it is the natural form here because pyLGM
reports densities rather than draws — no sampler, and no Monte-Carlo noise in
the statistic.

## Why two p-values

The two ways a marginal goes wrong leave different fingerprints, and one test
does not see both:

| failure | PIT histogram | caught by |
|---|---|---|
| posterior mean off | shifted sideways | `p(location)` |
| posterior variance off | symmetric, U- or dome-shaped | `p(dispersion)` |

A dispersion error leaves the PIT median at 0.5, so the ordinary KS statistic
barely moves: against a 10% inflated SD it has roughly 0.01 power even at 1024
replicates. Folding the PIT about its centre — testing $|u - 0.5|\cdot 2$, which
is also Uniform(0,1) — turns that symmetric deviation into the one-sided kind KS
detects. Both are reported, and a component passes only if both do.

### Sizing `replicates`

Calibration checking is not very sensitive at small `replicates`, and it is worth
knowing by how much before reading a `PASS` as reassurance. Measured detection
probability, at the default Bonferroni threshold with five tracked components:

| corruption | R=128 | R=256 | R=512 | R=1024 |
|---|---|---|---|---|
| mean off by 0.1σ | 0.01 | 0.02 | 0.08 | 0.27 |
| mean off by 0.2σ | 0.08 | 0.27 | 0.71 | 0.98 |
| mean off by 0.3σ | 0.34 | 0.80 | 1.00 | 1.00 |
| SD off by 15% (0.85×) | 0.07 | 0.25 | 0.66 | 0.98 |
| SD off by 25% (1.25×) | 0.22 | 0.64 | 0.99 | 1.00 |

So the default `replicates=128` is a smoke test: it catches gross breakage, not
subtle miscalibration. Use 512–1024 when a `PASS` is meant to carry weight, and
read a `PASS` at 128 as "nothing obviously broken" rather than "calibrated".

## What a failure means

A `FAIL` says the reported marginals are not calibrated *for this model at this
data size*. That is not automatically a bug:

- With a Gaussian likelihood and the default `latent_strategy="gaussian"` the
  posterior is **exact**, so a failure there is a genuine defect.
- With a non-Gaussian likelihood the Gaussian latent strategy is a known
  approximation. A failure at small counts is expected rather than alarming, and
  tells you how far the approximation is from honest at *your* data size — raise
  the counts, or treat the reported intervals as approximate.

## Hyperparameters

If the model declares a `Hyperparameter`, each replicate **draws it from its
prior** and the fit integrates over it (`hyperparameters="integrate"`), so the
check covers the marginal posterior rather than one conditional on a fixed
value. Every declared hyperparameter needs a prior for this; one without is
rejected, because falling back to empirical Bayes would re-estimate it from each
simulated dataset and quietly test a different quantity.

```python
model = LGM(
    response="y", likelihood=Gaussian(sigma=0.7),
    predictor=Fixed("1", prior_precision=1.0)
    + IID("u", index="g", precision=Hyperparameter(
        "tau", initial=2.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
)
calibrate(model, frame, replicates=512)
```

Draws are taken from the prior **truncated to the hyperparameter's
`lower`/`upper`**, because that is the prior the engine actually uses.

Each drawn value is also checked against its own reported marginal, listed as
`hyper:<name>`, and that row passes.

It did not always, and the history is worth knowing because it shows what this
check is for. The INLA grid used to span a fixed three Hessian-implied standard
deviations around the empirical-Bayes mode. For a weakly identified precision —
few groups, little information about the random-effect variance — the posterior's
right tail runs well past that, so the reported marginal was truncated: the grid
reached 0.18× the true 97.5th percentile, the reported mean was 0.32× the truth,
and the PIT mean sat at 0.62 instead of 0.5. The grid now explores outward until
the log density actually drops, so its extent follows the posterior:

| | fixed radius | adaptive |
|---|---|---|
| grid edge ÷ true 97.5th pct | 0.18 | 1.37 |
| reported mean ÷ true mean | 0.32 | 0.86 |
| median quantile error | 0.43 | 0.07 |
| PIT mean (0.5 is calibrated) | 0.62 | 0.53 |

The marginal is tabulated from the grid rather than moment-matched, so its shape
is right too: median relative quantile error 0.04 against a brute-force
reference, versus 0.51 for a moment-matched Gaussian.

With **two or more** hyperparameters the marginal is still moment-matched — the
grid is a lattice rotated onto the whitened Hessian's directions, so a single
axis's marginal cannot be read off it — and it is not checked here.

### Choosing a latent strategy

Integrating over hyperparameters makes the true latent marginal a *mixture* over
the θ grid. The default `latent_strategy="gaussian"` reports only that mixture's
first two moments — right variance, wrong shape — and calibration checking sees
it. On a Gaussian-likelihood IID model at 512 replicates:

| `latent_strategy` | worst p(dispersion) |
|---|---|
| `gaussian` | 4e-10 |
| `simplified_laplace` | 9e-07 |
| `laplace` | 7e-07 |

The location statistic barely moves; it is the dispersion statistic that reacts,
which is the fingerprint of a shape error rather than a bias. Refining the INLA
integration grid does **not** close the gap — the error is in how the marginal is
summarised, not in how accurately θ is integrated.

So if you use `hyperparameters="integrate"` and care about interval coverage
rather than just point estimates, prefer `simplified_laplace`. Comparing the two
reports on your own model is the intended way to decide.

## Limits

- **A hyperparameter's own marginal is checked, and currently fails** — see
  below. The latent-field result is the trustworthy part of the report.
- **Proper priors only.** `Fixed` defaults to `prior_precision=1e-6` — a prior SD
  of 1000, sensible for fitting and useless for simulating. `calibrate` rejects
  it rather than producing nonsense; pass a real `prior_precision`.
- **Intrinsic effects are supported.** `RW1`, `RW2` and `Besag` have singular
  precisions, and their constraints are a basis of the null space; the field is
  drawn on the subspace where the improper prior becomes proper, in the same
  parametrisation the engines use. (`BYM2` needs none of this — its precision is
  full rank.) A rank-deficient direction that *no* constraint pins down is still
  rejected, because that prior genuinely is improper.
- **Marginal, not joint.** Uniformity of each scalar marginal is necessary but
  not sufficient for joint correctness.
- **It cannot check the precision matrix itself.** The latent field is drawn
  from the *compiled* model, so an error in how $Q$ is assembled would corrupt
  both sides identically and stay invisible. That is deliberate — a separate
  hand-written generator would leave you unable to tell which side was wrong —
  and it is why the library also keeps closed-form oracle tests for $Q$. Oracles
  pin the model; calibration pins the inference given the model.

## Pieces

`simulate_latent`, `simulate_response` and `pit` are exported for anyone who
wants to drive the loop themselves — for example to calibrate a derived quantity
rather than a latent component.
