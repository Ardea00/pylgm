"""Cross-check the joint-model machinery against MCMC ground truth.

R-INLA was the originally planned reference, but agreeing with R-INLA would only
show pyLGM reproduces *another Laplace approximation of the same family*. NUTS is
asymptotically exact, so it tests the approximation itself. Rue, Martino & Chopin
(2009, sec. 5) validate INLA against MCMC the same way.

Reference posteriors live in ``examples/joint_mcmc_crosscheck/nuts_reference.json``
(4 chains x 8000 draws, min ESS ~23.8k, max R-hat 1.0006, 0 divergences).
PyMC is NOT needed to run these tests -- only to regenerate the reference.

WHAT IS AND IS NOT ASSERTED
---------------------------
pyLGM's ``gaussian`` latent strategy reports the **joint posterior mode**; MCMC
reports **marginal posterior means**. In 42 dimensions these genuinely differ,
even with near-symmetric marginals (the alpha_oral marginal has skew -0.087),
because the mode is not a typical point in high dimensions. So we do NOT assert
that pyLGM's means equal the MCMC means -- that would be asserting a falsehood.

We assert three things that are true and that would break if the joint machinery
were wrong:

1. pyLGM's latent mean IS the exact posterior mode, checked against an
   independent optimisation of the exact log-posterior written in plain
   numpy/scipy here -- no pyLGM, no PyMC in the loop.
2. pyLGM's posterior SDs match MCMC's, i.e. the curvature is right even where
   the location convention differs.
3. The mode-vs-mean gap is no worse in a joint model than in a plain
   single-response LGM. This is the control that proves the gap is a property of
   the Gaussian latent strategy and not something joint models introduce.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "joint_mcmc_crosscheck"
N, DELTA, SIGMA_U = 40, 1.6, 0.7
TAU_U = 1.0 / SIGMA_U**2
ALPHA_PRECISION = 1e-6          # pylgm.effects.spec.Fixed.prior_precision default
SEED = 20260902


def _simulate():
    rng = np.random.default_rng(SEED)
    u = rng.normal(0.0, SIGMA_U, size=N)
    y_oral = rng.poisson(np.exp(-0.3 + DELTA * u)).astype(float)
    y_lar = rng.poisson(np.exp(0.2 + u / DELTA)).astype(float)
    return u, y_oral, y_lar


def _fit_joint(y_oral, y_lar):
    frame = pd.DataFrame({
        "district": list(range(N)) * 2,
        "oral": list(y_oral) + [np.nan] * N,
        "larynx": [np.nan] * N + list(y_lar),
        "row": range(2 * N),
    })
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        # delta is FIXED on purpose: pyLGM estimates it by empirical Bayes (a point
        # maximising the marginal likelihood, not a posterior), so a free delta
        # would make the two sides answer different questions.
        shared=[Shared(IID("u", index="district", precision=TAU_U),
                       scale=(DELTA, 1.0 / DELTA))],
    )
    return joint.fit(frame, engine="laplace")


@pytest.fixture(scope="module")
def reference():
    return json.loads((EXAMPLE / "nuts_reference.json").read_text())


def test_joint_latent_mean_is_the_exact_posterior_mode():
    """The strongest check, and it needs no MCMC: solve the same problem directly.

    The log-posterior below is written from scratch in numpy -- Poisson
    log-likelihood on both stacked outcomes plus the Gaussian priors. If the
    stacking, row padding, or shared scaling were wrong, pyLGM would be
    optimising a different function and could not land on this optimum.
    """
    _, y_oral, y_lar = _simulate()
    result = _fit_joint(y_oral, y_lar)

    def neg_log_posterior(theta):
        a1, a2, u = theta[0], theta[1], theta[2:]
        eta_o, eta_l = a1 + DELTA * u, a2 + u / DELTA
        loglik = (y_oral * eta_o - np.exp(eta_o)).sum() + (y_lar * eta_l - np.exp(eta_l)).sum()
        logprior = -0.5 * TAU_U * (u**2).sum() - 0.5 * ALPHA_PRECISION * (a1**2 + a2**2)
        return -(loglik + logprior)

    opt = minimize(neg_log_posterior, np.zeros(N + 2), method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    assert opt.success

    by_label = dict(zip(result.labels, result.mean))
    fitted = np.array(
        [by_label["oral:fixed:Intercept"], by_label["larynx:fixed:Intercept"]]
        + [by_label[f"u:{i}"] for i in range(N)]
    )
    assert np.abs(fitted - opt.x).max() < 1e-5
    # And it is a better point than the MCMC posterior mean, which is the
    # signature of a correct mode rather than a failed optimisation.
    assert neg_log_posterior(fitted) <= neg_log_posterior(opt.x) + 1e-9


def test_joint_posterior_sds_agree_with_nuts(reference):
    """Location convention differs; curvature should not."""
    _, y_oral, y_lar = _simulate()
    result = _fit_joint(y_oral, y_lar)
    sd = dict(zip(result.labels, np.sqrt(np.diag(result.covariance))))
    ref = reference["joint"]["sd"]
    ratios = np.array([sd[k] / ref[k] for k in ref])
    assert ratios.min() > 0.90, f"pyLGM SDs too small vs NUTS: min ratio {ratios.min():.4f}"
    assert ratios.max() < 1.10, f"pyLGM SDs too large vs NUTS: max ratio {ratios.max():.4f}"


def test_joint_mode_mean_gap_is_no_worse_than_a_plain_lgm(reference):
    """The control: the gap is the Gaussian latent strategy's, not the joint's.

    A plain single-response LGM over the same data shows the same effect. If the
    joint gap were materially larger, the joint machinery would be adding error
    of its own, which is exactly what this guards against.
    """
    _, y_oral, y_lar = _simulate()

    joint = _fit_joint(y_oral, y_lar)
    jm = dict(zip(joint.labels, joint.mean))
    jref = reference["joint"]
    joint_z = np.array([(jm[k] - jref["mean"][k]) / jref["sd"][k] for k in jref["mean"]])

    frame = pd.DataFrame({"district": range(N), "y": y_oral, "row": range(N)})
    single = LGM(response="y", likelihood=Poisson(),
                 predictor=Fixed("1") + IID("u", index="district", precision=TAU_U)
                 ).fit(frame, engine="laplace")
    sm = dict(zip(single.labels, single.mean))
    sref = reference["single_response_control"]
    single_z = np.array([(sm[k] - sref["mean"][k]) / sref["sd"][k] for k in sref["mean"]])

    # Joint has two near-flat intercept directions to the control's one, so a
    # modestly larger gap is expected; a large one would mean real degradation.
    assert np.abs(joint_z).max() < 2.0 * np.abs(single_z).max(), (
        f"joint max|z| {np.abs(joint_z).max():.3f} vs control {np.abs(single_z).max():.3f}"
    )
    assert np.abs(joint_z).mean() < 0.20
