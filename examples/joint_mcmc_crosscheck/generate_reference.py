"""Regenerate nuts_reference.json. Requires PyMC; NOT needed to run the tests.

    python -m venv /tmp/venv-mcmc
    /tmp/venv-mcmc/bin/pip install pymc
    /tmp/venv-mcmc/bin/python generate_reference.py

Install PyMC in a SEPARATE environment. It pins PyTensor, which constrains
numpy/scipy, and installing it alongside pyLGM can move numpy under the project.
"""
import json
import pathlib

import numpy as np
import pymc as pm
import arviz as az

N, DELTA, SIGMA_U = 40, 1.6, 0.7
ALPHA_SD = 1.0 / np.sqrt(1e-6)      # pylgm Fixed.prior_precision default, as an SD
SEED = 20260902
SAMPLE = dict(draws=8000, tune=2000, chains=4, target_accept=0.95,
              random_seed=SEED, progressbar=False)


def simulate():
    rng = np.random.default_rng(SEED)
    u = rng.normal(0.0, SIGMA_U, size=N)
    return (rng.poisson(np.exp(-0.3 + DELTA * u)).astype(float),
            rng.poisson(np.exp(0.2 + u / DELTA)).astype(float))


def summarise(idata, mapping):
    ess, rhat = az.ess(idata), az.rhat(idata)
    out = {"mean": {}, "sd": {}, "skew": {}, "ess_bulk": {}, "r_hat": {}}
    for var, keys in mapping.items():
        arr = np.asarray(idata.posterior[var])
        for j, key in enumerate(keys):
            d = arr.ravel() if arr.ndim == 2 else arr[:, :, j].ravel()
            out["mean"][key] = float(d.mean())
            out["sd"][key] = float(d.std(ddof=1))
            out["skew"][key] = float(((d - d.mean())**3).mean() / d.std()**3)
            e, r = (ess[var], rhat[var]) if arr.ndim == 2 else (ess[var][j], rhat[var][j])
            out["ess_bulk"][key] = float(e)
            out["r_hat"][key] = float(r)
    return out


y_oral, y_lar = simulate()

with pm.Model():                                    # the joint shared-component model
    a1 = pm.Normal("alpha_oral", 0.0, sigma=ALPHA_SD)
    a2 = pm.Normal("alpha_larynx", 0.0, sigma=ALPHA_SD)
    u = pm.Normal("u", 0.0, sigma=SIGMA_U, shape=N)
    pm.Poisson("y_oral", mu=pm.math.exp(a1 + DELTA * u), observed=y_oral)
    pm.Poisson("y_larynx", mu=pm.math.exp(a2 + u / DELTA), observed=y_lar)
    joint = pm.sample(**SAMPLE)

with pm.Model():                                    # single-response control
    a = pm.Normal("a", 0.0, sigma=ALPHA_SD)
    u = pm.Normal("u", 0.0, sigma=SIGMA_U, shape=N)
    pm.Poisson("y", mu=pm.math.exp(a + u), observed=y_oral)
    control = pm.sample(**SAMPLE)

j = summarise(joint, {"alpha_oral": ["oral:fixed:Intercept"],
                      "alpha_larynx": ["larynx:fixed:Intercept"],
                      "u": [f"u:{i}" for i in range(N)]})
c = summarise(control, {"a": ["fixed:Intercept"], "u": [f"u:{i}" for i in range(N)]})

ref = {
    "_comment": ("NUTS reference posteriors. Regenerate with generate_reference.py "
                 f"(needs PyMC; see README). {SAMPLE['chains']} chains x "
                 f"{SAMPLE['draws']} draws, tune {SAMPLE['tune']}, "
                 f"target_accept {SAMPLE['target_accept']}, seed {SEED}."),
    "sampler": {"draws": SAMPLE["chains"] * SAMPLE["draws"],
                "min_ess_bulk": round(min(j["ess_bulk"].values())),
                "max_r_hat": round(max(j["r_hat"].values()), 5),
                "divergences": int(joint.sample_stats.diverging.sum())},
    "joint": {"mean": j["mean"], "sd": j["sd"], "skew": j["skew"]},
    "single_response_control": {"mean": c["mean"], "sd": c["sd"]},
}
path = pathlib.Path(__file__).with_name("nuts_reference.json")
path.write_text(json.dumps(ref, indent=1, sort_keys=True))
print(f"wrote {path}  min ESS {ref['sampler']['min_ess_bulk']}  "
      f"max R-hat {ref['sampler']['max_r_hat']}  div {ref['sampler']['divergences']}")
