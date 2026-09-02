"""A field entering one predictor twice, the second time scaled.

    log mu_k = alpha + u_{i(k)} + beta * u_{j(k)}

Recovering `beta` from data generated with it is what proves the copy folds
into the right columns with the right coefficient, rather than merely
compiling.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.parameters import Hyperparameter

N_LEVELS, N_ROWS, TRUE_BETA, SIGMA_U = 12, 600, 1.8, 0.5


def _simulate(seed=31):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, SIGMA_U, N_LEVELS)
    i = rng.integers(0, N_LEVELS, N_ROWS)
    j = rng.integers(0, N_LEVELS, N_ROWS)
    eta = 0.3 + u[i] + TRUE_BETA * u[j]
    return u, pd.DataFrame({
        "i": [f"L{k}" for k in i],
        "j": [f"L{k}" for k in j],
        "y": rng.poisson(np.exp(eta)).astype(float),
        "row": range(N_ROWS),
    })


def test_the_copy_scale_is_recovered():
    _, frame = _simulate()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1")
        + IID("u", index="i", precision=1.0 / SIGMA_U**2)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    ).fit(frame, engine="laplace")
    assert result.hyperparameters["beta"] == pytest.approx(TRUE_BETA, rel=0.4)


def test_a_unit_scale_copy_equals_stacking_the_index_columns_by_hand():
    """With beta = 1, u_{i(k)} + u_{j(k)} is what you would get by summing two
    one-hot designs over the same levels -- so the copy must reproduce exactly
    the design a hand-built sum gives."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    _, frame = _simulate()
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    copied = compile_lgm(
        LGM(response="y", likelihood=Poisson(),
            predictor=Fixed("1") + IID("u", index="i", precision=1.0)
            + Copy("u", index="j", scale=1.0)),
        panel,
    )
    base = compile_lgm(
        LGM(response="y", likelihood=Poisson(),
            predictor=Fixed("1") + IID("u", index="i", precision=1.0)),
        panel,
    )
    u_copied = [b for b in copied.blocks if b.name == "u"][0].design.toarray()
    u_base = [b for b in base.blocks if b.name == "u"][0].design.toarray()

    labels = [b for b in base.blocks if b.name == "u"][0].labels
    position = {label: k for k, label in enumerate(labels)}
    manual = u_base.copy()
    for row, level in enumerate(frame["j"].astype(str)):
        manual[row, position[level]] += 1.0
    assert np.allclose(u_copied, manual)
