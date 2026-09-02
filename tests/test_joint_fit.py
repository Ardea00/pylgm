import numpy as np
import pandas as pd
import pytest

from pylgm import Binomial, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.exceptions import CompilationError, UnsupportedEngineError
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def test_estimated_delta_is_recovered_from_simulated_shared_component_data(
    shared_component_frame,
):
    # The shared IID's precision is fixed at the simulation's true value
    # (1 / 0.7**2 ~= 2.041): a Hyperparameter here is rejected now (Finding 3 --
    # a shared effect's own precision cannot be estimated yet), and it was never
    # actually estimated before either -- it silently pinned at its initial 1.0,
    # letting `delta` (the only free knob) absorb the misspecification.
    frame, true_delta = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(
            IID("u", index="district", precision=1.0 / 0.7**2),
            scale=Hyperparameter("delta", initial=1.0),
        )],
    )
    result = joint.fit(frame, engine="laplace")
    assert result.hyperparameters["delta"] == pytest.approx(true_delta, rel=0.5)


def test_fixed_scales_need_no_hyperparameter_and_still_fit(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0), scale=(1.0, 1.0))],
    )
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)


def test_joint_rejects_the_exact_gaussian_engine(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    with pytest.raises(UnsupportedEngineError, match="exact_gaussian"):
        joint.fit(frame, engine="exact_gaussian")


def test_joint_with_estimated_gaussian_sigma_fits_and_reports_it(shared_component_frame):
    """Finding 1: an estimated Gaussian sigma must not crash compile_joint_family."""
    frame, _ = shared_component_frame()
    joint = Joint([
        LGM(
            response="oral",
            likelihood=Gaussian(sigma=Hyperparameter("sigma_oral", initial=1.0)),
            predictor=Fixed("1"),
        ),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.hyperparameters["sigma_oral"])
    assert result.hyperparameters["sigma_oral"] > 0


def test_shared_scale_name_colliding_with_submodel_hyperparameter_raises(
    shared_component_frame,
):
    """Finding 2: a name collision between a shared scale and a per-sub-model
    hyperparameter must raise, not silently alias the two unrelated quantities."""
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(
            response="oral", likelihood=Poisson(),
            predictor=IID("delta", index="district", precision=Hyperparameter("delta", initial=1.0)),
        ),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(
            IID("u", index="district", precision=1.0),
            scale=Hyperparameter("delta", initial=2.0),
        )],
    )
    with pytest.raises(CompilationError, match="delta"):
        joint.fit(frame, engine="laplace")


def test_shared_entries_reusing_the_same_hyperparameter_still_work(shared_component_frame):
    """Guard against over-rejecting: two Shared entries legitimately reusing the
    identical Hyperparameter object (not merely the same name) must still fit."""
    frame, _ = shared_component_frame()
    shared_scale = Hyperparameter("delta", initial=1.0)
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[
            Shared(IID("u", index="district", precision=1.0), scale=shared_scale),
            Shared(IID("v", index="district", precision=1.0), scale=(shared_scale, 1.0)),
        ],
    )
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.hyperparameters["delta"])


def test_joint_fit_keeps_nan_response_rows_as_unobserved_not_dropped():
    """Finding I4: Joint.fit used to drop each sub-model's NaN-response rows
    (`frame[frame[response].notna()]`) instead of holding them out, so the
    NaN hold-out idiom documented for LGM.fit -- unobserved but still fitted
    -- silently did nothing on a Joint sub-model. Every sub-model's panel
    must keep spanning the full input frame, with the previously-dropped
    rows now assigned a fitted value instead of vanishing."""
    frame = pd.DataFrame({
        "count_a": [1.0, 2.0, None, 4.0, 5.0],
        "count_b": [3.0, 1.0, 2.0, 4.0, None],
        "row": range(5),
    })
    joint = Joint([
        LGM(response="count_a", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="count_b", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    result = joint.fit(frame, engine="laplace")

    # Neither sub-model's rows were dropped: each still spans every input row
    # (5 rows, not 4), so the stacked prediction covers 2 * len(frame).
    assert len(result.predictive_mean) == 2 * len(frame)
    a_fitted, b_fitted = np.split(result.predictive_mean, 2)
    # The rows that used to be dropped (count_a row 2, count_b row 4) now get
    # a real fitted value on the linear predictor rather than being absent.
    assert np.isfinite(a_fitted[2])
    assert np.isfinite(b_fitted[4])
    assert np.all(np.isfinite(a_fitted))
    assert np.all(np.isfinite(b_fitted))


def test_joint_fit_with_binomial_submodel_and_unobserved_row():
    """Finding I3: compile_joint binds a part's aux (Binomial's trials) via
    `for_observations` over its own full sub-frame, but CompiledMixture's old
    `restrict` re-indexed only the masks, leaving that aux at full length.
    The moment I4 stops dropping NaN-response rows, a Binomial sub-model with
    any unobserved row hits the Laplace engine's `restrict(observed)` call
    and crashes with a broadcast error -- this is exactly that case, now
    routed through the full Joint.fit path rather than compile_joint
    directly, and must fit cleanly instead of crashing."""
    frame = pd.DataFrame({
        "y_bin": [3.0, 5.0, None, 2.0],
        "n_bin": [10.0, 10.0, 10.0, 10.0],
        "y_pois": [1.0, 2.0, 3.0, 4.0],
        "row": range(4),
    })
    joint = Joint([
        LGM(response="y_bin", likelihood=Binomial(trials="n_bin"), predictor=Fixed("1")),
        LGM(response="y_pois", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.predictive_mean) == 2 * len(frame)


def test_shared_effect_hyperparameter_precision_raises(shared_component_frame):
    """Finding 3: a Hyperparameter on a shared effect's own precision must fail
    loudly at compile time rather than silently pinning it at its initial value."""
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(
            IID("u", index="district", precision=Hyperparameter("tau_u", initial=1.0)),
            scale=Hyperparameter("delta", initial=1.0),
        )],
    )
    with pytest.raises(CompilationError, match="not supported"):
        joint.fit(frame, engine="laplace")
