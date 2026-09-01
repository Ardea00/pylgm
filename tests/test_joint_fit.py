import numpy as np
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson
from pylgm.exceptions import CompilationError
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
    with pytest.raises(Exception, match="exact_gaussian"):
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
