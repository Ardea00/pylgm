import numpy as np
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def test_estimated_delta_is_recovered_from_simulated_shared_component_data(
    shared_component_frame,
):
    frame, true_delta = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(
            IID("u", index="district", precision=Hyperparameter("tau_u", initial=1.0)),
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
