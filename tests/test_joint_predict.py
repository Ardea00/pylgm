import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _fitted(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0),
                       scale=Hyperparameter("delta", initial=1.0))],
    )
    return frame, joint.fit(frame, engine="laplace")


def test_predict_on_fit_rows_reproduces_the_fitted_means_per_outcome(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    prediction = result.predict(oral_rows, outcome="oral")
    fitted = result.predictive_mean[: len(oral_rows)]
    assert prediction.predictive_mean == pytest.approx(fitted, rel=1e-6, abs=1e-8)


def test_predict_requires_an_outcome_on_a_joint_result(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    with pytest.raises(ValueError, match="outcome"):
        result.predict(oral_rows)


def test_predict_rejects_an_unknown_outcome_and_lists_the_valid_ones(shared_component_frame):
    frame, result = _fitted(shared_component_frame)
    oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
    with pytest.raises(ValueError, match="oral"):
        result.predict(oral_rows, outcome="nonsense")


def test_predict_rejects_outcome_on_a_single_response_result(shared_component_frame):
    frame, _ = shared_component_frame()
    sub = frame[frame["oral"].notna()].reset_index(drop=True)
    result = LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")).fit(
        sub, engine="laplace"
    )
    with pytest.raises(ValueError, match="not a joint"):
        result.predict(sub, outcome="oral")
