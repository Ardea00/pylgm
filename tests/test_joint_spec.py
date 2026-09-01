import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _sub(response):
    return LGM(response=response, likelihood=Poisson(), predictor=Fixed("1"))


def test_joint_exposes_outcomes_in_declaration_order():
    joint = Joint([_sub("oral"), _sub("larynx")])
    assert joint.outcomes == ("oral", "larynx")


def test_joint_rejects_duplicate_response_names():
    with pytest.raises(ValueError, match="unique"):
        Joint([_sub("oral"), _sub("oral")])


def test_joint_requires_at_least_two_submodels():
    with pytest.raises(ValueError, match="at least two"):
        Joint([_sub("oral")])


def test_scalar_hyperparameter_scale_expands_to_delta_and_inverse():
    shared = Shared(IID("u", index="district"), scale=Hyperparameter("delta", initial=1.0))
    scales = shared.scales_for(2)
    assert scales[0].name == "delta"
    assert scales[1] == ("delta", "inverse")


def test_scalar_hyperparameter_scale_rejected_beyond_two_submodels():
    shared = Shared(IID("u", index="district"), scale=Hyperparameter("delta", initial=1.0))
    with pytest.raises(ValueError, match="exactly two"):
        shared.scales_for(3)


def test_explicit_tuple_scale_passes_through():
    shared = Shared(IID("u", index="district"), scale=(1.0, 2.0, 0.5))
    assert shared.scales_for(3) == (1.0, 2.0, 0.5)


def test_scale_tuple_length_must_match_submodel_count():
    shared = Shared(IID("u", index="district"), scale=(1.0, 2.0))
    with pytest.raises(ValueError, match="length"):
        shared.scales_for(3)


def test_float_scale_broadcasts_to_every_submodel():
    shared = Shared(IID("u", index="district"), scale=1.0)
    assert shared.scales_for(3) == (1.0, 1.0, 1.0)


def test_allow_ragged_defaults_to_false_and_must_be_a_bool():
    assert Shared(IID("u", index="district")).allow_ragged is False
    with pytest.raises(TypeError, match="allow_ragged"):
        Shared(IID("u", index="district"), allow_ragged="yes")


def test_mixed_likelihoods_are_allowed():
    joint = Joint([
        LGM(response="cd4", likelihood=Gaussian(sigma=1.0), predictor=Fixed("1")),
        LGM(response="event", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    assert joint.outcomes == ("cd4", "event")
