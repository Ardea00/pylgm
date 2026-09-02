import pandas as pd
import pytest

from pylgm import Copy, Fixed, Gaussian, IID, LGM, Poisson, Weighted
from pylgm.compiler import compile_joint
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.joint import Joint, Shared
from pylgm.likelihoods import CompiledGaussian, CompiledMixture, CompiledPoisson
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


def test_shared_rejects_an_effect_with_no_index_at_declaration_time():
    # Finding I2: Fixed (and MIDAS/MIDASParametric/SpaceTime/
    # DynamicSpatialPanel) pass the __post_init__ `hasattr(effect, "name")`
    # check but have no `.index` -- Shared(Fixed("1")) used to construct fine
    # and only die later, at fit time, with an unrelated AttributeError.
    with pytest.raises(TypeError, match="index"):
        Shared(Fixed("1"))


def test_shared_rejects_a_weighted_effect_naming_the_wrapper():
    # Final-slice finding I2: Weighted also has no `.index` (deliberately --
    # it wraps an indexed effect but doesn't expose one itself), so it hits
    # the same branch as Fixed above. The message must name Weighted as the
    # unsupported part rather than only describing what a base model needs,
    # since the base model (IID here) is in fact a valid Shared effect.
    with pytest.raises(TypeError, match="Weighted"):
        Shared(Weighted(IID("u", index="district"), by="z"))


def test_shared_rejects_a_copy_at_construction_time():
    # Final-slice finding M3: Copy has both `.name` and `.index` (they proxy
    # its target), so it used to pass the __post_init__ checks above and only
    # fail at compile time -- asymmetric with Weighted(Copy(...)), which is a
    # TypeError at construction. A copy folds into an existing target block
    # and produces none of its own, so Joint has no target for a shared copy
    # to fold into either.
    with pytest.raises(TypeError, match="Copy"):
        Shared(Copy("u", index="j"))


def test_mixed_likelihoods_are_allowed():
    # A Joint duplicating test_joint_exposes_outcomes_in_declaration_order's
    # assertion would prove nothing about "mixed": this one actually compiles a
    # Gaussian + Poisson joint and checks the resulting CompiledMixture really
    # dispatches two different likelihood types, not just two outcome names.
    joint = Joint([
        LGM(response="cd4", likelihood=Gaussian(sigma=1.0), predictor=Fixed("1")),
        LGM(response="event", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    assert joint.outcomes == ("cd4", "event")

    frame = pd.DataFrame({
        "cd4": [1.0, 2.0, 3.0, None, None, None],
        "event": [None, None, None, 1.0, 0.0, 2.0],
        "row": range(6),
    })

    def panel(response):
        sub = frame[frame[response].notna()].reset_index(drop=True)
        return CanonicalPanel.from_frame(sub, DataConfig(time="row", response=response, panel=()))

    panels = {"cd4": panel("cd4"), "event": panel("event")}
    compiled = compile_joint(joint, panels)

    assert isinstance(compiled.likelihood, CompiledMixture)
    kinds = {type(part_likelihood) for _, part_likelihood in compiled.likelihood.parts}
    assert kinds == {CompiledGaussian, CompiledPoisson}
