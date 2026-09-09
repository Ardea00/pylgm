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


def test_joint_fit_drops_each_submodels_nan_response_rows():
    """Pins the documented asymmetry with LGM.fit so it cannot drift silently.

    LGM.fit keeps a NaN-response row as unobserved-but-fitted. Joint.fit drops
    it, because in the long-stacked layout joint models are normally given, a
    NaN means "this row belongs to another outcome" rather than "hold this
    observation out" -- keeping them would double the stacked design and fit
    observations that do not exist. See docs/joint-models.md, Not supported yet.
    """
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

    # 4 observed count_a rows + 4 observed count_b rows, not 2 * 5.
    assert len(result.predictive_mean) == 8
    assert np.all(np.isfinite(result.predictive_mean))

    # The canonical long-stacked layout therefore stays at one row per
    # (outcome, unit) pair rather than inflating to two.
    n = 6
    long_frame = pd.DataFrame({
        "district": list(range(n)) * 2,
        "oral": list(np.arange(1.0, n + 1)) + [np.nan] * n,
        "larynx": [np.nan] * n + list(np.arange(1.0, n + 1)),
        "row": range(2 * n),
    })
    long_joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0), scale=(1.0, 1.0))],
    )
    assert len(long_joint.fit(long_frame, engine="laplace").predictive_mean) == 2 * n


def test_mixture_restrict_reslices_bound_aux_on_unobserved_rows():
    """Finding I3, exercised where it is still reachable.

    compile_joint binds a part's aux (Binomial's per-row `trials`) over that
    sub-model's whole sub-frame. CompiledMixture.restrict must re-slice that aux
    alongside the masks; re-indexing only the masks leaves the aux at full
    length and the Laplace engine dies with a broadcast error.

    Joint.fit no longer produces unobserved rows (it drops them, see the test
    above), so this goes through compile_joint + fit_laplace directly -- the
    path that still admits an unobserved row, and the one a future change to
    the row-handling policy would re-expose.
    """
    from pylgm.compiler import compile_joint
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.inference.laplace import fit_laplace

    binomial = pd.DataFrame({
        "y_bin": [3.0, 5.0, np.nan, 2.0],       # row 2 unobserved
        "n_bin": [10.0, 10.0, 10.0, 10.0],
        "row": range(4),
    })
    poisson = pd.DataFrame({"y_pois": [1.0, 2.0, 3.0, 4.0], "row": range(4)})
    panels = {
        "y_bin": CanonicalPanel.from_frame(
            binomial, DataConfig(time="row", response="y_bin", panel=())),
        "y_pois": CanonicalPanel.from_frame(
            poisson, DataConfig(time="row", response="y_pois", panel=())),
    }
    joint = Joint([
        LGM(response="y_bin", likelihood=Binomial(trials="n_bin"), predictor=Fixed("1")),
        LGM(response="y_pois", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    compiled = compile_joint(joint, panels)
    assert not compiled.observed.all(), "fixture must carry an unobserved row"

    result = fit_laplace(compiled)
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.predictive_mean) == 8


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
