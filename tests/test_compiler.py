import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from typing import cast

from pylgm import AR1, Bernoulli, BYM2, Fixed, Gaussian, IID, LGM, Poisson, ProperCAR
from pylgm.effects import Predictor
from pylgm.compiler import compile_family, compile_gaussian_family, compile_lgm, compile_model
from pylgm.config.experiment import EvaluationConfig, ExperimentDataConfig, OriginConfig
from pylgm.config.schema import DataConfig, RunConfig
from pylgm.data import CanonicalPanel
from pylgm.evaluation import build_fold_definitions, materialize_fold
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.ir import CompiledFamily, LatentBlock
from pylgm.likelihoods import CompiledPoisson
from pylgm.parameters import Hyperparameter


def test_compiler_assembles_named_blocks() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y", "panel": ["region"]},
            "model": {
                "fixed": "1 + x",
                "fixed_prior_precision": 0.5,
                "sigma": 1.5,
                "effects": [
                    {
                        "name": "group",
                        "type": "iid",
                        "index": "group",
                        "precision": 3.0,
                    },
                    {
                        "name": "trend",
                        "type": "rw2",
                        "index": "month",
                        "precision": 2.0,
                    },
                ],
            },
        }
    )
    frame = pd.DataFrame(
        {
            "month": [1, 2, 3],
            "region": ["A", "A", "A"],
            "group": ["B", "A", "B"],
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, None, 3.0],
        }
    )
    panel = CanonicalPanel.from_frame(frame, config.data)

    model = compile_model(config, panel)

    assert [block.name for block in model.blocks] == ["fixed", "group", "trend"]
    assert model.labels == (
        "fixed:Intercept",
        "fixed:x",
        "group:A",
        "group:B",
        "trend:1",
        "trend:2",
        "trend:3",
    )
    np.testing.assert_allclose(
        model.design.toarray(),
        [[1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [1, 2, 0, 1, 0, 0, 1]],
    )
    np.testing.assert_allclose(
        model.precision.toarray(),
        [
            [0.5, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 2, -4, 2],
            [0, 0, 0, 0, -4, 8, -4],
            [0, 0, 0, 0, 2, -4, 2],
        ],
    )
    np.testing.assert_allclose(
        model.constraints,
        [[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, -1, 0, 1]],
    )
    assert model.y.tolist() == [1.0, 0.0, 3.0]
    assert model.observed.tolist() == [True, False, True]
    assert model.sigma == 1.5


def test_compiler_rejects_duplicate_qualified_labels() -> None:
    with pytest.raises(ValueError, match="reserved"):
        RunConfig.model_validate(
            {
                "schema_version": 1,
                "data": {"time": "month", "response": "y"},
                "model": {
                    "sigma": 1.0,
                    "effects": [
                        {
                            "name": "fixed",
                            "type": "iid",
                            "index": "group",
                            "precision": 1.0,
                        }
                    ],
                },
            }
        )


def _config_with_effect_index(index: str = "group") -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {
                "fixed": "1 + x",
                "sigma": 1.0,
                "effects": [{"name": "group", "type": "iid", "index": index}],
            },
        }
    )


def test_compiler_rejects_panel_missing_configured_data_columns() -> None:
    config = _config_with_effect_index()
    panel = CanonicalPanel(
        pd.DataFrame({"x": [0.0], "group": ["A"]}),
        np.array([True]),
        ("month",),
        "y",
    )

    with pytest.raises(DataContractError, match="missing configured data columns"):
        compile_model(config, panel)


def test_compiler_rejects_missing_effect_index_with_typed_error() -> None:
    config = _config_with_effect_index("missing_group")
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "y": [1.0], "x": [0.0]}), config.data
    )

    with pytest.raises(DataContractError, match="effect index.*missing_group"):
        compile_model(config, panel)


@pytest.mark.parametrize("formula", ["1 + missing", "1 + ("])
def test_compiler_wraps_fixed_formula_failures(formula: str) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": formula, "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1], "y": [1.0]}), config.data)

    with pytest.raises(CompilationError, match="fixed formula"):
        compile_model(config, panel)


def test_compiler_rejects_fixed_formula_row_dropping() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": "1 + x", "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0], "x": [0.0, None]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="missing covariates|row"):
        compile_model(config, panel)


def test_compiler_rejects_block_with_wrong_panel_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": "1", "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), config.data)
    bad_block = LatentBlock(
        "bad",
        ("level",),
        csr_matrix([[1.0]]),
        csr_matrix([[1.0]]),
        np.empty((0, 1)),
    )
    monkeypatch.setattr("pylgm.compiler._structured_blocks", lambda config, panel: [bad_block])

    with pytest.raises(CompilationError, match="row count"):
        compile_model(config, panel)


class _MetadataOnlyPanel:
    def __init__(self, response: str, key_columns: tuple[str, ...]) -> None:
        self.response = response
        self.key_columns = key_columns

    @property
    def frame(self) -> pd.DataFrame:
        raise AssertionError("frame accessed before panel metadata validation")


def test_compiler_rejects_panel_response_metadata_mismatch_before_frame_access() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y"},
            "model": {"sigma": 1.0},
        }
    )
    panel = cast(CanonicalPanel, _MetadataOnlyPanel("other_y", ("time",)))

    with pytest.raises(DataContractError, match="response metadata"):
        compile_model(config, panel)


def test_compiler_rejects_panel_key_metadata_mismatch_before_frame_access() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y", "panel": ["region"]},
            "model": {"sigma": 1.0},
        }
    )
    panel = cast(CanonicalPanel, _MetadataOnlyPanel("y", ("time", "region")))

    with pytest.raises(DataContractError, match="key metadata"):
        compile_model(config, panel)


def test_qualified_label_collision_is_a_compilation_error() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y"},
            "model": {
                "fixed": "0",
                "sigma": 1.0,
                "effects": [
                    {"name": "a:b", "type": "iid", "index": "first"},
                    {"name": "a", "type": "iid", "index": "second"},
                ],
            },
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"time": [1], "y": [1.0], "first": ["c"], "second": ["b:c"]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="duplicate latent labels"):
        compile_model(config, panel)


@pytest.mark.parametrize("optimized", [("unknown",), ("sigma", "sigma"), (1,)])
def test_gaussian_family_compiler_rejects_invalid_optimized_names(
    optimized: tuple[object, ...],
) -> None:
    config = _config_with_effect_index()
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "y": [1.0], "x": [0.0], "group": ["A"]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="optimized parameter names|unknown"):
        compile_gaussian_family(
            config.data,
            config.model,
            panel,
            optimized=optimized,  # type: ignore[arg-type]
        )


def test_nonlexical_categorical_order_aligns_folds_labels_and_rw2_precision() -> None:
    data = ExperimentDataConfig(time="phase", response="y")
    evaluation = EvaluationConfig(horizons=(2,), origins=OriginConfig(values=("middle",)))
    frame = pd.DataFrame(
        {
            "phase": pd.Categorical(
                ["early", "middle", "late"],
                categories=["middle", "late", "unused", "early"],
                ordered=True,
            ),
            "y": [3.0, 1.0, 2.0],
        }
    )
    definition = build_fold_definitions(frame, data, evaluation)[0]
    fold = materialize_fold(frame, data, evaluation, definition)
    model = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "phase", "response": "y"},
            "model": {
                "fixed": "1",
                "sigma": 1.0,
                "effects": [{"name": "trend", "type": "rw2", "index": "phase", "precision": 2.0}],
            },
        }
    )

    compiled = compile_model(model, CanonicalPanel.from_frame(fold.model_frame, model.data))
    trend = compiled.blocks[1]

    assert definition.origin == "middle"
    assert definition.target == "early"
    assert trend.labels == ("middle", "late", "early")
    np.testing.assert_allclose(
        trend.precision.toarray(),
        [[2, -4, 2], [-4, 8, -4], [2, -4, 2]],
    )


def test_compiler_translates_unordered_rw_categorical_index_to_typed_error() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "phase", "response": "y"},
            "model": {
                "sigma": 1.0,
                "effects": [{"name": "trend", "type": "rw1", "index": "phase"}],
            },
        }
    )
    frame = pd.DataFrame(
        {
            "phase": pd.Categorical(
                ["middle", "late"], categories=["middle", "late"], ordered=False
            ),
            "y": [1.0, 2.0],
        }
    )

    with pytest.raises(CompilationError, match="categorical.*ordered"):
        compile_model(config, CanonicalPanel.from_frame(frame, config.data))


@pytest.mark.parametrize("error", [RuntimeError("fixed bug"), MemoryError("fixed oom")])
def test_unexpected_fixed_builder_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1], "y": [1.0]}), config.data)

    def broken_fixed(*args: object, **kwargs: object) -> LatentBlock:
        raise error

    monkeypatch.setattr("pylgm.compiler.build_fixed", broken_fixed)
    with pytest.raises(type(error), match="fixed"):
        compile_model(config, panel)


@pytest.mark.parametrize("error", [RuntimeError("effect bug"), MemoryError("effect oom")])
def test_unexpected_effect_builder_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {
                "sigma": 1.0,
                "effects": [{"name": "group", "type": "iid", "index": "group"}],
            },
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "group": ["A"], "y": [1.0]}), config.data
    )

    def broken_effect(*args: object, **kwargs: object) -> LatentBlock:
        raise error

    monkeypatch.setattr("pylgm.compiler.build_iid", broken_effect)
    with pytest.raises(type(error), match="effect"):
        compile_model(config, panel)


def _panel(frame):
    return CanonicalPanel.from_frame(frame, DataConfig(time="t", response="y", panel=()))


def test_compile_poisson_uses_non_gaussian_likelihood_and_offset():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 4.0], "logexp": [0.0, 0.1, 0.2]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t", offset="logexp")
    compiled = compile_lgm(model, _panel(frame))
    assert isinstance(compiled.likelihood, CompiledPoisson)
    np.testing.assert_allclose(compiled.offset, [0.0, 0.1, 0.2])


def test_compile_poisson_rejects_non_count_response():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 1.5]})
    model = LGM("y", Poisson(), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="non-negative integer"):
        compile_lgm(model, _panel(frame))


def test_compile_bernoulli_default_offset_is_zero():
    frame = pd.DataFrame({"t": [1, 2], "y": [0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1"), time="t")
    compiled = compile_lgm(model, _panel(frame))
    np.testing.assert_allclose(compiled.offset, [0.0, 0.0])


def test_compile_rejects_missing_offset_column():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    model = LGM("y", Poisson(), Fixed("1"), time="t", offset="does_not_exist")
    with pytest.raises(DataContractError, match="offset column not found"):
        compile_lgm(model, _panel(frame))


def test_compile_rejects_empty_predictor_for_non_gaussian():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    model = LGM("y", Poisson(), Predictor(()), time="t")
    with pytest.raises(CompilationError, match="model must contain at least one latent effect"):
        compile_lgm(model, _panel(frame))


def test_compile_family_returns_none_without_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"), time="t")
    assert compile_family(model, _panel(frame)) is None


def test_compile_family_binds_declared_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "region": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})
    model = LGM(
        "y", Gaussian(1.0),
        Fixed("1") + IID("region", index="region", precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    family = compile_family(model, CanonicalPanel.from_frame(frame, DataConfig(time="t", response="y", panel=("region",))))
    assert isinstance(family, CompiledFamily)
    assert family.parameter_names == ("region_prec",)
    compiled = family.materialize({"region_prec": 2.0})
    assert compiled.y.shape[0] == 4
    region_block = next(block for block in compiled.blocks if block.name == "region")
    np.testing.assert_allclose(region_block.precision.toarray(), 2.0 * np.eye(2))


def test_unrecognized_effect_is_rejected_not_compiled_as_a_random_walk():
    # The dispatch chains used to end in a bare `else` that built ANY unhandled
    # effect as an RW2. That silently mis-compiles a new effect type instead of
    # failing -- it is exactly how BYM2 appeared to "work" before it was wired.
    import pandas as pd

    from pylgm import Fixed, Gaussian, Hyperparameter, LGM
    from pylgm.compiler import compile_family, compile_lgm
    from pylgm.effects.spec import Predictor, _ComposableEffect
    from pylgm.exceptions import CompilationError

    class _UnknownEffect(_ComposableEffect):
        name = "mystery"
        index = "region"
        # a declared Hyperparameter so compile_family reaches its dispatch loop
        precision = Hyperparameter("mystery.precision", initial=1.0)

    frame = pd.DataFrame({"region": ["a", "b", "c"], "y": [0.1, 0.2, 0.3]})
    model = LGM(response="y", predictor=Fixed("1"), likelihood=Gaussian(sigma=0.5))
    panel = CanonicalPanel(frame, np.array([True, True, True]), ("region",), "y")
    # bypass Predictor's closed membership check the way an unwired effect would
    object.__setattr__(model, "predictor", Predictor.__new__(Predictor))
    object.__setattr__(model.predictor, "effects", (Fixed("1"), _UnknownEffect()))

    with pytest.raises(CompilationError, match="unsupported effect type"):
        compile_lgm(model, panel)
    with pytest.raises(CompilationError, match="unsupported effect type"):
        compile_family(model, panel)


def test_ar1_builder_failure_is_a_compilation_error_via_both_compile_paths() -> None:
    """A single-level AR1 fails the same way whether or not a Hyperparameter is declared.

    ``compile_lgm`` wraps builder failures as ``CompilationError``; before this
    fix, ``compile_family`` had no equivalent wrapping, so the identical
    malformed effect raised a bare ``ValueError`` there merely because a
    precision ``Hyperparameter`` routes it through ``compile_family`` too.
    """
    frame = pd.DataFrame({"t": [1], "y": [1.0]})
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.5),
        predictor=Fixed("1") + AR1(
            "trend", index="t", precision=Hyperparameter("trend.precision", initial=1.0), rho=0.5
        ),
    )
    panel = _panel(frame)
    with pytest.raises(CompilationError, match="failed to compile effect 'trend'"):
        compile_family(model, panel)
    with pytest.raises(CompilationError, match="failed to compile effect 'trend'"):
        compile_lgm(model, panel)


def test_build_prediction_context_matches_the_compiled_blocks():
    import pandas as pd

    from pylgm import Fixed, Gaussian, IID, LGM, RW1
    from pylgm.compiler import build_prediction_context, compile_lgm
    from pylgm.data import CanonicalPanel

    frame = pd.DataFrame({
        "y": [0.1, 0.2, 0.3, 0.4],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
        "t": [0, 1, 2, 3],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region") + RW1("trend", index="t"),
        likelihood=Gaussian(sigma=0.5),
    )
    panel = CanonicalPanel(frame, np.array([True] * 4), ("region",), "y")
    compiled = compile_lgm(model, panel)
    context = build_prediction_context(model, panel, compiled)

    assert context.width == compiled.design.shape[1]
    kinds = [kind for kind, _ in context.entries]
    assert kinds == ["fixed", "structured", "structured"]
    # structured entries carry the compiled labels, in compiled block order
    structured = [payload for kind, payload in context.entries if kind == "structured"]
    assert structured[0][:2] == ("region", "region")
    assert structured[0][2] == compiled.blocks[1].labels
    assert structured[1][:2] == ("trend", "t")
    assert structured[1][2] == compiled.blocks[2].labels


def test_prediction_context_guard_catches_misordered_entries():
    # predict_from only validates the TOTAL design width, so entries built in a
    # different order than the compiled blocks would misassign columns with no
    # error. Two same-width blocks make the width check useless; the label
    # alignment guard is what must catch it.
    import pandas as pd

    from pylgm import Fixed, Gaussian, IID, LGM
    from pylgm.compiler import build_prediction_context, compile_lgm
    from pylgm.data import CanonicalPanel
    from pylgm.exceptions import CompilationError

    frame = pd.DataFrame({
        "y": [0.1, 0.2, 0.3, 0.4],
        "region": ["a", "b", "a", "b"],
        "grp": ["p", "q", "p", "q"],
    })
    panel = CanonicalPanel(frame, np.array([True] * 4), ("region",), "y")
    ordered = LGM(
        response="y",
        predictor=Fixed("1") + IID("region", index="region") + IID("grp", index="grp"),
        likelihood=Gaussian(sigma=0.5),
    )
    compiled = compile_lgm(ordered, panel)
    assert [b.name for b in compiled.blocks] == ["fixed", "region", "grp"]
    # happy path still works
    build_prediction_context(ordered, panel, compiled)

    swapped = LGM(
        response="y",
        predictor=Fixed("1") + IID("grp", index="grp") + IID("region", index="region"),
        likelihood=Gaussian(sigma=0.5),
    )
    with pytest.raises(CompilationError, match="order"):
        build_prediction_context(swapped, panel, compiled)


def _chain_graph(n: int) -> dict[str, list[str]]:
    return {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}


def _region_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": [0.1, -0.2, 0.05, 0.3][:n]})


def _time_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"t": list(range(n)), "y": [0.1, -0.2, 0.05, 0.3][:n]})


def _proper_car_model(rho: Hyperparameter) -> LGM:
    return LGM(
        response="y",
        predictor=Fixed("1") + ProperCAR("region", index="region", graph=_chain_graph(4), rho=rho),
        likelihood=Gaussian(sigma=0.1),
    )


def _bym2_model(phi: Hyperparameter) -> LGM:
    return LGM(
        response="y",
        predictor=Fixed("1") + BYM2("region", index="region", graph=_chain_graph(4), phi=phi),
        likelihood=Gaussian(sigma=0.1),
    )


def _ar1_model(rho: Hyperparameter) -> LGM:
    return LGM(
        response="y",
        predictor=Fixed("1") + AR1("trend", index="t", rho=rho),
        likelihood=Gaussian(sigma=0.1),
    )


# Safety net for extracting the shared bounded-parameter bounds construction
# (proper CAR's rho, BYM2's phi, AR1's rho): each is a hyperparameter confined
# to an open interval on a logit scale. These pin the behaviour -- reject a
# non-logit transform naming the effect, reject an out-of-interval initial
# naming the effect and the interval -- through the real LGM.fit path, so an
# extraction that changes behaviour (not just wording) fails loudly.
@pytest.mark.parametrize(
    "build_model, frame_factory, label",
    [
        (_proper_car_model, _region_frame, "proper CAR rho"),
        (_bym2_model, _region_frame, "BYM2 phi"),
        (_ar1_model, _time_frame, "AR1 rho"),
    ],
)
def test_bounded_parameter_rejects_non_logit_transform(
    build_model, frame_factory, label: str
) -> None:
    hp = Hyperparameter("bounded", initial=0.5)  # transform defaults to "log"
    with pytest.raises(CompilationError, match=f"{label}.*transform='logit'"):
        build_model(hp).fit(frame_factory())


@pytest.mark.parametrize(
    "build_model, frame_factory, label",
    [
        (_proper_car_model, _region_frame, "proper CAR rho"),
        (_bym2_model, _region_frame, "BYM2 phi"),
        (_ar1_model, _time_frame, "AR1 rho"),
    ],
)
def test_bounded_parameter_rejects_out_of_interval_initial(
    build_model, frame_factory, label: str
) -> None:
    hp = Hyperparameter("bounded", initial=5.0, transform="logit")
    with pytest.raises(CompilationError, match=f"{label}.*must lie in"):
        build_model(hp).fit(frame_factory())
