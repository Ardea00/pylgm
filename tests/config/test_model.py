from pathlib import Path

import pytest

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson, RW1
from pylgm.config import load_model
from pylgm.exceptions import ConfigurationError


def test_load_model_builds_the_same_declaration_as_python(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
response: y
likelihood: {family: gaussian, sigma: 0.5}
data: {panel: [region], time: time}
predictor:
  fixed: "1 + x"
  effects:
    - {name: region_effect, type: iid, index: region, precision: 2.0}
    - {name: trend, type: rw1, index: time, precision: 3.0}
""".strip()
    )

    assert load_model(path) == LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1 + x")
        + IID("region_effect", "region", 2.0)
        + RW1("trend", "time", 3.0),
        panel=("region",),
        time="time",
    )


def test_load_model_forbids_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1'}\nunknown: true"
    )

    with pytest.raises(ConfigurationError, match="unknown"):
        load_model(path)


@pytest.mark.parametrize(
    "document",
    [
        "response: y\nlikelihood: {family: gaussian}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 0}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: .inf}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: true}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: region, type: iid, index: region}]",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: trend, type: rw2, index: time, precision: -1}]",
    ],
)
def test_load_model_requires_finite_positive_explicit_hyperparameters(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(document)

    with pytest.raises(ConfigurationError, match="sigma|precision|finite|greater than 0|Field required"):
        load_model(path)


@pytest.mark.parametrize(
    "document",
    [
        "response: y\nlikelihood: {family: poisson, sigma: 1}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor:\n  fixed: '1'\n  effects: [{name: trend, type: ar1, index: time, precision: 1}]",
        "response: y\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1', precision: 1}",
    ],
)
def test_load_model_allows_only_the_supported_yaml_vocabulary(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(document)

    with pytest.raises(ConfigurationError):
        load_model(path)


def test_load_model_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "response: y\nresponse: other\nlikelihood: {family: gaussian, sigma: 1}\npredictor: {fixed: '1'}"
    )

    with pytest.raises(ConfigurationError, match="duplicate mapping key"):
        load_model(path)


def test_load_poisson_model_with_offset(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: poisson\n"
        "offset: logexp\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1 + x'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Poisson)
    assert model.offset == "logexp"


def test_load_bernoulli_model(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: bernoulli\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Bernoulli)
    assert model.offset is None


@pytest.mark.parametrize(
    "family_name, cls_name",
    [("nbinomial", "NegativeBinomial"), ("gamma", "Gamma"), ("beta", "Beta")],
)
def test_load_phi_family_models(tmp_path, family_name, cls_name):
    import pylgm

    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        f"likelihood: {{family: {family_name}, phi: 2.5}}\n"
        "data: {time: t}\n"
        "predictor: {fixed: '1'}\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, getattr(pylgm, cls_name))
    assert model.likelihood.phi == 2.5


def test_load_phi_family_defaults_phi_when_omitted(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\nlikelihood: {family: gamma}\ndata: {time: t}\npredictor: {fixed: '1'}\n"
    )
    assert load_model(path).likelihood.phi == 1.0


@pytest.mark.parametrize(
    "document",
    [
        # sigma is gaussian-only
        "response: y\nlikelihood: {family: gamma, sigma: 1.0}\npredictor: {fixed: '1'}",
        # phi is not valid for poisson/bernoulli/gaussian
        "response: y\nlikelihood: {family: poisson, phi: 2.0}\npredictor: {fixed: '1'}",
        "response: y\nlikelihood: {family: gaussian, sigma: 1, phi: 2.0}\npredictor: {fixed: '1'}",
        # phi must be finite-positive
        "response: y\nlikelihood: {family: beta, phi: 0}\npredictor: {fixed: '1'}",
    ],
)
def test_load_model_rejects_invalid_phi_usage(tmp_path, document):
    path = tmp_path / "m.yaml"
    path.write_text(document)
    with pytest.raises(ConfigurationError):
        load_model(path)


_SPATIAL_GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
_INLINE_GRAPH = "{a: [b], b: [a, c], c: [b]}"


def _spatial_doc(effect_body: str) -> str:
    return (
        "response: y\n"
        "likelihood: {family: gaussian, sigma: 1.0}\n"
        "data: {time: time}\n"
        "predictor:\n"
        "  fixed: '1'\n"
        "  effects:\n"
        f"    - {{{effect_body}}}\n"
    )


def _spatial_lgm(effect):
    return LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=Fixed("1") + effect,
        time="time",
    )


def test_load_model_builds_besag_from_inline_graph(tmp_path: Path) -> None:
    from pylgm import Besag

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: besag, index: region, precision: 2.0, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(Besag("area", "region", _SPATIAL_GRAPH, 2.0))


def test_load_model_builds_besag_from_graph_file(tmp_path: Path) -> None:
    from pylgm import Besag, load_graph_file

    graph_path = tmp_path / "nb.graph"
    graph_path.write_text("3\n1 1 2\n2 2 1 3\n3 1 2\n")  # path graph 1-2-3
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc("name: area, type: besag, index: region, graph_file: nb.graph"))

    assert load_model(path) == _spatial_lgm(
        Besag("area", "region", load_graph_file(graph_path), 1.0)
    )


def test_load_model_builds_proper_car_with_fixed_rho(tmp_path: Path) -> None:
    from pylgm import ProperCAR

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: proper_car, index: region, rho: 0.4, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(ProperCAR("area", "region", _SPATIAL_GRAPH, 0.4))


def test_load_model_builds_bym2_with_default_phi(tmp_path: Path) -> None:
    from pylgm import BYM2

    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: bym2, index: region, graph: {_INLINE_GRAPH}")
    )

    assert load_model(path) == _spatial_lgm(BYM2("area", "region", _SPATIAL_GRAPH))


def test_load_model_builds_weighted_graph_bym2(tmp_path: Path) -> None:
    from pylgm import BYM2

    weighted = {"a": {"b": 5.0}, "b": {"a": 5.0, "c": 0.1}, "c": {"b": 0.1}}
    inline = "{a: {b: 5.0}, b: {a: 5.0, c: 0.1}, c: {b: 0.1}}"
    path = tmp_path / "m.yaml"
    path.write_text(
        _spatial_doc(f"name: area, type: bym2, index: region, graph: {inline}")
    )

    # YAML weighted mapping builds the identical spec as the Python API, hence
    # the same fit — the weighted graph rides through untouched (S4).
    assert load_model(path) == _spatial_lgm(BYM2("area", "region", weighted))


@pytest.mark.parametrize(
    "effect_body, match",
    [
        # proper_car without its required rho
        (f"name: area, type: proper_car, index: region, graph: {_INLINE_GRAPH}", "rho"),
        # rho on a non-proper_car spatial effect
        (f"name: area, type: besag, index: region, rho: 0.5, graph: {_INLINE_GRAPH}", "rho"),
        # graph field on a simple effect
        (f"name: r, type: iid, index: region, precision: 1.0, graph: {_INLINE_GRAPH}", "not valid"),
        # spatial effect with neither graph nor graph_file
        ("name: area, type: besag, index: region", "exactly one"),
        # spatial effect with both graph and graph_file
        (f"name: area, type: besag, index: region, graph_file: x.graph, graph: {_INLINE_GRAPH}", "exactly one"),
    ],
)
def test_load_model_rejects_mismatched_spatial_fields(
    tmp_path: Path, effect_body: str, match: str
) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc(effect_body))

    with pytest.raises(ConfigurationError, match=match):
        load_model(path)


def test_load_model_reports_missing_graph_file(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(_spatial_doc("name: area, type: besag, index: region, graph_file: nope.graph"))

    with pytest.raises(ConfigurationError):
        load_model(path)


def test_load_binomial_model(tmp_path: Path) -> None:
    from pylgm import Binomial

    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: binomial\n  trials: n\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Binomial)
    assert model.likelihood.trials == "n"


@pytest.mark.parametrize(
    "likelihood_body, match",
    [
        # binomial without its required trials column
        ("family: binomial", "trials is required"),
        # trials on a non-binomial family
        ("family: poisson\n  trials: n", "trials is not a valid"),
    ],
)
def test_load_model_rejects_invalid_trials_usage(
    tmp_path: Path, likelihood_body: str, match: str
) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        f"likelihood:\n  {likelihood_body}\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    with pytest.raises(ConfigurationError, match=match):
        load_model(path)


def test_sar_config_builds_effect():
    from pylgm.config.model import _EffectModelConfig, _build_effect
    from pathlib import Path
    from pylgm.effects.spec import SAR

    config = _EffectModelConfig(
        name="s", type="sar", index="region",
        graph={"a": ["b"], "b": ["a"]}, rho=0.4,
    )
    effect = _build_effect(config, Path("."))
    assert isinstance(effect, SAR)
    assert effect.index == "region"


def test_sar_config_requires_rho():
    from pylgm.config.model import _EffectModelConfig

    with pytest.raises(ValueError, match="rho"):
        _EffectModelConfig(name="s", type="sar", index="region", graph={"a": ["b"]})


def test_weibullsurv_config_builds_model(tmp_path: Path) -> None:
    from pylgm import WeibullSurv

    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: weibullsurv\n  event: d\n  shape: 1.5\n  entry: v\n"
        "data:\n  time: t\n  panel: [region]\n"
        "predictor:\n  fixed: '1'\n"
        "  effects: [{name: b, type: iid, index: region, precision: 1.0}]\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, WeibullSurv)
    assert model.likelihood.event == "d"
    assert model.likelihood.shape == 1.5
    assert model.likelihood.entry == "v"


def test_exponentialsurv_config_rejects_shape(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: exponentialsurv\n  event: d\n  shape: 2.0\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    with pytest.raises(ConfigurationError, match="shape"):
        load_model(path)
